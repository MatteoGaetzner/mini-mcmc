//! Batch-native HMC using the BatchVector trait for zero GPU regression.
//!
//! This module provides a truly batch-native HMC implementation where the entire
//! batch of chains is treated as a single vector in phase space.

use crate::euclidean::BatchVector;
use crate::stats::RunStats;
use ndarray::Array3;
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use rand::distr::Distribution as RandDistribution;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::error::Error;

/// A batched target density that returns per-chain log-probabilities.
///
/// This trait enables GPU-parallel execution by computing gradients for
/// all chains simultaneously.
pub trait BatchedHamiltonianTarget<V: BatchVector> {
    /// Returns per-chain log-densities and writes per-chain gradients into `grad`.
    /// For Tensor<B, 2>: position is [n_chains, dim], returns [n_chains] log-probs.
    fn logp_and_grad(&self, position: &V, grad: &mut V) -> V::Energy;
}

/// Batch-native HMC engine where "The Batch IS the Particle".
///
/// This struct stores all chains as a single batch vector `V`. For GPU backends,
/// this enables parallel execution without serialization.
#[derive(Clone, Debug)]
pub struct BatchedGenericHMC<V, Target>
where
    V: BatchVector,
    Target: BatchedHamiltonianTarget<V>,
{
    target: Target,
    step_size: V::Scalar,
    n_leapfrog: usize,
    /// ALL chains stored as single batch [n_chains, dim]
    position: V,
    /// Momentum buffer [n_chains, dim]
    momentum: V,
    /// Gradient buffer [n_chains, dim]
    grad: V,
    /// Proposal position buffer [n_chains, dim]
    proposal_pos: V,
    /// Proposal momentum buffer [n_chains, dim]
    proposal_mom: V,
    rng: SmallRng,
    n_chains: usize,
    dim: usize,
}

type RunResult<T> = Result<(Array3<T>, RunStats), Box<dyn Error + Send + Sync>>;

impl<V, Target> BatchedGenericHMC<V, Target>
where
    V: BatchVector,
    V::Scalar: Float + FromPrimitive + ToPrimitive + Zero,
    Target: BatchedHamiltonianTarget<V> + Clone,
    StandardNormal: RandDistribution<V::Scalar>,
{
    /// Create a new batch-native HMC sampler.
    ///
    /// `initial_position` is a batch tensor of shape [n_chains, dim].
    pub fn new(
        target: Target,
        initial_position: V,
        step_size: V::Scalar,
        n_leapfrog: usize,
    ) -> Self {
        let n_chains = initial_position.n_chains();
        let dim = initial_position.dim_per_chain();

        Self {
            target,
            step_size,
            n_leapfrog,
            momentum: initial_position.zeros_like(),
            grad: initial_position.zeros_like(),
            proposal_pos: initial_position.zeros_like(),
            proposal_mom: initial_position.zeros_like(),
            position: initial_position,
            rng: SmallRng::from_rng(&mut rand::rng()),
            n_chains,
            dim,
        }
    }

    /// Set the random seed for reproducibility.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    /// Run the sampler, collecting `n_collect` samples after discarding `n_discard`.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array3<V::Scalar> {
        // Burn-in
        (0..n_discard).for_each(|_| self.step());

        // Collect samples
        let mut out = Array3::<V::Scalar>::zeros((self.n_chains, n_collect, self.dim));
        let mut scratch = vec![V::Scalar::zero(); self.n_chains * self.dim];

        for step_idx in 0..n_collect {
            self.step();
            self.position.write_to_slice(&mut scratch);
            // Copy from flat slice to [n_chains, n_collect, dim]
            for chain_idx in 0..self.n_chains {
                for d in 0..self.dim {
                    out[[chain_idx, step_idx, d]] = scratch[chain_idx * self.dim + d];
                }
            }
        }
        out
    }

    /// Perform one HMC step on ALL chains simultaneously.
    ///
    /// This is GPU-parallel: no loops over chains, uses device-native RNG,
    /// and vectorized acceptance via masking.
    pub fn step(&mut self) {
        // 1. Sample momentum for all chains (device-native RNG)
        self.momentum.fill_random_normal(&mut self.rng);

        // 2. Compute current kinetic energy [n_chains]
        let ke_current = self.momentum.kinetic_energy();

        // 3. Compute current potential energy [n_chains]
        self.grad.fill_zero();
        let logp_current = self.target.logp_and_grad(&self.position, &mut self.grad);

        // 4. Copy current state to proposal buffers
        self.proposal_pos.assign(&self.position);
        self.proposal_mom.assign(&self.momentum);

        // 5. Leapfrog integration (all chains in parallel)
        let logp_proposed = self.leapfrog();

        // 6. Compute proposed kinetic energy [n_chains]
        let ke_proposed = self.proposal_mom.kinetic_energy();

        // 7. Compute log acceptance probability (element-wise for batch)
        // log_accept = (logp_proposed - logp_current) + (ke_current - ke_proposed)
        let delta_logp = V::energy_sub(&logp_proposed, &logp_current);
        let delta_ke = V::energy_sub(&ke_current, &ke_proposed);
        let log_accept = V::energy_add(&delta_logp, &delta_ke);

        // 8. Sample uniform [0,1] for acceptance test [n_chains]
        let u = self.position.sample_uniform(&mut self.rng);
        let ln_u = V::energy_ln(&u);

        // 9. Create acceptance mask and update positions (GPU-friendly masking)
        let mask = V::accept_mask(&log_accept, &ln_u);
        self.position.masked_assign(&self.proposal_pos, &mask);
    }

    /// Leapfrog integration on the entire batch.
    fn leapfrog(&mut self) -> V::Energy {
        let half = V::Scalar::from_f64(0.5).unwrap() * self.step_size;

        let mut logp = self
            .target
            .logp_and_grad(&self.proposal_pos, &mut self.grad);

        for _ in 0..self.n_leapfrog {
            // Half momentum update
            self.proposal_mom.add_scaled_assign(&self.grad, half);

            // Full position update
            self.proposal_pos
                .add_scaled_assign(&self.proposal_mom, self.step_size);

            // Recompute gradient at new position
            logp = self
                .target
                .logp_and_grad(&self.proposal_pos, &mut self.grad);

            // Half momentum update
            self.proposal_mom.add_scaled_assign(&self.grad, half);
        }
        logp
    }

    /// Get a reference to the current positions.
    pub fn positions(&self) -> &V {
        &self.position
    }

    /// Clone the RNG state.
    pub fn rng_clone(&self) -> SmallRng {
        self.rng.clone()
    }
}
