//! A simple Hamiltonian (Hybrid) Monte Carlo sampler using the `burn` crate for autodiff.
//!
//! This is modeled similarly to your Metropolis–Hastings approach but uses gradient-based proposals.

use burn::prelude::Backend;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use num_traits::Float;
use rand::prelude::*;
use rand::rngs::SmallRng;
use std::marker::PhantomData;

use crate::core::{HasChains, MarkovChain};

/// Trait for distributions that can compute both the unnormalized log-prob *and* its gradient.
///
/// - `log_prob_and_grad` takes a D-dimensional `Tensor` and returns a scalar `Tensor`
///   (the log probability) plus it enables gradient computation via Burn's AD.
pub trait GradientTarget<T: Float, B: AutodiffBackend> {
    /// Evaluates the unnormalized log-prob at `theta`, returning a 1-element `Tensor` that
    /// supports autodiff so we can call `backward()` on it to get the gradient.
    fn log_prob_tensor(&self, theta: &Tensor<B, 1>) -> Tensor<B, 1>;
}

/// A single HMC Markov chain.
///
/// - `target` must implement `GradientTarget` so we can compute log-probs and gradients.
/// - `step_size`: The step size used for leapfrog integration.
/// - `n_leapfrog`: The number of leapfrog steps.
/// - `current_pos`: The current position in parameter space (as a `[D]` 1D vector). We store it as a 2D tensor `[1, D]` for convenience.
/// - `rng`: A chain-specific RNG.
pub struct HMCMarkovChain<T, B, GTarget> {
    pub target: GTarget,
    pub step_size: T,
    pub n_leapfrog: usize,
    pub current_pos: Vec<T>,
    pub rng: SmallRng,

    phantom_data: PhantomData<B>,
}

impl<T, B, GTarget> HMCMarkovChain<T, B, GTarget>
where
    T: Float + burn::tensor::ElementConversion + burn::tensor::Element,
    GTarget: GradientTarget<T, B>,
    B: AutodiffBackend,
{
    /// Create a new chain at the specified `initial_pos` (dimension D),
    /// with given step size and leapfrog steps.
    pub fn new(
        target: GTarget,
        initial_pos: Vec<T>,
        step_size: T,
        n_leapfrog: usize,
        seed: u64,
    ) -> Self {
        let rng = SmallRng::seed_from_u64(seed);
        Self {
            target,
            step_size,
            n_leapfrog,
            current_pos: initial_pos,
            rng,
            phantom_data: PhantomData,
        }
    }

    /// A single HMC leapfrog integration.
    ///
    /// Returns `(new_pos, new_momentum, log_prob_of_new_pos)`.
    fn leapfrog(
        &mut self,
        pos: Tensor<B, 1>,
        mut mom: Tensor<B, 1>,
    ) -> (Tensor<B, 1>, Tensor<B, 1>, T) {
        // Ensure `pos` is AD-enabled and tracking gradients.
        // (Depending on your Burn version, you might have a method like `requires_grad()`.)
        let mut pos = pos.require_grad();

        // --- First half-step momentum update ---
        let logp = self.target.log_prob_tensor(&pos);
        let grads = logp.backward();
        // Here, grad_pos is likely of the inner type. Convert it back into an AD tensor.
        let grad_pos =
            Tensor::<B, 1>::from_data(pos.grad(&grads).unwrap().to_data(), &pos.device());

        // Use Burn’s scalar multiplication and addition methods.
        mom = mom.add(grad_pos.mul_scalar(self.step_size * T::from(0.5).unwrap()));

        // --- Full-step position update ---
        // -log N(mom; 0, I) = - (-0.5 mom^T mom)  + c =>  nabla_mom ... = mom
        pos = pos
            .add(mom.clone().mul_scalar(self.step_size))
            .detach()
            .require_grad();

        // --- Second half-step momentum update ---
        let logp2 = self.target.log_prob_tensor(&pos);
        let grads2 = logp2.backward();
        let grad_pos2 =
            Tensor::<B, 1>::from_data(pos.grad(&grads2).unwrap().to_data(), &pos.device());
        mom = mom.add(grad_pos2.mul_scalar(self.step_size * T::from(0.5).unwrap()));

        // Retrieve the final log-probability for the updated position.
        let final_logp = *logp2
            .to_data()
            .as_slice()
            .expect("Expected being able to convert to slice")
            .first()
            .expect("Expected slice to be non-empty");

        // When you want to save the resulting state, detach it (convert to plain numbers).
        let pos_detached = pos.to_data();
        // Optionally, re-wrap it into an AD tensor for the next iteration:
        let pos_for_next =
            Tensor::<B, 1>::from_data(pos_detached.clone(), &pos.device()).require_grad();

        (pos_for_next, mom, final_logp)
    }
}

/// For chaining in your existing framework, implement `MarkovChain`.
impl<T, B, GTarget> MarkovChain<T> for HMCMarkovChain<T, B, GTarget>
where
    T: Float + burn::tensor::Element + std::iter::Sum,
    GTarget: GradientTarget<T, B>,
    B: AutodiffBackend + Backend,
    rand_distr::StandardNormal: rand_distr::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// One HMC step = sample momentum, run leapfrog, accept/reject.
    fn step(&mut self) -> &Vec<T> {
        // --- Sample initial momentum from N(0, I) ---
        let mut momentum_data: Vec<T> = vec![T::zero(); self.current_pos.len()];
        // println!("New traj");
        for val in momentum_data.iter_mut() {
            // Standard normal
            let z: T = self.rng.sample(rand_distr::StandardNormal);
            *val = z;
        }
        let momentum_0 = Tensor::<B, 1>::from_data(momentum_data.as_slice(), &B::Device::default());

        // Current log-prob
        let current_pos_tensor: Tensor<B, 1> =
            Tensor::<B, 1>::from_floats(self.current_pos.as_slice(), &B::Device::default());

        let logp_current: T = *self
            .target
            .log_prob_tensor(&current_pos_tensor)
            .to_data()
            .as_slice()
            .expect("Expected getting slice to succeed")
            .first()
            .expect("Expected to be able to take first");

        // Kinetic energy of momentum: 0.5 * p^2
        let ke_current =
            T::from(0.5).unwrap() * momentum_data.iter().map(|x| (*x) * (*x)).sum::<T>();
        let h_current = -(logp_current) + ke_current; // negative log-prob + kinetic

        // --- Run leapfrog integrator for n steps ---
        // println!("  {:?}", self.current_state().as_slice());
        let (proposed_pos, proposed_mom, logp_proposed) = (0..self.n_leapfrog).fold(
            (
                Tensor::<B, 1>::from_floats(self.current_pos.as_slice(), &B::Device::default()),
                momentum_0,
                logp_current,
            ),
            |(pos, mom, _), _| {
                let (new_pos, new_mom, new_logp) = self.leapfrog(pos, mom);
                // println!(
                //     "  {:?}, log p(x) = {:?}",
                //     new_pos.to_data().as_slice::<f32>().unwrap(),
                //     self.target
                //         .log_prob_tensor(&new_pos)
                //         .to_data()
                //         .as_slice::<f32>()
                //         .unwrap()
                // );
                (new_pos, new_mom, new_logp)
            },
        );
        // println!("End traj");

        // Proposed kinetic
        let ke_proposed =
            T::from(0.5).unwrap() * proposed_mom.to_data().iter().map(|x: T| x * x).sum::<T>();
        let h_proposed = -(logp_proposed) + ke_proposed;

        // --- Accept/Reject ---
        let accept_logp = -(h_proposed - h_current); // i.e. -ΔH
        let u: T = self.rng.gen();
        if accept_logp >= u.ln() {
            // accept
            self.current_pos = proposed_pos.to_data().to_vec().unwrap();
        }

        self.current_state()
    }

    fn current_state(&self) -> &Vec<T> {
        &self.current_pos
    }
}

/// The top-level HMC sampler that, like Metropolis–Hastings, holds multiple parallel chains.
pub struct HamiltonianSampler<T, B, GTarget> {
    pub target: GTarget,
    pub step_size: T,
    pub n_leapfrog: usize,
    pub chains: Vec<HMCMarkovChain<T, B, GTarget>>,
    pub seed: u64,
    pub phantom_data: PhantomData<B>,
}

impl<T, B, GTarget> HamiltonianSampler<T, B, GTarget>
where
    B: AutodiffBackend + Backend,
    GTarget: GradientTarget<T, B> + Clone,
    T: Float + burn::tensor::Element,
{
    /// Construct a new HMC sampler with `n_chains`, each with the same step size, leapfrog steps,
    /// and initial position. We give each chain a different seed: `global_seed + chain_idx`.
    pub fn new(
        target: GTarget,
        initial_pos: Vec<T>,
        step_size: T,
        n_leapfrog: usize,
        n_chains: usize,
    ) -> Self {
        let seed_global = thread_rng().gen::<u64>();
        let chains = (0..n_chains)
            .map(|i| {
                HMCMarkovChain::new(
                    target.clone(),
                    initial_pos.clone(),
                    step_size,
                    n_leapfrog,
                    seed_global + i as u64,
                )
            })
            .collect();
        Self {
            target,
            step_size,
            n_leapfrog,
            chains,
            seed: seed_global,
            phantom_data: PhantomData,
        }
    }

    /// Set a new global seed and reseed each chain with `(seed + chain_index)`.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        for (i, chain) in self.chains.iter_mut().enumerate() {
            chain.rng = SmallRng::seed_from_u64(seed + i as u64);
        }
        self
    }
}

/// Let’s also implement `HasChains` for compatibility with your existing code.
impl<T, B, GTarget> HasChains<T> for HamiltonianSampler<T, B, GTarget>
where
    B: AutodiffBackend + Backend,
    GTarget: GradientTarget<T, B> + Clone + std::marker::Send,
    T: Float + std::marker::Send,
    HMCMarkovChain<T, B, GTarget>: MarkovChain<T>,
{
    type Chain = HMCMarkovChain<T, B, GTarget>;

    fn chains_mut(&mut self) -> &mut Vec<Self::Chain> {
        &mut self.chains
    }
}
