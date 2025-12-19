use crate::euclidean::EuclideanVector;
use crate::stats::{MultiChainTracker, RunStats};
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::{s, Array3, ArrayView1};
use num_traits::{Float, FromPrimitive, ToPrimitive, Zero};
use rand::distr::Distribution as RandDistribution;
// rand_distr types implement rand::distr::Distribution for rand 0.9; use this trait to avoid conflicts.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{StandardNormal, StandardUniform};
use std::error::Error;

/// A target density that can write its gradient in-place for a given position.
pub trait HamiltonianTarget<V: EuclideanVector> {
    /// Returns the log-density at `position` and writes the gradient into `grad`.
    fn logp_and_grad(&self, position: &V, grad: &mut V) -> V::Scalar;
}

/// Backend-agnostic, in-place Hamiltonian Monte Carlo engine.
#[derive(Clone, Debug)]
pub struct GenericHMC<V, Target>
where
    V: EuclideanVector,
    Target: HamiltonianTarget<V>,
{
    target: Target,
    step_size: V::Scalar,
    n_leapfrog: usize,
    positions: Vec<V>,
    rng: SmallRng,
    grad_buffers: Vec<V>,
    momentum_buffers: Vec<V>,
    proposal_positions: Vec<V>,
    proposal_momenta: Vec<V>,
    dim: usize,
}

type RunResult<T> = Result<(Array3<T>, RunStats), Box<dyn Error>>;

impl<V, Target> GenericHMC<V, Target>
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive + ToPrimitive,
    Target: HamiltonianTarget<V> + Clone,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
{
    pub fn new(
        target: Target,
        initial_positions: Vec<V>,
        step_size: V::Scalar,
        n_leapfrog: usize,
    ) -> Self {
        assert!(
            !initial_positions.is_empty(),
            "initial_positions must not be empty"
        );
        let dim = initial_positions[0].len();
        let template = initial_positions[0].zeros_like();
        let n_chains = initial_positions.len();
        let grad_buffers = (0..n_chains)
            .map(|_| template.zeros_like())
            .collect::<Vec<_>>();
        let momentum_buffers = (0..n_chains)
            .map(|_| template.zeros_like())
            .collect::<Vec<_>>();
        let proposal_positions = initial_positions
            .iter()
            .map(|p| p.zeros_like())
            .collect::<Vec<_>>();
        let proposal_momenta = (0..n_chains)
            .map(|_| template.zeros_like())
            .collect::<Vec<_>>();
        let mut thread_rng = rand::rng();
        let rng = SmallRng::from_rng(&mut thread_rng);

        Self {
            target,
            step_size,
            n_leapfrog,
            positions: initial_positions,
            rng,
            grad_buffers,
            momentum_buffers,
            proposal_positions,
            proposal_momenta,
            dim,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array3<V::Scalar> {
        (0..n_discard).for_each(|_| self.step());
        let n_chains = self.positions.len();
        let mut out = Array3::<V::Scalar>::zeros((n_chains, n_collect, self.dim));
        let mut scratch = vec![V::Scalar::zero(); self.dim];

        for step_idx in 0..n_collect {
            self.step();
            for (chain_idx, pos) in self.positions.iter().enumerate() {
                pos.write_to_slice(&mut scratch);
                let view = ArrayView1::from(&scratch);
                out.slice_mut(s![chain_idx, step_idx, ..]).assign(&view);
            }
        }
        out
    }

    pub fn run_progress(&mut self, n_collect: usize, n_discard: usize) -> RunResult<V::Scalar> {
        (0..n_discard).for_each(|_| self.step());

        let n_chains = self.positions.len();
        let mut out = Array3::<V::Scalar>::zeros((n_chains, n_collect, self.dim));
        let mut scratch = vec![V::Scalar::zero(); self.dim];
        let mut flattened = vec![V::Scalar::zero(); n_chains * self.dim];

        let mut tracker = MultiChainTracker::new(n_chains, self.dim);
        self.flatten_positions(&mut flattened);
        tracker.step(&flattened)?;

        let pb = ProgressBar::new(n_collect as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC");

        for step_idx in 0..n_collect {
            self.step();
            for (chain_idx, pos) in self.positions.iter().enumerate() {
                pos.write_to_slice(&mut scratch);
                let view = ArrayView1::from(&scratch);
                out.slice_mut(s![chain_idx, step_idx, ..]).assign(&view);
            }
            self.flatten_positions(&mut flattened);
            tracker.step(&flattened)?;
            if let Ok(max_rhat) = tracker.max_rhat() {
                pb.set_message(format!(
                    "p(accept)≈{:.2} max(rhat)≈{:.2}",
                    tracker.p_accept, max_rhat
                ));
            }
            pb.inc(1);
        }
        pb.finish_with_message("Done!");

        let stats = RunStats::from(out.view());
        Ok((out, stats))
    }

    fn flatten_positions(&self, out: &mut [V::Scalar]) {
        let dim = self.dim;
        for (i, pos) in self.positions.iter().enumerate() {
            let start = i * dim;
            let end = start + dim;
            pos.write_to_slice(&mut out[start..end]);
        }
    }

    pub(crate) fn step(&mut self) {
        let half = self.step_size * V::Scalar::from_f64(0.5).unwrap();
        let n_chains = self.positions.len();

        for i in 0..n_chains {
            let grad = &mut self.grad_buffers[i];
            grad.fill_zero();
            let logp_current = self.target.logp_and_grad(&self.positions[i], grad);

            let momentum = &mut self.momentum_buffers[i];
            momentum.fill_standard_normal(&mut self.rng);
            let ke_current = momentum.dot(momentum) * half;

            let proposal_pos = &mut self.proposal_positions[i];
            proposal_pos.assign(&self.positions[i]);
            let proposal_mom = &mut self.proposal_momenta[i];
            proposal_mom.assign(momentum);

            let logp_proposed = Self::leapfrog_chain(
                &self.target,
                proposal_pos,
                proposal_mom,
                grad,
                self.step_size,
                self.n_leapfrog,
                logp_current,
            );

            let ke_proposed = proposal_mom.dot(proposal_mom) * half;
            let log_accept = (logp_proposed - logp_current) + (ke_current - ke_proposed);
            let ln_u: V::Scalar = self.rng.sample(StandardUniform).ln();
            if ln_u <= log_accept {
                self.positions[i].assign(proposal_pos);
            }
        }
    }

    fn leapfrog_chain(
        target: &Target,
        position: &mut V,
        momentum: &mut V,
        grad: &mut V,
        step_size: V::Scalar,
        n_leapfrog: usize,
        mut logp: V::Scalar,
    ) -> V::Scalar {
        let half = V::Scalar::from_f64(0.5).unwrap() * step_size;
        for _ in 0..n_leapfrog {
            momentum.add_scaled_assign(grad, half);
            position.add_scaled_assign(momentum, step_size);
            logp = target.logp_and_grad(position, grad);
            momentum.add_scaled_assign(grad, half);
        }
        logp
    }

    pub(crate) fn positions(&self) -> &[V] {
        &self.positions
    }

    pub(crate) fn rng_clone(&self) -> SmallRng {
        self.rng.clone()
    }
}
