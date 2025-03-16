//! A simple Hamiltonian (Hybrid) Monte Carlo sampler using the `burn` crate for autodiff.
//!
//! This is modeled similarly to a Metropolis–Hastings sampler but uses gradient-based proposals
//! for improved efficiency. The sampler works in a data-parallel fashion and can update multiple
//! chains simultaneously.
//!
//! The code relies on a target distribution provided via the `GradientTarget` trait, which computes
//! the unnormalized log probability for a batch of positions. The HMC implementation uses the leapfrog
//! integrator to simulate Hamiltonian dynamics, and the standard accept/reject step for proposal
//! validation.

use crate::stats::RhatMulti;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Tensor;
use indicatif::{ProgressBar, ProgressStyle};
use num_traits::Float;
use rand::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;
use std::collections::VecDeque;
use std::error::Error;

/// A batched target trait for computing the unnormalized log probability (and gradients) for a
/// collection of positions.
///
/// Implement this trait for your target distribution to enable gradient-based sampling.
///
/// # Type Parameters
///
/// * `T`: The floating-point type (e.g., f32 or f64).
/// * `B`: The autodiff backend from the `burn` crate.
pub trait GradientTarget<T: Float, B: AutodiffBackend> {
    /// Compute the log probability for a batch of positions.
    ///
    /// # Parameters
    ///
    /// * `positions`: A tensor of shape `[n_chains, D]` representing the current positions for each chain.
    ///
    /// # Returns
    ///
    /// A 1D tensor of shape `[n_chains]` containing the log probabilities for each chain.
    fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1>;
}

/// A data-parallel Hamiltonian Monte Carlo (HMC) sampler.
///
/// This struct encapsulates the HMC algorithm, including the leapfrog integrator and the
/// accept/reject mechanism, for sampling from a target distribution in a batched manner.
///
/// # Type Parameters
///
/// * `T`: Floating-point type for numerical calculations.
/// * `B`: Autodiff backend from the `burn` crate.
/// * `GTarget`: The target distribution type implementing the `GradientTarget` trait.
#[derive(Debug, Clone)]
pub struct HMC<T, B, GTarget>
where
    B: AutodiffBackend,
{
    /// The target distribution which provides log probability evaluations and gradients.
    pub target: GTarget,
    /// The step size for the leapfrog integrator.
    pub step_size: T,
    /// The number of leapfrog steps to take per HMC update.
    pub n_leapfrog: usize,
    /// The current positions for all chains, stored as a tensor of shape `[n_chains, D]`.
    pub positions: Tensor<B, 2>,
    /// A random number generator for sampling momenta and uniform random numbers for the
    /// Metropolis acceptance test.
    pub rng: SmallRng,
}

impl<T, B, GTarget> HMC<T, B, GTarget>
where
    T: Float
        + burn::tensor::ElementConversion
        + burn::tensor::Element
        + rand_distr::uniform::SampleUniform
        + num_traits::FromPrimitive,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + std::marker::Sync,
    StandardNormal: rand::distributions::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// Create a new data-parallel HMC sampler.
    ///
    /// This method initializes the sampler with the target distribution, initial positions,
    /// step size, number of leapfrog steps, and a random seed for reproducibility.
    ///
    /// # Parameters
    ///
    /// * `target`: The target distribution implementing the `GradientTarget` trait.
    /// * `initial_positions`: A vector of vectors containing the initial positions for each chain,
    ///    with shape `[n_chains][D]`.
    /// * `step_size`: The step size used in the leapfrog integrator.
    /// * `n_leapfrog`: The number of leapfrog steps per update.
    /// * `seed`: A seed for initializing the random number generator.
    ///
    /// # Returns
    ///
    /// A new instance of `HMC`.
    pub fn new(
        target: GTarget,
        initial_positions: Vec<Vec<T>>,
        step_size: T,
        n_leapfrog: usize,
    ) -> Self {
        // Build a [n_chains, D] tensor from the flattened initial positions.
        let (n_chains, dim) = (initial_positions.len(), initial_positions[0].len());
        let td: TensorData = TensorData::new(
            initial_positions.into_iter().flatten().collect(),
            [n_chains, dim],
        );
        let positions = Tensor::<B, 2>::from_data(td, &B::Device::default());
        let rng = SmallRng::seed_from_u64(thread_rng().gen::<u64>());
        Self {
            target,
            step_size,
            n_leapfrog,
            positions,
            rng,
        }
    }

    /// Sets a new random seed.
    ///
    /// This method ensures reproducibility across runs.
    ///
    /// # Arguments
    ///
    /// * `seed` - The new random seed value.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    /// Run the HMC sampler for `n_collect` + `n_discard` steps.
    ///
    /// First, the sampler takes `n_discard` burn-in steps, then takes
    /// `n_collect` further steps and collects those samples in a 3D tensor of
    /// shape `[n_collect, n_chains, D]`.
    ///
    /// # Parameters
    ///
    /// * `n_collect` - The number of samples to collect and return.
    /// * `n_discard` - The number of samples to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A tensor containing the collected samples.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        let (n_chains, dim) = (self.positions.dims()[0], self.positions.dims()[1]);
        let mut out = Tensor::<B, 3>::empty(
            [n_collect, n_chains, self.positions.dims()[1]],
            &B::Device::default(),
        );

        // Discard the first `discard` positions.
        (0..n_discard).for_each(|_| self.step());

        // Collect samples.
        for step in 1..(n_collect + 1) {
            self.step();
            out.inplace(|_out| {
                _out.slice_assign(
                    [step - 1..step, 0..n_chains, 0..dim],
                    self.positions.clone().unsqueeze_dim(0),
                )
            });
        }
        out
    }

    /// Run the HMC sampler for `n_collect` + `n_discard` steps and displays progress with
    /// convergence statistics.
    ///
    /// First, the sampler takes `n_discard` burn-in steps, then takes
    /// `n_collect` further steps and collects those samples in a 3D tensor of
    /// shape `[n_collect, n_chains, D]`.
    ///
    /// This function displays a progress bar (using the `indicatif` crate) that is updated
    /// with an approximate acceptance probability computed over a sliding window of 100 iterations
    /// as well as the potential scale reduction factor, see [Stan Reference Manual.][1]
    ///
    /// # Parameters
    ///
    /// * `n_collect` - The number of samples to collect and return.
    /// * `n_discard` - The number of samples to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A tensor of shape `[n_collect, n_chains, D]` containing the collected samples.
    ///
    /// [1]: https://mc-stan.org/docs/2_18/reference-manual/notation-for-samples-chains-and-draws.html
    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<Tensor<B, 3>, Box<dyn Error>> {
        // Discard initial burn-in samples.
        (0..n_discard).for_each(|_| self.step());

        let (n_chains, dim) = (self.positions.dims()[0], self.positions.dims()[1]);
        let mut out = Tensor::<B, 3>::empty([n_collect, n_chains, dim], &B::Device::default());

        let pb = ProgressBar::new(n_collect as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.white} ETA {eta:3} | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC");

        // Use a sliding window of 100 iterations to estimate the acceptance probability.
        let window_size = 100;
        let mut accept_window: VecDeque<f32> = VecDeque::with_capacity(window_size);

        let mut psr = RhatMulti::new(n_chains, dim);

        let mut last_state = self.positions.clone();

        let mut last_state_data = last_state.to_data();
        psr.step(last_state_data.as_slice::<T>().unwrap())?;

        for i in 0..n_collect {
            self.step();
            let current_state = self.positions.clone();

            // For each chain, check if its state changed.
            let accepted_count = last_state
                .clone()
                .not_equal(current_state.clone())
                .all_dim(1)
                .int()
                .sum()
                .into_scalar()
                .to_f32();

            let iter_accept_rate = accepted_count / n_chains as f32;

            // Update the sliding window.
            accept_window.push_front(iter_accept_rate);
            if accept_window.len() > window_size {
                accept_window.pop_back();
            }

            // Store the current state.
            out.inplace(|_out| {
                _out.slice_assign(
                    [i..i + 1, 0..n_chains, 0..dim],
                    current_state.clone().unsqueeze_dim(0),
                )
            });
            pb.inc(1);
            last_state = current_state;

            last_state_data = last_state.to_data();
            psr.step(last_state_data.as_slice::<T>().unwrap())?;
            let maxrhat = psr.max()?;

            // Compute average acceptance rate over the sliding window.
            let avg_accept_rate: f32 =
                accept_window.iter().sum::<f32>() / accept_window.len() as f32;
            pb.set_message(format!(
                "p(accept)≈{:.2} max(rhat)≈{:.2}",
                avg_accept_rate, maxrhat
            ));
        }
        pb.finish_with_message("Done!");
        Ok(out)
    }

    /// Perform one batched HMC update for all chains in parallel.
    ///
    /// The update consists of:
    /// 1) Sampling momenta from a standard normal distribution.
    /// 2) Running the leapfrog integrator to propose new positions.
    /// 3) Performing an accept/reject step for each chain.
    ///
    /// This method updates `self.positions` in-place.
    pub fn step(&mut self) {
        let shape = self.positions.shape();
        let (n_chains, dim) = (shape.dims[0], shape.dims[1]);

        // 1) Sample momenta: shape [n_chains, D]
        let momentum_0 = Tensor::<B, 2>::random(
            Shape::new([n_chains, dim]),
            burn::tensor::Distribution::Normal(0., 1.),
            &B::Device::default(),
        );

        // Current log probability: shape [n_chains]
        let logp_current = self.target.log_prob_batch(&self.positions);

        // Compute kinetic energy: 0.5 * sum_{d} (p^2) for each chain.
        let ke_current = momentum_0
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1) // Sum over dimension 1 => shape [n_chains]
            .squeeze(1)
            .mul_scalar(T::from(0.5).unwrap());

        // Compute the Hamiltonian: -logp + kinetic energy, shape [n_chains]
        let h_current: Tensor<B, 1> = -logp_current + ke_current;

        // 2) Run the leapfrog integrator.
        let (proposed_positions, proposed_momenta, logp_proposed) =
            self.leapfrog(self.positions.clone(), momentum_0);

        // Compute proposed kinetic energy.
        let ke_proposed = proposed_momenta
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze(1)
            .mul_scalar(T::from(0.5).unwrap());

        let h_proposed = -logp_proposed + ke_proposed;

        // 3) Accept/Reject each proposal.
        let accept_logp = h_current.sub(h_proposed);

        // Draw a uniform random number for each chain.
        let mut uniform_data = Vec::with_capacity(n_chains);
        for _ in 0..n_chains {
            uniform_data.push(self.rng.gen::<T>());
        }
        let uniform = Tensor::<B, 1>::random(
            Shape::new([n_chains]),
            burn::tensor::Distribution::Default,
            &B::Device::default(),
        );

        // Accept the proposal if accept_logp >= ln(u).
        let ln_u = uniform.log(); // shape [n_chains]
        let accept_mask = accept_logp.greater_equal(ln_u); // Boolean mask of shape [n_chains]
        let mut accept_mask_big: Tensor<B, 2, Bool> = accept_mask.clone().unsqueeze_dim(1);
        accept_mask_big = accept_mask_big.expand([n_chains, dim]);

        // Update positions: for accepted chains, replace current positions with proposed positions.
        self.positions.inplace(|x| {
            x.clone()
                .mask_where(accept_mask_big, proposed_positions)
                .detach()
        });
    }

    /// Perform the leapfrog integrator steps in a batched manner.
    ///
    /// This method performs `n_leapfrog` iterations of the leapfrog update:
    /// - A half-step update of the momentum.
    /// - A full-step update of the positions.
    /// - Another half-step update of the momentum.
    ///
    /// # Parameters
    ///
    /// * `pos`: The current positions, a tensor of shape `[n_chains, D]`.
    /// * `mom`: The initial momenta, a tensor of shape `[n_chains, D]`.
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - The new positions (tensor of shape `[n_chains, D]`),
    /// - The new momenta (tensor of shape `[n_chains, D]`),
    /// - The log probability evaluated at the new positions (tensor of shape `[n_chains]`).
    fn leapfrog(
        &mut self,
        mut pos: Tensor<B, 2>,
        mut mom: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>) {
        let half = T::from(0.5).unwrap();
        for _step_i in 0..self.n_leapfrog {
            // Detach pos to ensure it's AD-enabled for the gradient computation.
            pos = pos.detach().require_grad();

            // Compute gradient of log probability with respect to pos (batched over chains).
            let logp = self.target.log_prob_batch(&pos); // shape [n_chains]
            let grads = pos.grad(&logp.backward()).unwrap();

            // Update momentum by a half-step using the computed gradients.
            mom.inplace(|_mom| {
                _mom.add(Tensor::<B, 2>::from_inner(
                    grads.mul_scalar(self.step_size * half),
                ))
            });

            // Full-step update for positions.
            pos.inplace(|_pos| {
                _pos.add(mom.clone().mul_scalar(self.step_size))
                    .detach()
                    .require_grad()
            });

            // Compute gradient at the new positions.
            let logp2 = self.target.log_prob_batch(&pos);
            let grads2 = pos.grad(&logp2.backward()).unwrap();

            // Update momentum by another half-step using the new gradients.
            mom.inplace(|_mom| {
                _mom.add(Tensor::<B, 2>::from_inner(
                    grads2.mul_scalar(self.step_size * half),
                ))
            });
        }

        // Compute final log probability at the updated positions.
        let logp_final = self.target.log_prob_batch(&pos);
        (pos.detach(), mom.detach(), logp_final.detach())
    }
}

#[cfg(test)]
mod tests {
    use crate::dev_tools::Timer;

    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Element, Tensor},
    };
    use num_traits::Float;

    // Define the Rosenbrock distribution.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct Rosenbrock2D<T: Float> {
        a: T,
        b: T,
    }

    // For the batched version we need to implement BatchGradientTarget.
    impl<T, B> GradientTarget<T, B> for Rosenbrock2D<T>
    where
        T: Float + std::fmt::Debug + Element,
        B: burn::tensor::backend::AutodiffBackend,
    {
        fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1> {
            let n = positions.dims()[0] as i64;
            let x = positions.clone().slice([(0, n), (0, 1)]);
            let y = positions.clone().slice([(0, n), (1, 2)]);

            // Compute (a - x)^2 in place.
            let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);

            // Compute (y - x^2)^2 in place.
            let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);

            // Return the negative sum as a flattened 1D tensor.
            -(term_1 + term_2).flatten(0, 1)
        }
    }

    // Define the Rosenbrock distribution.
    // From: https://arxiv.org/pdf/1903.09556.
    struct RosenbrockND {}

    // For the batched version we need to implement BatchGradientTarget.
    impl<T, B> GradientTarget<T, B> for RosenbrockND
    where
        T: Float + std::fmt::Debug + Element,
        B: burn::tensor::backend::AutodiffBackend,
    {
        fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1> {
            let k = positions.dims()[0] as i64;
            let n = positions.dims()[1] as i64;
            let low = positions.clone().slice([(0, k), (0, (n - 1))]);
            let high = positions.clone().slice([(0, k), (1, n)]);
            let term_1 = (high - low.clone().powi_scalar(2))
                .powi_scalar(2)
                .mul_scalar(100);
            let term_2 = low.neg().add_scalar(1).powi_scalar(2);
            -(term_1 + term_2).sum_dim(1).squeeze(1)
        }
    }

    #[test]
    fn test_single() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define initial positions for a single chain (2-dimensional).
        let initial_positions = vec![vec![0.0_f32, 0.0]];
        let n_collect = 3;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            2,    // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run the sampler for n_collect steps.
        let mut timer = Timer::new();
        let samples: Tensor<BackendType, 3> = sampler.run(n_collect, 0);
        timer.log(format!(
            "Collected samples (10 chains) with shape: {:?}",
            samples.dims()
        ))
    }

    #[test]
    fn test_10_chains() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define 10 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 10];
        let n_collect = 1000;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            10,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run the sampler for n_collect.
        let mut timer = Timer::new();
        let samples: Tensor<BackendType, 3> = sampler.run(n_collect, 0);
        timer.log(format!(
            "Collected samples (10 chains) with shape: {:?}",
            samples.dims()
        ))
    }

    #[test]
    fn test_progress_10_chains() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define 10 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 10];
        let n_collect = 1000;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.05, // step size
            10,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run the sampler for n_collect with no discard.
        let mut timer = Timer::new();
        let samples: Tensor<BackendType, 3> = sampler.run_progress(n_collect, 100).unwrap();
        timer.log(format!(
            "Collected samples (10 chains) with shape: {:?}",
            samples.dims()
        ))
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 6];
        let n_collect = 5000;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for `n_collect` steps.
        let mut timer = Timer::new();
        let samples = sampler.run(n_collect, 0);
        timer.log(format!(
            "HMC sampler: generated {} samples.",
            samples.dims()[0..2].iter().product::<usize>()
        ))
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_progress_bench() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 6];
        let n_collect = 5000;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for n_collect steps.
        let mut timer = Timer::new();
        let samples = sampler.run_progress(n_collect, 0).unwrap();
        timer.log(format!(
            "HMC sampler: generated {} samples.",
            samples.dims()[0..2].iter().product::<usize>()
        ))
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_10000d() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        let seed = 42;
        let d = 10000;

        let rng = SmallRng::seed_from_u64(seed);
        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions: Vec<Vec<f32>> =
            vec![rng.sample_iter(StandardNormal).take(d).collect(); 6];
        let n_collect = 500;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
            RosenbrockND {},
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for n_collect steps.
        let mut timer = Timer::new();
        let samples = sampler.run(n_collect, 0);
        timer.log(format!(
            "HMC sampler: generated {} samples.",
            samples.dims()[0..2].iter().product::<usize>()
        ))
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    #[cfg(feature = "wgpu")]
    fn test_progress_10000d_bench() {
        type BackendType = Autodiff<burn::backend::Wgpu>;

        let seed = 42;
        let d = 10000;

        let rng = SmallRng::seed_from_u64(seed);
        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions: Vec<Vec<f32>> =
            vec![rng.sample_iter(StandardNormal).take(d).collect(); 6];
        let n_collect = 5000;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, RosenbrockND>::new(
            RosenbrockND {},
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run HMC for n_collect steps.
        let mut timer = Timer::new();
        let samples = sampler.run_progress(n_collect, 0).unwrap();
        timer.log(format!(
            "HMC sampler: generated {} samples.",
            samples.dims()[0..2].iter().product::<usize>()
        ))
    }
}
