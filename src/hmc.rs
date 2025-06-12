//! Hamiltonian Monte Carlo (HMC) sampler.
//!
//! This is modeled similarly to a Metropolis–Hastings sampler but uses gradient-based proposals
//! for improved efficiency. The sampler works in a data-parallel fashion and can update multiple
//! chains simultaneously.
//!
//! The code relies on a target distribution provided via the `BatchedGradientTarget` trait, which computes
//! the unnormalized log probability for a batch of positions. The HMC implementation uses the leapfrog
//! integrator to simulate Hamiltonian dynamics, and the standard accept/reject step for proposal
//! validation.

use crate::distributions::BatchedGradientTarget;
use crate::stats::MultiChainTracker;
use crate::stats::RunStats;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use indicatif::{ProgressBar, ProgressStyle};
use num_traits::Float;
use rand::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;
use std::error::Error;

/// A data-parallel Hamiltonian Monte Carlo (HMC) sampler.
///
/// This struct encapsulates the HMC algorithm, including the leapfrog integrator and the
/// accept/reject mechanism, for sampling from a target distribution in a batched manner.
///
/// # Type Parameters
///
/// * `T`: Floating-point type for numerical calculations.
/// * `B`: Autodiff backend from the `burn` crate.
/// * `GTarget`: The target distribution type implementing the `BatchedGradientTarget` trait.
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
    /// The current positions for all chains, stored as a tensor of shape `[n_chains, D]` where:
    /// - `n_chains`: number of parallel chains
    /// - `D`: dimensionality of the state space
    pub positions: Tensor<B, 2>,

    /// Last step's position gradient
    last_grad_summands: Tensor<B, 2>,

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
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync,
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
    /// * `target`: The target distribution implementing the `BatchedGradientTarget` trait.
    /// * `initial_positions`: A vector of vectors containing the initial positions for each chain, with shape `[n_chains][D]`.
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
            last_grad_summands: Tensor::<B, 2>::zeros_like(&positions),
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
    /// `n_collect` further steps and collects those observations in a 3D tensor of
    /// shape `[n_chains, n_collect, D]`.
    ///
    /// # Parameters
    ///
    /// * `n_collect` - The number of observations to collect and return.
    /// * `n_discard` - The number of observations to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A tensor containing the collected observations.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        let (n_chains, dim) = (self.positions.dims()[0], self.positions.dims()[1]);
        let mut out = Tensor::<B, 3>::empty(
            [n_collect, n_chains, self.positions.dims()[1]],
            &B::Device::default(),
        );

        // Discard the first `discard` positions.
        (0..n_discard).for_each(|_| self.step());

        // Collect observations.
        for step in 1..(n_collect + 1) {
            self.step();
            out.inplace(|_out| {
                _out.slice_assign(
                    [step - 1..step, 0..n_chains, 0..dim],
                    self.positions.clone().unsqueeze_dim(0),
                )
            });
        }
        out.permute([1, 0, 2])
    }

    /// Run the HMC sampler for `n_collect` + `n_discard` steps and displays progress with
    /// convergence statistics.
    ///
    /// First, the sampler takes `n_discard` burn-in steps, then takes
    /// `n_collect` further steps and collects those observations in a 3D tensor of
    /// shape `[n_chains, n_collect, D]`.
    ///
    /// This function displays a progress bar (using the `indicatif` crate) that is updated
    /// with an approximate acceptance probability computed over a sliding window of 100 iterations
    /// as well as the potential scale reduction factor, see [Stan Reference Manual.][1]
    ///
    /// # Parameters
    ///
    /// * `n_collect` - The number of observations to collect and return.
    /// * `n_discard` - The number of observations to discard (burn-in).
    ///
    /// # Returns
    ///
    /// A tuple containing:
    /// - A tensor of shape `[n_chains, n_collect, D]` containing the collected observations.
    /// - A `RunStats` object containing convergence statistics including:
    ///   - Acceptance probability
    ///   - Potential scale reduction factor (R-hat)
    ///   - Effective sample size (ESS)
    ///   - Other convergence diagnostics
    ///
    /// # Example
    ///
    /// ```rust
    /// use mini_mcmc::hmc::HMC;
    /// use mini_mcmc::distributions::DiffableGaussian2D;
    /// use burn::backend::{Autodiff, NdArray};
    /// use burn::prelude::*;
    ///
    /// // Create a 2D Gaussian target distribution
    /// let target = DiffableGaussian2D::new(
    ///     [0.0_f32, 1.0],  // mean
    ///     [[4.0, 2.0],     // covariance
    ///      [2.0, 3.0]]
    /// );
    ///
    /// // Create HMC sampler with:
    /// // - target distribution
    /// // - initial positions for each chain
    /// // - step size for leapfrog integration
    /// // - number of leapfrog steps
    /// type BackendType = Autodiff<NdArray>;
    /// let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
    ///     target,
    ///     vec![vec![0.0; 2]; 4],    // Initial positions for 4 chains
    ///     0.1,                      // Step size
    ///     5,                       // Number of leapfrog steps
    /// );
    ///
    /// // Run sampler with progress tracking
    /// let (sample, stats) = sampler.run_progress(12, 34).unwrap();
    ///
    /// // Print convergence statistics
    /// println!("{stats}");
    /// ```
    ///
    /// [1]: https://mc-stan.org/docs/2_18/reference-manual/notation-for-samples-chains-and-draws.html
    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        // Discard initial burn-in observations.
        (0..n_discard).for_each(|_| self.step());

        let (n_chains, dim) = (self.positions.dims()[0], self.positions.dims()[1]);
        let mut out = Tensor::<B, 3>::empty([n_collect, n_chains, dim], &B::Device::default());

        let pb = ProgressBar::new(n_collect as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC");

        let mut tracker = MultiChainTracker::new(n_chains, dim);

        let mut last_state = self.positions.clone();

        let mut last_state_data = last_state.to_data();
        if let Err(e) = tracker.step(last_state_data.as_slice::<T>().unwrap()) {
            eprintln!("Warning: Shown progress statistics may be unreliable since updating them failed with: {}", e);
        }

        for i in 0..n_collect {
            self.step();
            let current_state = self.positions.clone();

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
            if let Err(e) = tracker.step(last_state_data.as_slice::<T>().unwrap()) {
                eprintln!("Warning: Shown progress statistics may be unreliable since updating them failed with: {}", e);
            }

            match tracker.max_rhat() {
                Err(e) => {
                    eprintln!("Computing max(rhat) failed with: {}", e);
                }
                Ok(max_rhat) => {
                    pb.set_message(format!(
                        "p(accept)≈{:.2} max(rhat)≈{:.2}",
                        tracker.p_accept, max_rhat
                    ));
                }
            }
        }
        pb.finish_with_message("Done!");
        let sample = out.permute([1, 0, 2]);

        let stats = match tracker.stats(sample.clone()) {
            Ok(stats) => stats,
            Err(e) => {
                eprintln!("Getting run statistics failed with: {}", e);
                return Err(e);
            }
        };

        Ok((sample, stats))
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
        // Detach pos to ensure it's AD-enabled for the gradient computation.
        let pos = self.positions.clone().detach().require_grad();
        let logp_current = self.target.unnorm_logp_batch(pos.clone());

        // Compute gradient of log probability with respect to pos.
        // First gradient step in leapfrog needs it.
        let grads = pos.grad(&logp_current.backward()).unwrap();
        let grad_summands =
            Tensor::<B, 2>::from_inner(grads.mul_scalar(self.step_size * T::from(0.5).unwrap()));
        self.last_grad_summands = grad_summands;

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
        // dbg!(&pos, &mom);
        for _step_i in 0..self.n_leapfrog {
            // Detach pos to ensure it's AD-enabled for the gradient computation.
            pos = pos.detach().require_grad();

            // Update momentum by a half-step using the computed gradients.
            mom.inplace(|_mom| _mom.add(self.last_grad_summands.clone()));

            // Full-step update for positions.
            pos.inplace(|_pos| {
                _pos.add(mom.clone().mul_scalar(self.step_size))
                    .detach()
                    .require_grad()
            });

            // Compute gradient at the new positions.
            let logp = self.target.unnorm_logp_batch(pos.clone());
            let grads = pos.grad(&logp.backward()).unwrap();
            let grad_summands = Tensor::<B, 2>::from_inner(grads.mul_scalar(self.step_size * half));

            // Update momentum by another half-step using the new gradients.
            mom.inplace(|_mom| _mom.add(grad_summands.clone()));

            self.last_grad_summands = grad_summands;
        }

        // Compute final log probability at the updated positions.
        let logp_final = self.target.unnorm_logp_batch(pos.clone());
        (pos.detach(), mom.detach(), logp_final.detach())
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        core::init,
        dev_tools::Timer,
        distributions::{DiffableGaussian2D, Rosenbrock2D, RosenbrockND},
        stats::split_rhat_mean_ess,
    };
    use ndarray::ArrayView3;
    use ndarray_stats::QuantileExt;

    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::Tensor,
    };

    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<NdArray>;

    #[test]
    fn test_hmc_single() {
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
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, 0);
        timer.log(format!(
            "Collected sample (10 chains) with shape: {:?}",
            sample.dims()
        ));
        assert_eq!(sample.dims(), [1, 3, 2]);
    }

    #[test]
    fn test_3_chains() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define 3 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 3];
        let n_collect = 10;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            2,    // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run the sampler for n_collect.
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, 0);
        timer.log(format!(
            "Collected sample (3 chains) with shape: {:?}",
            sample.dims()
        ));
        assert_eq!(sample.dims(), [3, 10, 2]);
    }

    #[test]
    fn test_progress_3_chains() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define 3 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 3];
        let n_collect = 10;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.05, // step size
            2,    // number of leapfrog steps per update
        )
        .set_seed(42);

        // Run the sampler for n_collect with no discard.
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run_progress(n_collect, 3).unwrap().0;
        timer.log(format!(
            "Collected sample (10 chains) with shape: {:?}",
            sample.dims()
        ));
        assert_eq!(sample.dims(), [3, 10, 2]);
    }

    #[test]
    fn test_gaussian_2d_hmc_debug() {
        let n_chains = 1;
        let n_discard = 1;
        let n_collect = 1;

        let target = DiffableGaussian2D::new([0.0, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![vec![0.0_f32, 0.0_f32]];

        type BackendType = Autodiff<NdArray>;
        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            0.1,
            1,
        )
        .set_seed(42);

        let sample_3d = sampler.run(n_collect, n_discard);

        assert_eq!(sample_3d.dims(), [n_chains, n_collect, 2]);
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_gaussian_2d_hmc_single_run() {
        // Each experiment uses 3 chains:
        let n_chains = 3;

        let n_discard = 500;
        let n_collect = 1000;

        // 1) Define the 2D Gaussian target distribution:
        //    mean: [0.0, 1.0], cov: [[4.0, 2.0], [2.0, 3.0]]
        let target = DiffableGaussian2D::new([0.0, 1.0], [[4.0, 2.0], [2.0, 3.0]]);

        // 2) Define 3 chains, each chain is 2-dimensional:
        let initial_positions = vec![
            vec![1.0_f32, 2.0_f32],
            vec![1.0_f32, 2.0_f32],
            vec![1.0_f32, 2.0_f32],
        ];

        // 3) Create the HMC sampler using NdArray backend with autodiff
        type BackendType = Autodiff<NdArray>;
        let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
            target,
            initial_positions,
            0.1, // step size
            10,  // leapfrog steps
        )
        .set_seed(42);

        // 4) Run the sampler for (burn_in + collected) steps, discard the first `burn_in`
        //    The shape of `sample` will be [n_chains, collected, 2]
        let sample_3d = sampler.run(n_collect, n_discard);

        // Check shape is as expected
        assert_eq!(sample_3d.dims(), [n_chains, n_collect, 2]);

        // 5) Convert the sample into an ndarray view
        let data = sample_3d.to_data();
        let arr = ArrayView3::from_shape(sample_3d.dims(), data.as_slice().unwrap()).unwrap();

        // 6) Compute split-Rhat and ESS
        let (rhat, ess_vals) = split_rhat_mean_ess(arr.view());
        let ess1 = ess_vals[0];
        let ess2 = ess_vals[1];

        println!("\nSingle Run Results:");
        println!("Rhat: {:?}", rhat);
        println!("ESS(Param1): {:.2}", ess1);
        println!("ESS(Param2): {:.2}", ess2);

        // Optionally, add some asserts about expected minimal ESS
        assert!(ess1 > 50.0, "Expected param1 ESS > 50, got {:.2}", ess1);
        assert!(ess2 > 50.0, "Expected param2 ESS > 50, got {:.2}", ess2);
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_gaussian_2d_hmc_ess_stats() {
        use crate::stats::basic_stats;
        use indicatif::{ProgressBar, ProgressStyle};
        use ndarray::Array1;

        let n_runs = 100;
        let n_chains = 3;
        let n_discard = 500;
        let n_collect = 1000;
        let mut rng = SmallRng::seed_from_u64(42);

        // We'll store the ESS and R-hat values for each parameter across all runs
        let mut ess_param1s = Vec::with_capacity(n_runs);
        let mut ess_param2s = Vec::with_capacity(n_runs);
        let mut rhat_param1s = Vec::with_capacity(n_runs);
        let mut rhat_param2s = Vec::with_capacity(n_runs);

        // Set up the progress bar
        let pb = ProgressBar::new(n_runs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        pb.set_prefix("HMC Test");

        for run in 0..n_runs {
            // 1) Define the 2D Gaussian target distribution:
            //    mean: [0.0, 1.0], cov: [[4.0, 2.0], [2.0, 3.0]]
            let target = DiffableGaussian2D::new([0.0_f32, 1.0], [[4.0, 2.0], [2.0, 3.0]]);

            // 2) Define 3 chains, each chain is 2-dimensional:
            // Create a seeded RNG for reproducible initial positions
            let initial_positions: Vec<Vec<f32>> = (0..n_chains)
                .map(|_| {
                    // Sample 2D position from standard normal
                    vec![
                        rng.sample::<f32, _>(StandardNormal),
                        rng.sample::<f32, _>(StandardNormal),
                    ]
                })
                .collect();

            // 3) Create the HMC sampler using NdArray backend with autodiff
            type BackendType = Autodiff<NdArray>;
            let mut sampler = HMC::<f32, BackendType, DiffableGaussian2D<f32>>::new(
                target,
                initial_positions,
                0.1, // step size
                10,  // leapfrog steps
            )
            .set_seed(run as u64); // Use run number as seed for reproducibility

            // 4) Run the sampler for (n_discard + n_collect) steps, discard the first `n_discard`
            //    observations
            let sample_3d = sampler.run(n_collect, n_discard);

            // Check shape is as expected
            assert_eq!(sample_3d.dims(), [n_chains, n_collect, 2]);

            // 5) Convert the sample into an ndarray view
            let data = sample_3d.to_data();
            let arr = ArrayView3::from_shape(sample_3d.dims(), data.as_slice().unwrap()).unwrap();

            // 6) Compute split-Rhat and ESS
            let (rhat, ess_vals) = split_rhat_mean_ess(arr.view());
            let ess1 = ess_vals[0];
            let ess2 = ess_vals[1];

            // Store ESS values
            ess_param1s.push(ess1);
            ess_param2s.push(ess2);

            // Store R-hat values from stats object
            rhat_param1s.push(rhat[0]);
            rhat_param2s.push(rhat[1]);

            pb.inc(1);

            // Update progress bar with current ESS statistics across runs
            if run > 0 {
                // Calculate mean and std of ESS for both parameters across all runs so far
                let mean_ess1 = ess_param1s.iter().sum::<f32>() / (run as f32 + 1.0);
                let mean_ess2 = ess_param2s.iter().sum::<f32>() / (run as f32 + 1.0);

                // Calculate standard deviations
                let var_ess1 = ess_param1s
                    .iter()
                    .map(|&x| (x - mean_ess1).powi(2))
                    .sum::<f32>()
                    / (run as f32 + 1.0);
                let var_ess2 = ess_param2s
                    .iter()
                    .map(|&x| (x - mean_ess2).powi(2))
                    .sum::<f32>()
                    / (run as f32 + 1.0);

                let std_ess1 = var_ess1.sqrt();
                let std_ess2 = var_ess2.sqrt();

                pb.set_message(format!(
                    "ESS1={:.0}±{:.0} ESS2={:.0}±{:.0}",
                    mean_ess1, std_ess1, mean_ess2, std_ess2
                ));
            } else {
                // For the first run, just show the current values
                pb.set_message(format!("ESS1={:.0} ESS2={:.0}", ess1, ess2));
            }
        }
        pb.finish_with_message("All runs complete!");

        // Convert to ndarray for statistics
        let ess_param1_array = Array1::from_vec(ess_param1s);
        let ess_param2_array = Array1::from_vec(ess_param2s);
        let rhat_param1_array = Array1::from_vec(rhat_param1s);
        let rhat_param2_array = Array1::from_vec(rhat_param2s);

        // Compute and print statistics
        let stats_p1_ess = basic_stats("ESS(Param1)", ess_param1_array);
        let stats_p2_ess = basic_stats("ESS(Param2)", ess_param2_array);
        let stats_p1_rhat = basic_stats("R-hat(Param1)", rhat_param1_array);
        let stats_p2_rhat = basic_stats("R-hat(Param2)", rhat_param2_array);

        println!("\nStatistics over {} runs:", n_runs);
        println!("\nESS Statistics:");
        println!("{stats_p1_ess}\n{stats_p2_ess}");
        println!("\nR-hat Statistics:");
        println!("{stats_p1_rhat}\n{stats_p2_rhat}");

        // Assertions for ESS
        assert!(
            (135.0..=185.0).contains(&stats_p1_ess.mean),
            "Expected param1 ESS to average in [135, 185], got {:.2}",
            stats_p1_ess.mean
        );
        assert!(
            (141.0..=191.0).contains(&stats_p2_ess.mean),
            "Expected param2 ESS to average in [141, 191], got {:.2}",
            stats_p2_ess.mean
        );

        // Assertions for R-hat (should be close to 1.0)
        assert!(
            (0.95..=1.05).contains(&stats_p1_rhat.mean),
            "Expected param1 R-hat to be in [0.95, 1.05], got {:.2}",
            stats_p1_rhat.mean
        );
        assert!(
            (0.95..=1.05).contains(&stats_p2_rhat.mean),
            "Expected param2 R-hat to be in [0.95, 1.05], got {:.2}",
            stats_p2_rhat.mean
        );
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_noprogress() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init(6, 2);
        let n_collect = 5000;
        let n_discard = 500;

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
        let sample = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [6, 5000, 2]);

        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice().unwrap()).unwrap();
        let (split_rhat, ess) = split_rhat_mean_ess(array);
        println!("MIN Split Rhat: {}", split_rhat.min().unwrap());
        println!("MIN ESS: {}", ess.min().unwrap());
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_progress_bench() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;
        BackendType::seed(42);

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let n_chains = 6;
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; n_chains];
        let n_collect = 1000;
        let n_discard = 1000;

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
        let sample = sampler.run_progress(n_collect, n_discard).unwrap().0;
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        println!(
            "Chain 1, first 10: {}",
            sample.clone().slice([0..1, 0..10, 0..1])
        );
        println!(
            "Chain 2, first 10: {}",
            sample.clone().slice([2..3, 0..10, 0..1])
        );

        #[cfg(feature = "arrow")]
        crate::io::csv::save_csv_tensor(sample.clone(), "data.csv")
            .expect("Expected saving to succeed");

        assert_eq!(sample.dims(), [n_chains, n_collect, 2]);
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_10000d() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        let seed = 42;
        let d = 10000;
        let n_chains = 6;
        let n_collect = 100;
        let n_discard = 100;

        let rng = SmallRng::seed_from_u64(seed);
        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions: Vec<Vec<f32>> =
            vec![rng.sample_iter(StandardNormal).take(d).collect(); n_chains];

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
        let sample = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [n_chains, n_collect, d]);
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    #[cfg(feature = "wgpu")]
    fn test_progress_10000d_bench() {
        type BackendType = Autodiff<burn::backend::Wgpu>;

        let seed = 42;
        let d = 10000;
        let n_chains = 6;

        let rng = SmallRng::seed_from_u64(seed);
        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions: Vec<Vec<f32>> =
            vec![rng.sample_iter(StandardNormal).take(d).collect(); n_chains];
        let n_collect = 100;
        let n_discard = 100;

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
        let sample = sampler.run_progress(n_collect, n_discard).unwrap().0;
        timer.log(format!(
            "HMC sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [n_chains, n_collect, d]);
    }
}
