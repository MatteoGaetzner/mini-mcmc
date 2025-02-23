//! A simple Hamiltonian (Hybrid) Monte Carlo sampler using the `burn` crate for autodiff.
//!
//! This is modeled similarly to your Metropolis–Hastings approach but uses gradient-based
//! proposals.

use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use num_traits::Float;
use rand::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;
use std::marker::PhantomData;

use crate::dev_tools::Timer;

// -- 1) A batched target trait (see above) --
pub trait BatchGradientTarget<T: Float, B: AutodiffBackend> {
    fn log_prob_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1>;
}

// -- 2) The data-parallel HMC struct --
pub struct DataParallelHMC<T, B, GTarget>
where
    B: AutodiffBackend,
{
    pub target: GTarget,
    pub step_size: T,
    pub n_leapfrog: usize,
    /// Positions for all chains, shape `[n_chains, D]`.
    pub positions: Tensor<B, 2>,
    /// A random-number generator for sampling momenta & accept tests.
    pub rng: SmallRng,
    phantom: PhantomData<(T, B)>,
}

impl<T, B, GTarget> DataParallelHMC<T, B, GTarget>
where
    T: Float
        + burn::tensor::ElementConversion
        + burn::tensor::Element
        + rand_distr::uniform::SampleUniform,
    B: AutodiffBackend,
    GTarget: BatchGradientTarget<T, B>,
    StandardNormal: rand::distributions::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    /// Create a new data-parallel HMC, using `[n_chains, D]` initial positions.
    ///
    /// `initial_positions`: a `Vec<Vec<T>>` of shape `[n_chains][D]`.
    pub fn new(
        target: GTarget,
        initial_positions: Vec<Vec<T>>,
        step_size: T,
        n_leapfrog: usize,
        seed: u64,
    ) -> Self {
        // Build a [n_chains, D] tensor
        let (n_chains, dim) = (initial_positions.len(), initial_positions[0].len());
        let td: TensorData = TensorData::new(
            initial_positions.into_iter().flatten().collect(),
            [n_chains, dim],
        );
        // dbg!(&td.as_bytes());
        let positions = Tensor::<B, 2>::from_data(td, &B::Device::default());

        let rng = SmallRng::seed_from_u64(seed);

        Self {
            target,
            step_size,
            n_leapfrog,
            positions,
            rng,
            phantom: PhantomData,
        }
    }

    /// Perform one *batched* HMC update for all chains in parallel:
    /// 1) Sample momenta from N(0, I).
    /// 2) Run leapfrog steps in batch.
    /// 3) Accept/reject per chain.
    pub fn step(&mut self) {
        let mut timer = Timer::new();
        let shape = self.positions.shape();
        let (n_chains, dim) = (shape.dims[0], shape.dims[1]);

        // 1) Sample momenta: shape [n_chains, D]
        let momentum_0 = Tensor::<B, 2>::random(
            Shape::new([n_chains, dim]),
            burn::tensor::Distribution::Normal(0., 1.),
            &B::Device::default(),
        );

        timer.log("Momentum 0");

        // Current log-prob, shape [n_chains]
        let logp_current = self.target.log_prob_batch(self.positions.clone());
        timer.log("logp_current");

        // Kinetic energy: 0.5 * sum_{d} (p^2) per chain => shape [n_chains]
        let ke_current = momentum_0
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1) // sum over dimension=1 => shape [n_chains]
            .squeeze(1)
            .mul_scalar(T::from(0.5).unwrap());
        timer.log("ke_current");

        // "Hamiltonian" = -logp + KE, shape [n_chains]
        let h_current: Tensor<B, 1> = -logp_current + ke_current;
        timer.log("h_current");

        // 2) Run leapfrog integrator
        let (proposed_positions, proposed_momenta, logp_proposed) =
            self.leapfrog(self.positions.clone(), momentum_0);
        timer.log("leapfrog");

        // Proposed kinetic
        let ke_proposed = proposed_momenta
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze(1)
            .mul_scalar(T::from(0.5).unwrap());
        timer.log("ke_proposed");

        let h_proposed = -logp_proposed + ke_proposed;
        timer.log("h_proposed");

        // 3) Accept/Reject per chain
        //    accept_logp = -(h_proposed - h_current) = h_current - h_proposed
        let accept_logp = h_current.sub(h_proposed);
        timer.log("accept_logp");

        // We draw uniform(0,1) for each chain => shape [n_chains]
        let mut uniform_data = Vec::with_capacity(n_chains);
        for _i in 0..n_chains {
            uniform_data.push(self.rng.gen::<T>());
        }
        timer.log("uniform_data");
        let uniform = Tensor::<B, 1>::random(
            Shape::new([n_chains]),
            burn::tensor::Distribution::Default,
            &B::Device::default(),
        );
        timer.log("random");

        // Condition: accept_logp >= ln(u)
        let ln_u = uniform.log(); // shape [n_chains]
        let accept_mask = accept_logp.greater_equal(ln_u); // Boolean mask: shape [n_chains]
        let mut accept_mask_big: Tensor<B, 2, Bool> = accept_mask.clone().unsqueeze_dim(1);
        accept_mask_big = accept_mask_big.expand([n_chains, dim]);
        timer.log("accept_mask_big");

        self.positions.clone_from(
            &self
                .positions
                .clone()
                .mask_where(accept_mask_big, proposed_positions)
                .detach(),
        );
        timer.log("clone to positions");
    }

    /// A batched leapfrog step (one iteration). Usually you do `n_leapfrog` steps in a loop.
    /// We’ll do `n_leapfrog` inside here for simplicity.
    ///
    /// Returns `(positions, momenta, logp)` all shape `[n_chains, D]` or `[n_chains]`.
    fn leapfrog(
        &mut self,
        mut pos: Tensor<B, 2>,
        mut mom: Tensor<B, 2>,
    ) -> (Tensor<B, 2>, Tensor<B, 2>, Tensor<B, 1>) {
        let mut timer = Timer::new();
        let half = T::from(0.5).unwrap();
        for _step_i in 0..self.n_leapfrog {
            // Make sure pos is AD-enabled
            pos = pos.detach().require_grad();
            timer.log("pos 1");

            // Compute gradient of log_prob wrt pos (all chains in parallel!)
            let logp = self.target.log_prob_batch(pos.clone()); // shape [n_chains]
            timer.log("logp");
            let grads = pos.grad(&logp.backward()).unwrap();
            timer.log("grads");

            // First half-step for momentum
            let mom_inner = mom
                .clone()
                .inner()
                .add(grads.mul_scalar(self.step_size * half));
            timer.log("mom 1");

            // Full step in position
            pos = Tensor::<B, 2>::from_inner(pos.inner().add(mom_inner.mul_scalar(self.step_size)))
                .detach()
                .require_grad();

            timer.log("pos update");
            // Second half-step for momentum
            let logp2 = self.target.log_prob_batch(pos.clone());
            timer.log("logp2");
            let grads2 = pos.grad(&logp2.backward()).unwrap();
            timer.log("grads2");
            mom = Tensor::<B, 2>::from_inner(
                mom.inner().add(grads2.mul_scalar(self.step_size * half)),
            );
            timer.log("mom 2");
        }

        // Final logp for these positions
        let logp_final = self.target.log_prob_batch(pos.clone());
        timer.log("prep out leapfrog");
        // println!("\n");
        (pos.detach(), mom.detach(), logp_final.detach())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra as na;

    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Element, Tensor},
    };
    use num_traits::Float;
    use std::time::Instant;

    // Define the Rosenbrock distribution.
    #[derive(Clone, Copy)]
    struct Rosenbrock<T: Float> {
        a: T,
        b: T,
    }

    // For the batched version we need to implement BatchGradientTarget.
    impl<T, B> BatchGradientTarget<T, B> for Rosenbrock<T>
    where
        T: Float + std::fmt::Debug + Element,
        B: burn::tensor::backend::AutodiffBackend,
    {
        fn log_prob_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1> {
            // Assume `positions` is a Tensor<B, 2> with shape [n, 2].
            let n = positions.dims()[0] as i64;
            // dbg!(&positions);
            let x = positions.clone().slice([(0, n), (0, 1)]); // shape: [n, 1]
            let y = positions.slice([(0, n), (1, 2)]); // shape: [n, 1]
                                                       // dbg!(&x);
                                                       // dbg!(&y);

            // Compute (a - x)^2 vectorized:
            let term1 = (-x.clone()).add_scalar(self.a).powi_scalar(2.0);

            // Compute (y - x^2)^2 vectorized:
            let term2 = y.sub(x.clone().mul(x)).powi_scalar(2.0).mul_scalar(self.b);

            // Return the negative sum as a 2D tensor [n, 1] (optionally flatten it to [n]):
            -(term1 + term2).flatten(0, 1)
        }
    }

    /// Test that runs the data-parallel HMC sampler, collects all chain samples, and saves them to a Parquet file.
    #[test]
    fn test_collect_hmc_samples_and_save_parquet_single() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define initial positions for 4 chains, each 2-dimensional.
        let initial_positions = vec![vec![0.0_f32, 0.0]];
        let n_chains = initial_positions.len();
        let dim = initial_positions[0].len();
        let n_steps = 3;

        // Create the data-parallel HMC sampler.
        let mut sampler = DataParallelHMC::<f32, BackendType, Rosenbrock<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            2,    // number of leapfrog steps per update
            42,   // RNG seed
        );

        // Prepare a history container: one Vec<f32> per chain.
        // Each chain's sample (a [dim] vector) will be appended consecutively.
        let mut history: Vec<Vec<f32>> = vec![Vec::with_capacity(n_steps * dim); n_chains];

        let start = Instant::now();
        // Run HMC for n_steps, collecting samples.
        for step in 0..n_steps {
            let start_step = Instant::now();
            // Run HMC for n_steps, collecting samples.
            sampler.step();
            if step % 10 == 0 {
                println!("Step: {step}");
                println!("Step runtime: {:?}", start_step.elapsed());
            }

            // sampler.positions has shape [n_chains, dim]
            let start_store = Instant::now();
            let flat: Vec<f32> = sampler.positions.to_data().to_vec::<f32>().unwrap();
            (0..n_chains).for_each(|chain_idx| {
                let start_idx = chain_idx * dim;
                let end_idx = start_idx + dim;
                history[chain_idx].extend_from_slice(&flat[start_idx..end_idx]);
            });
            if step % 10 == 0 {
                println!("Store runtime: {:?}", start_store.elapsed());
            }
        }
        let duration = start.elapsed();
        println!("HMC sampler: {} steps took {:?}", n_steps, duration);

        // Convert each chain's history (a Vec<f32>) into a DMatrix.
        // Each chain's matrix will have shape [n_steps, dim].
        let mut matrices = Vec::with_capacity(n_chains);
        for chain_data in history.iter() {
            let matrix = na::DMatrix::from_row_slice(n_steps, dim, chain_data);
            matrices.push(matrix);
        }
        println!(
            "Number of samples: {}",
            matrices.len() * matrices[0].nrows()
        );

        // Save the collected samples to a Parquet file using the provided I/O utility.
        // save_parquet(&matrices, "rosenbrock_hmc.parquet").unwrap();
    }

    #[test]
    fn test_collect_hmc_samples_and_save_parquet_10() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 10 chains all initialized to (1.0, 2.0).
        let mut initial_positions = Vec::with_capacity(10);
        for _ in 0..10 {
            initial_positions.push(vec![1.0_f32, 2.0_f32]);
        }
        let n_chains = initial_positions.len(); // 1000
        let dim = initial_positions[0].len(); // 2
        let n_steps = 1000;

        // Create the data-parallel HMC sampler.
        let mut sampler = DataParallelHMC::<f32, BackendType, Rosenbrock<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            10,   // number of leapfrog steps per update
            42,   // RNG seed
        );

        // Prepare a history container: one Vec<f32> per chain.
        // Each chain's sample (a [dim] vector) will be appended consecutively.
        let mut history: Vec<Vec<f32>> = vec![Vec::with_capacity(n_steps * dim); n_chains];

        let start = Instant::now();
        // Run HMC for n_steps, collecting samples.
        for step in 0..n_steps {
            let start_step = Instant::now();
            sampler.step();
            if step % 10 == 0 {
                println!("Step: {step}");
                println!("Step runtime: {:?}", start_step.elapsed());
            }

            // `positions` has shape [n_chains, dim]
            let start_store = Instant::now();
            let flat: Vec<f32> = sampler.positions.to_data().to_vec::<f32>().unwrap();
            (0..n_chains).for_each(|chain_idx| {
                let start_idx = chain_idx * dim;
                let end_idx = start_idx + dim;
                history[chain_idx].extend_from_slice(&flat[start_idx..end_idx]);
            });
            if step % 10 == 0 {
                println!("Store runtime: {:?}", start_store.elapsed());
            }
        }
        let duration = start.elapsed();
        println!("HMC sampler: {} steps took {:?}", n_steps, duration);

        // Convert each chain's history (a Vec<f32>) into an nalgebra DMatrix.
        // Each chain's matrix will have shape [n_steps, dim].
        let mut matrices = Vec::with_capacity(n_chains);
        for chain_data in &history {
            let matrix = na::DMatrix::from_row_slice(n_steps, dim, chain_data);
            matrices.push(matrix);
        }
        println!(
            "Number of samples: {} ({} chains × {} steps)",
            matrices.len() * matrices[0].nrows(),
            n_chains,
            n_steps
        );

        // save_parquet(&matrices, "rosenbrock_hmc_1000.parquet").unwrap();
    }

    #[test]
    fn test_collect_hmc_samples_benchmark() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<burn::backend::NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let mut initial_positions = Vec::with_capacity(6);
        for _ in 0..6 {
            initial_positions.push(vec![1.0_f32, 2.0_f32]);
        }
        let n_chains = initial_positions.len();
        let dim = initial_positions[0].len();
        let n_steps = 5000;

        // Create the data-parallel HMC sampler.
        let mut sampler = DataParallelHMC::<f32, BackendType, Rosenbrock<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
            42,   // RNG seed
        );

        // Prepare a history container: one Vec<f32> per chain.
        // Each chain's sample (a [dim] vector) will be appended consecutively.
        let mut history: Vec<Vec<f32>> = vec![Vec::with_capacity(n_steps * dim); n_chains];

        let start = Instant::now();
        // Run HMC for n_steps, collecting samples.
        for _ in 0..n_steps {
            sampler.step();
            let flat: Vec<f32> = sampler.positions.to_data().to_vec::<f32>().unwrap();
            (0..n_chains).for_each(|chain_idx| {
                let start_idx = chain_idx * dim;
                let end_idx = start_idx + dim;
                history[chain_idx].extend_from_slice(&flat[start_idx..end_idx]);
            });
        }
        let duration = start.elapsed();
        println!("HMC sampler: {} steps took {:?}", n_steps, duration);
    }
}
