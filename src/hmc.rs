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

// -- 1) A batched target trait (see above) --
pub trait GradientTarget<T: Float, B: AutodiffBackend> {
    fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1>;
}

// -- 2) The data-parallel HMC struct --
pub struct HMC<T, B, GTarget>
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

impl<T, B, GTarget> HMC<T, B, GTarget>
where
    T: Float
        + burn::tensor::ElementConversion
        + burn::tensor::Element
        + rand_distr::uniform::SampleUniform,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B>,
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

    pub fn run(&mut self, n_steps: usize, discard: usize) -> Tensor<B, 3> {
        let (n_chains, dim) = (self.positions.dims()[0], self.positions.dims()[1]);
        let mut out = Tensor::<B, 3>::empty(
            [n_steps, n_chains, self.positions.dims()[1]],
            &B::Device::default(),
        );

        // Discard first `discard` positions
        (0..discard).for_each(|_| self.step());

        // Collect samples
        for step in 1..(n_steps + 1) {
            self.step();
            out.inplace(|x| {
                x.slice_assign(
                    [step - 1..step, 0..n_chains, 0..dim],
                    self.positions.clone().unsqueeze_dim(0),
                )
            });
        }
        out
    }

    /// Perform one *batched* HMC update for all chains in parallel:
    /// 1) Sample momenta from N(0, I).
    /// 2) Run leapfrog steps in batch.
    /// 3) Accept/reject per chain.
    pub fn step(&mut self) {
        let shape = self.positions.shape();
        let (n_chains, dim) = (shape.dims[0], shape.dims[1]);

        // 1) Sample momenta: shape [n_chains, D]
        let momentum_0 = Tensor::<B, 2>::random(
            Shape::new([n_chains, dim]),
            burn::tensor::Distribution::Normal(0., 1.),
            &B::Device::default(),
        );

        // Current log-prob, shape [n_chains]
        let logp_current = self.target.log_prob_batch(&self.positions);

        // Kinetic energy: 0.5 * sum_{d} (p^2) per chain => shape [n_chains]
        let ke_current = momentum_0
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1) // sum over dimension=1 => shape [n_chains]
            .squeeze(1)
            .mul_scalar(T::from(0.5).unwrap());

        // "Hamiltonian" = -logp + KE, shape [n_chains]
        let h_current: Tensor<B, 1> = -logp_current + ke_current;

        // 2) Run leapfrog integrator
        let (proposed_positions, proposed_momenta, logp_proposed) =
            self.leapfrog(self.positions.clone(), momentum_0);

        // Proposed kinetic
        let ke_proposed = proposed_momenta
            .powf_scalar(2.0)
            .sum_dim(1)
            .squeeze(1)
            .mul_scalar(T::from(0.5).unwrap());

        let h_proposed = -logp_proposed + ke_proposed;

        // 3) Accept/Reject per chain
        //    accept_logp = -(h_proposed - h_current) = h_current - h_proposed
        let accept_logp = h_current.sub(h_proposed);

        // We draw uniform(0,1) for each chain => shape [n_chains]
        let mut uniform_data = Vec::with_capacity(n_chains);
        for _i in 0..n_chains {
            uniform_data.push(self.rng.gen::<T>());
        }
        let uniform = Tensor::<B, 1>::random(
            Shape::new([n_chains]),
            burn::tensor::Distribution::Default,
            &B::Device::default(),
        );

        // Condition: accept_logp >= ln(u)
        let ln_u = uniform.log(); // shape [n_chains]
        let accept_mask = accept_logp.greater_equal(ln_u); // Boolean mask: shape [n_chains]
        let mut accept_mask_big: Tensor<B, 2, Bool> = accept_mask.clone().unsqueeze_dim(1);
        accept_mask_big = accept_mask_big.expand([n_chains, dim]);

        self.positions.inplace(|x| {
            x.clone()
                .mask_where(accept_mask_big, proposed_positions)
                .detach()
        });
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
        let half = T::from(0.5).unwrap();
        for _step_i in 0..self.n_leapfrog {
            // Make sure pos is AD-enabled
            pos = pos.detach().require_grad();

            // Compute gradient of log_prob wrt pos (all chains in parallel!)
            let logp = self.target.log_prob_batch(&pos); // shape [n_chains]
            let grads = pos.grad(&logp.backward()).unwrap();

            // First half-step for momentum
            let mom_inner = mom
                .clone()
                .inner()
                .add(grads.mul_scalar(self.step_size * half));

            // Full step in position
            pos = Tensor::<B, 2>::from_inner(pos.inner().add(mom_inner.mul_scalar(self.step_size)))
                .detach()
                .require_grad();

            // Second half-step for momentum
            let logp2 = self.target.log_prob_batch(&pos);
            let grads2 = pos.grad(&logp2.backward()).unwrap();
            mom = Tensor::<B, 2>::from_inner(
                mom.inner().add(grads2.mul_scalar(self.step_size * half)),
            );
        }

        // Final logp for these positions
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
    #[derive(Clone, Copy)]
    struct Rosenbrock<T: Float> {
        a: T,
        b: T,
    }

    // For the batched version we need to implement BatchGradientTarget.
    impl<T, B> GradientTarget<T, B> for Rosenbrock<T>
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

    #[test]
    fn test_collect_hmc_samples_single() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define initial positions for a single chain (2-dimensional).
        let initial_positions = vec![vec![0.0_f32, 0.0]];
        let n_steps = 3;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            2,    // number of leapfrog steps per update
            42,   // RNG seed
        );

        // Run the sampler for n_steps with no discard.
        let mut timer = Timer::new();
        let samples: Tensor<BackendType, 3> = sampler.run(n_steps, 0);
        timer.log(format!(
            "Collected samples (10 chains) with shape: {:?}",
            samples.dims()
        ))
    }

    #[test]
    fn test_collect_hmc_samples_10_chains() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define 10 chains all initialized to (1.0, 2.0).
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 10];
        let n_steps = 1000;

        // Create the HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            10,   // number of leapfrog steps per update
            42,   // RNG seed
        );

        // Run the sampler for n_steps with no discard.
        let mut timer = Timer::new();
        let samples: Tensor<BackendType, 3> = sampler.run(n_steps, 0);
        timer.log(format!(
            "Collected samples (10 chains) with shape: {:?}",
            samples.dims()
        ))
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
        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 6];
        let n_steps = 5000;

        // Create the data-parallel HMC sampler.
        let mut sampler = HMC::<f32, BackendType, Rosenbrock<f32>>::new(
            target,
            initial_positions,
            0.01, // step size
            50,   // number of leapfrog steps per update
            42,   // RNG seed
        );

        // Run HMC for n_steps, collecting samples.
        let mut timer = Timer::new();
        let samples = sampler.run(n_steps, 0);
        timer.log(format!(
            "HMC sampler: generated {} samples.",
            samples.dims()[0..2].iter().product::<usize>()
        ))
    }
}
