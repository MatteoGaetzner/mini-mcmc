//! A simple Hamiltonian (Hybrid) Monte Carlo sampler using the `burn` crate for autodiff.
//!
//! This is modeled similarly to your Metropolis–Hastings approach but uses gradient-based
//! proposals.

use burn::prelude::Backend;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Tensor;
use num_traits::Float;
use rand::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;
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
/// - `current_pos`: The current position in parameter space (as a `[D]` 1D vector).
///    We store it as a 2D tensor `[1, D]` for convenience.
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
            Tensor::<B, 1>::from_data(self.current_pos.as_slice(), &B::Device::default());

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
                (new_pos, new_mom, new_logp)
            },
        );

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

// -- 1) A batched target trait (see above) --
pub trait BatchGradientTarget<T: Float, B: AutodiffBackend> {
    fn log_prob_batch(&self, positions: Tensor<B, 2>) -> Tensor<B, 1>;
}

// -- 2) The data-parallel HMC struct --
pub struct DataParallelHMC<T, B, GTarget>
where
    B: Backend,
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
    B: AutodiffBackend + Backend,
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
        let shape = self.positions.shape();
        let (n_chains, dim) = (shape.dims[0], shape.dims[1]);

        // 1) Sample momenta: shape [n_chains, D]
        let momentum_0 = Tensor::<B, 2>::random(
            Shape::new([n_chains, dim]),
            burn::tensor::Distribution::Normal(0., 1.),
            &B::Device::default(),
        );

        // Current log-prob, shape [n_chains]
        let logp_current = self.target.log_prob_batch(self.positions.clone());

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

        self.positions.clone_from(
            &self
                .positions
                .clone()
                .mask_where(accept_mask_big, proposed_positions)
                .detach(),
        );
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
            pos = pos.detach().clone().require_grad();

            // Compute gradient of log_prob wrt pos (all chains in parallel!)
            let logp = self.target.log_prob_batch(pos.clone()); // shape [n_chains]
            let grads = {
                let _tape = logp.backward(); // gather grads
                                             // gradient wrt pos => shape [n_chains, D]
                let grad_pos_data = pos.grad(&_tape).unwrap().to_data();
                Tensor::<B, 2>::from_data(grad_pos_data, &pos.device())
            };

            // First half-step for momentum
            mom = mom.add(grads.mul_scalar(self.step_size * half));

            // Full step in position
            pos = pos
                .add(mom.clone().mul_scalar(self.step_size))
                .detach() // remove old gradient info
                .require_grad(); // but we still want to track new gradients
                                 // Second half-step for momentum
            let logp2 = self.target.log_prob_batch(pos.clone());
            let grads2 = {
                let _tape = logp2.backward();
                let grad_pos_data2 = pos.grad(&_tape).unwrap().to_data();
                Tensor::<B, 2>::from_data(grad_pos_data2, &pos.device())
            };
            mom = mom.add(grads2.mul_scalar(self.step_size * half));
        }

        // Final logp for these positions
        let logp_final = self.target.log_prob_batch(pos.clone());

        (pos.detach(), mom.detach(), logp_final.detach())
    }
}

#[cfg(test)]
mod tests {
    use crate::io::save_parquet;
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
    fn test_collect_hmc_samples_and_save_parquet() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // Define initial positions for 4 chains, each 2-dimensional.
        let initial_positions = vec![
            vec![0.0_f32, 0.0],
            vec![1.0, 1.0],
            vec![-1.0, 1.0],
            vec![1.0, -1.0],
        ];
        let n_chains = initial_positions.len();
        let dim = initial_positions[0].len();
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
            // Run HMC for n_steps, collecting samples.
            sampler.step();
            if step % 100 == 0 {
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
            if step % 100 == 0 {
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
        save_parquet(&matrices, "rosenbrock_hmc.parquet").unwrap();
    }

    #[test]
    fn test_collect_hmc_samples_and_save_parquet_1000_chains() {
        // Use the CPU backend (NdArray) wrapped in Autodiff.
        type BackendType = Autodiff<NdArray>;

        // Create the Rosenbrock target (a = 1, b = 100)
        let target = Rosenbrock {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 1000 chains all initialized to (1.0, 2.0).
        let mut initial_positions = Vec::with_capacity(1000);
        for _ in 0..1000 {
            initial_positions.push(vec![1.0_f32, 2.0_f32]);
        }
        let n_chains = initial_positions.len(); // 1000
        let dim = initial_positions[0].len(); // 2
        let n_steps = 100;

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

        // Save the collected samples to a Parquet file (assuming you have `save_parquet`).
        // You can change the filename as desired.
        save_parquet(&matrices, "rosenbrock_hmc_1000.parquet").unwrap();
    }
}
