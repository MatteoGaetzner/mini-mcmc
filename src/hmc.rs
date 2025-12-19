//! Hamiltonian Monte Carlo (HMC) sampler.
//!
//! This module now routes the public `HMC` API through a backend-agnostic,
//! in-place core (`GenericHMC`). When the `burn` feature is enabled (default),
//! the familiar `HMC<T, B, GTarget>` wrapper is available and backwards
//! compatible with previous versions while benefiting from the allocation-free
//! integrator underneath.

use crate::distributions::BatchedGradientTarget;
use crate::generic_hmc::{GenericHMC, HamiltonianTarget};
use crate::stats::RunStats;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::Element;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::distr::Distribution as RandDistribution;
// Keep trait bounds on rand's Distribution to avoid mixed-rand version mismatches.
use rand_distr::uniform::SampleUniform;
use rand_distr::{StandardNormal, StandardUniform};
use rand::rngs::SmallRng;
use std::error::Error;

#[derive(Clone, Debug)]
struct BurnBatchedTarget<GTarget> {
    inner: GTarget,
}

impl<T, B, GTarget> HamiltonianTarget<Tensor<B, 1>> for BurnBatchedTarget<GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync,
    StandardNormal: RandDistribution<T>,
{
    fn logp_and_grad(&self, position: &Tensor<B, 1>, grad: &mut Tensor<B, 1>) -> T {
        let pos = position.clone().unsqueeze_dim(0).detach().require_grad();
        let logp = self.inner.unnorm_logp_batch(pos.clone());
        let logp_scalar = logp.clone().into_scalar();

        let grads_inner = pos.grad(&logp.backward()).expect("grad computation to succeed");
        let grad_tensor = Tensor::<B, 2>::from_inner(grads_inner).squeeze(0);
        grad.inplace(|_| grad_tensor.clone());

        logp_scalar
    }
}

/// A data-parallel Hamiltonian Monte Carlo (HMC) sampler using the burn backend.
#[derive(Debug, Clone)]
pub struct HMC<T, B, GTarget>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive + ToPrimitive + Copy,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync + Clone,
    StandardNormal: RandDistribution<T>,
    StandardUniform: RandDistribution<T>,
{
    inner: GenericHMC<Tensor<B, 1>, BurnBatchedTarget<GTarget>>,
    /// The target distribution which provides log probability evaluations and gradients.
    pub target: GTarget,
    /// The step size for the leapfrog integrator.
    pub step_size: T,
    /// The number of leapfrog steps to take per HMC update.
    pub n_leapfrog: usize,
    /// Exposed RNG for compatibility; mirrors the internal engine RNG.
    pub rng: SmallRng,
    /// Current positions stacked as a `[n_chains, dim]` tensor.
    pub positions: Tensor<B, 2>,
    dim: usize,
}

impl<T, B, GTarget> HMC<T, B, GTarget>
where
    T: Float
        + burn::tensor::ElementConversion
        + burn::tensor::Element
        + SampleUniform
        + num_traits::FromPrimitive
        + num_traits::ToPrimitive,
    B: AutodiffBackend<FloatElem = T>,
    GTarget: BatchedGradientTarget<T, B> + std::marker::Sync + Clone,
    StandardNormal: RandDistribution<T>,
    StandardUniform: RandDistribution<T>,
{
    pub fn new(
        target: GTarget,
        initial_positions: Vec<Vec<T>>,
        step_size: T,
        n_leapfrog: usize,
    ) -> Self {
        let positions_vec: Vec<Tensor<B, 1>> = initial_positions
            .into_iter()
            .map(|pos| {
                let len = pos.len();
                let td: TensorData = TensorData::new(pos, [len]);
                Tensor::<B, 1>::from_data(td, &B::Device::default())
            })
            .collect();
        let dim = positions_vec[0].dims()[0];
        let inner = GenericHMC::new(
            BurnBatchedTarget {
                inner: target.clone(),
            },
            positions_vec,
            step_size,
            n_leapfrog,
        );
        let positions = stack_positions(inner.positions());
        let rng = inner.rng_clone();
        Self {
            inner,
            target,
            step_size,
            n_leapfrog,
            rng,
            positions,
            dim,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.set_seed(seed);
        self.rng = self.inner.rng_clone();
        self
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        let sample = self.inner.run(n_collect, n_discard);
        self.refresh_positions();
        array3_to_tensor(sample)
    }

    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        let (sample, stats) = self.inner.run_progress(n_collect, n_discard)?;
        self.refresh_positions();
        Ok((array3_to_tensor(sample), stats))
    }

    pub fn step(&mut self) {
        self.inner.step();
        self.refresh_positions();
        self.rng = self.inner.rng_clone();
    }

    fn refresh_positions(&mut self) {
        self.positions = stack_positions(self.inner.positions());
    }
}

fn array3_to_tensor<B, T>(arr: ndarray::Array3<T>) -> Tensor<B, 3>
where
    B: AutodiffBackend<FloatElem = T>,
    T: Float + Element + ElementConversion,
{
    let shape = arr.raw_dim();
    let td = TensorData::new(arr.into_raw_vec(), [shape[0], shape[1], shape[2]]);
    Tensor::<B, 3>::from_data(td, &B::Device::default())
}

fn stack_positions<B, T>(positions: &[Tensor<B, 1>]) -> Tensor<B, 2>
where
    B: AutodiffBackend<FloatElem = T>,
    T: Float + Element + ElementConversion,
{
    Tensor::<B, 1>::stack(positions.to_vec(), 0)
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
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = vec![vec![0.0_f32, 0.0]];
        let n_collect = 3;

        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01,
            2,
        )
        .set_seed(42);

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
        type BackendType = Autodiff<NdArray>;

        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 3];
        let n_collect = 10;

        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01,
            2,
        )
        .set_seed(42);

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
        type BackendType = Autodiff<NdArray>;

        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = vec![vec![1.0_f32, 2.0_f32]; 3];
        let n_collect = 10;

        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.05,
            2,
        )
        .set_seed(42);

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
    fn test_bench_noprogress() {
        type BackendType = Autodiff<burn::backend::NdArray>;

        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        let initial_positions = init(6, 2);
        let n_collect = 5000;
        let n_discard = 500;

        let mut sampler = HMC::<f32, BackendType, Rosenbrock2D<f32>>::new(
            target,
            initial_positions,
            0.01,
            50,
        )
        .set_seed(42);

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
}
