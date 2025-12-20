//! Burn-backed NUTS wrapper over the backend-agnostic core.

use crate::distributions::GradientTarget;
use crate::generic_hmc::HamiltonianTarget;
use crate::generic_nuts::{GenericNUTS, GenericNUTSChain};
use crate::stats::RunStats;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
#[cfg(test)]
use burn::tensor::Tensor;
use burn::tensor::{Element, ElementConversion};
use num_traits::{Float, FromPrimitive};
use rand::distr::Distribution as RandDistribution;
// Bind to rand's Distribution to avoid mismatches from transitive rand 0.8 deps.
#[cfg(test)]
use rand::rngs::SmallRng;
#[cfg(test)]
use rand::SeedableRng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use std::error::Error;
use std::marker::PhantomData;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

#[derive(Debug)]
struct BurnGradientTarget<GTarget, T> {
    inner: GTarget,
    _marker: PhantomData<T>,
}

impl<T, B, GTarget> HamiltonianTarget<Tensor<B, 1>> for BurnGradientTarget<GTarget, T>
where
    T: Float + Element + ElementConversion + SampleUniform + FromPrimitive,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync,
    StandardNormal: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    fn logp_and_grad(&self, position: &Tensor<B, 1>, grad: &mut Tensor<B, 1>) -> B::FloatElem {
        let (logp, grad_tensor) = self.inner.unnorm_logp_and_grad(position.clone());
        grad.inplace(|_| grad_tensor.clone());
        logp.into_scalar()
    }
}

/// Burn-backed No-U-Turn Sampler (NUTS).
pub struct NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    inner: GenericNUTS<Tensor<B, 1>, BurnGradientTarget<GTarget, T>>,
    _phantom: PhantomData<T>,
}

impl<T, B, GTarget> NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    pub fn new(target: GTarget, initial_positions: Vec<Vec<T>>, target_accept_p: T) -> Self {
        let positions_vec: Vec<Tensor<B, 1>> = initial_positions
            .into_iter()
            .map(|pos| {
                let len = pos.len();
                let pos_elem: Vec<B::FloatElem> =
                    pos.into_iter().map(B::FloatElem::from_elem).collect();
                let td: TensorData = TensorData::new(pos_elem, [len]);
                Tensor::<B, 1>::from_data(td, &B::Device::default())
            })
            .collect();
        let target_accept_p_elem = B::FloatElem::from_elem(target_accept_p);
        let inner = GenericNUTS::new(
            BurnGradientTarget {
                inner: target,
                _marker: PhantomData,
            },
            positions_vec,
            target_accept_p_elem,
        );
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        if n_collect == 0 {
            let (n_chains, dim) = {
                let chains = self.inner.chains_mut();
                let n_chains = chains.len();
                let dim = chains[0].position().dims()[0];
                (n_chains, dim)
            };
            return Tensor::<B, 3>::empty([n_chains, 0, dim], &B::Device::default());
        }

        let chain_samples: Vec<Tensor<B, 2>> = self
            .inner
            .chains_mut()
            .par_iter_mut()
            .map(|chain| chain.run(n_collect, n_discard))
            .collect();
        Tensor::<B, 2>::stack(chain_samples, 0)
    }

    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        let (sample, stats) = self.inner.run_progress(n_collect, n_discard)?;
        Ok((array3_to_tensor(sample), stats))
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        B::seed(seed);
        self.inner = self.inner.set_seed(seed);
        self
    }
}

/// Burn-backed single-chain NUTS wrapper.
pub struct NUTSChain<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    inner: GenericNUTSChain<Tensor<B, 1>, BurnGradientTarget<GTarget, T>>,
    _phantom: PhantomData<T>,
}

impl<T, B, GTarget> NUTSChain<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Send,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync + Send,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    pub fn new(target: GTarget, initial_position: Vec<T>, target_accept_p: T) -> Self {
        let len = initial_position.len();
        let position_elem: Vec<B::FloatElem> = initial_position
            .into_iter()
            .map(B::FloatElem::from_elem)
            .collect();
        let td: TensorData = TensorData::new(position_elem, [len]);
        let position = Tensor::<B, 1>::from_data(td, &B::Device::default());
        let inner = GenericNUTSChain::new(
            BurnGradientTarget {
                inner: target,
                _marker: PhantomData,
            },
            position.clone(),
            B::FloatElem::from_elem(target_accept_p),
        );
        Self {
            inner,
            _phantom: PhantomData,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        B::seed(seed);
        self.inner = self.inner.set_seed(seed);
        self
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        if n_collect == 0 {
            let dim = self.inner.position().dims()[0];
            return Tensor::<B, 2>::empty([0, dim], &B::Device::default());
        }

        let dim = self.inner.position().dims()[0];
        let mut out = Tensor::<B, 2>::empty([n_collect, dim], &B::Device::default());
        let total = n_collect + n_discard;

        for step_idx in 0..total {
            if step_idx > 0 {
                self.inner.step();
            }
            if step_idx >= n_discard {
                let row = self.inner.position().clone().unsqueeze_dim(0);
                out.inplace(|tensor| {
                    tensor.slice_assign(
                        [step_idx - n_discard..step_idx - n_discard + 1, 0..dim],
                        row,
                    )
                });
            }
        }
        out
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn position(&self) -> &Tensor<B, 1> {
        self.inner.position()
    }
}

fn array3_to_tensor<B, T>(arr: ndarray::Array3<T>) -> Tensor<B, 3>
where
    B: AutodiffBackend<FloatElem = T>,
    T: Float + Element + ElementConversion,
{
    let shape = arr.raw_dim();
    let (mut data, offset) = arr.into_raw_vec_and_offset();
    if let Some(offset) = offset {
        if offset != 0 {
            data.rotate_left(offset);
        }
    }
    let td = TensorData::new(data, [shape[0], shape[1], shape[2]]);
    Tensor::<B, 3>::from_data(td, &B::Device::default())
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;
    use std::marker::PhantomData;

    use crate::{
        core::init,
        dev_tools::Timer,
        distributions::{DiffableGaussian2D, Rosenbrock2D},
        generic_nuts::{build_tree, find_reasonable_epsilon},
        stats::split_rhat_mean_ess,
    };

    #[cfg(feature = "csv")]
    use crate::io::csv::save_csv_tensor;

    use super::*;
    use burn::{
        backend::Autodiff,
        tensor::{Tensor, Tolerance},
    };
    use ndarray::ArrayView3;
    use ndarray_stats::QuantileExt;
    use num_traits::Float;

    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<burn::backend::NdArray<f64>>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StandardNormal;

    impl<T, B> GradientTarget<T, B> for StandardNormal
    where
        T: Float + Debug + ElementConversion + Element,
        B: AutodiffBackend,
    {
        fn unnorm_logp(&self, positions: Tensor<B, 1>) -> Tensor<B, 1> {
            let sq = positions.clone().powi_scalar(2);
            let half = T::from(0.5).unwrap();
            -(sq.mul_scalar(half)).sum()
        }
    }

    fn assert_tensor_approx_eq<T: Backend, F: Float + burn::tensor::Element>(
        actual: Tensor<T, 1>,
        expected: &[f64],
        tol: Tolerance<F>,
    ) {
        let a = actual.clone().to_data();
        let e = Tensor::<T, 1>::from(expected).to_data();
        a.assert_approx_eq(&e, tol);
    }

    #[test]
    fn test_find_reasonable_epsilon() {
        // Use BurnGradientTarget to wrap StandardNormal for the generic function
        let target = BurnGradientTarget::<StandardNormal, f64> {
            inner: StandardNormal,
            _marker: PhantomData,
        };
        let position = Tensor::<BackendType, 1>::from([0.0, 1.0]);
        let mom = Tensor::<BackendType, 1>::from([1.0, 0.0]);
        let epsilon: f64 = find_reasonable_epsilon(&position, &mom, &target);
        assert_eq!(epsilon, 2.0);
    }

    #[test]
    fn test_build_tree() {
        // Use BurnGradientTarget to wrap DiffableGaussian2D for the generic function
        let gradient_target = BurnGradientTarget {
            inner: DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]),
            _marker: PhantomData::<f64>,
        };
        let position = Tensor::<BackendType, 1>::from([0.0, 1.0]);
        let mom = Tensor::<BackendType, 1>::from([2.0, 3.0]);
        let grad = Tensor::<BackendType, 1>::from([4.0, 5.0]);
        let logu = -2.0;
        let v: i8 = -1;
        let j: usize = 3;
        let epsilon: f64 = 0.01;
        let joint_0 = 0.1_f64;
        let mut rng = SmallRng::seed_from_u64(0);
        let (
            position_minus,
            mom_minus,
            grad_minus,
            position_plus,
            mom_plus,
            grad_plus,
            position_prime,
            grad_prime,
            logp_prime,
            n_prime,
            s_prime,
            alpha_prime,
            n_alpha_prime,
        ) = build_tree(
            position,
            mom,
            grad,
            logu,
            v,
            j,
            epsilon,
            &gradient_target,
            joint_0,
            &mut rng,
        );
        let tol = Tolerance::<f64>::default()
            .set_relative(1e-5)
            .set_absolute(1e-6);

        assert_tensor_approx_eq(position_minus, &[-0.1584001, 0.76208336], tol);
        assert_tensor_approx_eq(mom_minus, &[1.980_003_6, 2.971_825_3], tol);
        assert_tensor_approx_eq(grad_minus, &[-7.912_36e-5, 7.935_829_5e-2], tol);

        assert_tensor_approx_eq(position_plus, &[-0.0198, 0.97025], tol);
        assert_tensor_approx_eq(mom_plus, &[1.98, 2.974_950_3], tol);
        assert_tensor_approx_eq(grad_plus, &[-1.250e-05, 9.925e-03], tol);

        assert_tensor_approx_eq(position_prime, &[-0.0198, 0.97025], tol);
        assert_tensor_approx_eq(grad_prime, &[-1.250e-05, 9.925e-03], tol);

        assert_eq!(n_prime, 0);
        assert!(s_prime);
        assert_eq!(n_alpha_prime, 8);

        let logp_exp = -2.877_745_4_f64;
        let alpha_exp = 0.000_686_661_7_f64;
        assert!((logp_prime - logp_exp).abs() < 1e-6, "logp mismatch");
        assert!((alpha_prime - alpha_exp).abs() < 1e-8, "alpha mismatch");
    }

    #[test]
    fn test_chain_1() {
        let target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![0.0_f64, 1.0];
        let n_discard = 0;
        let n_collect = 1;
        let mut sampler = NUTSChain::new(target, initial_positions, 0.8).set_seed(42);
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [n_collect, 2]);
        let tol = Tolerance::<f64>::default()
            .set_relative(1e-5)
            .set_absolute(1e-6);
        assert_tensor_approx_eq(sample.flatten(0, 1), &[0.0, 1.0], tol);
    }

    #[test]
    fn test_chain_2() {
        let target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
        let initial_positions = vec![0.0_f64, 1.0];
        let n_discard = 3;
        let n_collect = 3;
        let mut sampler = NUTSChain::new(target, initial_positions, 0.8).set_seed(42);
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [n_collect, 2]);

        // Statistical assertion: samples should be finite and reasonable
        let data = sample.to_data();
        let values: &[f64] = data.as_slice().expect("dense data");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "All samples should be finite"
        );
        assert!(
            values.iter().all(|v| v.abs() < 100.0),
            "Samples should be reasonable magnitude"
        );
    }

    #[test]
    fn test_chain_3() {
        let target = DiffableGaussian2D::new([1.0_f64, 2.0], [[1.0, 2.0], [2.0, 5.0]]);
        let initial_positions = vec![-2.0_f64, 1.0];
        let n_discard = 5;
        let n_collect = 5;
        let mut sampler = NUTSChain::new(target, initial_positions, 0.8).set_seed(42);
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [n_collect, 2]);

        // Statistical assertion: samples should be finite and reasonable
        let data = sample.to_data();
        let values: &[f64] = data.as_slice().expect("dense data");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "All samples should be finite"
        );
        assert!(
            values.iter().all(|v| v.abs() < 100.0),
            "Samples should be reasonable magnitude"
        );
    }

    #[test]
    fn test_run_1() {
        let target = DiffableGaussian2D::new([1.0_f64, 2.0], [[1.0, 2.0], [2.0, 5.0]]);
        let initial_positions = vec![vec![-2_f64, 1.0]];
        let n_discard = 5;
        let n_collect = 5;
        let mut sampler = NUTS::new(target, initial_positions, 0.8).set_seed(41);
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
        assert_eq!(sample.dims(), [1, n_collect, 2]);

        // Statistical assertion: samples should be finite and reasonable
        let data = sample.to_data();
        let values: &[f64] = data.as_slice().expect("dense data");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "All samples should be finite"
        );
        assert!(
            values.iter().all(|v| v.abs() < 100.0),
            "Samples should be reasonable magnitude"
        );
    }

    #[test]
    fn test_progress_1() {
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init::<f32>(6, 2);
        let n_collect = 10;
        let n_discard = 10;

        let mut sampler =
            NUTS::<_, BackendType, _>::new(target, initial_positions, 0.95).set_seed(42);
        let (sample, stats) = sampler.run_progress(n_collect, n_discard).unwrap();
        println!(
            "NUTS sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        );
        assert_eq!(sample.dims(), [6, n_collect, 2]);

        println!("Statistics: {stats}");

        #[cfg(feature = "csv")]
        save_csv_tensor(sample, "/tmp/nuts-sample.csv").expect("saving data should succeed")
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_noprogress_1() {
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init::<f32>(6, 2);
        let n_collect = 5000;
        let n_discard = 500;

        let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "NUTS sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [6, 5000, 2]);

        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice().unwrap()).unwrap();
        let (split_rhat, ess) = split_rhat_mean_ess(array);
        println!("AVG Split Rhat: {}", split_rhat.mean().unwrap());
        println!("AVG ESS: {}", ess.mean().unwrap());

        #[cfg(feature = "csv")]
        save_csv_tensor(sample, "/tmp/nuts-sample.csv").expect("saving data should succeed")
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_bench_noprogress_2() {
        let target = Rosenbrock2D {
            a: 1.0_f32,
            b: 100.0_f32,
        };

        // We'll define 6 chains all initialized to (1.0, 2.0).
        let initial_positions = init::<f32>(6, 2);
        let n_collect = 1000;
        let n_discard = 1000;

        let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "NUTS sampler: generated {} observations.",
            sample.dims()[0..2].iter().product::<usize>()
        ));
        assert_eq!(sample.dims(), [6, 1000, 2]);

        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice().unwrap()).unwrap();
        let (split_rhat, ess) = split_rhat_mean_ess(array);
        println!("MIN Split Rhat: {}", split_rhat.min().unwrap());
        println!("MIN ESS: {}", ess.min().unwrap());

        #[cfg(feature = "csv")]
        save_csv_tensor(sample, "/tmp/nuts-sample.csv").expect("saving data should succeed")
    }
}
