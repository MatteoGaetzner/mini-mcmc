//! Burn-backed NUTS wrapper over the backend-agnostic core.

use crate::distributions::GradientTarget;
use crate::generic_hmc::HamiltonianTarget;
use crate::generic_nuts::{GenericNUTS, GenericNUTSChain};
use crate::stats::RunStats;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
#[cfg(test)]
use burn::tensor::cast::ToElement;
use burn::tensor::{Element, ElementConversion, Tensor};
use num_traits::{Float, FromPrimitive};
use rand::distr::Distribution as RandDistribution;
// Bind to rand's Distribution to avoid mismatches from transitive rand 0.8 deps.
#[cfg(test)]
use rand::rngs::SmallRng;
#[cfg(test)]
use rand::{Rng, SeedableRng};
use rand_distr::uniform::SampleUniform;
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use std::error::Error;
use std::marker::PhantomData;

#[derive(Clone, Debug)]
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
#[derive(Clone)]
pub struct NUTS<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Copy
        + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Clone + Send,
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
        + Copy
        + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Clone + Send,
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
                inner: target.clone(),
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
        let sample = self.inner.run(n_collect, n_discard);
        array3_to_tensor(sample)
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
        self.inner = self.inner.set_seed(seed);
        self
    }
}

/// Burn-backed single-chain NUTS wrapper.
#[derive(Clone)]
pub struct NUTSChain<T, B, GTarget>
where
    T: Float
        + Element
        + ElementConversion
        + SampleUniform
        + FromPrimitive
        + num_traits::ToPrimitive
        + Copy,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync + Clone,
    StandardNormal: RandDistribution<B::FloatElem>,
    StandardUniform: RandDistribution<B::FloatElem>,
    Exp1: RandDistribution<B::FloatElem>,
    B::FloatElem: Float + Element + ElementConversion + SampleUniform + FromPrimitive + Copy,
{
    inner: GenericNUTSChain<Tensor<B, 1>, BurnGradientTarget<GTarget, T>>,
    pub position: Tensor<B, 1>,
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
        + Copy,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync + Clone,
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
            position,
            _phantom: PhantomData,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.inner = self.inner.set_seed(seed);
        self.position = self.inner.position().clone();
        self
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        let sample = self.inner.run(n_collect, n_discard);
        self.position = self.inner.position().clone();
        array2_to_tensor(sample)
    }

    pub fn step(&mut self) {
        self.inner.step();
        self.position = self.inner.position().clone();
    }
}

fn array2_to_tensor<B, T>(arr: ndarray::Array2<T>) -> Tensor<B, 2>
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
    let td = TensorData::new(data, [shape[0], shape[1]]);
    Tensor::<B, 2>::from_data(td, &B::Device::default())
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
fn all_real<B, T>(x: Tensor<B, 1>) -> bool
where
    T: Float + Element,
    B: AutodiffBackend,
{
    x.clone()
        .equal_elem(T::infinity())
        .bool_or(x.clone().equal_elem(T::neg_infinity()))
        .bool_or(x.is_nan())
        .any()
        .bool_not()
        .into_scalar()
        .to_bool()
}

#[cfg(test)]
#[allow(dead_code)]
fn find_reasonable_epsilon<B, T, GTarget>(
    position: Tensor<B, 1>,
    mom: Tensor<B, 1>,
    gradient_target: &GTarget,
) -> T
where
    T: Float + Element,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync,
{
    let mut epsilon = T::one();
    let half = T::from(0.5).unwrap();
    let (ulogp, grad) = gradient_target.unnorm_logp_and_grad(position.clone());
    let (_, mut mom_prime, grad_prime, mut ulogp_prime) = leapfrog(
        position.clone(),
        mom.clone(),
        grad.clone(),
        epsilon,
        gradient_target,
    );
    let mut k = T::one();

    while !all_real::<B, T>(ulogp_prime.clone()) && !all_real::<B, T>(grad_prime.clone()) {
        k = k * half;
        (_, mom_prime, _, ulogp_prime) = leapfrog(
            position.clone(),
            mom.clone(),
            grad.clone(),
            epsilon * k,
            gradient_target,
        );
    }

    epsilon = half * k * epsilon;
    let log_accept_prob = ulogp_prime
        - ulogp.clone()
        - ((mom_prime.clone() * mom_prime).sum() - (mom.clone() * mom.clone()).sum()) * half;
    let mut log_accept_prob = T::from(log_accept_prob.into_scalar().to_f64()).unwrap();

    let a = if log_accept_prob > half.ln() {
        T::one()
    } else {
        -T::one()
    };

    while a * log_accept_prob > -a * T::from(2.0).unwrap().ln() {
        epsilon = epsilon * T::from(2.0).unwrap().powf(a);
        (_, mom_prime, _, ulogp_prime) = leapfrog(
            position.clone(),
            mom.clone(),
            grad.clone(),
            epsilon,
            gradient_target,
        );
        log_accept_prob = T::from(
            (ulogp_prime
                - ulogp.clone()
                - ((mom_prime.clone() * mom_prime).sum() - (mom.clone() * mom.clone()).sum())
                    * 0.5)
                .into_scalar()
                .to_f64(),
        )
        .unwrap();
    }

    epsilon
}

#[cfg(test)]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_tree<B, T, GTarget>(
    position: Tensor<B, 1>,
    mom: Tensor<B, 1>,
    grad: Tensor<B, 1>,
    logu: T,
    v: i8,
    j: usize,
    epsilon: T,
    gradient_target: &GTarget,
    joint_0: T,
    rng: &mut SmallRng,
) -> (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    usize,
    bool,
    T,
    usize,
)
where
    T: Float + Element,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync,
{
    if j == 0 {
        let (position_prime, mom_prime, grad_prime, logp_prime) = leapfrog(
            position.clone(),
            mom.clone(),
            grad.clone(),
            T::from(v as i32).unwrap() * epsilon,
            gradient_target,
        );
        let joint = logp_prime.clone() - (mom_prime.clone() * mom_prime.clone()).sum() * 0.5;
        let joint = T::from(joint.into_scalar().to_f64())
            .expect("type conversion from joint tensor to scalar type T to succeed");
        let n_prime = (logu < joint) as usize;
        let s_prime = (logu - T::from(1000.0).unwrap()) < joint;
        let position_minus = position_prime.clone();
        let position_plus = position_prime.clone();
        let mom_minus = mom_prime.clone();
        let mom_plus = mom_prime.clone();
        let grad_minus = grad_prime.clone();
        let grad_plus = grad_prime.clone();
        let alpha_prime = T::min(T::one(), (joint - joint_0).exp());
        let n_alpha_prime = 1_usize;
        (
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
        )
    } else {
        let (
            mut position_minus,
            mut mom_minus,
            mut grad_minus,
            mut position_plus,
            mut mom_plus,
            mut grad_plus,
            mut position_prime,
            mut grad_prime,
            mut logp_prime,
            mut n_prime,
            mut s_prime,
            mut alpha_prime,
            mut n_alpha_prime,
        ) = build_tree(
            position,
            mom,
            grad,
            logu,
            v,
            j - 1,
            epsilon,
            gradient_target,
            joint_0,
            rng,
        );
        if s_prime {
            let (
                position_minus_2,
                mom_minus_2,
                grad_minus_2,
                position_plus_2,
                mom_plus_2,
                grad_plus_2,
                position_prime_2,
                grad_prime_2,
                logp_prime_2,
                n_prime_2,
                s_prime_2,
                alpha_prime_2,
                n_alpha_prime_2,
            ) = if v == -1 {
                build_tree(
                    position_minus.clone(),
                    mom_minus.clone(),
                    grad_minus.clone(),
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    gradient_target,
                    joint_0,
                    rng,
                )
            } else {
                build_tree(
                    position_plus.clone(),
                    mom_plus.clone(),
                    grad_plus.clone(),
                    logu,
                    v,
                    j - 1,
                    epsilon,
                    gradient_target,
                    joint_0,
                    rng,
                )
            };
            if v == -1 {
                position_minus = position_minus_2;
                mom_minus = mom_minus_2;
                grad_minus = grad_minus_2;
            } else {
                position_plus = position_plus_2;
                mom_plus = mom_plus_2;
                grad_plus = grad_plus_2;
            }

            let u_build_tree: f64 = (*rng).random::<f64>();
            if u_build_tree < (n_prime_2 as f64 / (n_prime + n_prime_2).max(1) as f64) {
                position_prime = position_prime_2;
                grad_prime = grad_prime_2;
                logp_prime = logp_prime_2;
            }

            n_prime += n_prime_2;

            s_prime = s_prime
                && s_prime_2
                && stop_criterion(
                    position_minus.clone(),
                    position_plus.clone(),
                    mom_minus.clone(),
                    mom_plus.clone(),
                );
            alpha_prime = alpha_prime + alpha_prime_2;
            n_alpha_prime += n_alpha_prime_2;
        }
        (
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
        )
    }
}

#[cfg(test)]
fn stop_criterion<B>(
    position_minus: Tensor<B, 1>,
    position_plus: Tensor<B, 1>,
    mom_minus: Tensor<B, 1>,
    mom_plus: Tensor<B, 1>,
) -> bool
where
    B: AutodiffBackend,
{
    let diff = position_plus - position_minus;
    let dot_minus = (diff.clone() * mom_minus).sum();
    let dot_plus = (diff * mom_plus).sum();
    dot_minus.greater_equal_elem(0).into_scalar().to_bool()
        && dot_plus.greater_equal_elem(0).into_scalar().to_bool()
}

#[cfg(test)]
fn leapfrog<B, T, GTarget>(
    position: Tensor<B, 1>,
    mom: Tensor<B, 1>,
    grad: Tensor<B, 1>,
    epsilon: T,
    gradient_target: &GTarget,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>)
where
    T: Float + ElementConversion,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B>,
{
    let mom_prime = mom + grad * epsilon * 0.5;
    let position_prime = position + mom_prime.clone() * epsilon;
    let (ulogp_prime, grad_prime) = gradient_target.unnorm_logp_and_grad(position_prime.clone());
    let mom_prime = mom_prime + grad_prime.clone() * epsilon * 0.5;
    (position_prime, mom_prime, grad_prime, ulogp_prime)
}

#[cfg(test)]
mod tests {
    use std::fmt::Debug;

    use crate::{
        core::init,
        dev_tools::Timer,
        distributions::{DiffableGaussian2D, Rosenbrock2D},
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
        let position = Tensor::<BackendType, 1>::from([0.0, 1.0]);
        let mom = Tensor::<BackendType, 1>::from([1.0, 0.0]);
        let epsilon = find_reasonable_epsilon::<_, f64, _>(position, mom, &StandardNormal);
        assert_eq!(epsilon, 2.0);
    }

    #[test]
    fn test_build_tree() {
        let gradient_target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);
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
        ) = build_tree::<BackendType, f64, _>(
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
        assert!(
            (logp_prime.into_scalar().to_f64() - logp_exp).abs() < 1e-6,
            "logp mismatch"
        );
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
        let tol = Tolerance::<f64>::default()
            .set_relative(1e-5)
            .set_absolute(1e-6);
        assert_tensor_approx_eq(
            sample.flatten(0, 1),
            &[
                -1.168318748474121,
                -0.4077277183532715,
                -1.8463939428329468,
                0.19176559150218964,
                -1.0662782192230225,
                -0.3948383331298828,
            ],
            tol,
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
        let tol = Tolerance::<f64>::default()
            .set_relative(1e-5)
            .set_absolute(1e-6);
        assert_tensor_approx_eq(
            sample.flatten(0, 1),
            &[
                2.653707265853882,
                5.560618877410889,
                2.9760334491729736,
                6.325948715209961,
                2.187873125076294,
                5.611990928649902,
                2.1512224674224854,
                5.416507720947266,
                2.4165120124816895,
                3.9120564460754395,
            ],
            tol,
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
        let tol = Tolerance::<f64>::default()
            .set_relative(1e-5)
            .set_absolute(1e-6);
        assert_tensor_approx_eq(
            sample.flatten(0, 2),
            &[
                2.653707265853882,
                5.560618877410889,
                2.9760334491729736,
                6.325948715209961,
                2.187873125076294,
                5.611990928649902,
                2.1512224674224854,
                5.416507720947266,
                2.4165120124816895,
                3.9120564460754395,
            ],
            tol,
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
