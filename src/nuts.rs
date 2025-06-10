use crate::distributions::GradientTarget;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Tensor;
use num_traits::Float;
use rand::prelude::*;
use rand::Rng;
use rand_distr::Exp1;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct NUTSChain<T, B, GTarget>
where
    B: AutodiffBackend,
{
    pub target: GTarget,
    pub position: Tensor<B, 1>,

    target_accept_p: T,
    epsilon: T,

    pub rng: SmallRng,
    phantom_data: std::marker::PhantomData<T>,
}

impl<T, B, GTarget> NUTSChain<T, B, GTarget>
where
    T: Float
        + burn::tensor::ElementConversion
        + burn::tensor::Element
        + rand_distr::uniform::SampleUniform
        + num_traits::FromPrimitive,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + std::marker::Sync,
    StandardNormal: rand::distributions::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>, rand_distr::Exp1: rand_distr::Distribution<T>
{
    pub fn new(target: GTarget, initial_position: Vec<T>, target_accept_p: T) -> Self {
        let dim = initial_position.len();
        let td: TensorData = TensorData::new(initial_position, [dim]);
        let position = Tensor::<B, 1>::from_data(td, &B::Device::default());
        let rng = SmallRng::seed_from_u64(thread_rng().gen::<u64>());
        Self {
            target,
            position,
            target_accept_p,
            epsilon: -T::one(),
            rng,
            phantom_data: std::marker::PhantomData,
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

    #[allow(clippy::single_range_in_vec_init)]
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        let dim = self.position.dims()[0];

        let mut sample = Tensor::<B, 2>::empty([n_collect + n_discard, dim], &B::Device::default());
        let mut ln_prob = Tensor::<B, 1>::empty([n_collect + n_discard], &B::Device::default());

        let (mut logp, mut grad) = self.target.unnorm_logp_and_grad(self.position.clone());
        sample = sample.slice_assign([0..1, 0..dim], self.position.clone().unsqueeze());
        ln_prob = ln_prob.slice_assign([0..1], logp.clone());
        // dbg!(&logp, &grad);
        // dbg!(&sample);
        let mom_0_data: Vec<T> = (&mut self.rng).sample_iter(StandardNormal).take(dim).collect();
        let mom_0 = Tensor::<B, 1>::from_data(mom_0_data.as_slice(), &B::Device::default());
        dbg!(&mom_0);
        // dbg!(&self.position, &self.epsilon);
        if T::abs(self.epsilon + T::one()) <= T::epsilon() {
            self.epsilon = find_reasonable_epsilon(self.position.clone(), mom_0, &self.target);
        }
        // dbg!(&self.epsilon);
        let gamma = T::from(0.05).unwrap();
        let t_0 = 10;
        let kappa = T::from(0.75).unwrap();
        let mu = (T::from(10.0).unwrap() * self.epsilon).ln();
        let mut epsilon_bar = T::one();
        let mut h_bar = T::zero();

        for m in 1..(n_collect + n_discard) {
            let mom_0 = (&mut self.rng).sample_iter(StandardNormal).take(dim).collect::<Vec<T>>();
            dbg!(&mom_0);
            let mom_0 = Tensor::<B, 1>::from_data(mom_0.as_slice(), &B::Device::default());
            let joint = logp.clone() - (mom_0.clone() * mom_0.clone()).sum() * 0.5;
            let joint = T::from_f64(joint.into_scalar().to_f64()).expect("successful conversion from 64 to T");
            // dbg!(&logp);
            // dbg!(&joint);
            let exp1_obs = self.rng.sample(Exp1);
            dbg!(exp1_obs);
            let logu = joint - exp1_obs;
            // dbg!(&logu);
            sample = sample.clone().slice_assign([m..m+1, 0..dim], sample.clone().slice([m-1..m, 0..dim]));

            ln_prob = ln_prob.clone().slice_assign([m..m+1], ln_prob.clone().slice([m-1..m]));
            // dbg!(&sample);
            // dbg!(&ln_prob);

            let mut position_minus = sample.clone().slice([m-1..m, 0..dim]).flatten(0, 1);
            let mut position_plus = position_minus.clone();
            let mut mom_minus = mom_0.clone();
            let mut mom_plus = mom_0.clone();
            let mut grad_minus = grad.clone();
            let mut grad_plus = grad.clone();
            // dbg!(&position_minus, &position_plus, &mom_minus, &mom_plus, &grad_minus, &grad_plus);
            let mut j = 0;
            let mut n = 1;
            let mut s = true; // 's' stands for 'stop', indicating the stopping of inner while loop
            let mut alpha: T = T::zero();
            let mut n_alpha: u64 = 0;

            while s {
                let u_run_1: T = self.rng.gen::<T>();
                dbg!(u_run_1);
                let v = (2 * (u_run_1 < T::from(0.5).unwrap()) as i8) - 1;
                // dbg!(v);

                let (position_prime, grad_prime, logp_prime, n_prime, s_prime) = {
                    if v == -1 {
                        let (position_minus_2, mom_minus_2, grad_minus_2, _, _, _, position_prime_2, grad_prime_2, logp_prime_2, n_prime_2, s_prime_2, alpha_2, n_alpha_2) = build_tree(
                        position_minus.clone(), mom_minus.clone(), grad_minus.clone(), logu, v, j, self.epsilon, &self.target, joint, &mut self.rng);

                        position_minus = position_minus_2;
                        mom_minus = mom_minus_2;
                        grad_minus = grad_minus_2;
                        alpha = alpha_2;
                        n_alpha = n_alpha_2;

                        (position_prime_2, grad_prime_2, logp_prime_2, n_prime_2, s_prime_2)
                    } else {
                        let (_, _, _, position_plus_2, mom_plus_2, grad_plus_2, position_prime_2, grad_prime_2, logp_prime_2, n_prime_2, s_prime_2, alpha_2, n_alpha_2) = build_tree( position_plus.clone(), mom_plus.clone(), grad_plus.clone(), logu, v, j, self.epsilon, &self.target, joint, &mut self.rng);

                        position_plus = position_plus_2;
                        mom_plus = mom_plus_2;
                        grad_plus = grad_plus_2;
                        alpha = alpha_2;
                        n_alpha = n_alpha_2;

                        (position_prime_2, grad_prime_2, logp_prime_2, n_prime_2, s_prime_2)
                    }
                };
                // dbg!(&position_prime, &grad_prime, &logp_prime, &n_prime, &s_prime);

                let tmp = T::one().min(T::from(n_prime).expect("successful conversion of n_prime from u64 to T") / T::from(n).expect("successful conversion of n from u64 to T"));
                let u_run_2 = self.rng.gen::<T>();
                dbg!(u_run_2);
                // dbg!(tmp, n, n_prime);
                if s_prime && (u_run_2 < tmp) {
                    sample = sample.slice_assign([m..m+1, 0..dim], position_prime.unsqueeze_dim(0));
                    ln_prob = ln_prob.slice_assign([m..m+1], logp_prime.clone());
                    logp = logp_prime;
                    grad = grad_prime;
                }
                n += n_prime;

                // dbg!(&position_minus, &position_plus, &mom_minus, &mom_plus);
                s = s_prime && stop_criterion(position_minus.clone(), position_plus.clone(), mom_minus.clone(), mom_plus.clone());
                j += 1
            }

            let mut eta = T::one() / T::from(m + t_0).expect("successful conversion of m + t_0 to T");
            // dbg!(eta);
            h_bar = (T::one() - eta) * h_bar + eta * (self.target_accept_p - alpha / T::from(n_alpha).expect("successful conversion of n_alpha to T"));
            // dbg!(h_bar);
            if m <= n_discard {
                let _m = T::from(m).expect("successful conversion of m to T");
                self.epsilon = T::exp(mu - T::sqrt(_m) / gamma * h_bar);
                eta = _m.powf(-kappa);
                // dbg!(eta);
                epsilon_bar = T::exp((T::one() - eta) * T::ln(epsilon_bar) + eta * T::ln(self.epsilon));
                // dbg!(epsilon_bar);
            } else {
                self.epsilon = epsilon_bar;
            }
            // dbg!(self.epsilon);
        }
        // dbg!(&sample);
        sample = sample.slice([n_discard..]);
        sample
    }

    pub fn step(&mut self) {
        todo!()
    }
}

#[allow(dead_code)]
fn find_reasonable_epsilon<B, T, GTarget>(
    position: Tensor<B, 1>,
    mom: Tensor<B, 1>,
    gradient_target: &GTarget,
) -> T
where
    T: Float + burn::tensor::Element,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + std::marker::Sync,
{
    let mut epsilon = T::one();
    let half = T::from(0.5).unwrap();
    let (ulogp, grad) = gradient_target.unnorm_logp_and_grad(position.clone());
    // dbg!(&ulogp, &grad);
    let (_, mut mom_prime, grad_prime, mut ulogp_prime) = leapfrog(
        position.clone(),
        mom.clone(),
        grad.clone(),
        epsilon,
        gradient_target,
    );
    // dbg!(&mom_prime, &grad_prime, &ulogp_prime);
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
    // dbg!(&log_accept_prob);
    let mut log_accept_prob = T::from(log_accept_prob.into_scalar().to_f64()).unwrap();

    let a = if log_accept_prob > half.ln() {
        T::one()
    } else {
        -T::one()
    };

    while a * log_accept_prob > -a * T::from(2.0).unwrap().ln() {
        epsilon = epsilon * T::from(2.0).unwrap().powf(a);
        // dbg!("in loop", &epsilon);
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
                - ((mom_prime.clone() * mom_prime).sum()
                - (mom.clone() * mom.clone()).sum()) * 0.5)
            .into_scalar()
            .to_f64(),
        )
        .unwrap();
        // dbg!(&log_accept_prob);
    }

    epsilon
}

#[allow(clippy::too_many_arguments,clippy::type_complexity)]
fn build_tree<B, T, GTarget>(position: Tensor<B, 1>, mom: Tensor<B, 1>, grad: Tensor<B, 1>, logu: T, v: i8, j: u64, epsilon: T, gradient_target: &GTarget, joint_0: T, rng: &mut SmallRng) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, u64, bool, T, u64) 
where
    T: Float + burn::tensor::Element,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + std::marker::Sync,
{
    if j == 0 {
            let (position_prime, mom_prime, grad_prime, logp_prime) = leapfrog(position.clone(), mom.clone(), grad.clone(), T::from(v as i32).unwrap() * epsilon, gradient_target);
            let joint = logp_prime.clone() - (mom_prime.clone() * mom_prime.clone()).sum() * 0.5;
            let joint = T::from(joint.into_scalar().to_f64()).expect("type conversion from joint tensor to scalar type T to succeed");
            let n_prime = (logu < joint) as u64;
            let s_prime = (logu - T::from(1000.0).unwrap()) < joint;
            let position_minus = position_prime.clone();
            let position_plus = position_prime.clone();
            let mom_minus = mom_prime.clone();
            let mom_plus = mom_prime.clone();
            let grad_minus = grad_prime.clone();
            let grad_plus = grad_prime.clone();
            let alpha_prime = T::min(T::one(), (joint - joint_0).exp());
            let n_alpha_prime = 1_u64;
            (position_minus, mom_minus, grad_minus, position_plus, mom_plus, grad_plus, position_prime, grad_prime, logp_prime, n_prime, s_prime, alpha_prime, n_alpha_prime)
        }
    else {
        let (mut position_minus, mut mom_minus, mut grad_minus, mut position_plus, mut mom_plus, mut grad_plus, mut position_prime, mut grad_prime, mut logp_prime, mut n_prime, mut s_prime, mut alpha_prime, mut n_alpha_prime) = build_tree(position, mom, grad, logu, v, j-1, epsilon, gradient_target, joint_0, rng);
        if s_prime {
            let (position_minus_2, mom_minus_2, grad_minus_2, position_plus_2, mom_plus_2, grad_plus_2, position_prime_2, grad_prime_2, logp_prime_2, n_prime_2, s_prime_2, alpha_prime_2, n_alpha_prime_2) = if v == -1 {
                build_tree(position_minus.clone(), mom_minus.clone(), grad_minus.clone(), logu, v, j-1, epsilon, gradient_target, joint_0, rng)
            } else {
                build_tree(position_plus.clone(), mom_plus.clone(), grad_plus.clone(), logu, v, j-1, epsilon, gradient_target, joint_0, rng)
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

            let u_build_tree: f64 = (*rng).gen::<f64>();
            dbg!(u_build_tree);
            if u_build_tree < (n_prime_2 as f64 / (n_prime + n_prime_2).max(1) as f64) {
                position_prime = position_prime_2;
                grad_prime = grad_prime_2;
                logp_prime = logp_prime_2;
            }

            n_prime += n_prime_2;

            s_prime = s_prime && s_prime_2 && stop_criterion(position_minus.clone(), position_plus.clone(), mom_minus.clone(), mom_plus.clone());
            alpha_prime = alpha_prime + alpha_prime_2;
            n_alpha_prime += n_alpha_prime_2;
        } 
        (position_minus, mom_minus, grad_minus, position_plus, mom_plus, grad_plus, position_prime, grad_prime, logp_prime, n_prime, s_prime, alpha_prime, n_alpha_prime)
    }
}

fn all_real<B, T>(x: Tensor<B, 1>) -> bool
where
    T: Float + burn::tensor::Element,
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

fn stop_criterion<B>(position_minus: Tensor<B, 1>, position_plus: Tensor<B, 1>, mom_minus: Tensor<B, 1>, mom_plus: Tensor<B, 1>) -> bool 
where
    B: AutodiffBackend,
{
    let diff = position_plus - position_minus;
    let dot_minus = (diff.clone() * mom_minus).sum();
    let dot_plus = (diff * mom_plus).sum();
    dot_minus.greater_equal_elem(0).into_scalar().to_bool() && dot_plus.greater_equal_elem(0).into_scalar().to_bool() 
}

fn leapfrog<B, T, GTarget>(
    position: Tensor<B, 1>,
    mom: Tensor<B, 1>,
    grad: Tensor<B, 1>,
    epsilon: T,
    gradient_target: &GTarget,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>)
where
    T: Float + burn::tensor::ElementConversion,
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

    use crate::{dev_tools::Timer, distributions::DiffableGaussian2D};

    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Tensor, Tolerance},
    };
    use num_traits::Float;

    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<NdArray>;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct StandardNormal;

    impl<T, B> GradientTarget<T, B> for StandardNormal
    where
        T: Float + Debug + ElementConversion + burn::tensor::Element,
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
        expected: [f64; 2],
        tol: Tolerance<F>
    ) {
        let a = actual.clone().to_data();
        let e = Tensor::<T, 1>::from(expected).to_data();
        a.assert_approx_eq(&e, tol);
    }

    #[test]
    fn test_find_reasonable_epsilon() {
        // Define initial positions for a single chain (2-dimensional).
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
        let j: u64 = 3;
        let epsilon: f64 = 0.01;
        let joint_0 = 0.1_f64;
        let mut rng = SmallRng::seed_from_u64(0);
        let (position_minus, mom_minus, grad_minus, position_plus, mom_plus, grad_plus, position_prime, grad_prime, logp_prime, n_prime, s_prime, alpha_prime, n_alpha_prime) = build_tree::<BackendType, f64, _>(position, mom, grad, logu, v, j, epsilon, &gradient_target, joint_0, &mut rng);
        let tol = Tolerance::<f64>::default().set_relative(1e-5).set_absolute(1e-6);

        assert_tensor_approx_eq(position_minus, [-0.1584001, 0.76208336], tol);
        assert_tensor_approx_eq(mom_minus,      [1.980_003_6,2.971_825_3], tol);
        assert_tensor_approx_eq(grad_minus,     [-7.912_36e-5, 7.935_829_5e-2], tol);

        assert_tensor_approx_eq(position_plus,  [-0.0198, 0.97025], tol);
        assert_tensor_approx_eq(mom_plus,       [1.98, 2.974_950_3], tol);
        assert_tensor_approx_eq(grad_plus,      [-1.250e-05, 9.925e-03], tol);

        assert_tensor_approx_eq(position_prime, [-0.0198, 0.97025], tol);
        assert_tensor_approx_eq(grad_prime,     [-1.250e-05, 9.925e-03], tol);

        assert_eq!(n_prime, 0);
        assert!(s_prime);
        assert_eq!(n_alpha_prime, 8);

        let logp_exp = -2.877_745_4_f64;
        let alpha_exp =  0.000_686_661_7_f64;
        assert!((logp_prime.into_scalar().to_f64()  - logp_exp).abs()  < 1e-6, "logp mismatch");
        assert!((alpha_prime - alpha_exp).abs() < 1e-8, "alpha mismatch");
    }


    #[test]
    fn test_nuts_single() {
        let target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);

        // Define initial positions for a single chain (2-dimensional).
        let initial_positions = vec![0.0_f64, 1.0];
        let n_discard = 1;
        let n_collect = 1;

        // Create the HMC sampler.
        let mut sampler =
            NUTSChain::new(target, initial_positions, 0.8)
                .set_seed(42);

        // Run the sampler for n_collect steps.
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "Collected sample with shape: {:?}",
            sample.dims()
        ));
        dbg!(&sample);
        assert_eq!(sample.dims(), [n_collect, 2]);
        let tol = Tolerance::<f64>::default().set_relative(1e-5).set_absolute(1e-6);
        assert_tensor_approx_eq(sample.flatten(0, 1), [1.6786678_f64 , 2.73184293], tol);
    }

    #[test]
    fn test_nuts_single_2() {
        let target = DiffableGaussian2D::new([0.0_f64, 1.0], [[4.0, 2.0], [2.0, 3.0]]);

        // Define initial positions for a single chain (2-dimensional).
        let initial_positions = vec![0.0_f64, 1.0];
        let n_discard = 3;
        let n_collect = 3;

        // Create the HMC sampler.
        let mut sampler =
            NUTSChain::new(target, initial_positions, 0.8)
                .set_seed(42);

        // Run the sampler for n_collect steps.
        let mut timer = Timer::new();
        let sample: Tensor<BackendType, 2> = sampler.run(n_collect, n_discard);
        timer.log(format!(
            "Collected sample with shape: {:?}",
            sample.dims()
        ));
        dbg!(&sample);
        assert_eq!(sample.dims(), [n_collect, 2]);
        // let tol = Tolerance::<f64>::default().set_relative(1e-5).set_absolute(1e-6);
        // assert_tensor_approx_eq(sample.flatten(0, 1), [1.6786678_f64 , 2.73184293], tol);
    }
}
