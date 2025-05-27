use crate::distributions::GradientTarget;
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Tensor;
use num_traits::Float;
use rand::prelude::*;
use rand::Rng;
use rand_distr::StandardNormal;

#[derive(Debug, Clone)]
pub struct NUTSChain<T, B, GTarget>
where
    B: AutodiffBackend,
{
    pub target: GTarget,
    pub position: Tensor<B, 1>,

    #[allow(dead_code)]
    target_accept_p: f32,

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
    rand_distr::Standard: rand_distr::Distribution<T>,
{
    pub fn new(target: GTarget, initial_position: Vec<T>, target_accept_p: f32) -> Self {
        let dim = initial_position.len();
        let td: TensorData = TensorData::new(initial_position, [dim]);
        let positions = Tensor::<B, 1>::from_data(td, &B::Device::default());
        let rng = SmallRng::seed_from_u64(thread_rng().gen::<u64>());
        Self {
            target,
            position: positions,
            target_accept_p,
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

    #[allow(unused_variables)]
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        todo!()
    }

    pub fn step(&mut self) {
        todo!()
    }
}

#[allow(dead_code)]
fn find_reasonable_epsilon<B, T, GTarget>(
    position: Tensor<B, 1>,
    mom: Tensor<B, 1>,
    gradient_target: GTarget,
) -> T
where
    T: Float + burn::tensor::Element,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + std::marker::Sync,
{
    let mut epsilon = T::one();
    let half = T::from(0.5).unwrap();
    let (ulogp, grad) = gradient_target.unnorm_logp_and_grad(position.clone());
    let (_, mut mom_prime, grad_prime, mut ulogp_prime) = leapfrog(
        position.clone(),
        mom.clone(),
        grad.clone(),
        epsilon,
        &gradient_target,
    );
    let mut k = T::one();

    while !all_real::<B, T>(ulogp_prime.clone()) && !all_real::<B, T>(grad_prime.clone()) {
        k = k * half;
        (_, mom_prime, _, ulogp_prime) = leapfrog(
            position.clone(),
            mom.clone(),
            grad.clone(),
            epsilon * k,
            &gradient_target,
        );
    }

    epsilon = half * k * epsilon;
    let log_accept_prob = ulogp_prime
        - ulogp.clone()
        - ((mom_prime.clone() * mom_prime).sum() - (mom.clone() * mom.clone()).sum()) * half;
    let mut log_accept_prob = T::from(log_accept_prob.into_scalar().to_f32()).unwrap();

    let a = if log_accept_prob > half.ln() {
        T::one()
    } else {
        -T::one()
    };

    while a * log_accept_prob > -a * T::from(2.0).unwrap() {
        epsilon = epsilon * T::from(2.0).unwrap().powf(a);
        (_, mom_prime, _, ulogp_prime) = leapfrog(
            position.clone(),
            mom.clone(),
            grad.clone(),
            epsilon,
            &gradient_target,
        );
        log_accept_prob = T::from(
            (ulogp_prime
                - ulogp.clone()
                - (mom_prime.clone() * mom_prime).sum() * half
                - (mom.clone() * mom.clone()).sum())
            .into_scalar()
            .to_f32(),
        )
        .unwrap();
    }

    epsilon
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

    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::{Element, Tensor},
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

    // Define the Rosenbrock distribution.
    #[allow(dead_code)]
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
        fn unnorm_logp(&self, positions: Tensor<B, 1>) -> Tensor<B, 1> {
            let x = positions.clone().slice([0]);
            let y = positions.slice([1]);

            // Compute (a - x)^2 in place.
            let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);

            // Compute (y - x^2)^2 in place.
            let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);

            // Return the negative sum as a flattened 1D tensor.
            -(term_1 + term_2).flatten(0, 1)
        }
    }

    #[test]
    fn test_find_reasonable_epsilon() {
        // Define initial positions for a single chain (2-dimensional).
        let position = Tensor::<BackendType, 1>::from([0.0, 1.0]);
        let mom = Tensor::<BackendType, 1>::from([1.0, 0.0]);
        let epsilon = find_reasonable_epsilon::<_, f32, _>(position, mom, StandardNormal);
        assert_eq!(epsilon, 2.0);
    }

    // #[test]
    // fn test_single() {
    //     // Create the Rosenbrock target (a = 1, b = 100)
    //     let target = Rosenbrock2D {
    //         a: 1.0_f32,
    //         b: 100.0_f32,
    //     };
    //
    //     // Define initial positions for a single chain (2-dimensional).
    //     let initial_positions = vec![0.0_f32, 0.0];
    //     let n_collect = 3;
    //
    //     // Create the HMC sampler.
    //     let mut sampler =
    //         NUTSChain::<f32, BackendType, Rosenbrock2D<f32>>::new(target, initial_positions, 0.8)
    //             .set_seed(42);
    //
    //     // Run the sampler for n_collect steps.
    //     let mut timer = Timer::new();
    //     let samples: Tensor<BackendType, 2> = sampler.run(n_collect, 0);
    //     timer.log(format!(
    //         "Collected samples with shape: {:?}",
    //         samples.dims()
    //     ));
    //     assert_eq!(samples.dims(), [3, 2]);
    // }
}
