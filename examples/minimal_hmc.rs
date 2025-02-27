use burn::tensor::Element;
use burn::{backend::Autodiff, prelude::Tensor};
use mini_mcmc::hmc::{GradientTarget, HMC};
use num_traits::Float;

/// The 3D Rosenbrock distribution.
///
/// For a point x = (x₁, x₂, x₃), the log probability is defined as the negative of
/// the sum of two Rosenbrock terms:
///
///   f(x) = 100*(x₂ - x₁²)² + (1 - x₁)² + 100*(x₃ - x₂²)² + (1 - x₂)²
///
/// This implementation generalizes to d dimensions, but here we use it for 3D.
struct RosenbrockND {}

impl<T, B> GradientTarget<T, B> for RosenbrockND
where
    T: Float + std::fmt::Debug + Element,
    B: burn::tensor::backend::AutodiffBackend,
{
    fn log_prob_batch(&self, positions: &Tensor<B, 2>) -> Tensor<B, 1> {
        // Assume positions has shape [n_chains, d] with d = 3.
        let k = positions.dims()[0] as i64;
        let n = positions.dims()[1] as i64;
        let low = positions.clone().slice([(0, k), (0, n - 1)]);
        let high = positions.clone().slice([(0, k), (1, n)]);
        let term_1 = (high - low.clone().powi_scalar(2))
            .powi_scalar(2)
            .mul_scalar(100);
        let term_2 = low.neg().add_scalar(1).powi_scalar(2);
        -(term_1 + term_2).sum_dim(1).squeeze(1)
    }
}

fn main() {
    // Use the CPU backend wrapped in Autodiff (e.g., NdArray).
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 3D Rosenbrock target.
    let target = RosenbrockND {};

    // Define initial positions for 6 chains (each a 3D point).
    let initial_positions = vec![vec![1.0_f32, 2.0_f32, 3.0_f32]; 6];

    // Create the HMC sampler with a step size of 0.01 and 50 leapfrog steps.
    let mut sampler =
        HMC::<f32, BackendType, RosenbrockND>::new(target, initial_positions, 0.032, 50);

    // Run the sampler for 1000 iterations, discard 100
    let samples = sampler.run(1000, 100);

    // Print the shape of the collected samples.
    println!("Collected samples with shape: {:?}", samples.dims());
}
