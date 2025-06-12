use burn::backend::Autodiff;
use burn::prelude::Tensor;
use mini_mcmc::core::init;
use mini_mcmc::distributions::Rosenbrock2D;
use mini_mcmc::nuts::NUTS;
use mini_mcmc::stats::split_rhat_mean_ess;
use ndarray::ArrayView3;
use ndarray_stats::QuantileExt;

fn main() {
    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<burn::backend::NdArray>;

    // Create the 2D Rosenbrock target (a = 1, b = 100).
    let target = Rosenbrock2D {
        a: 1.0_f32,
        b: 100.0_f32,
    };

    // Define 6 chains all initialized to (1.0, 2.0).
    let initial_positions = init::<f32>(4, 2);

    // Configure and seed the NUTS sampler with target_accept_p = 0.95.
    let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);

    // Number of samples to collect and to discard (burn-in).
    let n_collect = 400;
    let n_discard = 400;

    // Run sampler and time it.
    let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
    println!(
        "NUTS sampler: generated {} observations.",
        sample.dims()[0..2].iter().product::<usize>()
    );

    // Verify dimensions: [n_chains, n_collect, dim]
    assert_eq!(sample.dims(), [4, 400, 2]);

    // Compute convergence diagnostics
    let (split_rhat, ess) = {
        let data = sample.to_data();
        let array = ArrayView3::from_shape(sample.dims(), data.as_slice().unwrap()).unwrap();
        split_rhat_mean_ess(array)
    };
    println!("MIN Split Rhat: {:.3}", split_rhat.min().unwrap());
    println!("MIN ESS: {:.1}", ess.min().unwrap());
}
