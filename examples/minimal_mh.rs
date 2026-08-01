use mini_mcmc::core::ChainRunner;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use ndarray::{arr1, arr2};

fn main() {
    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);
    let initial_states = vec![vec![0.0, 0.0]; 4]; // 4 chains, each starting at [0,0]

    // Create a MH sampler with 4 parallel chains
    let mut mh = MetropolisHastings::new(target, proposal, initial_states);

    // Run the sampler for 1,100 steps, discarding the first 100 as burn-in
    let (samples, stats) = mh.run_progress(1000, 100).unwrap();

    // Print convergence statistics
    println!("{stats}");

    // We should have 4 chains, each with 1000 samples of 2 dimensions
    assert_eq!(samples.shape(), [4, 1000, 2]);
}

#[cfg(test)]
mod tests {
    use super::main;

    #[test]
    fn test_main() {
        main();
    }
}
