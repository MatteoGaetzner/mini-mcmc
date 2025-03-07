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
    let initial_state = [0.0, 0.0];

    // Create a MH sampler with 4 parallel chains
    let mut mh = MetropolisHastings::new(target, proposal, &initial_state, 4);

    // Run the sampler for 1,000 steps, discarding the first 100 as burn-in
    let samples = mh.run(1000, 100).unwrap();

    // We should have 900 * 4 = 3600 samples
    assert_eq!(samples.shape()[0], 4);
    assert_eq!(samples.shape()[1], 900);
}
