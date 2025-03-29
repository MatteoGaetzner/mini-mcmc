use mini_mcmc::core::{init_det, ChainRunner};
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian};
use mini_mcmc::metropolis_hastings::MetropolisHastings;
use ndarray::{arr1, arr2};

fn main() {
    let target = Gaussian2D {
        mean: arr1(&[0.0, 0.0]),
        cov: arr2(&[[1.0, 0.0], [0.0, 1.0]]),
    };
    let proposal = IsotropicGaussian::new(1.0);

    // Create a MH sampler with 4 parallel chains
    let mut mh = MetropolisHastings::new(target, proposal, init_det(4, 2));

    // Run the sampler for 1,100 steps, discarding the first 100 as burn-in
    let samples = mh.run(1000, 100).unwrap();

    // We should have 1000 * 4 = 3600 samples
    assert_eq!(samples.shape()[0], 4);
    assert_eq!(samples.shape()[1], 1000);
}
