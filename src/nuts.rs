//! Parallel No-U-Turn Sampler (NUTS).
//!
//! A parallel implementation of NUTS running independent Markov chains via Rayon.
//!
//! ## Example: custom 2D Rosenbrock target
//! ```rust
//! use mini_mcmc::core::init;
//! use mini_mcmc::distributions::GradientTarget;
//! use mini_mcmc::nuts::NUTS;
//! use burn::backend::{Autodiff, NdArray};
//! use burn::prelude::*;
//!
//! type B = Autodiff<NdArray>;
//!
//! #[derive(Clone)]
//! struct Rosenbrock2D { a: f32, b: f32 }
//!
//! impl GradientTarget<f32, B> for Rosenbrock2D {
//!     fn unnorm_logp(&self, position: Tensor<B, 1>) -> Tensor<B, 1> {
//!         let x = position.clone().slice(s![0..1]);
//!         let y = position.slice(s![1..2]);
//!         let term_1 = (-x.clone()).add_scalar(self.a).powi_scalar(2);
//!         let term_2 = y.sub(x.powi_scalar(2)).powi_scalar(2).mul_scalar(self.b);
//!         -(term_1 + term_2)
//!     }
//! }
//!
//! let target = Rosenbrock2D { a: 1.0, b: 100.0 };
//! let initial_positions = init::<f32>(4, 2);    // 4 chains in 2D
//! let mut sampler = NUTS::new(target, initial_positions, 0.9);
//! let (samples, stats) = sampler.run_progress(100, 20).unwrap();
//! ```
//!
//! ## Inspiration
//! Borrowed ideas from [mfouesneau/NUTS](https://github.com/mfouesneau/NUTS).

use std::error::Error;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use crate::distributions::GradientTarget;
use crate::stats::{collect_rhat, ChainStats, ChainTracker, RunStats};
use burn::prelude::*;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::cast::ToElement;
use burn::tensor::Element;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::ArrayView3;
use ndarray_stats::QuantileExt;
use num_traits::{Float, FromPrimitive};
use rand::prelude::*;
use rand::Rng;
use rand_distr::uniform::SampleUniform;
use rand_distr::{Exp1, StandardNormal};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

/// No-U-Turn Sampler (NUTS).
///
/// Encapsulates multiple independent Markov chains using the NUTS algorithm. Utilizes dual-averaging
/// step size adaptation and dynamic trajectory lengths to efficiently explore complex posterior geometries.
/// Chains are executed concurrently via Rayon, each evolving independently.
///
/// # Type Parameters
/// - `T`: Floating-point type for numerical calculations.
/// - `B`: Autodiff backend from the `burn` crate.
/// - `GTarget`: Target distribution type implementing the `GradientTarget` trait.
#[derive(Debug, Clone)]
pub struct NUTS<T, B, GTarget>
where
    T: Float + ElementConversion + Element + SampleUniform + FromPrimitive,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + Sync,
    StandardNormal: rand::distributions::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>,
    rand_distr::Exp1: rand_distr::Distribution<T>,
{
    /// The vector of independent Markov chains.
    chains: Vec<NUTSChain<T, B, GTarget>>,
}

impl<T, B, GTarget> NUTS<T, B, GTarget>
where
    T: Float + ElementConversion + Element + SampleUniform + FromPrimitive + Send,
    B: AutodiffBackend + Send,
    GTarget: GradientTarget<T, B> + Sync + Clone + Send,
    StandardNormal: rand::distributions::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>,
    rand_distr::Exp1: rand_distr::Distribution<T>,
{
    /// Creates a new NUTS sampler with the given target distribution and initial state for each chain.
    ///
    /// # Parameters
    /// - `target`: The target distribution implementing `GradientTarget`.
    /// - `initial_positions`: A vector of initial positions for each chain, shape `[n_chains, D]`.
    /// - `target_accept_p`: Desired average acceptance probability for the dual-averaging adaptation. Try values between 0.6 and 0.95.
    ///
    /// # Returns
    /// A newly initialized `NUTS` instance.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use burn::backend::{Autodiff, NdArray};
    /// # use mini_mcmc::nuts::NUTS;
    /// # use mini_mcmc::distributions::DiffableGaussian2D;
    /// type B = Autodiff<NdArray>;
    ///
    /// // Create a 2D Gaussian with mean [0,0] and identity covariance
    /// let gauss = DiffableGaussian2D::new([0.0_f64, 0.0], [[1.0, 0.0], [0.0, 1.0]]);
    ///
    /// // Initialize 3 chains in 2D at different starting points
    /// let init_positions = vec![
    ///     vec![-1.0, -1.0],
    ///     vec![ 0.0,  0.0],
    ///     vec![ 1.0,  1.0],
    /// ];
    ///
    /// // Build the sampler targeting 85% acceptance probability
    /// let sampler: NUTS<f64, B, _> = NUTS::new(gauss, init_positions, 0.85);
    /// ```
    pub fn new(target: GTarget, initial_positions: Vec<Vec<T>>, target_accept_p: T) -> Self {
        let chains = initial_positions
            .into_iter()
            .map(|pos| NUTSChain::new(target.clone(), pos, target_accept_p))
            .collect();
        Self { chains }
    }

    /// Runs all chains for a total of `n_collect + n_discard` steps and collects samples.
    ///
    /// First discards `n_discard` warm-up steps for each chain (during which adaptation occurs),
    /// then collects `n_collect` samples per chain.
    ///
    /// # Parameters
    /// - `n_collect`: Number of samples to collect after warm-up per chain.
    /// - `n_discard`: Number of warm-up (burn-in) steps to discard per chain.
    ///
    /// # Returns
    /// A 3D tensor of shape `[n_chains, n_collect, D]` containing the collected samples.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use burn::backend::{Autodiff, NdArray};
    /// # use mini_mcmc::nuts::NUTS;
    /// # use mini_mcmc::distributions::DiffableGaussian2D;
    /// type B = Autodiff<NdArray>;
    ///
    /// // As above, construct the sampler
    /// let gauss = DiffableGaussian2D::new([0.0, 0.0], [[1.0,0.0],[0.0,1.0]]);
    /// let mut sampler = NUTS::new(gauss, vec![vec![0.0,0.0]; 2], 0.8)
    ///     .set_seed(2025);
    ///
    /// // Discard 50 warm-up steps, then collect 150 samples per chain
    /// let samples = sampler.run(150, 50);
    ///
    /// // samples.dims() == [2 chains, 150 samples, 2 dimensions]
    /// assert_eq!(samples.dims(), [2, 150, 2]);
    /// ```
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 3> {
        let chain_samples: Vec<Tensor<B, 2>> = self
            .chains
            .par_iter_mut()
            .map(|chain| chain.run(n_collect, n_discard))
            .collect();
        Tensor::<B, 2>::stack(chain_samples, 0)
    }

    /// Run with live progress bars and collect summary stats.
    ///
    /// Spawns a background thread to render per-chain and global bars,
    /// then returns `(samples, RunStats)` when done.
    ///
    /// # Example
    ///
    /// ```rust
    /// use burn::backend::{Autodiff, NdArray};
    /// use mini_mcmc::distributions::Rosenbrock2D;
    /// use mini_mcmc::nuts::NUTS;
    /// use mini_mcmc::core::init;
    ///
    /// type B = Autodiff<NdArray>;
    ///
    /// let target = Rosenbrock2D { a: 1.0, b: 100.0 };
    /// let init   = init::<f64>(4, 2);    // 4 chains in 2D
    /// let mut sampler = NUTS::<f64, B, RosenBrock2D>::new(target, init, 0.9);
    /// let (samples, stats) = sampler.run_progress(100, 20).unwrap();
    /// ```
    ///
    /// You can swap in any other [`GradientTarget`] just as easily.
    pub fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<(Tensor<B, 3>, RunStats), Box<dyn Error>> {
        let chains = &mut self.chains;

        let mut rxs: Vec<Receiver<ChainStats>> = vec![];
        let mut txs: Vec<Sender<ChainStats>> = vec![];
        (0..chains.len()).for_each(|_| {
            let (tx, rx) = mpsc::channel();
            rxs.push(rx);
            txs.push(tx);
        });

        let progress_handle = thread::spawn(move || {
            let sleep_ms = Duration::from_millis(250);
            let timeout_ms = Duration::from_millis(0);
            let multi = MultiProgress::new();

            let pb_style = ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.cyan/blue} {pos}/{len} ({eta}) | {msg}")
                .unwrap()
                .progress_chars("=>-");
            let total: u64 = (n_collect + n_discard).try_into().unwrap();

            // Global Progress bar
            let global_pb = multi.add(ProgressBar::new((rxs.len() as u64) * total));
            global_pb.set_style(pb_style.clone());
            global_pb.set_prefix("Global");

            let mut active: Vec<(usize, ProgressBar)> = (0..rxs.len().min(5))
                .map(|chain_idx| {
                    let pb = multi.add(ProgressBar::new(total));
                    pb.set_style(pb_style.clone());
                    pb.set_prefix(format!("Chain {chain_idx}"));
                    (chain_idx, pb)
                })
                .collect();
            let mut next_active = active.len();
            let mut n_finished = 0;
            let mut most_recent = vec![None; rxs.len()];
            let mut total_progress;

            loop {
                for (i, rx) in rxs.iter().enumerate() {
                    while let Ok(stats) = rx.recv_timeout(timeout_ms) {
                        most_recent[i] = Some(stats)
                    }
                }

                // Update chain progress bar messages
                // and compute average acceptance probability
                let mut to_replace = vec![false; active.len()];
                let mut avg_p_accept = 0.0;
                let mut n_available_stats = 0.0;
                for (vec_idx, (i, pb)) in active.iter().enumerate() {
                    if let Some(stats) = &most_recent[*i] {
                        pb.set_position(stats.n);
                        pb.set_message(format!("p(accept)≈{:.2}", stats.p_accept));
                        avg_p_accept += stats.p_accept;
                        n_available_stats += 1.0;

                        if stats.n == total {
                            to_replace[vec_idx] = true;
                            n_finished += 1;
                        }
                    }
                }
                avg_p_accept /= n_available_stats;

                // Update global progress bar
                total_progress = 0;
                for stats in most_recent.iter().flatten() {
                    total_progress += stats.n;
                }
                global_pb.set_position(total_progress);
                let valid: Vec<&ChainStats> = most_recent.iter().flatten().collect();
                if valid.len() >= 2 {
                    let rhats = collect_rhat(valid.as_slice());
                    let max = rhats.max_skipnan();
                    global_pb.set_message(format!(
                        "p(accept)≈{:.2} max(rhat)≈{:.2}",
                        avg_p_accept, max
                    ))
                }

                let mut to_remove = vec![];
                for (i, replace) in to_replace.iter().enumerate() {
                    if *replace && next_active < most_recent.len() {
                        let pb = multi.add(ProgressBar::new(total));
                        pb.set_style(pb_style.clone());
                        pb.set_prefix(format!("Chain {next_active}"));
                        active[i] = (next_active, pb);
                        next_active += 1;
                    } else if *replace {
                        to_remove.push(i);
                    }
                }

                to_remove.sort();
                for i in to_remove.iter().rev() {
                    active.remove(*i);
                }

                if n_finished >= most_recent.len() {
                    break;
                }
                std::thread::sleep(sleep_ms);
            }
        });

        let chain_sample: Vec<Tensor<B, 2>> = thread::scope(|s| {
            let handles: Vec<thread::ScopedJoinHandle<Tensor<B, 2>>> = chains
                .iter_mut()
                .zip(txs)
                .map(|(chain, tx)| {
                    s.spawn(|| {
                        chain
                            .run_progress(n_collect, n_discard, tx)
                            .expect("Expected running chain to succeed.")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .expect("Expected thread to succeed in generating observation.")
                })
                .collect()
        });
        let sample = Tensor::<B, 2>::stack(chain_sample, 0);

        if let Err(e) = progress_handle.join() {
            eprintln!("Progress bar thread emitted error message: {:?}", e);
        }

        let sample_f32 = sample.to_data();
        let view =
            ArrayView3::<f32>::from_shape(sample.dims(), sample_f32.as_slice().unwrap()).unwrap();
        let run_stats = RunStats::from(view);

        Ok((sample, run_stats))
    }

    /// Sets a new random seed for all chains to ensure reproducibility.
    ///
    /// # Parameters
    /// - `seed`: Base seed value. Each chain will derive its own seed for independence.
    ///
    /// # Returns
    /// `self` with the RNGs re-seeded.
    pub fn set_seed(mut self, seed: u64) -> Self {
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = seed + i as u64 + 1;
            chain.rng = SmallRng::seed_from_u64(chain_seed);
        }
        self
    }
}

/// Single-chain state and adaptation for NUTS.
///
/// Manages the dynamic trajectory building, dual-averaging adaptation of step size,
/// and current position for one chain.
#[derive(Debug, Clone)]
pub struct NUTSChain<T, B, GTarget>
where
    B: AutodiffBackend,
{
    /// Target distribution providing gradients and log-probabilities.
    target: GTarget,

    /// Current position in parameter space.
    pub position: Tensor<B, 1>,

    /// Desired average acceptance probability.
    target_accept_p: T,

    /// Current step size (epsilon).
    epsilon: T,

    // Internal variables
    m: usize,
    n_collect: usize,
    n_discard: usize,
    gamma: T,
    t_0: usize,
    kappa: T,
    mu: T,
    epsilon_bar: T,
    h_bar: T,

    rng: SmallRng,
    phantom_data: std::marker::PhantomData<T>,
}

impl<T, B, GTarget> NUTSChain<T, B, GTarget>
where
    T: Float + ElementConversion + Element + SampleUniform + FromPrimitive,
    B: AutodiffBackend,
    GTarget: GradientTarget<T, B> + std::marker::Sync,
    StandardNormal: rand::distributions::Distribution<T>,
    rand_distr::Standard: rand_distr::Distribution<T>,
    rand_distr::Exp1: rand_distr::Distribution<T>,
{
    /// Constructs a new NUTSChain for a single chain with the given initial position.
    ///
    /// # Parameters
    /// - `target`: The target distribution implementing `GradientTarget`.
    /// - `initial_position`: Initial position vector of length `D`.
    /// - `target_accept_p`: Desired average acceptance probability for adaptation.
    ///
    /// # Returns
    /// An initialized `NUTSChain`.
    pub fn new(target: GTarget, initial_position: Vec<T>, target_accept_p: T) -> Self {
        let dim = initial_position.len();
        let td: TensorData = TensorData::new(initial_position, [dim]);
        let position = Tensor::<B, 1>::from_data(td, &B::Device::default());
        let rng = SmallRng::from_entropy();
        let epsilon = -T::one();

        Self {
            target,
            position,
            target_accept_p,
            epsilon,
            m: 0,
            n_collect: 0,
            n_discard: 0,
            gamma: T::from(0.05).unwrap(),
            t_0: 10,
            kappa: T::from(0.75).unwrap(),
            mu: (T::from(10.0).unwrap() * T::one()).ln(),
            epsilon_bar: T::one(),
            h_bar: T::zero(),
            rng,
            phantom_data: std::marker::PhantomData,
        }
    }

    /// Sets a new random seed for this chain to ensure reproducibility.
    ///
    /// # Parameters
    /// - `seed`: Seed value for the chain's RNG.
    ///
    /// # Returns
    /// `self` with the RNG re-seeded.
    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    /// Runs the chain for `n_collect + n_discard` steps, adapting during burn-in and
    /// returning collected samples.
    ///
    /// # Parameters
    /// - `n_collect`: Number of samples to collect after adaptation.
    /// - `n_discard`: Number of burn-in steps for adaptation.
    ///
    /// # Returns
    /// A 2D tensor of shape `[n_collect, D]` containing collected samples.
    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Tensor<B, 2> {
        let (dim, mut sample) = self.init_chain(n_collect, n_discard);

        for m in 1..(n_collect + n_discard) {
            self.step();

            if m >= n_discard {
                sample = sample.slice_assign(
                    [m - n_discard..m - n_discard + 1, 0..dim],
                    self.position.clone().unsqueeze(),
                );
            }
        }
        sample
    }

    fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
        tx: Sender<ChainStats>,
    ) -> Result<Tensor<B, 2>, Box<dyn Error>> {
        let (dim, mut sample) = self.init_chain(n_collect, n_discard);
        // let mut tracker = ChainTracker::new(dim, self.position.clone().to_element::<f32>(|x| x.to_f32()).to_data().as_slice().unwrap());
        let pos_0: Vec<f32> = self
            .position
            .to_data()
            .iter()
            .map(|x: T| ToElement::to_f32(&x))
            .collect();
        let mut tracker = ChainTracker::new(dim, &pos_0);
        let mut last = Instant::now();
        let freq = Duration::from_secs(1);
        let total = n_discard + n_collect;

        for i in 0..total {
            self.step();
            let pos_i: Vec<f32> = self
                .position
                .to_data()
                .iter()
                .map(|x: T| ToElement::to_f32(&x))
                .collect();
            tracker.step(&pos_i).map_err(|e| {
                let msg = format!(
                "Chain statistics tracker caused error: {}.\nAborting generation of further observations.",
                e
                );
                println!("{}", msg);
                msg
            })?;

            let now = Instant::now();
            if (now >= last + freq) | (i == total - 1) {
                if let Err(e) = tx.send(tracker.stats()) {
                    eprintln!("Sending chain statistics failed: {e}");
                }
                last = now;
            }

            if i >= n_discard {
                sample = sample.slice_assign(
                    [i - n_discard..i - n_discard + 1, 0..dim],
                    self.position.clone().unsqueeze(),
                );
            }
        }

        // TODO: Somehow save state of the chains and enable continuing runs
        Ok(sample)
    }

    fn init_chain(&mut self, n_collect: usize, n_discard: usize) -> (usize, Tensor<B, 2>) {
        let dim = self.position.dims()[0];
        self.n_collect = n_collect;
        self.n_discard = n_discard;

        let mut sample = Tensor::<B, 2>::empty([n_collect, dim], &B::Device::default());
        sample = sample.slice_assign([0..1, 0..dim], self.position.clone().unsqueeze());
        let mom_0_data: Vec<T> = (&mut self.rng)
            .sample_iter(StandardNormal)
            .take(dim)
            .collect();
        let mom_0 = Tensor::<B, 1>::from_data(mom_0_data.as_slice(), &B::Device::default());
        if T::abs(self.epsilon + T::one()) <= T::epsilon() {
            self.epsilon = find_reasonable_epsilon(self.position.clone(), mom_0, &self.target);
        }
        self.mu = T::ln(T::from(10).unwrap() * self.epsilon);
        (dim, sample)
    }

    /// Performs one NUTS update step, including tree expansion and adaptation updates.
    ///
    /// This method updates `self.position` and adaptation statistics in-place.
    pub fn step(&mut self) {
        self.m += 1;

        let dim = self.position.dims()[0];
        let mom_0 = (&mut self.rng)
            .sample_iter(StandardNormal)
            .take(dim)
            .collect::<Vec<T>>();
        let mom_0 = Tensor::<B, 1>::from_data(mom_0.as_slice(), &B::Device::default());
        let (ulogp, grad) = self.target.unnorm_logp_and_grad(self.position.clone());
        let joint = ulogp.clone() - (mom_0.clone() * mom_0.clone()).sum() * 0.5;
        let joint =
            T::from_f64(joint.into_scalar().to_f64()).expect("successful conversion from 64 to T");
        let exp1_obs = self.rng.sample(Exp1);
        let logu = joint - exp1_obs;

        let mut position_minus = self.position.clone();
        let mut position_plus = self.position.clone();
        let mut mom_minus = mom_0.clone();
        let mut mom_plus = mom_0.clone();
        let mut grad_minus = grad.clone();
        let mut grad_plus = grad.clone();
        let mut j = 0;
        let mut n = 1;
        let mut s = true; // 's' stands for 'stop', indicating the stopping of inner while loop
        let mut alpha: T = T::zero();
        let mut n_alpha: usize = 0;

        while s {
            let u_run_1: T = self.rng.gen::<T>();
            let v = (2 * (u_run_1 < T::from(0.5).unwrap()) as i8) - 1;

            let (position_prime, n_prime, s_prime) = {
                if v == -1 {
                    let (
                        position_minus_2,
                        mom_minus_2,
                        grad_minus_2,
                        _,
                        _,
                        _,
                        position_prime_2,
                        _,
                        _,
                        n_prime_2,
                        s_prime_2,
                        alpha_2,
                        n_alpha_2,
                    ) = build_tree(
                        position_minus.clone(),
                        mom_minus.clone(),
                        grad_minus.clone(),
                        logu,
                        v,
                        j,
                        self.epsilon,
                        &self.target,
                        joint,
                        &mut self.rng,
                    );

                    position_minus = position_minus_2;
                    mom_minus = mom_minus_2;
                    grad_minus = grad_minus_2;
                    alpha = alpha_2;
                    n_alpha = n_alpha_2;

                    (position_prime_2, n_prime_2, s_prime_2)
                } else {
                    let (
                        _,
                        _,
                        _,
                        position_plus_2,
                        mom_plus_2,
                        grad_plus_2,
                        position_prime_2,
                        _,
                        _,
                        n_prime_2,
                        s_prime_2,
                        alpha_2,
                        n_alpha_2,
                    ) = build_tree(
                        position_plus.clone(),
                        mom_plus.clone(),
                        grad_plus.clone(),
                        logu,
                        v,
                        j,
                        self.epsilon,
                        &self.target,
                        joint,
                        &mut self.rng,
                    );

                    position_plus = position_plus_2;
                    mom_plus = mom_plus_2;
                    grad_plus = grad_plus_2;
                    alpha = alpha_2;
                    n_alpha = n_alpha_2;

                    (position_prime_2, n_prime_2, s_prime_2)
                }
            };

            let tmp = T::one().min(
                T::from(n_prime).expect("successful conversion of n_prime from usize to T")
                    / T::from(n).expect("successful conversion of n from usize to T"),
            );
            let u_run_2 = self.rng.gen::<T>();
            if s_prime && (u_run_2 < tmp) {
                self.position = position_prime;
            }
            n += n_prime;

            s = s_prime
                && stop_criterion(
                    position_minus.clone(),
                    position_plus.clone(),
                    mom_minus.clone(),
                    mom_plus.clone(),
                );
            j += 1
        }

        let mut eta =
            T::one() / T::from(self.m + self.t_0).expect("successful conversion of m + t_0 to T");
        self.h_bar = (T::one() - eta) * self.h_bar
            + eta
                * (self.target_accept_p
                    - alpha / T::from(n_alpha).expect("successful conversion of n_alpha to T"));
        if self.m <= self.n_discard {
            let _m = T::from(self.m).expect("successful conversion of m to T");
            self.epsilon = T::exp(self.mu - T::sqrt(_m) / self.gamma * self.h_bar);
            eta = _m.powf(-self.kappa);
            self.epsilon_bar =
                T::exp((T::one() - eta) * T::ln(self.epsilon_bar) + eta * T::ln(self.epsilon));
        } else {
            self.epsilon = self.epsilon_bar;
        }
    }
}

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

            let u_build_tree: f64 = (*rng).gen::<f64>();
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
        backend::{Autodiff, NdArray},
        tensor::{Tensor, Tolerance},
    };
    use ndarray::ArrayView3;
    use ndarray_stats::QuantileExt;
    use num_traits::Float;

    // Use the CPU backend (NdArray) wrapped in Autodiff.
    type BackendType = Autodiff<NdArray>;

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
        let n_discard = 1;
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
                0.44937651,
                -0.1170752,
                0.75795663,
                -0.32304322,
                0.58511739,
                1.60916189,
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
                1.2114488, 1.2120318, 1.2114488, 1.2120318, 1.2114488, 1.2120318, 0.6581087,
                2.5633106, 1.2620386, 1.5624053,
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
                1.2114488, 1.2120318, 1.2114488, 1.2120318, 1.2114488, 1.2120318, 0.6581087,
                2.5633106, 1.2620386, 1.5624053,
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
        let n_collect = 100;
        let n_discard = 20;

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
