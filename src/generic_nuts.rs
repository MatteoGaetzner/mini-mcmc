//! Backend-agnostic No-U-Turn Sampler (NUTS) core.

use crate::euclidean::EuclideanVector;
use crate::generic_hmc::HamiltonianTarget;
use crate::stats::{collect_rhat, ChainStats, ChainTracker, RunStats};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use ndarray::{s, Array2, Array3, ArrayView1, ArrayView2, Axis};
use ndarray_stats::QuantileExt;
use num_traits::{Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::distr::Distribution as RandDistribution;
// rand_distr provides the distributions, but we rely on rand's Distribution trait for compatibility.
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Exp1, StandardNormal, StandardUniform};
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use std::error::Error;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// Backend-agnostic No-U-Turn Sampler (NUTS) spanning multiple chains.
#[derive(Clone)]
pub struct GenericNUTS<V, Target>
where
    V: EuclideanVector,
    Target: HamiltonianTarget<V>,
{
    chains: Vec<GenericNUTSChain<V, Target>>,
}

type RunResult<T> = Result<(Array3<T>, RunStats), Box<dyn Error>>;

impl<V, Target> GenericNUTS<V, Target>
where
    V: EuclideanVector + Send,
    V::Scalar: Float + FromPrimitive + ToPrimitive + Send,
    Target: HamiltonianTarget<V> + Sync + Clone + Send,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
    Exp1: RandDistribution<V::Scalar>,
{
    pub fn new(target: Target, initial_positions: Vec<V>, target_accept_p: V::Scalar) -> Self {
        let chains = initial_positions
            .into_iter()
            .map(|pos| GenericNUTSChain::new(target.clone(), pos, target_accept_p))
            .collect();
        Self { chains }
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array3<V::Scalar> {
        let chain_samples: Vec<Array2<V::Scalar>> = self
            .chains
            .par_iter_mut()
            .map(|chain| chain.run(n_collect, n_discard))
            .collect();
        let views: Vec<ArrayView2<V::Scalar>> = chain_samples.iter().map(|s| s.view()).collect();
        ndarray::stack(Axis(0), &views).expect("expected stacking chain samples to succeed")
    }

    pub fn run_progress(&mut self, n_collect: usize, n_discard: usize) -> RunResult<V::Scalar> {
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

            loop {
                for (i, rx) in rxs.iter().enumerate() {
                    while let Ok(stats) = rx.recv_timeout(timeout_ms) {
                        most_recent[i] = Some(stats)
                    }
                }

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
                if n_available_stats > 0.0 {
                    avg_p_accept /= n_available_stats;
                }

                let mut total_progress = 0;
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

        let chain_sample: Vec<Array2<V::Scalar>> = thread::scope(|s| {
            let handles: Vec<thread::ScopedJoinHandle<Array2<V::Scalar>>> = chains
                .iter_mut()
                .zip(txs)
                .map(|(chain, tx)| {
                    s.spawn(|| {
                        chain
                            .run_progress(n_collect, n_discard, tx)
                            .expect("expected running chain to succeed.")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .expect("expected thread to succeed in generating observation.")
                })
                .collect()
        });
        let views: Vec<ArrayView2<V::Scalar>> = chain_sample.iter().map(|s| s.view()).collect();
        let sample = ndarray::stack(Axis(0), &views).expect("expected stacking sample to succeed");

        if let Err(e) = progress_handle.join() {
            eprintln!("Progress bar thread emitted error message: {:?}", e);
        }

        let run_stats = RunStats::from(sample.view());
        Ok((sample, run_stats))
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        for (i, chain) in self.chains.iter_mut().enumerate() {
            let chain_seed = seed + i as u64 + 1;
            chain.rng = SmallRng::seed_from_u64(chain_seed);
        }
        self
    }
}

/// Single-chain state and adaptation for NUTS.
#[derive(Clone)]
pub struct GenericNUTSChain<V, Target>
where
    V: EuclideanVector,
    Target: HamiltonianTarget<V>,
{
    target: Target,
    position: V,
    target_accept_p: V::Scalar,
    epsilon: V::Scalar,
    m: usize,
    n_collect: usize,
    n_discard: usize,
    gamma: V::Scalar,
    t_0: usize,
    kappa: V::Scalar,
    mu: V::Scalar,
    epsilon_bar: V::Scalar,
    h_bar: V::Scalar,
    rng: SmallRng,
}

impl<V, Target> GenericNUTSChain<V, Target>
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive + ToPrimitive,
    Target: HamiltonianTarget<V> + Sync,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
    Exp1: RandDistribution<V::Scalar>,
{
    pub fn new(target: Target, initial_position: V, target_accept_p: V::Scalar) -> Self {
        let mut thread_rng = rand::rng();
        let rng = SmallRng::from_rng(&mut thread_rng);
        let epsilon = -V::Scalar::one();

        Self {
            target,
            position: initial_position,
            target_accept_p,
            epsilon,
            m: 0,
            n_collect: 0,
            n_discard: 0,
            gamma: V::Scalar::from_f64(0.05).unwrap(),
            t_0: 10,
            kappa: V::Scalar::from_f64(0.75).unwrap(),
            mu: (V::Scalar::from_f64(10.0).unwrap() * V::Scalar::one()).ln(),
            epsilon_bar: V::Scalar::one(),
            h_bar: V::Scalar::zero(),
            rng,
        }
    }

    pub fn set_seed(mut self, seed: u64) -> Self {
        self.rng = SmallRng::seed_from_u64(seed);
        self
    }

    pub fn position(&self) -> &V {
        &self.position
    }

    pub fn run(&mut self, n_collect: usize, n_discard: usize) -> Array2<V::Scalar> {
        let (dim, mut sample) = self.init_chain(n_collect, n_discard);
        let mut scratch = vec![V::Scalar::zero(); dim];

        for m in 1..(n_collect + n_discard) {
            self.step();

            if m >= n_discard {
                self.position.write_to_slice(&mut scratch);
                let view = ArrayView1::from(&scratch);
                sample.slice_mut(s![m - n_discard, ..]).assign(&view);
            }
        }
        sample
    }

    fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
        tx: Sender<ChainStats>,
    ) -> Result<Array2<V::Scalar>, Box<dyn Error>> {
        let (dim, mut sample) = self.init_chain(n_collect, n_discard);
        let mut scratch = vec![V::Scalar::zero(); dim];
        self.position.write_to_slice(&mut scratch);

        let mut tracker = ChainTracker::new(dim, &scratch);
        let mut last = Instant::now();
        let freq = Duration::from_secs(1);
        let total = n_discard + n_collect;

        for i in 0..total {
            self.step();
            self.position.write_to_slice(&mut scratch);
            tracker.step(&scratch).map_err(|e| {
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
                let view = ArrayView1::from(&scratch);
                sample.slice_mut(s![i - n_discard, ..]).assign(&view);
            }
        }

        Ok(sample)
    }

    fn init_chain(&mut self, n_collect: usize, n_discard: usize) -> (usize, Array2<V::Scalar>) {
        let dim = self.position.len();
        self.n_collect = n_collect;
        self.n_discard = n_discard;

        let mut sample = Array2::<V::Scalar>::zeros((n_collect, dim));
        let mut scratch = vec![V::Scalar::zero(); dim];
        self.position.write_to_slice(&mut scratch);
        let view = ArrayView1::from(&scratch);
        sample.slice_mut(s![0, ..]).assign(&view);

        let mut mom_0 = self.position.zeros_like();
        mom_0.fill_standard_normal(&mut self.rng);
        if V::Scalar::abs(self.epsilon + V::Scalar::one()) <= V::Scalar::epsilon() {
            self.epsilon = find_reasonable_epsilon(&self.position, &mom_0, &self.target);
        }
        self.mu = (V::Scalar::from_f64(10.0).unwrap() * self.epsilon).ln();
        (dim, sample)
    }

    pub fn step(&mut self) {
        self.m += 1;

        let mut mom_0 = self.position.zeros_like();
        mom_0.fill_standard_normal(&mut self.rng);

        let mut grad = self.position.zeros_like();
        let logp = self.target.logp_and_grad(&self.position, &mut grad);
        let joint = logp - mom_0.dot(&mom_0) * V::Scalar::from_f64(0.5).unwrap();
        let exp1_obs: V::Scalar = self.rng.sample(Exp1);
        let logu = joint - exp1_obs;

        let mut position_minus = self.position.clone();
        let mut position_plus = self.position.clone();
        let mut mom_minus = mom_0.clone();
        let mut mom_plus = mom_0.clone();
        let mut grad_minus = grad.clone();
        let mut grad_plus = grad.clone();
        let mut j = 0;
        let mut n = 1;
        let mut s = true;
        let mut alpha: V::Scalar = V::Scalar::zero();
        let mut n_alpha: usize = 0;

        while s {
            let u_run_1: V::Scalar = self.rng.random();
            let v = (2 * (u_run_1 < V::Scalar::from_f64(0.5).unwrap()) as i8) - 1;

            let (position_prime, n_prime, s_prime) = if v == -1 {
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
            };

            let tmp = V::Scalar::one().min(
                V::Scalar::from_usize(n_prime)
                    .expect("successful conversion of n_prime from usize")
                    / V::Scalar::from_usize(n).expect("successful conversion of n from usize"),
            );
            let u_run_2: V::Scalar = self.rng.random();
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

        let mut eta = V::Scalar::one()
            / V::Scalar::from_usize(self.m + self.t_0).expect("successful conversion of m + t_0");
        self.h_bar = (V::Scalar::one() - eta) * self.h_bar
            + eta
                * (self.target_accept_p
                    - alpha
                        / V::Scalar::from_usize(n_alpha)
                            .expect("successful conversion of n_alpha"));
        if self.m <= self.n_discard {
            let m = V::Scalar::from_usize(self.m).expect("successful conversion of m");
            self.epsilon = (self.mu - m.sqrt() / self.gamma * self.h_bar).exp();
            eta = m.powf(-self.kappa);
            self.epsilon_bar =
                ((V::Scalar::one() - eta) * self.epsilon_bar.ln() + eta * self.epsilon.ln()).exp();
        } else {
            self.epsilon = self.epsilon_bar;
        }
    }
}

fn all_real_vec<V: EuclideanVector>(v: &V) -> bool
where
    V::Scalar: Float,
{
    let mut scratch = vec![V::Scalar::zero(); v.len()];
    v.write_to_slice(&mut scratch);
    scratch.iter().all(|x: &V::Scalar| x.is_finite())
}

#[allow(dead_code)]
pub(crate) fn find_reasonable_epsilon<V, Target>(
    position: &V,
    mom: &V,
    gradient_target: &Target,
) -> V::Scalar
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V> + Sync,
    StandardNormal: RandDistribution<V::Scalar>,
    StandardUniform: RandDistribution<V::Scalar>,
{
    let mut epsilon = V::Scalar::one();
    let half = V::Scalar::from_f64(0.5).unwrap();

    let mut grad = position.zeros_like();
    let ulogp = gradient_target.logp_and_grad(position, &mut grad);

    let mut position_prime = position.clone();
    let mut mom_prime = mom.clone();
    let mut grad_prime = grad.clone();
    let mut ulogp_prime = leapfrog(
        &mut position_prime,
        &mut mom_prime,
        &mut grad_prime,
        epsilon,
        gradient_target,
    );
    let mut k = V::Scalar::one();

    while !ulogp_prime.is_finite() && !all_real_vec(&grad_prime) {
        k = k * half;
        position_prime.assign(position);
        mom_prime.assign(mom);
        grad_prime.assign(&grad);
        ulogp_prime = leapfrog(
            &mut position_prime,
            &mut mom_prime,
            &mut grad_prime,
            epsilon * k,
            gradient_target,
        );
    }

    epsilon = half * k * epsilon;
    let log_accept_prob = ulogp_prime - ulogp - (mom_prime.dot(&mom_prime) - mom.dot(mom)) * half;
    let mut log_accept_prob = log_accept_prob;

    let a = if log_accept_prob > half.ln() {
        V::Scalar::one()
    } else {
        -V::Scalar::one()
    };

    while a * log_accept_prob > -a * V::Scalar::from_f64(2.0).unwrap().ln() {
        epsilon = epsilon * V::Scalar::from_f64(2.0).unwrap().powf(a);
        position_prime.assign(position);
        mom_prime.assign(mom);
        grad_prime.assign(&grad);
        ulogp_prime = leapfrog(
            &mut position_prime,
            &mut mom_prime,
            &mut grad_prime,
            epsilon,
            gradient_target,
        );
        log_accept_prob = ulogp_prime - ulogp - (mom_prime.dot(&mom_prime) - mom.dot(mom)) * half;
    }

    epsilon
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub(crate) fn build_tree<V, Target>(
    position: V,
    mom: V,
    grad: V,
    logu: V::Scalar,
    v: i8,
    j: usize,
    epsilon: V::Scalar,
    gradient_target: &Target,
    joint_0: V::Scalar,
    rng: &mut SmallRng,
) -> (
    V,
    V,
    V,
    V,
    V,
    V,
    V,
    V,
    V::Scalar,
    usize,
    bool,
    V::Scalar,
    usize,
)
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V> + Sync,
{
    if j == 0 {
        let mut position_prime = position.clone();
        let mut mom_prime = mom.clone();
        let mut grad_prime = grad.clone();
        let logp_prime = leapfrog(
            &mut position_prime,
            &mut mom_prime,
            &mut grad_prime,
            V::Scalar::from_i64(v as i64).unwrap() * epsilon,
            gradient_target,
        );
        let joint = logp_prime - mom_prime.dot(&mom_prime) * V::Scalar::from_f64(0.5).unwrap();
        let n_prime = (logu < joint) as usize;
        let s_prime = (logu - V::Scalar::from_f64(1000.0).unwrap()) < joint;
        let position_minus = position_prime.clone();
        let position_plus = position_prime.clone();
        let mom_minus = mom_prime.clone();
        let mom_plus = mom_prime.clone();
        let grad_minus = grad_prime.clone();
        let grad_plus = grad_prime.clone();
        let alpha_prime = V::Scalar::one().min((joint - joint_0).exp());
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

            let u_build_tree: f64 = rng.random();
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

pub(crate) fn stop_criterion<V>(
    position_minus: V,
    position_plus: V,
    mom_minus: V,
    mom_plus: V,
) -> bool
where
    V: EuclideanVector,
    V::Scalar: Float,
{
    let mut diff = position_plus.clone();
    diff.add_scaled_assign(&position_minus, -V::Scalar::one());
    let dot_minus = diff.dot(&mom_minus);
    let dot_plus = diff.dot(&mom_plus);
    dot_minus >= V::Scalar::zero() && dot_plus >= V::Scalar::zero()
}

pub(crate) fn leapfrog<V, Target>(
    position: &mut V,
    momentum: &mut V,
    grad: &mut V,
    epsilon: V::Scalar,
    gradient_target: &Target,
) -> V::Scalar
where
    V: EuclideanVector,
    V::Scalar: Float + FromPrimitive,
    Target: HamiltonianTarget<V>,
{
    let half = V::Scalar::from_f64(0.5).unwrap() * epsilon;
    momentum.add_scaled_assign(grad, half);
    position.add_scaled_assign(momentum, epsilon);
    let logp = gradient_target.logp_and_grad(position, grad);
    momentum.add_scaled_assign(grad, half);
    logp
}
