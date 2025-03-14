/*!
# Core MCMC Utilities

This module provides core functionality for running Markov Chain Monte Carlo (MCMC) chains in parallel.
It includes:
- The [`MarkovChain<T>`] trait, which abstracts a single MCMC chain.
- Utility functions [`run_chain`] and [`run_chain_progress`] for executing a single chain and collecting its states.
- The [`HasChains<T>`] trait for types that own multiple Markov chains.
- The [`ChainRunner<T>`] trait that extends `HasChains<T>` with methods to run chains in parallel (using Rayon), discarding burn-in and optionally displaying progress bars.

Any type implementing `HasChains<T>` (with the required trait bounds) automatically implements `ChainRunner<T>` via a blanket implementation.

This module is generic over the state type using [`ndarray::LinalgScalar`].
*/

use crate::stats::{ChainStats, ChainTracker};
use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use ndarray::stack;
use ndarray::{prelude::*, LinalgScalar, ShapeError};
use rayon::prelude::*;
use std::cmp::PartialEq;
use std::error::Error;
use std::marker::Send;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self};
use std::time::{Duration, Instant};

/// A trait that abstracts a single MCMC chain.
///
/// A type implementing `MarkovChain<T>` must provide:
/// - `step()`: advances the chain one iteration and returns a reference to the updated state.
/// - `current_state()`: returns a reference to the current state without modifying the chain.
pub trait MarkovChain<T> {
    /// Performs one iteration of the chain and returns a reference to the new state.
    fn step(&mut self) -> &Vec<T>;

    /// Returns a reference to the current state of the chain without advancing it.
    fn current_state(&self) -> &Vec<T>;
}

/// Runs a single MCMC chain for a specified number of steps.
///
/// This function repeatedly calls the chain's `step()` method and collects each state into a
/// [`ndarray::Array2<T>`], where each row corresponds to one iteration of the chain.
///
/// # Arguments
///
/// * `chain` - A mutable reference to an object implementing [`MarkovChain<T>`].
/// * `n_steps` - The total number of iterations to run.
///
/// # Returns
///
/// A [`ndarray::Array2<T>`] where the number of rows equals `n_steps` and the number of columns equals
/// the dimensionality of the chain's state.
pub fn run_chain<T, M>(chain: &mut M, n_collect: usize, n_discard: usize) -> Array2<T>
where
    M: MarkovChain<T>,
    T: LinalgScalar,
{
    let dim = chain.current_state().len();
    let mut out = Array2::<T>::zeros((n_collect, dim));
    let total = n_collect + n_discard;

    for i in 0..total {
        let state = chain.step();
        if i >= n_discard {
            let state_arr = ArrayView::from_shape(state.len(), state.as_slice()).unwrap();
            out.row_mut(i - n_discard).assign(&state_arr);
        }
    }

    out
}

/// Runs a single MCMC chain for a specified number of steps while displaying progress.
///
/// This function is similar to [`run_chain`], but it accepts an [`indicatif::ProgressBar`]
/// that is updated as the chain advances.
///
/// # Arguments
///
/// * `chain` - A mutable reference to an object implementing [`MarkovChain<T>`].
/// * `n_steps` - The total number of iterations to run.
/// * `pb` - A progress bar used to display progress.
///
/// # Returns
///
/// A [`ndarray::Array2<T>`] containing the chain's states (one row per iteration).
pub fn run_chain_progress<T, M>(
    chain: &mut M,
    n_collect: usize,
    n_discard: usize,
    tx: Sender<ChainStats>,
) -> Result<Array2<T>, String>
where
    M: MarkovChain<T>,
    T: LinalgScalar + PartialEq + num_traits::ToPrimitive,
{
    let n_params = chain.current_state().len();
    let mut out = Array2::<T>::zeros((n_collect, n_params));

    let mut tracker = ChainTracker::new(n_params, chain.current_state());
    let mut last = Instant::now();
    let freq = Duration::from_millis(50);
    let total = n_discard + n_collect;

    for i in 0..total {
        let current_state = chain.step();
        tracker.step(current_state).map_err(|e| {
            let msg = format!(
            "Chain statistics tracker caused error: {}.\nAborting generation of further samples.",
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
            out.row_mut(i - n_discard).assign(
                &ArrayView1::from_shape(current_state.len(), current_state.as_slice()).unwrap(),
            );
        }
    }

    Ok(out)
}

// pub fn run_chain_progress<T, M>(
//     chain: &mut M,
//     n_steps: usize,
//     tx: Sender<ChainStats>,
// ) -> Array2<T>
// where
//     M: MarkovChain<T>,
//     T: LinalgScalar + PartialEq + num_traits::ToPrimitive,
// {
//     let n_params = chain.current_state().len();
//     let mut out = Array2::<T>::zeros((n_steps, n_params));
//
//     let mut n_accept = 0;
//     let mut accept_q = VecDeque::<bool>::new();
//     let mut last_state = chain.current_state().clone();
//     let mut tracker = ChainTracker::new(n_params, &last_state);
//
//     for i in 0..n_steps {
//         let current_state = chain.step();
//         tracker.step(current_state);
//         if last_state != *current_state {
//             n_accept += 1;
//             accept_q.push_front(true)
//         } else {
//             accept_q.push_front(false)
//         }
//         if i >= 50 {
//             if accept_q
//                 .pop_back()
//                 .expect("Expected popping back to yield something")
//             {
//                 n_accept -= 1;
//             }
//
//             if i % pbar_update_mod == 0 {
//                 let p_accept = n_accept as f32 / 50.0;
//                 pb.set_message(format!("p(accept) ≈ {p_accept}"));
//             }
//         }
//
//         out.row_mut(i).assign(
//             &ArrayView1::from_shape(current_state.len(), current_state.as_slice()).unwrap(),
//         );
//
//         last_state.clone_from(chain.current_state());
//     }
//
//     out
// }
//
/// A trait for types that own multiple MCMC chains.
///
/// - `T` is the type of the state elements (e.g., `f64`).
/// - `Chain` is the concrete type of the individual chain, which must implement [`MarkovChain<T>`]
///   and be `Send`.
///
/// Implementors must provide a method to access the internal vector of chains.
pub trait HasChains<S> {
    type Chain: MarkovChain<S> + Send;

    /// Returns a mutable reference to the vector of chains.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain>;
}

/// An extension trait for types that own multiple MCMC chains.
///
/// `ChainRunner<T>` extends [`HasChains<T>`] by providing default methods to run all chains
/// in parallel using Rayon. These methods allow you to:
/// - Run all chains for a specified number of iterations and discard an initial burn-in period.
/// - Optionally display progress bars for each chain during execution.
///
/// Any type that implements [`HasChains<T>`] (with appropriate bounds on `T`) automatically implements
/// `ChainRunner<T>`.
pub trait ChainRunner<T>: HasChains<T>
where
    T: LinalgScalar + PartialEq + Send + num_traits::ToPrimitive,
{
    /// Runs all chains in parallel, discarding the first `discard` iterations (burn-in).
    ///
    /// # Arguments
    ///
    /// * `n_steps` - The total number of iterations to run for each chain.
    /// * `discard` - The number of initial iterations to discard from each chain.
    ///
    /// # Returns
    ///
    /// A [`ndarray::Array3`] tensor with the first axis representing the chain, the second one the
    /// step and the last one the parameter dimension.
    fn run(&mut self, n_collect: usize, n_discard: usize) -> Result<Array3<T>, ShapeError> {
        // Run them all in parallel
        let results: Vec<Array2<T>> = self
            .chains_mut()
            .par_iter_mut()
            .map(|chain| run_chain(chain, n_collect, n_discard))
            .collect();
        let views: Vec<ArrayView2<T>> = results.iter().map(|x| x.view()).collect();
        let out: Array3<T> = stack(Axis(0), &views)?;
        Ok(out)
    }

    /// Runs all chains in parallel with progress bars, discarding the burn-in.
    ///
    /// Each chain is run concurrently with its own progress bar. After execution, the first `discard`
    /// iterations are discarded.
    ///
    /// # Arguments
    ///
    /// * `n_steps` - The total number of iterations to run for each chain.
    /// * `discard` - The number of initial iterations to discard.
    ///
    /// # Returns
    ///
    /// Returns a [`ndarray::Array3`] tensor with the first axis representing the chain, the second one the
    /// step and the last one the parameter dimension.
    fn run_progress(
        &mut self,
        n_collect: usize,
        n_discard: usize,
    ) -> Result<Array3<T>, Box<dyn Error>> {
        if n_collect < 50 {
            let out = self.run(n_collect, n_discard)?;
            return Ok(out);
        }

        // Channels.
        // Each chain gets its own channel. Hence, we have `n_chains` channels.
        // The objects sent over channels are Array2<f32>s ($s_m^2$, $\bar{\theta}_m^{(\bullet)}$).
        // The child thread sends it's respective one to the parent thread.
        // The parent thread assemples the tuples it receives to compute Rhat.

        let chains = self.chains_mut();

        let mut rxs: Vec<Receiver<ChainStats>> = vec![];
        let mut txs: Vec<Sender<ChainStats>> = vec![];
        (0..chains.len()).for_each(|_| {
            let (tx, rx) = mpsc::channel();
            rxs.push(rx);
            txs.push(tx);
        });

        let progress_handle = thread::spawn(move || {
            let sleep_ms = Duration::from_millis(50);
            let timeout_ms = Duration::from_millis(0);
            let multi = MultiProgress::new();

            let pb_style = ProgressStyle::default_bar()
                .template("{prefix:8} {bar:40.white} ETA {eta:3} | {msg}")
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
                    if let Ok(stats) = rx.recv_timeout(timeout_ms) {
                        most_recent[i] = Some(stats)
                    }
                }

                total_progress = 0;
                for stats in most_recent.iter().flatten() {
                    total_progress += stats.n;
                }
                global_pb.set_position(total_progress);
                global_pb.tick();

                let mut to_replace = vec![false; active.len()];
                for (vec_idx, (i, pb)) in active.iter().enumerate() {
                    if let Some(stats) = &most_recent[*i] {
                        pb.set_position(stats.n);

                        if stats.n == total {
                            to_replace[vec_idx] = true;
                            n_finished += 1;
                        }
                    }
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

        let samples: Vec<Array2<T>> = thread::scope(|s| {
            let handles: Vec<thread::ScopedJoinHandle<Array2<T>>> = chains
                .iter_mut()
                .zip(txs)
                .map(|(chain, tx)| {
                    std::thread::sleep(Duration::from_millis(50));
                    s.spawn(|| {
                        run_chain_progress(chain, n_collect, n_discard, tx)
                            .expect("Expected running chain to succeed.")
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| {
                    h.join()
                        .expect("Expected thread to succeed in generating sample.")
                })
                .collect()
        });
        let out: Array3<T> = stack(
            Axis(0),
            &samples
                .iter()
                .map(|x| x.view())
                .collect::<Vec<ArrayView2<T>>>(),
        )?;

        if let Err(e) = progress_handle.join() {
            println!("Progress bar thread emitted error message: {:?}", e);
        }
        Ok(out)
    }
}

impl<T: LinalgScalar + Send + PartialEq + num_traits::ToPrimitive, R: HasChains<T>> ChainRunner<T>
    for R
{
}
