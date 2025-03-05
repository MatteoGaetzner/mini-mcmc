/*!
# Core MCMC Utilities

This module provides core functionality for running Markov Chain Monte Carlo (MCMC) chains in parallel.
It includes:
- The [`MarkovChain<T>`] trait, which abstracts a single MCMC chain.
- Utility functions [`run_chain`] and [`run_chain_with_progress`] for executing a single chain and collecting its states.
- The [`HasChains<T>`] trait for types that own multiple Markov chains.
- The [`ChainRunner<T>`] trait that extends `HasChains<T>` with methods to run chains in parallel (using Rayon), discarding burn-in and optionally displaying progress bars.

Any type implementing `HasChains<T>` (with the required trait bounds) automatically implements `ChainRunner<T>` via a blanket implementation.

This module is generic over the state type using [`ndarray::LinalgScalar`].
*/

use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use ndarray::{prelude::*, LinalgScalar, ShapeError};
use ndarray::{stack, Slice};
use rayon::prelude::*;
use std::cmp::PartialEq;
use std::collections::VecDeque;
use std::marker::Send;

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
pub fn run_chain<T, M>(chain: &mut M, n_steps: usize) -> Array2<T>
where
    M: MarkovChain<T>,
    T: LinalgScalar,
{
    let dim = chain.current_state().len();
    let mut out = Array2::<T>::zeros((n_steps, dim));

    for i in 0..n_steps {
        let state = chain.step();
        let state_arr = ArrayView::from_shape(state.len(), state.as_slice()).unwrap();
        out.row_mut(i).assign(&state_arr);
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
pub fn run_chain_with_progress<T, M>(chain: &mut M, n_steps: usize, pb: &ProgressBar) -> Array2<T>
where
    M: MarkovChain<T>,
    T: LinalgScalar + PartialEq,
{
    let dim = chain.current_state().len();
    let mut out = Array2::<T>::zeros((n_steps, dim));

    pb.set_length(n_steps as u64);
    let mut n_accept = 0;
    let mut accept_q = VecDeque::<bool>::new();
    let mut last_state = chain.current_state().clone();
    let mut pbar_update_mod: usize = 0;

    for i in 0..n_steps {
        let current_state = chain.step();
        if last_state != *current_state {
            n_accept += 1;
            accept_q.push_front(true)
        } else {
            accept_q.push_front(false)
        }
        if i >= 50 {
            if accept_q
                .pop_back()
                .expect("Expected popping back to yield something")
            {
                n_accept -= 1;
            }

            if pbar_update_mod == 0 {
                // updates_per_sec = iter_per_sec / pbar_update_mod
                // <=> pbar_update_mod = iter_per_sec / updates_per_sec
                // pbar_update_mod =
                let iter_per_sec = (i as f32) / (pb.elapsed().as_secs() as f32);
                pbar_update_mod = (iter_per_sec / 10.0).ceil() as usize;
            }

            if i % pbar_update_mod == 0 {
                let p_accept = n_accept as f32 / 50.0;
                pb.set_message(format!("p(accept) ≈ {p_accept}"));
            }
        }

        out.row_mut(i).assign(
            &ArrayView1::from_shape(current_state.len(), current_state.as_slice()).unwrap(),
        );

        // Update progress bar
        pb.inc(1);
        last_state.clone_from(chain.current_state());
    }

    out
}

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
    T: LinalgScalar + PartialEq + Send,
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
    fn run(&mut self, n_steps: usize, discard: usize) -> Result<Array3<T>, ShapeError> {
        // Run them all in parallel
        let results: Vec<Array2<T>> = self
            .chains_mut()
            .par_iter_mut()
            .map(|chain| run_chain(chain, n_steps))
            .collect();
        let views: Vec<ArrayView2<T>> = results
            .iter()
            .map(|x| x.slice_axis(Axis(0), Slice::from(discard..discard + n_steps)))
            .collect();
        let out: Array3<T> = stack(Axis(0), views.as_slice())?;
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
    fn run_progress(&mut self, n_steps: usize, discard: usize) -> Result<Array3<T>, ShapeError> {
        let multi = MultiProgress::new();
        let pb_style = ProgressStyle::default_bar()
            .template("{prefix} [{elapsed_precise}, {eta}] {bar:40.white} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("=>-");

        // Run each chain in parallel
        let results: Vec<Array2<T>> = self
            .chains_mut()
            .par_iter_mut()
            .map(|chain| {
                let pb = multi.add(ProgressBar::new(n_steps as u64));
                pb.set_style(pb_style.clone());

                let samples = run_chain_with_progress(chain, n_steps, &pb);

                pb.finish_with_message("Done!");

                samples
            })
            .collect();

        let views: Vec<ArrayView2<T>> = results
            .iter()
            .map(|x| x.slice_axis(Axis(0), Slice::from(discard..discard + n_steps)))
            .collect();
        let out: Array3<T> = stack(Axis(0), views.as_slice())?;
        Ok(out)
    }
}

impl<T: LinalgScalar + Send + PartialEq, R: HasChains<T>> ChainRunner<T> for R {}
