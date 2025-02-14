/*!
# Core MCMC Utilities

This module provides core functionality for running Markov Chain Monte Carlo (MCMC) chains in parallel.
It includes:
- The [`MarkovChain<S>`] trait, which abstracts a single MCMC chain.
- Utility functions [`run_chain`] and [`run_chain_with_progress`] for executing a single chain and collecting its states.
- The [`HasChains<S>`] trait for types that own multiple Markov chains.
- The [`ChainRunner<S>`] trait that extends `HasChains<S>` with methods to run chains in parallel (using Rayon), discarding burn-in and optionally displaying progress bars.

Any type implementing `HasChains<S>` (with the required trait bounds) automatically implements `ChainRunner<S>` via a blanket implementation.

This module is generic over the state type using [`num_traits::Float`].
*/

use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use nalgebra as na;
use num_traits::Zero;
use rayon::prelude::*;

/// A trait that abstracts a single MCMC chain.
///
/// A type implementing `MarkovChain<S>` must provide:
/// - `step()`: advances the chain one iteration and returns a reference to the updated state.
/// - `current_state()`: returns a reference to the current state without modifying the chain.
pub trait MarkovChain<S> {
    /// Performs one iteration of the chain and returns a reference to the new state.
    fn step(&mut self) -> &Vec<S>;

    /// Returns a reference to the current state of the chain without advancing it.
    fn current_state(&self) -> &Vec<S>;
}

/// Runs a single MCMC chain for a specified number of steps.
///
/// This function repeatedly calls the chain's `step()` method and collects each state into a
/// [`nalgebra::DMatrix`], where each row corresponds to one iteration of the chain.
///
/// # Arguments
///
/// * `chain` - A mutable reference to an object implementing [`MarkovChain<S>`].
/// * `n_steps` - The total number of iterations to run.
///
/// # Returns
///
/// A [`nalgebra::DMatrix<S>`] where the number of rows equals `n_steps` and the number of columns equals
/// the dimensionality of the chain's state.
pub fn run_chain<S, M>(chain: &mut M, n_steps: usize) -> na::DMatrix<S>
where
    M: MarkovChain<S>,
    S: Clone + na::Scalar + Zero,
{
    let dim = chain.current_state().len();
    let mut out = na::DMatrix::<S>::zeros(n_steps, dim);

    for i in 0..n_steps {
        let state = chain.step();
        out.row_mut(i).copy_from_slice(state);
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
/// * `chain` - A mutable reference to an object implementing [`MarkovChain<S>`].
/// * `n_steps` - The total number of iterations to run.
/// * `pb` - A progress bar used to display progress.
///
/// # Returns
///
/// A [`nalgebra::DMatrix<S>`] containing the chain's states (one row per iteration).
pub fn run_chain_with_progress<S, M>(
    chain: &mut M,
    n_steps: usize,
    pb: &ProgressBar,
) -> na::DMatrix<S>
where
    M: MarkovChain<S>,
    S: Clone + na::Scalar + Zero,
{
    let dim = chain.current_state().len();
    let mut out = na::DMatrix::<S>::zeros(n_steps, dim);

    pb.set_length(n_steps as u64);

    for i in 0..n_steps {
        let state = chain.step();
        out.row_mut(i).copy_from_slice(state);

        // Update progress bar
        pb.inc(1);
    }

    out
}

/// A trait for types that own multiple MCMC chains.
///
/// - `S` is the type of the state elements (e.g., `f64`).
/// - `Chain` is the concrete type of the individual chain, which must implement [`MarkovChain<S>`]
///   and be `Send`.
///
/// Implementors must provide a method to access the internal vector of chains.
pub trait HasChains<S> {
    type Chain: MarkovChain<S> + std::marker::Send;

    /// Returns a mutable reference to the vector of chains.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain>;
}

/// An extension trait for types that own multiple MCMC chains.
///
/// `ChainRunner<S>` extends [`HasChains<S>`] by providing default methods to run all chains
/// in parallel using Rayon. These methods allow you to:
/// - Run all chains for a specified number of iterations and discard an initial burn-in period.
/// - Optionally display progress bars for each chain during execution.
///
/// Any type that implements [`HasChains<S>`] (with appropriate bounds on `S`) automatically implements
/// `ChainRunner<S>`.
pub trait ChainRunner<S>: HasChains<S>
where
    S: std::clone::Clone
        + num_traits::Zero
        + std::marker::Send
        + std::cmp::PartialEq
        + std::marker::Sync
        + std::fmt::Debug
        + 'static,
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
    /// A vector of [`nalgebra::DMatrix<S>`] matrices, one for each chain, containing the samples
    /// after burn-in.
    fn run(&mut self, n_steps: usize, discard: usize) -> Vec<na::DMatrix<S>> {
        // Run them all in parallel
        let results: Vec<na::DMatrix<S>> = self
            .chains_mut()
            .par_iter_mut()
            .map(|chain| run_chain(chain, n_steps))
            .collect();

        // Now discard the burn-in rows from each matrix
        results
            .into_iter()
            .map(|mat| {
                let nrows = mat.nrows();
                let keep = nrows - discard;
                mat.rows(discard, keep).into()
            })
            .collect()
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
    /// A vector of sample matrices (one per chain) containing only the samples after burn-in.
    fn run_with_progress(&mut self, n_steps: usize, discard: usize) -> Vec<na::DMatrix<S>> {
        let multi = MultiProgress::new();
        let pb_style = ProgressStyle::default_bar()
            .template("{prefix} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("##-");

        // Run each chain in parallel
        let results: Vec<(Vec<S>, na::DMatrix<S>)> = self
            .chains_mut()
            .par_iter_mut()
            .enumerate()
            .map(|(i, chain)| {
                let pb = multi.add(ProgressBar::new(n_steps as u64));
                pb.set_prefix(format!("Chain {i}"));
                pb.set_style(pb_style.clone());

                let samples = run_chain_with_progress(chain, n_steps, &pb);

                pb.finish_with_message("Done!");

                (chain.current_state().clone(), samples)
            })
            .collect();

        results
            .into_par_iter()
            .map(|(_, samples)| {
                let keep_rows = samples.nrows().saturating_sub(discard);
                samples.rows(discard, keep_rows).into()
            })
            .collect()
    }
}

impl<
        S: std::fmt::Debug
            + std::marker::Sync
            + std::cmp::PartialEq
            + std::marker::Send
            + num_traits::Zero
            + std::clone::Clone
            + 'static,
        T: HasChains<S>,
    > ChainRunner<S> for T
{
}
