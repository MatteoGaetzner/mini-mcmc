use indicatif::ProgressBar;
use indicatif::{MultiProgress, ProgressStyle};
use nalgebra as na;
use num_traits::Zero;
use rayon::prelude::*;

pub trait MarkovChain<S> {
    /// Does one iteration of the chain, returning the new current state.
    /// (This could be a reference or a clone, depending on your design.)
    fn step(&mut self) -> &Vec<S>;

    /// Optional: get the current state without stepping.
    fn current_state(&self) -> &Vec<S>;
}

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

/// A trait for "anything that owns multiple MarkovChains".
/// - `S` is the state element type (e.g. f64).
/// - `Chain` is the MarkovChain type stored by this struct.
pub trait HasChains<S> {
    type Chain: MarkovChain<S> + std::marker::Send;

    /// Returns a mutable reference to the vector of chains.
    fn chains_mut(&mut self) -> &mut Vec<Self::Chain>;
}

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
    /// Runs the chains in parallel, discarding burn-in.
    fn run(&mut self, n_steps: usize, discard: usize) -> Vec<na::DMatrix<S>> {
        // Run them all in parallel
        let results: Vec<na::DMatrix<S>> = self
            .chains_mut()
            .par_iter_mut()
            .map(|chain| run_chain(chain, n_steps))
            .collect();

        // Now we can discard burn-in rows from each matrix
        results
            .into_iter()
            .map(|mat| {
                let nrows = mat.nrows();
                let keep = nrows - discard;
                mat.rows(discard, keep).into()
            })
            .collect()
    }

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
                // Create a progress bar for this chain
                let pb = multi.add(indicatif::ProgressBar::new(n_steps as u64));
                pb.set_prefix(format!("Chain {i}"));
                pb.set_style(pb_style.clone());

                // Run the chain with progress
                let samples = run_chain_with_progress(chain, n_steps, &pb);

                pb.finish_with_message("Done!");

                // Return final state + all samples
                (chain.current_state().clone(), samples)
            })
            .collect();

        // Discard burn-in
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
