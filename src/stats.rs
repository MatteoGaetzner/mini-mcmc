//! Computation and tracking of MCMC statistics like acceptance probability and Potential Scale
//! Reduction.

use burn::prelude::*;
use ndarray::{concatenate, prelude::*, stack};
use ndarray_stats::QuantileExt;
use num_traits::Num;
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};
use std::{cmp::Ordering, error::Error};

const ALPHA: f32 = 0.01;

/// Tracks statistics for a single MCMC chain.
///
/// # Fields
/// - `n_params`: Number of parameters in the chain.
/// - `n`: Number of steps taken.
/// - `p_accept`: Acceptance probability.
/// - `mean`: Mean of the parameters.
/// - `mean_sq`: Mean of the squared parameters.
/// - `last_state`: Last state of the chain.
/// - `accept_queue`: Queue tracking acceptance history.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainTracker {
    n_params: usize,
    n: u64,
    p_accept: f32,
    last_state: Array1<f32>,
    mean: Array1<f32>,    // n_params
    mean_sq: Array1<f32>, // n_params
}

/// Statistics of an MCMC chain.
///
/// # Fields
/// - `n`: Number of steps taken.
/// - `p_accept`: Acceptance probability.
/// - `mean`: Mean of the parameters.
/// - `sm2`: Variance of the parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct ChainStats {
    pub n: u64,
    pub p_accept: f32,
    pub mean: Array1<f32>, // n_params
    pub sm2: Array1<f32>,  // n_params
}

impl ChainTracker {
    /// Creates a new `ChainTracker` with the given number of parameters and initial state.
    ///
    /// # Arguments
    /// - `n_params`: Number of parameters in the chain.
    /// - `initial_state`: Initial state of the chain.
    ///
    /// # Returns
    /// A new `ChainTracker` instance.
    pub fn new<T>(n_params: usize, initial_state: &[T]) -> Self
    where
        T: num_traits::ToPrimitive + Clone,
    {
        let mean_sq = Array1::<f32>::zeros(n_params);
        let mean = Array1::<f32>::zeros(n_params);
        let last_state = ArrayView1::from_shape(n_params, initial_state)
            .expect("Expected being able to convert initial state to a NdArray")
            .mapv(|x| {
                x.to_f32()
                    .expect("Expected conversion of elements to f32's to succeed")
            });

        Self {
            n_params,
            n: 0,
            p_accept: -1.0,
            last_state,
            mean,
            mean_sq,
        }
    }

    /// Updates the tracker with a new state.
    ///
    /// # Arguments
    /// - `x`: New state of the chain.
    ///
    /// # Returns
    /// `Ok(())` if successful; an error otherwise.
    pub fn step<T>(&mut self, x: &[T]) -> Result<(), Box<dyn Error>>
    where
        T: num_traits::ToPrimitive + Clone,
    {
        self.n += 1;

        let n = self.n as f32;
        let x_arr =
            ndarray::ArrayView1::<T>::from_shape(self.n_params, x)?.mapv(|x| x.to_f32().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_arr.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_arr.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_arr.pow2())) / n;
        };

        //  x_1 = (1 - a) x_0 + a x_1
        // <=> x_1 (1 - a) = (1 - a) x_0
        // <=> x_1 = x_0
        // So set initial p_accept to 1 if the transition state was an 'accept' and 0 otherwise
        let p_start = if self.p_accept >= 0.0 {
            self.p_accept
        } else {
            x_arr
                .index_axis(Axis(0), 0)
                .ne(&self.last_state.index_axis(Axis(0), 0)) as i32 as f32
        };
        self.p_accept = ndarray::Zip::from(x_arr.rows())
            .and(self.last_state.rows())
            .fold(p_start, |p_accept, a, b| {
                let accepted = (a.ne(&b) as i32) as f32;
                (1.0 - ALPHA) * p_accept + ALPHA * accepted
            });
        self.last_state = x_arr;

        Ok(())
    }

    /// Retrieves the current statistics of the chain.
    ///
    /// # Returns
    /// A `ChainStats` struct containing the current statistics.
    pub fn stats(&self) -> ChainStats {
        let n = self.n as f32;
        ChainStats {
            n: self.n,
            p_accept: self.p_accept,
            mean: self.mean.clone(),
            sm2: (self.mean_sq.clone() - self.mean.pow2()) * n / (n - 1.0),
        }
    }
}

/// Computes the Potential Scale Reduction Factor (R-hat) for multiple chains.
///
/// # Arguments
/// - `chain_stats`: Slice of references to `ChainStats` from multiple chains.
///
/// # Returns
/// An array containing the R-hat values for each parameter.
pub fn collect_rhat(chain_stats: &[&ChainStats]) -> Array1<f32> {
    let (within, var) = withinvar_from_cs(chain_stats);
    (var / within).sqrt()
}

fn withinvar_from_cs(chain_stats: &[&ChainStats]) -> (Array1<f32>, Array1<f32>) {
    let means: Vec<ArrayView1<f32>> = chain_stats.iter().map(|x| x.mean.view()).collect();
    let means = ndarray::stack(Axis(0), &means).expect("Expected stacking means to succeed");
    let sm2s: Vec<ArrayView1<f32>> = chain_stats.iter().map(|x| x.sm2.view()).collect();
    let sm2s = ndarray::stack(Axis(0), &sm2s).expect("Expected stacking sm2 arrays to succeed");

    let within = sm2s
        .mean_axis(Axis(0))
        .expect("Expected computing within-chain variances to succeed");
    let global_means = means
        .mean_axis(Axis(0))
        .expect("Expected computing global means to succeed");
    let diffs: Array2<f32> = (means.clone()
        - global_means
            .broadcast(means.shape())
            .expect("Expected broadcasting to succeed"))
    .into_dimensionality()
    .expect("Expected casting dimensionality to Array1 to succeed");
    let between = diffs.pow2().sum_axis(Axis(0)) / (diffs.len() - 1) as f32;

    let n: f32 = chain_stats.iter().map(|x| x.n as f32).sum::<f32>() / chain_stats.len() as f32;
    let var = between + within.clone() * ((n - 1.0) / n);
    (within, var)
}

/// Tracks statistics across multiple MCMC chains.
///
/// # Fields
/// - `n`: Number of steps taken.
/// - `mean`: Mean of the parameters across chains.
/// - `mean_sq`: Mean of the squared parameters across chains.
/// - `n_chains`: Number of chains.
/// - `n_params`: Number of parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiChainTracker {
    n: usize,
    pub p_accept: f32,
    last_state: Array2<f32>,
    mean: Array2<f32>,    // n_chains x n_params
    mean_sq: Array2<f32>, // n_chains x n_params
    n_chains: usize,
    n_params: usize,
}

impl MultiChainTracker {
    /// Creates a new `MultiChainTracker` for the given number of chains and parameters.
    ///
    /// # Arguments
    /// - `n_chains`: Number of chains.
    /// - `n_params`: Number of parameters.
    ///
    /// # Returns
    /// A new `MultiChainTracker` instance.
    pub fn new(n_chains: usize, n_params: usize) -> Self {
        let mean_sq = Array2::<f32>::zeros((n_chains, n_params));
        Self {
            n: 0,
            p_accept: 0.0,
            last_state: Array2::<f32>::zeros((n_chains, n_params)),
            mean: Array2::<f32>::zeros((n_chains, n_params)),
            mean_sq,
            n_chains,
            n_params,
        }
    }

    /// Updates the tracker with new states from all chains.
    ///
    /// # Arguments
    /// - `x`: New states of the chains, flattened into a single slice.
    ///
    /// # Returns
    /// `Ok(())` if successful; an error otherwise.
    pub fn step<T>(&mut self, x: &[T]) -> Result<(), Box<dyn Error>>
    where
        T: Num
            + num_traits::ToPrimitive
            + num_traits::FromPrimitive
            + std::clone::Clone
            + std::cmp::PartialOrd,
    {
        self.n += 1;

        let n = self.n as f32;
        let x_arr = ndarray::ArrayView2::<T>::from_shape((self.n_chains, self.n_params), x)?
            .mapv(|x| x.to_f32().unwrap());

        self.mean = (self.mean.clone() * (n - 1.0) + x_arr.clone()) / n;
        if self.n == 1 {
            self.mean_sq = x_arr.pow2();
        } else {
            self.mean_sq = (self.mean_sq.clone() * (n - 1.0) + (x_arr.pow2())) / n;
        };

        // Update self.p_accept and last state
        self.p_accept = ndarray::Zip::from(x_arr.rows())
            .and(self.last_state.rows())
            .fold(self.p_accept, |p_accept, a, b| {
                let accepted = (a.ne(&b) as i32) as f32;
                (1.0 - ALPHA) * p_accept + ALPHA * accepted
            });
        self.last_state = x_arr;

        Ok(())
    }

    pub fn stats<B: Backend>(&self, sample: Tensor<B, 3>) -> Result<RunStats, Box<dyn Error>> {
        let sample_data = sample.to_data();
        let sample_ndarray =
            ArrayView3::from_shape(sample.dims(), sample_data.as_slice().unwrap())?;
        let (rhat, ess) = split_rhat_mean_ess(sample_ndarray);
        let ess = basic_stats(ess);
        let rhat = basic_stats(rhat);
        Ok(RunStats { ess, rhat })
    }

    /// Computes the maximum R-hat value across all parameters.
    ///
    /// # Returns
    /// The maximum R-hat value, or an error if computation fails.
    pub fn max_rhat(&self) -> Result<f32, Box<dyn Error>> {
        let all: Array1<f32> = self.rhat()?;
        let max = *all.max()?;
        Ok(max)
    }

    /// Computes the R-hat values for all parameters.
    ///
    /// # Returns
    /// An array containing the R-hat values for each parameter, or an error if computation fails.
    pub fn rhat(&self) -> Result<Array1<f32>, Box<dyn Error>> {
        let (within, var) = self.within_and_var()?;
        let rhat = (var / within).sqrt();
        Ok(rhat)
    }

    fn within_and_var(&self) -> Result<(Array1<f32>, Array1<f32>), Box<dyn Error>> {
        let mean_chain = self
            .mean
            .mean_axis(Axis(0))
            .ok_or("Mean reduction across chains for mean failed.")?;
        let n_chains = self.mean.shape()[0] as f32;
        let n = self.n as f32;
        let fac = n / (n_chains - 1.0);
        let between = (self.mean.clone() - mean_chain.insert_axis(Axis(0)))
            .pow2()
            .sum_axis(Axis(0))
            * fac;
        let sm2 = (self.mean_sq.clone() - self.mean.pow2()) * n / (n - 1.0);
        let within = sm2
            .mean_axis(Axis(0))
            .ok_or("Mean reduction across chains for mean of squares failed.")?;
        let var = within.clone() * ((n - 1.0) / n) + between * (1.0 / n);
        Ok((within, var))
    }
}

fn basic_stats(data: Array1<f32>) -> BasicStats {
    let mut data = data.clone();
    data.as_slice_mut()
        .unwrap()
        .sort_by(|a, b| match b.partial_cmp(a) {
            Some(x) => x,
            None => Ordering::Equal,
        });
    let (min, median, max) = (
        *data
            .last()
            .expect("Expected getting first element from ess array succeed"),
        data[data.len() / 2],
        *data
            .first()
            .expect("Expected getting last element from ess array succeed"),
    );
    let mean = data.mean().expect("Expected computing mean ess to succeed");
    let std = data.std(1.0);
    BasicStats {
        min,
        median,
        max,
        mean,
        std,
    }
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct RunStats {
    pub ess: BasicStats,
    pub rhat: BasicStats,
}

#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct BasicStats {
    pub min: f32,
    pub median: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

/// Takes a (chains, samples, parameters) view and returns a new
/// (2*chains, samples/2, parameters) array by splitting each chain in half.
fn splitcat(sample: ArrayView3<f32>) -> Array3<f32> {
    let n = sample.shape()[1];
    let half = (n / 2) as i32;
    let half_1 = sample.slice(s![.., ..half, ..]);
    let half_2 = sample.slice(s![.., -half.., ..]);
    concatenate(Axis(0), &[half_1, half_2]).expect("Expected stacking two halves to succeed")
}

pub fn withinvar(sample: ArrayView3<f32>) -> (Array1<f32>, Array1<f32>) {
    let c = sample.shape()[0];
    let n = sample.shape()[1];
    let p = sample.shape()[2];

    // 2) For each parameter, compute the chain-split stats
    //    shape for data_p is (2c, n/2)
    let (within, var): (Vec<f32>, Vec<f32>) = (0..p)
        .into_par_iter()
        .map(|param_idx| {
            let data_p = sample.slice(s![.., .., param_idx]);

            // chain means => shape (2c,)
            let chain_means = data_p.mean_axis(Axis(1)).unwrap();
            let overall_mean = chain_means.mean().unwrap();

            // B => between chain var
            //   = (n/2 / (2c - 1)) * sum( (chain_means - overall_mean)^2 )
            // (We assume 2c > 1)
            let diff = &chain_means - overall_mean;
            let b = diff.pow2().sum() * ((n as f32) / ((c - 1) as f32));
            dbg!(param_idx, &chain_means, &overall_mean, &diff, b);

            // within => we broadcast chain_means to shape (2c, n/2) to subtract
            // but an easier approach might be:
            // squares[i] = mean over that chain's (x_i - chain_mean[i])^2
            // then average squares across all chains
            let mut squares = Vec::with_capacity(c);
            for chain_i in 0..c {
                let row = data_p.slice(s![chain_i, ..]);
                let cm = chain_means[chain_i];
                let sq = row.iter().map(|v| (v - cm) * (v - cm)).sum::<f32>() / (n as f32);
                squares.push(sq);
            }
            let squares = Array1::from(squares); // shape (2c,)
            let w = squares.mean().unwrap(); // within = average across chains
                                             // var => ((n/2 - 1)/(n/2)) * w + b/(n/2)
            let v = ((n as f32 - 1.0) / (n as f32)) * w + b / (n as f32);

            (w, v)
        })
        .collect::<Vec<(f32, f32)>>()
        .into_iter()
        .fold((vec![], vec![]), |(mut within, mut var), (w, v)| {
            within.push(w);
            var.push(v);
            (within, var)
        });
    (Array1::from_vec(within), Array1::from_vec(var))
}

/// Computes ESS after splitting each chain into two halves.
pub fn split_rhat_mean_ess(sample: ArrayView3<f32>) -> (Array1<f32>, Array1<f32>) {
    let splitted = splitcat(sample); // shape: (2c, n/2, p)
    let (within, var) = withinvar(splitted.view());
    dbg!(&within, &var);
    (
        rhat(within.view(), var.view()),
        ess(splitted.view(), within.view(), var.view()),
    )
}

fn rhat(within: ArrayView1<f32>, var: ArrayView1<f32>) -> Array1<f32> {
    (within.to_owned() / var).sqrt()
}

/// Computes the Effective Sample Size (ESS) from chain statistics.
///
/// # Arguments
/// - `sample`: 3D array of samples with shape (chains, samples, parameters).
/// - `chain_stats`: Slice of references to `ChainStats` from multiple chains.
///
/// # Returns
/// An array containing the ESS for each parameter.
pub fn ess_from_chainstats(sample: ArrayView3<f32>, chain_stats: &[&ChainStats]) -> Array1<f32> {
    let (within, var) = withinvar_from_cs(chain_stats);
    ess(sample, within.view(), var.view())
}

fn ess(sample: ArrayView3<f32>, within: ArrayView1<f32>, var: ArrayView1<f32>) -> Array1<f32> {
    let shape = sample.shape();
    let (n_chains, n_steps, n_params) = (shape[0], shape[1], shape[2]);
    let chain_rho: Vec<Array2<f32>> = (0..n_chains)
        .map(|c| {
            let chain_samples = sample.index_axis(Axis(0), c);
            autocov(chain_samples)
        })
        .collect();
    let chain_rho: Vec<ArrayView2<f32>> = chain_rho.iter().map(|x| x.view()).collect();
    let chain_rho = stack(Axis(0), &chain_rho)
        .expect("Expected stacking chain-specific autocovariance matrices to succeed");
    let avg_rho = chain_rho.mean_axis(Axis(0)).unwrap();
    let diff = -avg_rho
        + within
            .broadcast((n_steps, n_params))
            .expect("Expected broadcasting to succeed");
    let rho = -(diff
        / var
            .broadcast((n_steps, n_params))
            .expect("Expected broadcasting to succeed"))
        + 1.0;
    let tau: Vec<f32> = (0..n_params)
        .into_par_iter()
        .map(|d| {
            let rho_d = rho.index_axis(Axis(1), d).to_owned();

            let mut min = if rho_d.len() >= 2 {
                rho_d[[0]] + rho_d[[1]]
            } else {
                0.0
            };

            let mut out = 0.0;
            for rho_t in rho_d.windows_with_stride(2, 2) {
                let mut p_t = rho_t[0] + rho_t[1];
                if p_t <= 0.0 {
                    break;
                }
                if p_t > min {
                    p_t = min;
                }
                min = p_t;
                out += p_t;
            }
            -1.0 + 2.0 * out
        })
        .collect();
    let tau = Array1::from_vec(tau);
    tau.recip() * n_chains as f32 * n_steps as f32
}

fn autocov(sample: ArrayView2<f32>) -> Array2<f32> {
    if sample.nrows() <= 100 {
        autocov_bf(sample)
    } else {
        autocov_fft(sample)
    }
}

/// Compute the autocovariance of multiple sequences (each column represents a distinct sequence)
/// using FFT for efficient calculation.
///
/// # Arguments
///
/// * `sample` - A 2-dimensional array view (`ArrayView2<f32>`) of shape `(n, d)`, where:
///     - `n`: length of each sequence.
///     - `d`: number of sequences (each column is treated independently).
///
/// # Returns
///
/// An `Array2<f32>` of shape `(n, d)` containing the autocovariance results.
/// Each column contains the autocovariance values for the corresponding input sequence.
///
/// # Notes
///
/// * Uses zero-padding to avoid circular convolution effects (wrap-around).
/// * FFT and inverse FFT are performed using the `rustfft` crate.
/// * Computation is parallelized across sequences using Rayon.
/// * Normalization (`1/n_padded`) is applied explicitly, as `rustfft` does not normalize results.
fn autocov_fft(sample: ArrayView2<f32>) -> Array2<f32> {
    let (n, d) = (sample.shape()[0], sample.shape()[1]);
    let mut planner = FftPlanner::new();

    // Next power of 2 >= 2*n - 1 for zero-padding to avoid wrap-around.
    let mut n_padded = 1;
    while n_padded < 2 * n - 1 {
        n_padded <<= 1;
    }
    let fft = planner.plan_fft_forward(n_padded);
    let ffti = planner.plan_fft_inverse(n_padded);
    let out: Vec<f32> = sample
        .axis_iter(Axis(1))
        .into_par_iter()
        .map(|traj| {
            let traj_mean = traj.sum() / traj.len() as f32;
            let mut x: Vec<Complex<f32>> = traj
                .iter()
                .map(|xi| Complex {
                    re: (*xi - traj_mean),
                    im: 0.0f32,
                })
                .chain(
                    [Complex {
                        re: 0.0f32,
                        im: 0.0f32,
                    }]
                    .repeat(n_padded - n),
                )
                .collect();
            fft.process(x.as_mut_slice());
            x.iter_mut().for_each(|xi| {
                *xi *= xi.conj();
            });
            ffti.process(x.as_mut_slice());
            x.iter_mut()
                .take(n)
                .map(|xi| xi.re / n_padded as f32 / n as f32) // rustfft doens't normalize for us
                .collect::<Vec<f32>>()
        })
        .flatten_iter()
        .collect();
    let out = Array2::from_shape_vec((d, n), out).expect("Expected creating dxn array to succeed");
    out.t().to_owned()
}

/// Brute force autocovariance on a 2D array of shape (n, d).
/// - `n` = number of time points (rows)
/// - `d` = number of parameters (columns)
///
/// For each column `col` and each lag `lag` (0..n), the function
/// computes:
/// $$
///    sum_{t=0..(n - lag - 1)} [ data[t, col] * data[t + lag, col] ]
/// $$
/// and stores it in `out[lag, col]`.
fn autocov_bf(data: ArrayView2<f32>) -> Array2<f32> {
    let (n, d) = data.dim();
    let mut out = Array2::<f32>::zeros((n, d));

    out.axis_iter_mut(Axis(1)) // mutable view of each column in `out`
        .into_par_iter() // make it parallel
        .enumerate() // get (col_index, col_view_mut)
        .for_each(|(col_idx, mut out_col)| {
            let col_data = data.column(col_idx);
            let col_data = col_data.to_owned() - col_data.mean().unwrap();

            // For each lag, compute sum_{t=0..(n-lag-1)} [ data[t, col] * data[t + lag, col] ]
            for lag in 0..n {
                let mut sum_lag = 0.0;
                for t in 0..(n - lag) {
                    sum_lag += col_data[t] * col_data[t + lag];
                }
                // Write result into the current column
                out_col[lag] = sum_lag / n as f32;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::{f32, fs::File, time::Instant};

    use approx::assert_abs_diff_eq;
    use rand::Rng;

    use super::*;

    // Generic helper function to run the Rhat test.
    fn run_rhat_test_generic<T>(data0: Array2<T>, data1: Array2<T>, expected: Array1<f32>, tol: f32)
    where
        T: ndarray::NdFloat + num_traits::FromPrimitive,
    {
        let mut psr = MultiChainTracker::new(3, 4);
        psr.step(data0.as_slice().unwrap()).unwrap();
        psr.step(data1.as_slice().unwrap()).unwrap();
        let rhat = psr.rhat().unwrap();
        let diff = *(rhat.clone() - expected.clone()).abs().max().unwrap();
        assert!(
            diff < tol,
            "Mismatch in Rhat. Got {:?}, expected {:?}, diff = {:?}",
            rhat,
            expected,
            diff
        );
    }

    #[test]
    fn test_rhat_f32_1() {
        // Step 0 data (chains x params)
        let data_step_0: Array2<f32> = arr2(&[
            [0.0, 1.0, 0.0, 1.0], // chain 0
            [1.0, 2.0, 0.0, 2.0], // chain 1
            [0.0, 0.0, 0.0, 2.0], // chain 2
        ]);

        // Step 1 data (chains x params)
        let data_step_1: Array2<f32> = arr2(&[
            [1.0, 2.0, 2.0, 0.0], // chain 0
            [1.0, 1.0, 1.0, 1.0], // chain 1
            [0.0, 1.0, 0.0, 0.0], // chain 2
        ]);
        let expected = array![f32::consts::SQRT_2, 1.080_123_4, 0.894_427_3, 0.8660254];
        run_rhat_test_generic(data_step_0, data_step_1, expected, f32::EPSILON * 10.0);
    }

    #[test]
    fn test_rhat_f64_1() {
        let data_step_0: Array2<f64> = arr2(&[
            [0.0, 1.0, 0.0, 1.0], // chain 0
            [1.0, 2.0, 0.0, 2.0], // chain 1
            [0.0, 0.0, 0.0, 2.0], // chain 2
        ]);
        let data_step_1: Array2<f64> = arr2(&[
            [1.0, 2.0, 2.0, 0.0], // chain 0
            [1.0, 1.0, 1.0, 1.0], // chain 1
            [0.0, 1.0, 0.0, 0.0], // chain 2
        ]);
        let expected = array![f32::consts::SQRT_2, 1.0801234, 0.8944271, 0.8660254];
        run_rhat_test_generic(data_step_0, data_step_1, expected, f32::EPSILON * 10.0);
    }

    #[test]
    fn test_rhat_f64_2() {
        let data_step_0 = arr2(&[
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 2.0],
        ]);
        let data_step_1 = arr2(&[
            [1.0, 2.0, 0.0, 2.0],
            [1.0, 2.0, 0.0, 0.0],
            [2.0, 0.0, 1.0, 2.0],
        ]);
        let expected = array![f32::consts::FRAC_1_SQRT_2, 0.74535599, 1.0, 1.5];
        run_rhat_test_generic(data_step_0, data_step_1, expected, f32::EPSILON * 10.0);
    }

    fn run_test_case(
        autocov_func: &dyn Fn(ArrayView2<f32>) -> Array2<f32>,
        data: &Array2<f32>,
        expected: &Array2<f32>,
        test_name: &str,
    ) {
        let result = autocov_func(data.view());
        assert_eq!(
            result.dim(),
            expected.dim(),
            "{}: shape mismatch; got {:?}, expected {:?}",
            test_name,
            result.dim(),
            expected.dim()
        );

        assert_abs_diff_eq!(result, *expected, epsilon = 1e-6);
        println!("Test: {test_name} succeeded");
    }

    // ----------------------------------------------------------
    // Test: single parameter, small integer sequence
    // ----------------------------------------------------------
    #[test]
    fn test_single_param() {
        let data = array![[1.0], [2.0], [3.0], [4.0],];
        let expected = array![[1.0], [0.25], [-0.3], [-0.45],];

        // Compare brute force
        run_test_case(&autocov_bf, &data, &expected, "BF: single_param_small");

        println!("Doing FFT test");
        // Compare FFT-based
        run_test_case(&autocov_fft, &data, &expected, "FFT: single_param_small");
        println!("FFT test succeeded");
    }

    // ----------------------------------------------------------
    // Test: two parameters, 4 time points
    // ----------------------------------------------------------
    #[test]
    fn test_two_params_1() {
        let data = array![[1.0, 0.3], [2.0, 2.0], [3.0, -2.0], [4.0, 5.0],];
        let expected = array![
            [1.0, 1.0],
            [0.25, -0.58139925],
            [-0.3, 0.2259039],
            [-0.45, -0.14450465],
        ];

        // Compare brute force
        run_test_case(&autocov_bf, &data, &expected, "BF: two_params_small");
        // Compare FFT-based
        run_test_case(&autocov_fft, &data, &expected, "FFT: two_params_small");
    }

    // ----------------------------------------------------------
    // Test: two datasets, 4 time points each (nested arrays)
    // ----------------------------------------------------------
    #[test]
    fn test_two_params_2() {
        let data = array![[1.0, 3.0], [-1.0, 2.0], [1.0, 1.0], [-1.0, 0.0]];
        let expected = array![[1.0, 1.0], [-0.75, 0.25], [0.5, -0.3], [-0.25, -0.45],];

        // Compare brute force
        run_test_case(&autocov_bf, &data, &expected, "BF: two_datasets_small");

        // Compare FFT-based
        run_test_case(&autocov_fft, &data, &expected, "FFT: two_datasets_small");
    }

    #[test]
    fn ess_1() {
        let m = 4;
        let n = 1000;

        // Initialize an empty 4 x 100 array
        let mut data = Array2::<f32>::zeros((m, n));

        // Use the built-in RNG to generate each row separately
        let mut rng = rand::thread_rng();
        for mut row in data.rows_mut() {
            for elem in row.iter_mut() {
                *elem = rng.gen::<f32>(); // generates uniform random number between 0 and 1
            }
        }
        // let chain_means = data.mean_axis(Axis(1)).unwrap();
        // let overall_mean = data.mean().unwrap();
        // let b = (chain_means.clone() - overall_mean).pow2().sum() * (n as f32 / (m - 1) as f32);
        // let big_chain_means_t = chain_means.broadcast((n, m)).unwrap();
        // let squares = (data.clone() - big_chain_means_t.t())
        //     .pow2()
        //     .mean_axis(Axis(1))
        //     .unwrap();
        // let within = squares.mean().unwrap();
        // let var = ((n as f32 - 1.0) / (n as f32)) * within + b / (n as f32);
        // let rhat = (within / var).sqrt();
        //
        // let within = array![within];
        // let var = array![var];
        let data = data
            .to_shape((data.shape()[0], data.shape()[1], 1))
            .unwrap();
        let (rhat, ess) = split_rhat_mean_ess(data.view());
        // let ess = ess(data.view(), within, var);
        println!("Samples: {}", m * n);
        println!("ESS: {ess}");
        println!("Rhat: {rhat}");
    }

    #[test]
    #[ignore = "Benchmark test: run only when explicitly requested"]
    fn test_autocov_perf_comp() {
        // Create output CSV
        let mut file =
            File::create("runtime_results.csv").expect("Unable to create runtime_results.csv");
        // Write header row
        writeln!(file, "length,rep,time,algorithm").expect("Unable to write CSV header");

        let mut rng = rand::thread_rng();

        for exp in 0..10 {
            let n = 1 << exp; // 2^exp
            for rep in 1..=10 {
                // Generate random data of size (n x 1)
                let sample_data: Vec<f32> = (0..n * 1000).map(|_| rng.gen()).collect();
                let sample = Array2::from_shape_vec((n, 1000), sample_data)
                    .expect("Failed to create Array2");

                // Measure FFT-based implementation
                let start_fft = Instant::now();
                autocov_fft(sample.view());
                let fft_time = start_fft.elapsed().as_nanos();

                // Measure brute-force implementation
                let start_brute = Instant::now();
                autocov_bf(sample.view());
                let brute_time = start_brute.elapsed().as_nanos();

                // Log results to CSV
                writeln!(file, "{},{},{},fft", n, rep, fft_time)
                    .expect("Unable to write test results to CSV");
                writeln!(file, "{},{},{},brute force", n, rep, brute_time)
                    .expect("Unable to write test results to CSV");

                // Print results for convenience
                println!(
                    "Length: {} | Rep: {} | FFT: {} ns | Brute: {} ns",
                    n, rep, fft_time, brute_time
                );
            }
        }
    }
}
