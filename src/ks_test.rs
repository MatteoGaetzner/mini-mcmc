/*!
A minimal two-sample Kolmogorov–Smirnov test implementation.

This module provides functionality to perform a two-sample KS test,
adapted from the [`kolmogorov_smirnov`](https://crates.io/crates/kolmogorov_smirnov)
crate under the Apache 2.0 License. The KS test compares two samples to determine
whether they come from the same distribution.

# Overview

The public API consists of:

- The [`TotalF64`] struct, a wrapper around `f64` that provides a total order (even when NaN values occur).
- The [`two_sample_ks_test`] function, which returns a [`TestResult`] containing the test statistic,
  p-value, and a boolean flag indicating if the null hypothesis is rejected at a given significance level.

The internal functions such as `compute_ks_statistic`, `ks_p_value`, `pks`, and `qks` perform the
necessary computations based on algorithms found in *Numerical Recipes (Third Edition)*.
*/
use std::cmp::Ordering;

use rand_distr::num_traits::ToPrimitive;
// use std::cmp::Ordering;

/**
A wrapper around `f64` that implements a total ordering.

This type is used for sorting and comparing floating-point values in a way that
treats NaN values as equal (and places them after all finite numbers).

# Examples

```rust
use mini_mcmc::ks_test::TotalF64;

let mut values = [TotalF64(3.0), TotalF64(f64::NAN), TotalF64(1.0)];
values.sort();
assert_eq!(values[0].0, 1.0);
assert_eq!(values[1].0, 3.0);
assert!(values[2].0.is_nan());
```
*/
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TotalF64(pub f64);

impl Eq for TotalF64 {}

impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &TotalF64) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/**
Performs a two-sample Kolmogorov–Smirnov test on two samples at the given significance level.

The function computes the maximum difference between the empirical distribution functions
of the two samples and then estimates a p-value. If the p-value is less than the provided
significance level, the null hypothesis (that the two samples are drawn from the same distribution)
is rejected.

# Type Parameters

* `T`: A type that implements `Ord`, `Clone`, and `Copy`. In practice, you may wrap your
  floating-point numbers with [`TotalF64`] to ensure a total ordering.

# Arguments

* `sample_1` - A slice containing the first sample.
* `sample_2` - A slice containing the second sample.
* `level` - The significance level at which to test the null hypothesis (e.g. 0.05).

# Returns

Returns a [`TestResult`] with the KS test statistic, p-value, and a flag indicating if the null
hypothesis is rejected.

# Errors

Returns an error `String` if either sample is empty or if the p-value cannot be computed.

# Examples

```rust
use mini_mcmc::ks_test::{two_sample_ks_test, TotalF64};

// Two identical samples should yield a KS statistic of 0 and a p-value of 1.
let sample_1: Vec<TotalF64> = (0..10).map(|x| TotalF64(x as f64)).collect();
let result = two_sample_ks_test(&sample_1, &sample_1, 0.05).unwrap();
assert_eq!(result.statistic, 0.0);
assert!((result.p_value - 1.0).abs() < 1e-10);

// For different samples, the statistic will be > 0 and the p-value will be less than 1.
let sample_2: Vec<TotalF64> = sample_1.iter().map(|x| TotalF64(x.0 * x.0)).collect();
let result_diff = two_sample_ks_test(&sample_1, &sample_2, 0.05).unwrap();
assert!(result_diff.statistic > 0.0);
assert!(result_diff.p_value < 1.0);
```
*/
pub fn two_sample_ks_test<T: Ord + Clone + Copy>(
    sample_1: &[T],
    sample_2: &[T],
    level: f64,
) -> Result<TestResult, String> {
    let statistic = compute_ks_statistic(sample_1, sample_2)?;
    let p_value = ks_p_value(statistic, sample_1.len(), sample_2.len())?;
    Ok(TestResult {
        is_rejected: p_value < level,
        statistic,
        p_value,
        level,
    })
}

/**
The result of a two-sample Kolmogorov–Smirnov test.

Contains the test statistic, the computed p-value, the significance level used for testing,
and a boolean flag `is_rejected` indicating whether the null hypothesis (that the two samples
come from the same distribution) is rejected.
*/
#[derive(Debug)]
pub struct TestResult {
    pub is_rejected: bool,
    pub statistic: f64,
    pub p_value: f64,
    pub level: f64,
}

/**
Computes the Kolmogorov–Smirnov p-value for the two-sample case.

This function uses an approximation based on the effective sample size and
the KS test statistic. It asserts that both samples have sizes greater than 7 for accuracy.

# Arguments

* `statistic` - The KS test statistic.
* `n1` - The size of the first sample.
* `n2` - The size of the second sample.

# Returns

Returns the p-value as an `f64` if successful.

*/
pub fn ks_p_value(statistic: f64, n1: usize, n2: usize) -> Result<f64, String> {
    if n1 <= 7 || n2 <= 7 {
        return Err(("Requires sample sizes > 7 for accuracy.").to_string());
    }

    let factor = ((n1 as f64 * n2 as f64) / (n1 as f64 + n2 as f64)).sqrt();
    let term = factor * statistic;

    // We call `qks` to get the complementary CDF of the KS distribution.
    let p_value = qks(term)?;
    assert!((0.0..=1.0).contains(&p_value));

    Ok(p_value)
}

/**
Computes the two-sample KS statistic as the maximum absolute difference between the
empirical distribution functions of the two samples.

The input samples are first sorted (in ascending order) before computing the statistic.

# Arguments

* `sample_1` - The first sample.
* `sample_2` - The second sample.

# Returns

Returns the KS statistic as an `f64` if both samples are non-empty.

# Errors

Returns an error if either sample is empty.
*/
pub fn compute_ks_statistic<T: Ord + Clone + Copy>(
    sample_1: &[T],
    sample_2: &[T],
) -> Result<f64, String> {
    if sample_1.is_empty() {
        return Err("Expected sample_1 to be non-empty.".into());
    }
    if sample_2.is_empty() {
        return Err("Expected sample_2 to be non-empty.".into());
    }

    // let (mut _sample_1, mut _sample_2) = (sample_1.clone(), sample_2.clone());
    let mut _sample_1 = sample_1.to_vec();
    let mut _sample_2 = sample_2.to_vec();

    _sample_1.sort_unstable();
    _sample_2.sort_unstable();

    let (n, m) = (_sample_1.len(), _sample_2.len());
    let (n_i32, m_i32) = (n as i32, m as i32);
    let (n_f64, m_f64) = (n as f64, m as f64);

    let (mut i, mut j) = (-1_i32, -1_i32);
    let mut max_diff: f64 = 0.0;
    let mut cur_x: T = _sample_1[0].min(_sample_2[0]);

    while i + 1 < n_i32 || j + 1 < m_i32 {
        advance(&mut i, n_i32, &_sample_1, &cur_x);
        advance(&mut j, m_i32, &_sample_2, &cur_x);

        let fi = if i < 0 { 0.0 } else { (i + 1) as f64 / n_f64 };
        let fj = if j < 0 { 0.0 } else { (j + 1) as f64 / m_f64 };

        max_diff = max_diff.max((fj - fi).abs());

        let ip = (i + 1).to_usize().unwrap();
        let jp = (j + 1).to_usize().unwrap();
        if ip < n && jp < m {
            cur_x = _sample_1[ip].min(_sample_2[jp]);
        } else {
            break;
        }
    }
    Ok(max_diff)
}

/**
Advances the index `i` while the next value in `sample` is less than or equal to `cur_x`.

This helper function is used in the computation of the KS statistic.

# Arguments

* `i` - A mutable reference to the current index.
* `n` - The total number of elements in the sample (as `i32`).
* `sample` - The sorted sample slice.
* `cur_x` - The current threshold value.

# Example

(This function is internal; see [`compute_ks_statistic`] for its usage.)
*/
fn advance<T: Ord + Clone>(i: &mut i32, n: i32, sample: &[T], cur_x: &T) {
    while *i + 1 < n {
        let next_val = &sample[(*i + 1) as usize];
        if *next_val <= *cur_x {
            *i += 1;
        } else {
            break;
        }
    }
}

/**
Computes the one-sided cumulative distribution function (CDF) of the KS distribution.

This function uses an algorithm adapted from *Numerical Recipes (Third Edition)*.

# Arguments

* `z` - The argument of the CDF (must be non-negative).

# Returns

Returns the CDF value for the KS distribution.

# Errors

Returns an error if `z` is negative.

# Examples

```rust
// For z = 0, the CDF should be 0.
let cdf = mini_mcmc::ks_test::pks(0.0).unwrap();
assert_eq!(cdf, 0.0);
```
*/
pub fn pks(z: f64) -> Result<f64, String> {
    if z < 0. {
        return Err("Bad z for KS distribution function.".into());
    }
    if z == 0. {
        return Ok(0.);
    }
    if z < 1.18 {
        let y = (-1.233_700_550_136_169_7 / z.powi(2)).exp();
        return Ok(2.256_758_334_191_025
            * (-y.ln()).sqrt()
            * (y + y.powf(9.) + y.powf(25.) + y.powf(49.)));
    }
    let x = (-2. * z.powi(2)).exp();
    Ok(1. - 2. * (x - x.powf(4.) + x.powf(9.)))
}

/**
Computes the complementary CDF (Q-function) of the KS distribution.

This function is also adapted from *Numerical Recipes (Third Edition)*.

# Arguments

* `z` - The argument of the Q-function (must be non-negative).

# Returns

Returns the complementary probability for the KS distribution.

# Errors

Returns an error if `z` is negative.

# Examples

```rust
// For z = 0, the Q-function should return 1.
let q = mini_mcmc::ks_test::qks(0.0).unwrap();
assert_eq!(q, 1.0);
```
*/
pub fn qks(z: f64) -> Result<f64, String> {
    if z < 0. {
        return Err("Bad z for KS distribution function.".into());
    }
    if z == 0. {
        return Ok(1.);
    }
    if z < 1.18 {
        return Ok(1. - pks(z)?);
    }
    let x = (-2. * z.powi(2)).exp();
    Ok(2. * (x - x.powf(4.) + x.powf(9.)))
}

#[cfg(test)]
mod tests {
    use super::*;

    use rand::{rngs::SmallRng, Rng, SeedableRng};

    #[test]
    fn test_ks_p_value_too_few() {
        let res = ks_p_value(1., 1, 1);
        assert!(res.is_err(), "Expected to get an Err object");
    }

    #[test]
    fn test_ks_p_value_ok() {
        let res = ks_p_value(1., 8, 8);
        assert!(res.is_ok(), "Expected to get a Ok object");
    }

    #[test]
    fn test_ks_simple_case() {
        // Three-element samples with partial overlap; we expect D ~ 1/3.
        let s1 = [1.0, 2.0, 3.0].map(TotalF64);
        let s2 = [2.0, 3.0, 4.0].map(TotalF64);
        let d = compute_ks_statistic(&s1, &s2).unwrap();
        assert!((d - 1.0 / 3.0).abs() < 1e-9, "Expected D ~ 1/3, got {}", d);
    }

    #[test]
    fn test_ks_identical_samples() {
        // Identical => D=0.
        let s1 = [1.0, 2.0, 3.0].map(TotalF64);
        let s2 = [1.0, 2.0, 3.0].map(TotalF64);
        let d = compute_ks_statistic(&s1, &s2).unwrap();
        assert_eq!(d, 0.0, "KS should be 0 for identical samples.");
    }

    #[test]
    fn test_ks_non_overlapping() {
        // Disjoint => D=1.
        let s1 = [1.0, 2.0, 3.0].map(TotalF64);
        let s2 = [10.0, 11.0, 12.0].map(TotalF64);
        let d = compute_ks_statistic(&s1, &s2).unwrap();
        assert_eq!(d, 1.0, "Non-overlapping samples => D=1.");
    }

    #[test]
    fn test_ks_single_element() {
        // s1=[2], s2=[5] => D=1.
        let s1 = [TotalF64(2.0)];
        let s2 = [TotalF64(5.0)];
        let d = compute_ks_statistic(&s1, &s2).unwrap();
        assert_eq!(d, 1.0);
    }

    #[test]
    fn test_ks_repeated_values() {
        // Tie-handling with repeated values; expect around 0.2 from R-like logic.
        let s1 = [1.0, 1.0, 1.0, 2.0, 2.0].map(TotalF64);
        let s2 = [1.0, 1.0, 2.0, 2.0, 2.0].map(TotalF64);
        let d = compute_ks_statistic(&s1, &s2).unwrap();
        assert!((d - 0.2).abs() < 1e-6, "Expected ~0.2, got {}", d);
    }

    #[test]
    fn test_ks_partial_overlap() {
        // Overlapping but not identical => D=0.25.
        let s1 = [0.0, 1.0, 2.0, 3.0].map(TotalF64);
        let s2 = [1.0, 2.0, 3.0, 4.0].map(TotalF64);
        let d = compute_ks_statistic(&s1, &s2).unwrap();
        assert!((d - 0.25).abs() < 1e-9, "Expected 0.25, got {}", d);
    }

    #[test]
    fn test_ks_rep_similar() {
        // Repeated pattern, slight difference => check statistic & p-value.
        let s1: Vec<TotalF64> = [0.12, 0.25, 0.25, 0.78, 0.99, 0.33, 0.15, 0.5]
            .iter()
            .cycle()
            .take(8 * 20)
            .copied()
            .map(TotalF64)
            .collect();
        let s2: Vec<TotalF64> = [0.12, 0.25, 0.25, 0.78, 0.99, 0.33, 0.15, 0.51]
            .iter()
            .cycle()
            .take(8 * 20)
            .copied()
            .map(TotalF64)
            .collect();

        let result = two_sample_ks_test(&s1, &s2, 0.05).unwrap();
        assert!((result.statistic - 0.125).abs() < 1e-9, "D mismatch");
        assert!((result.p_value - 0.1641).abs() < 1e-4, "p-value mismatch");
    }

    #[test]
    fn test_ks_empty_1() {
        let s1 = [];
        let s2 = [1.0, 2.0, 3.0, 4.0].map(TotalF64);
        let res = compute_ks_statistic(&s1, &s2);
        assert!(res.is_err(), "Expected compute_ks_statistic(...) to return an error since the first list is empty, got {:?}.", res);
    }

    #[test]
    fn test_ks_empty_2() {
        let s1 = [1.0, 2.0, 3.0, 4.0].map(TotalF64);
        let s2 = [];
        let res = compute_ks_statistic(&s1, &s2);
        assert!(res.is_err(), "Expected compute_ks_statistic(...) to return an error since the second list is empty, got {:?}.", res);
    }

    #[test]
    fn test_bad_z_for_pks() {
        let res = pks(-1.0);
        assert!(
            res.is_err(),
            "Expected pks(-1.0) to return an error, got {:?}.",
            res
        );
    }

    #[test]
    fn test_pks_zero() {
        match pks(0.0) {
            Err(msg) => panic!("Expected pks(0.0) == 0, got error message {:?}.", msg),
            Ok(val) => assert!(val == 0.0, "Expected pks(0.0) == 0, got {:?}.", val),
        }
    }

    #[test]
    fn test_pks_large_1() {
        match pks(1.23) {
            Err(msg) => panic!(
                "Expected pks(1.23), to not error out, got error message {:?}.",
                msg
            ),
            Ok(val) => assert!(
                (val - 0.9029731024047791).abs() < 1e-8,
                "Expected pks(1.23) ~= 0.9029731024047791, got {:?}.",
                val
            ),
        }
    }

    #[test]
    fn test_pks_large_2() {
        match pks(2.34) {
            Err(msg) => panic!(
                "Expected pks(2.34), to not error out, got error message {:?}.",
                msg
            ),
            Ok(val) => assert!(
                (val - 0.9999649260833611).abs() < 1e-8,
                "Expected pks(2.34) ~= 0.9999649260833611, got {:?}.",
                val
            ),
        }
    }

    #[test]
    fn test_pks_large_3() {
        match pks(3.45) {
            Err(msg) => panic!(
                "Expected pks(3.45), to not error out, got error message {:?}.",
                msg
            ),
            Ok(val) => assert!(
                (val - 1.0).abs() < 1e-8,
                "Expected pks(3.45) ~= 1.0, got {:?}.",
                val
            ),
        }
    }

    #[test]
    fn test_qks_zero() {
        match qks(0.0) {
            Err(msg) => panic!(
                "Expected qks(0.0), to not error out, got error message {:?}.",
                msg
            ),
            Ok(val) => assert!(val == 1.0, "Expected qks(0.0) = 0.0, got {:?}.", val),
        }
    }

    #[test]
    fn test_qks_large() {
        match qks(1.2) {
            Err(msg) => panic!(
                "Expected qks(1.2), to not error out, got error message {:?}.",
                msg
            ),
            Ok(val) => assert!(
                (val - 0.11224966667072497).abs() < 1e-8,
                "Expected qks(1.2) ~= 00.11224966667072497, got {:?}.",
                val
            ),
        }
    }

    #[test]
    fn test_bad_z_for_qks() {
        let res = qks(-1.0);
        assert!(
            res.is_err(),
            "Expected qks(-1.0) to return an error, got {:?}.",
            res
        );
    }

    #[test]
    fn test_cmp_f64_middle_nan() {
        let mut s = [1.0, f64::NAN, 3.0];
        s.sort_by(|a, b| a.total_cmp(b));
        assert!(
            s[0] == 1.0 && s[1] == 3.0 && s[2].is_nan(),
            "Expected sorting [1.0, NAN, 3.0] to give [1.0, 3.0, NAN], got {s:?}."
        );
    }
    #[test]
    fn test_cmp_f64_beginning_nan() {
        let mut s = [f64::NAN, 2.0, 3.0].map(TotalF64);
        s.sort();
        assert!(
            s[0].0 == 2.0 && s[1].0 == 3.0 && s[2].0.is_nan(),
            "Expected sorting [NAN, 2.0, 3.0] to give [2.0, 3.0, NAN], got {s:?}."
        );
    }

    #[test]
    fn test_cmp_f64_end_nan() {
        let mut s = [1.0, 2.0, f64::NAN].map(TotalF64);
        s.sort();
        assert!(
            s[0].0 == 1.0 && s[1].0 == 2.0 && s[2].0.is_nan(),
            "Expected sorting [NAN, 2.0, 3.0] to give [2.0, 3.0, NAN], got {s:?}."
        );
    }

    #[test]
    fn test_cmp_f64_double_nana() {
        let mut s = [f64::NAN, 2.0, f64::NAN].map(TotalF64);
        s.sort();
        assert!(
            s[0].0 == 2.0 && s[1].0.is_nan() && s[2].0.is_nan(),
            "Expected sorting [NAN, 2.0, NAN] to give [2.0, NAN, NAN], got {s:?}."
        );
    }

    #[test]
    fn test_cmp_f64_all_nana() {
        let mut s = [f64::NAN, f64::NAN, f64::NAN].map(TotalF64);
        s.sort();
        assert!(
            s[0].0.is_nan() && s[1].0.is_nan() && s[2].0.is_nan(),
            "Expected sorting [NAN, NAN, NAN] to give [NAN, NAN, NAN], got {s:?}."
        );
    }

    #[test]
    fn test_same_as_external() {
        let mut rng = SmallRng::seed_from_u64(42);

        let s1: Vec<TotalF64> = (0..100000).map(|_| rng.gen()).map(TotalF64).collect();
        let s2: Vec<TotalF64> = (0..100000).map(|_| rng.gen()).map(TotalF64).collect();
        let res_external = kolmogorov_smirnov::test(&s1, &s2, 0.95);
        let res_internal = two_sample_ks_test(&s1, &s2, 0.05).expect("Expected KS test to succeed");
        println!(
            "EXTERNAL:\n  statistic={:?}\n  is_rejected={:?}\n  reject_probability={:?}",
            res_external.statistic, res_external.is_rejected, res_external.reject_probability
        );
        println!(
            "INTERNAL:\n  statistic={:?}\n  is_rejected={:?}\n  reject_probability={:?}",
            res_internal.statistic,
            res_internal.is_rejected,
            1.0 - res_internal.p_value
        );
        println!("{res_internal:?}");
    }
}
