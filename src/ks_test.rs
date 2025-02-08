//! A minimal two-sample Kolmogorov–Smirnov test, adapted from the `kolmogorov_smirnov` crate
//! under the Apache 2.0 License (see references in doc comments).

use rand_distr::num_traits::ToPrimitive;

use std::cmp::Ordering;

#[derive(Debug, Copy, Clone)]
pub struct TotalF64(pub f64);

impl PartialEq for TotalF64 {
    fn eq(&self, other: &TotalF64) -> bool {
        if self.0.is_nan() && other.0.is_nan() {
            true
        } else {
            self.0 == other.0
        }
    }
}

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

/// Performs a two-sample KS test at the given significance level. Returns a `TestResult`
/// indicating whether the null hypothesis (same distribution) is rejected.
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

/// Stores the result of a two-sample KS test, indicating whether the null was rejected,
/// along with the test statistic, p-value, and the chosen level.
///
/// Based on `TestResult` from [kolmogorov_smirnov].
/// Modifications: Removed critical value field, renamed attributes.
#[derive(Debug)]
pub struct TestResult {
    pub is_rejected: bool,
    pub statistic: f64,
    pub p_value: f64,
    pub level: f64,
}

/// Computes the Kolmogorov–Smirnov p-value for the two-sample case.
/// If below `level`, the null hypothesis can be rejected.
fn ks_p_value(statistic: f64, n1: usize, n2: usize) -> Result<f64, String> {
    assert!(n1 > 7 && n2 > 7, "Requires sample sizes > 7 for accuracy.");
    let factor = ((n1 as f64 * n2 as f64) / (n1 as f64 + n2 as f64)).sqrt();
    let term = factor * statistic;

    // We call `qks` to get the complementary CDF of the KS distribution.
    let p_value = qks(term)?;
    assert!((0.0..=1.0).contains(&p_value));
    Ok(p_value)
}

/// Computes the two-sample KS statistic. Sorts both slices and computes the maximum
/// difference between their empirical distribution functions.
fn compute_ks_statistic<T: Ord + Clone + Copy>(
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

/// Advances the index `i` while the next value in `sample` is <= `cur_x`.
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

/// CDF of the Kolmogorov–Smirnov distribution (one-sided).
/// Uses the algorithm from *Numerical Recipes* (Third Edition).
fn pks(z: f64) -> Result<f64, String> {
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

/// Q-function (complementary CDF) of the Kolmogorov–Smirnov distribution.
/// Also from *Numerical Recipes*.
fn qks(z: f64) -> Result<f64, String> {
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
