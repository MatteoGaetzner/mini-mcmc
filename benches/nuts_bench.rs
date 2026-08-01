//! Wall-clock benchmarks for `NUTS` and `ManualNUTS`, migrated from `#[ignore]`d tests
//! in `src/nuts.rs`. Run with `cargo bench --bench nuts_bench`. These measure timing
//! only; correctness is covered by the crate's regular (non-ignored) test suite.

use burn::backend::{Autodiff, NdArray};
use burn::prelude::Tensor;
use criterion::{criterion_group, criterion_main, Criterion};
use mini_mcmc::core::{init, init_det, ChainRunner};
use mini_mcmc::distributions::Rosenbrock2D;
use mini_mcmc::nuts::{ManualNUTS, NUTS};
use std::hint::black_box;

type BackendType = Autodiff<NdArray>;

/// Ported from `nuts::tests::test_bench_noprogress_1`.
fn bench_noprogress_1(c: &mut Criterion) {
    let mut group = c.benchmark_group("nuts_noprogress");
    group.sample_size(10);
    group.bench_function("6chains_5000collect", |b| {
        b.iter(|| {
            let target = Rosenbrock2D {
                a: 1.0_f32,
                b: 100.0_f32,
            };
            let initial_positions = init::<f32>(6, 2);
            let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);
            let sample: Tensor<BackendType, 3> = sampler.run(5000, 500);
            assert_eq!(sample.dims(), [6, 5000, 2]);
            black_box(sample);
        });
    });
    group.finish();
}

/// Ported from `nuts::tests::test_bench_noprogress_2`.
fn bench_noprogress_2(c: &mut Criterion) {
    let mut group = c.benchmark_group("nuts_noprogress");
    group.sample_size(10);
    group.bench_function("6chains_1000collect_1000discard", |b| {
        b.iter(|| {
            let target = Rosenbrock2D {
                a: 1.0_f32,
                b: 100.0_f32,
            };
            let initial_positions = init::<f32>(6, 2);
            let mut sampler = NUTS::new(target, initial_positions, 0.95).set_seed(42);
            let sample: Tensor<BackendType, 3> = sampler.run(1000, 1000);
            assert_eq!(sample.dims(), [6, 1000, 2]);
            black_box(sample);
        });
    });
    group.finish();
}

/// Ported from `nuts::tests::manual_nuts_vs_nuts_bench`: compares `NUTS` (burn
/// autodiff) against `ManualNUTS` (analytic gradient) on the same target, as two named
/// benchmarks in the same group for direct comparison.
fn bench_manual_vs_burn(c: &mut Criterion) {
    let n_chains = 6;
    let n_collect = 1000;
    let n_discard = 500;
    let initial_positions: Vec<Vec<f32>> = init_det(n_chains, 2);

    let mut group = c.benchmark_group("nuts_manual_vs_burn");
    group.sample_size(10);

    group.bench_function("burn_autodiff", |b| {
        b.iter(|| {
            let mut sampler = NUTS::new(
                Rosenbrock2D { a: 1.0, b: 100.0 },
                initial_positions.clone(),
                0.8,
            )
            .set_seed(42);
            let sample: Tensor<BackendType, 3> = sampler.run(n_collect, n_discard);
            assert_eq!(sample.dims(), [n_chains, n_collect, 2]);
            black_box(sample);
        });
    });

    group.bench_function("manual_analytic_gradient", |b| {
        b.iter(|| {
            let mut sampler = ManualNUTS::new(
                Rosenbrock2D { a: 1.0, b: 100.0 },
                initial_positions.clone(),
                0.8,
                n_discard,
            )
            .set_seed(42);
            let sample = sampler.run(n_collect, n_discard).unwrap();
            assert_eq!(sample.shape(), [n_chains, n_collect, 2]);
            black_box(sample);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_noprogress_1,
    bench_noprogress_2,
    bench_manual_vs_burn,
);
criterion_main!(benches);
