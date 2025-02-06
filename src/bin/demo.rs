//! A small MCMC demo using Metropolis-Hastings to sample from a 2D Gaussian, then plotting the samples.

/// Contains Gaussian-related distributions and traits for sampling/log-prob evaluation.
// mod distributions;
use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian, ProposalDistribution};

/// Provides the MetropolisHastings struct for running MCMC.
// mod metrohast;
use mini_mcmc::metrohast::MetropolisHastings;

use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea};
use plotters::style::{Color, RGBAColor, BLACK, RED, WHITE};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::error::Error;

/// Main entry point: sets up a 2D Gaussian target, runs Metropolis-Hastings,
/// computes summary statistics, and generates a scatter plot of the samples.
fn main() -> Result<(), Box<dyn Error>> {
    const ITERATIONS: usize = 100_000;
    const BURNIN: usize = 10000;
    const N_CHAINS: usize = 8;
    const SEED: u64 = 42;

    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[2.0, 1.0], [1.0, 2.0]].into(),
    };
    let proposal = IsotropicGaussian::new(1.0).set_seed(SEED);
    let initial_state = vec![10.0, 12.0];

    let mut mh = MetropolisHastings::new(target, proposal, initial_state, N_CHAINS).set_seed(SEED);

    // Generate samples
    let mut samples = mh.run(BURNIN + ITERATIONS / N_CHAINS, BURNIN).concat();
    samples.shuffle(&mut SmallRng::from_entropy());
    println!("Generated {} samples", samples.len());
    println!("Last sample: {}", samples[2][samples[2].len() - 1]);

    // Basic statistics
    let mean_x = samples.iter().map(|p| p[0]).sum::<f64>() / samples.len() as f64;
    let mean_y = samples.iter().map(|p| p[1]).sum::<f64>() / samples.len() as f64;
    println!("Mean after burn-in: ({:.2}, {:.2})", mean_x, mean_y);

    // Compute quantiles for plotting ranges
    let mut x_coords: Vec<f64> = samples.iter().map(|p| p[0]).collect();
    let mut y_coords: Vec<f64> = samples.iter().map(|p| p[1]).collect();
    x_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    y_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (0.005 * samples.len() as f64) as usize;
    let upper_idx = (0.995 * samples.len() as f64) as usize;
    let x_range = x_coords[lower_idx]..x_coords[upper_idx];
    let y_range = y_coords[lower_idx]..y_coords[upper_idx];
    println!("x_range: {:?}", x_range);
    println!("y_range: {:?}", y_range);

    // Filter samples within the plotting range
    let filtered_samples: Vec<_> = samples
        .iter()
        .filter(|&point| x_range.contains(&point[0]) && y_range.contains(&point[1]))
        .collect();
    println!("Filtered samples: {}", filtered_samples.len());

    // Draw the scatter plot
    let root = BitMapBackend::new("samples.png", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("MCMC Samples from 2D Gaussian", ("sans-serif", 50))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .light_line_style(WHITE.mix(0.8))
        .bold_line_style(BLACK.mix(0.5))
        .draw()?;

    chart.draw_series(filtered_samples.iter().map(|&point| {
        Circle::new(
            (point[0], point[1]),
            2,
            RGBAColor(70, 130, 180, 0.5).filled(),
        )
    }))?;

    chart
        .draw_series(std::iter::once(Circle::new(
            (mean_x, mean_y),
            6,
            RED.filled(),
        )))?
        .label("Mean")
        .legend(|(x, y)| Circle::new((x, y), 6, RED.filled()));

    chart
        .configure_series_labels()
        .border_style(BLACK)
        .background_style(WHITE.mix(0.9))
        .label_font(("sans-serif", 35))
        .draw()?;

    println!("Saved scatter plot to samples.png");
    Ok(())
}

#[test]
fn test_main() {
    main().expect("Expected main to not return an error.");
    assert!(
        std::path::Path::new("samples.png").exists(),
        "Expected samples.png to exist."
    );
}
