//! A small MCMC demo using Metropolis-Hastings to sample from a 2D Gaussian, then plotting the samples.

use mini_mcmc::distributions::{Gaussian2D, IsotropicGaussian, ProposalDistribution};
use mini_mcmc::io::save_parquet;
use mini_mcmc::metrohast::MetropolisHastings;
use nalgebra as na;

use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea};
use plotters::style::{Color, RGBAColor, BLACK, RED, WHITE};
use rand::{thread_rng, Rng};
use std::error::Error;

/// Main entry point: sets up a 2D Gaussian target, runs Metropolis-Hastings,
/// computes summary statistics, and generates a scatter plot of the samples.
fn main() -> Result<(), Box<dyn Error>> {
    const SAMPLE_SIZE: usize = 100_000;
    const BURNIN: usize = 10_000;
    const N_CHAINS: usize = 8;
    let seed: u64 = thread_rng().gen();

    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[2.0, 1.0], [1.0, 2.0]].into(),
    };
    let proposal = IsotropicGaussian::new(2.0).set_seed(seed);
    let initial_state = [10.0, 12.0];

    let mut mh = MetropolisHastings::new(target, proposal, &initial_state, N_CHAINS).set_seed(seed);

    // Generate samples
    let samples = mh.run_with_progress(BURNIN + SAMPLE_SIZE / N_CHAINS, BURNIN);
    let mut pooled = na::DMatrix::<f64>::zeros(SAMPLE_SIZE, 2);
    samples.iter().enumerate().for_each(|(i, x)| {
        pooled
            .rows_mut(i * x.nrows(), x.nrows())
            .copy_from_slice(x.as_slice())
    });

    println!("Generated {} samples", pooled.len());

    // Basic statistics
    let row_mean = pooled.row_mean();
    println!(
        "Mean after burn-in: ({:.2}, {:.2})",
        row_mean[0], row_mean[1]
    );

    // Compute quantiles for plotting ranges
    let mut x_coords: Vec<f64> = Vec::from_iter(pooled.column(0).iter().copied());
    let mut y_coords: Vec<f64> = Vec::from_iter(pooled.column(1).iter().copied());
    x_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    y_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (0.005 * pooled.nrows() as f64) as usize;
    let upper_idx = (0.995 * pooled.nrows() as f64) as usize;
    let x_range = x_coords[lower_idx]..x_coords[upper_idx];
    let y_range = y_coords[lower_idx]..y_coords[upper_idx];

    // Filter samples within the plotting range
    let filtered: Vec<_> = pooled
        .row_iter()
        .filter(|point| x_range.contains(&point[0]) && y_range.contains(&point[1]))
        .collect();
    println!("Filtered samples: {}", filtered.len());

    // Draw the scatter plot
    let root = BitMapBackend::new("scatter_plot.png", (1200, 900)).into_drawing_area();
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

    chart.draw_series(filtered.iter().map(|&point| {
        Circle::new(
            (point[0], point[1]),
            2,
            RGBAColor(70, 130, 180, 0.5).filled(),
        )
    }))?;

    chart
        .draw_series(std::iter::once(Circle::new(
            (row_mean[0], row_mean[1]),
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

    println!("Saved scatter plot to scatter_plot.png");

    let _ = save_parquet(&samples, "samples.parquet");
    println!("Saved sampels in file samples.parquet.");

    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_main() {
        main().expect("Expected main to not return an error.");
        assert!(
            std::path::Path::new("scatter_plot.png").exists(),
            "Expected scatter_plot.png to exist."
        );
        assert!(
            std::path::Path::new("samples.parquet").exists(),
            "Expected sample.parquet to exist."
        );
    }
}
