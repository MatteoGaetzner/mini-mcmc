//! A small MCMC demo using Metropolis-Hastings to sample from a 2D Rosenbrock distribution,
//! then plotting the samples.

use mini_mcmc::core::ChainRunner;
use mini_mcmc::distributions::{IsotropicGaussian, Proposal, Target};
use mini_mcmc::metropolis_hastings::MetropolisHastings;

// Optionally, save samples to file (if you have an IO module).
// use mini_mcmc::io::save_parquet;

use ndarray::Axis;
use num_traits::Float;
use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea};
use plotters::style::{Color, RGBAColor, BLACK, RED, WHITE};
use std::error::Error;

/// The **Rosenbrock** distribution is a classic example with a narrow, curved valley.
/// Its unnormalized log-density is defined as:
///
/// \[ \log \pi(x,y) \propto -\Big[(a - x)^2 + b\,(y - x^2)^2\Big] \]
///
/// where we typically set `a = 1` and `b = 100`.
#[derive(Clone, Copy)]
pub struct Rosenbrock<T: Float> {
    pub a: T,
    pub b: T,
}

impl<T> Target<T, T> for Rosenbrock<T>
where
    T: Float,
{
    fn unnorm_log_prob(&self, theta: &[T]) -> T {
        let x = theta[0];
        let y = theta[1];
        let term1 = self.a - x;
        let term2 = y - x * x;
        -(term1 * term1 + self.b * term2 * term2)
    }
}

/// Main entry point: sets up a 2D Rosenbrock target, runs Metropolis-Hastings,
/// computes summary statistics, and generates a scatter plot of the samples.
fn main() -> Result<(), Box<dyn Error>> {
    const SAMPLE_SIZE: usize = 100_000;
    const BURNIN: usize = 10_000;
    const N_CHAINS: usize = 8;
    let seed: u64 = 42;

    // Define the Rosenbrock target distribution with parameters a=1, b=100.
    let target = Rosenbrock { a: 1.0, b: 100.0 };

    // Use an isotropic Gaussian as the proposal distribution.
    // The standard deviation is chosen to be small given the narrow valley of the target.
    let proposal = IsotropicGaussian::new(1.0).set_seed(seed);

    // Starting from an initial state (here, not at the mode).
    let initial_state = [0.0, 0.0];

    let mut mh = MetropolisHastings::new(target, proposal, &initial_state, N_CHAINS).set_seed(seed);

    // Generate samples
    let samples = mh
        .run_progress(SAMPLE_SIZE / N_CHAINS, BURNIN)
        .expect("Expected generating samples to succeed");
    let pooled = samples
        .to_shape((SAMPLE_SIZE, 2))
        .expect("Expected reshaping to succeed");

    println!("Generated {:?} samples", pooled.shape()[0]);

    // Basic statistics
    let row_mean = pooled.mean_axis(Axis(0)).unwrap();
    println!(
        "Mean after burn-in: ({:.2}, {:.2})",
        row_mean[0], row_mean[1]
    );

    // Compute quantiles for plotting ranges
    let mut x_coords: Vec<f64> = pooled.column(0).iter().copied().collect();
    let mut y_coords: Vec<f64> = pooled.column(1).iter().copied().collect();
    x_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    y_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let x_range = x_coords.first().unwrap().to_owned()..x_coords.last().unwrap().to_owned();
    let y_range = y_coords.first().unwrap().to_owned()..y_coords.last().unwrap().to_owned();

    // Draw the scatter plot
    let root = BitMapBackend::new("rosenbrock_scatter_plot.png", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("MCMC Samples from 2D Rosenbrock", ("sans-serif", 50))
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

    chart.draw_series(pooled.axis_iter(Axis(0)).map(|point| {
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

    println!("Saved scatter plot to rosenbrock_scatter_plot.png");

    // Optionally, save samples to file (if you have an IO module).
    // let _ = save_parquet(&samples, "rosenbrock_samples.parquet");
    // println!("Saved samples in file rosenbrock_samples.parquet.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_main() {
        main().expect("Expected main to not return an error.");
        assert!(
            std::path::Path::new("rosenbrock_scatter_plot.png").exists(),
            "Expected rosenbrock_scatter_plot.png to exist."
        );
        assert!(
            std::path::Path::new("rosenbrock_samples.parquet").exists(),
            "Expected rosenbrock_samples.parquet to exist."
        );
    }
}
