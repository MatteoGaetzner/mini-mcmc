//! A small MCMC demo using Gibbs sampling to sample from a 2D mixture distribution.
//! The target is a two-component Gaussian mixture (over a state [x, z]).

use mini_mcmc::core::ChainRunner;
use mini_mcmc::distributions::Conditional;
use mini_mcmc::gibbs::GibbsSampler;
use ndarray::Axis;
use plotters::chart::ChartBuilder;
use plotters::prelude::{BitMapBackend, Circle, IntoDrawingArea};
use plotters::style::Color;
use plotters::style::{RGBAColor, BLACK, RED, WHITE};
use rand::{thread_rng, Rng};
use std::error::Error;

// Define a conditional distribution for a two-component Gaussian mixture.
// The state is [x, z] where x ∈ ℝ and z ∈ {0.0, 1.0} is a latent indicator.
// When z == 0, x ~ N(mu0, sigma0²); when z == 1, x ~ N(mu1, sigma1²).
// The joint distribution is defined by p(x, z = 0) = π0 * N(x; mu0, sigma0²),
// p(x, z = 1) = π1 * N(x; mu1, sigma1²),
// When updating z, we compute:
//    p(z=0|x) ∝ π0 * N(x; mu0, sigma0²)
//    p(z=1|x) ∝ (1-π0) * N(x; mu1, sigma1²)
#[derive(Clone)]
struct MixtureConditional {
    mu0: f64,
    sigma0: f64,
    mu1: f64,
    sigma1: f64,
    pi0: f64,
}

impl MixtureConditional {
    fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
        let var = sigma * sigma;
        let coeff = 1.0 / ((2.0 * std::f64::consts::PI * var).sqrt());
        let exp_val = (-((x - mu).powi(2)) / (2.0 * var)).exp();
        coeff * exp_val
    }
}

impl Conditional<f64> for MixtureConditional {
    fn sample(&mut self, i: usize, given: &[f64]) -> f64 {
        // Our state is [x, z].
        if i == 0 {
            // Sample x conditionally on z.
            let z = given[1];
            if z < 0.5 {
                // Mode 0: x ~ N(mu0, sigma0²)
                let normal = rand_distr::Normal::new(self.mu0, self.sigma0).unwrap();
                rand::thread_rng().sample(normal)
            } else {
                // Mode 1: x ~ N(mu1, sigma1²)
                let normal = rand_distr::Normal::new(self.mu1, self.sigma1).unwrap();
                rand::thread_rng().sample(normal)
            }
        } else if i == 1 {
            // Sample z conditionally on x.
            let x = given[0];
            let p0 = self.pi0 * MixtureConditional::normal_pdf(x, self.mu0, self.sigma0);
            let p1 = (1.0 - self.pi0) * MixtureConditional::normal_pdf(x, self.mu1, self.sigma1);
            let total = p0 + p1;
            let prob_z1 = if total > 0.0 { p1 / total } else { 0.5 };
            if rand::thread_rng().gen::<f64>() < prob_z1 {
                1.0
            } else {
                0.0
            }
        } else {
            panic!("Invalid coordinate index in MixtureConditional");
        }
    }
}

/// Main entry point: sets up a two-component Gaussian mixture target,
/// runs Gibbs sampling, computes summary statistics, and plots the samples.
fn main() -> Result<(), Box<dyn Error>> {
    // Mixture parameters.
    let mu0 = -2.0;
    let sigma0 = 1.0;
    let mu1 = 3.0;
    let sigma1 = 1.5;
    let pi0 = 0.25;

    // Create the conditional distribution.
    let conditional = MixtureConditional {
        mu0,
        sigma0,
        mu1,
        sigma1,
        pi0,
    };

    // Our state is [x, z]. We start with x = 0.0 and z = 0.0.
    let initial_state = [0.0, 0.0];

    // Set up the Gibbs sampler.
    const N_CHAINS: usize = 4;
    const BURNIN: usize = 1000;
    const TOTAL_STEPS: usize = 1100;
    let seed: u64 = thread_rng().gen();

    let mut sampler = GibbsSampler::new(conditional, &initial_state, N_CHAINS).set_seed(seed);

    // Generate samples.
    let samples = sampler.run(TOTAL_STEPS, BURNIN).unwrap();
    let pooled = samples.to_shape((((TOTAL_STEPS - BURNIN) * 4), 2)).unwrap();
    println!("Generated {} samples", pooled.len());

    // Compute basic statistics.
    let row_mean = pooled.mean_axis(Axis(0)).unwrap();
    println!(
        "Mean after burn-in: ({:.2}, {:.2})",
        row_mean[0], row_mean[1]
    );

    // Compute quantiles for plotting ranges.
    let mut x_coords: Vec<f64> = pooled.column(0).iter().copied().collect();
    let mut y_coords: Vec<f64> = pooled.column(1).iter().copied().collect();
    x_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    y_coords.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (0.005 * pooled.nrows() as f64) as usize;
    let upper_idx = (0.995 * pooled.nrows() as f64) as usize;
    let x_range = x_coords[lower_idx]..x_coords[upper_idx];
    let y_range =
        (y_coords.first().unwrap().to_owned() - 0.1)..(y_coords.last().unwrap().to_owned() + 0.1);

    // Filter samples within the plotting range.
    let filtered: Vec<_> = pooled
        .axis_iter(Axis(0))
        .filter(|point| x_range.contains(&point[0]) && y_range.contains(&point[1]))
        .collect();
    println!("Filtered samples: {}", filtered.len());

    // Draw the scatter plot.
    let root = BitMapBackend::new("gibbs_scatter_plot.png", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Gibbs Sampling from a 2D Mixture", ("sans-serif", 50))
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
            10,
            RGBAColor(70, 130, 180, 0.25).filled(),
        )
    }))?;

    chart
        .draw_series(std::iter::once(Circle::new(
            (row_mean[0], row_mean[1]),
            15,
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

    println!("Saved scatter plot to gibbs_scatter_plot.png");

    // Optionally, save samples to file (if you have an IO module).
    let _ = mini_mcmc::io::save_parquet(&samples, "gibbs_samples.parquet");
    println!("Saved samples to gibbs_samples.parquet.");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_main() {
        main().expect("Main should execute without error");
        assert!(
            Path::new("gibbs_scatter_plot.png").exists(),
            "Expected gibbs_scatter_plot.png to exist"
        );
        assert!(
            Path::new("gibbs_samples.parquet").exists(),
            "Expected gibbs_samples.parquet to exist"
        );
    }
}
