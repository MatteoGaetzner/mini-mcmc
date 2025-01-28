mod mini_mcmc;
use mini_mcmc::{Gaussian2D, GaussianProposal, MetropolisHastings};

use plotters::prelude::*;
use plotters::style::RGBAColor; // For colors with alpha
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    let target = Gaussian2D {
        mean: [0.0, 0.0].into(),
        cov: [[2.0, 1.0], [1.0, 2.0]].into(),
    };
    let proposal = GaussianProposal::new(1.0);
    let initial_state = vec![10.0, 12.0];

    let mut mh = MetropolisHastings::new(target, proposal, initial_state);

    let iterations = 10000;
    let mut samples = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let new_state = mh.step();
        samples.push(new_state);
    }

    println!("Generated {} samples", samples.len());

    let burn_in = (0.2 * (iterations as f64)) as usize;
    let post_burnin_samples = &samples[burn_in..];

    // --- Compute basic statistics ---
    let mean_x: f64 =
        post_burnin_samples.iter().map(|p| p[0]).sum::<f64>() / post_burnin_samples.len() as f64;
    let mean_y: f64 =
        post_burnin_samples.iter().map(|p| p[1]).sum::<f64>() / post_burnin_samples.len() as f64;

    println!("Mean after burn-in: ({:.2}, {:.2})", mean_x, mean_y);

    // --- Compute quantiles for automatic range determination ---
    let mut x_coords: Vec<f64> = post_burnin_samples.iter().map(|point| point[0]).collect();
    let mut y_coords: Vec<f64> = post_burnin_samples.iter().map(|point| point[1]).collect();

    x_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_coords.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let lower_idx = (0.005 * post_burnin_samples.len() as f64) as usize;
    let upper_idx = (0.995 * post_burnin_samples.len() as f64) as usize;

    let x_range = x_coords[lower_idx]..x_coords[upper_idx];
    let y_range = y_coords[lower_idx]..y_coords[upper_idx];

    println!("x_range: {:?}", x_range);
    println!("y_range: {:?}", y_range);

    // --- Filter points within the range ---
    let filtered_samples: Vec<_> = post_burnin_samples
        .iter()
        .filter(|point| {
            let x = point[0];
            let y = point[1];
            x_range.contains(&x) && y_range.contains(&y)
        })
        .collect();

    println!("Filtered samples: {}", filtered_samples.len());

    // --- Plot the 2D samples ---
    // 1) Create a drawing area using BitMapBackend with higher resolution
    let root = BitMapBackend::new("samples.png", (1200, 900)).into_drawing_area();
    root.fill(&WHITE)?;

    // 2) Build a chart with the computed ranges
    let mut chart = ChartBuilder::on(&root)
        .caption("MCMC Samples from 2D Gaussian", ("sans-serif", 50))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(x_range.clone(), y_range.clone())?;

    // 3) Configure the mesh with additional grid lines
    chart
        .configure_mesh()
        .x_labels(10)
        .y_labels(10)
        .light_line_style(&WHITE.mix(0.8))
        .bold_line_style(&BLACK.mix(0.5))
        .draw()?;

    // 4) Plot each sample as a semi-transparent steelblue dot
    chart.draw_series(filtered_samples.iter().map(|point| {
        let x = point[0];
        let y = point[1];
        Circle::new((x, y), 2, RGBAColor(70, 130, 180, 0.5).filled())
    }))?;

    // 5) Highlight the mean with a distinct red circle
    chart
        .draw_series(std::iter::once(Circle::new(
            (mean_x, mean_y),
            6,
            RED.filled(),
        )))?
        .label("Mean")
        .legend(|(x, y)| Circle::new((x, y), 6, RED.filled()));

    // 6) Add a legend to the chart
    chart
        .configure_series_labels()
        .border_style(&BLACK)
        .background_style(&WHITE.mix(0.9))
        .label_font(("sans-serif", 35))
        .draw()?;

    println!("Saved scatter plot to samples.png");
    Ok(())
}
