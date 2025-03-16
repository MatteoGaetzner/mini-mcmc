/*!
# I/O Utilities for Saving MCMC Data to CSV

This module provides functions to save MCMC sample data to CSV files. Enable via the `csv` feature.
*/

use burn::prelude::*;
use ndarray::{Array3, Axis};
use std::error::Error;
use std::fs::File;

use csv::Writer;

/**
Saves MCMC sample data as a CSV file.

The data is expected to be in a shape of **chain × sample × dimension**.

The resulting CSV file will have:
- A header row containing `"chain"`, `"sample"`, and one column per dimension
  named `"dim_0"`, `"dim_1"`, etc.
- Each subsequent row will correspond to a single sample of a specific chain.

# Arguments

* `data` - An Array3<T> object containing the MCMC data.
* * `filename` - The file path where the CSV data will be written.

# Returns

Returns `Ok(())` if successful, or an error if any I/O or CSV formatting
issue occurs.

# Examples

```rust
use mini_mcmc::io::csv::save_csv;
use ndarray::arr3;

// This matrix has 2 rows (samples) and 4 columns (dimensions).
let data = arr3(&[[[1, 2, 3, 4], [5, 6, 7, 8]]]);

save_csv(&data, "/tmp/output.csv").expect("Expecting saving data to succeed");
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_csv<T: std::fmt::Display>(
    data: &Array3<T>,
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_writer(File::create(filename)?);
    let n_dims = data.shape()[2];

    let mut header: Vec<String> = vec!["chain".to_string(), "sample".to_string()];
    header.extend((0..n_dims).map(|i| format!("dim_{}", i)));
    wtr.write_record(&header)?;

    // Flatten and write data
    for (chain_idx, chain) in data.axis_iter(Axis(0)).enumerate() {
        for (sample_idx, sample) in chain.axis_iter(Axis(0)).enumerate() {
            let mut row = vec![chain_idx.to_string(), sample_idx.to_string()];
            row.extend(sample.iter().map(|v| v.to_string()));
            wtr.write_record(&row)?;
        }
    }

    wtr.flush()?;
    Ok(())
}

/**
Saves a 3D Burn tensor (sample × chain × dimension) as a CSV file.

The CSV file will contain a header row with columns:
  - `"sample"`: the sample index,
  - `"chain"`: the chain index,
  - `"dim_0"`, `"dim_1"`, … for each dimension.

Each subsequent row corresponds to one data point from the tensor, with its sample and chain indices
followed by the dimension values. For example, the coordinate `d` of data point `s` that chain `c`
generated is assumed to be in `tensor[n][s][d]`.

# Arguments
* `tensor` - A reference to a Burn tensor with shape `[num_samples, num_chains, num_dimensions]`.
* `filename` - The file path where the CSV data will be written.

# Type Parameters
* `B` - The backend type.
* `K` - The tensor kind.
* `T` - The scalar type; must implement [`burn::tensor::Element`]
# Returns
Returns `Ok(())` if successful or an error if any I/O or CSV formatting issue occurs.

# Example
```rust
use burn::tensor::Tensor;
use burn::backend::ndarray::{NdArray, NdArrayDevice};
use mini_mcmc::io::csv::save_csv_tensor;
let tensor = Tensor::<NdArray, 3>::from_floats(
    [
        [[1.0, 2.0], [1.1, 2.1], [1.2, 2.2]],
        [[1.01, 2.01], [1.11, 2.11], [1.21, 2.21]],
        [[1.02, 2.02], [1.12, 2.12], [1.22, 2.22]],
        [[1.03, 2.03], [1.13, 2.13], [1.23, 2.23]],
    ],
    &NdArrayDevice::Cpu,
);
save_csv_tensor::<NdArray, _, f32>(&tensor, "/tmp/output.csv")?;
# Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_csv_tensor<B, K, T>(
    tensor: &burn::tensor::Tensor<B, 3, K>,
    filename: &str,
) -> Result<(), Box<dyn Error>>
where
    B: Backend,
    K: burn::tensor::TensorKind<B>,
    T: burn::tensor::Element,
    K: burn::tensor::BasicOps<B>,
{
    use csv::Writer;
    use std::fs::File;
    // Extract data as TensorData and convert to a flat Vec<T>
    let shape = tensor.dims(); // expected to be [num_samples, num_chains, num_dimensions]
    let data = tensor.to_data();
    let (num_samples, num_chains, num_dims) = (shape[0], shape[1], shape[2]);
    let flat = data
        .to_vec::<T>()
        .map_err(|e| format!("Converting data to Vec failed.\nData: {data:?}.\nError: {e:?}"))?;

    let mut wtr = Writer::from_writer(File::create(filename)?);

    // Build header: "sample", "chain", "dim_0", "dim_1", ...
    let mut header = vec!["sample".to_string(), "chain".to_string()];
    header.extend((0..num_dims).map(|i| format!("dim_{}", i)));
    wtr.write_record(&header)?;

    // Iterate over sample and chain indices; each row corresponds to one data point.
    for sample in 0..num_samples {
        for chain in 0..num_chains {
            let mut row = vec![sample.to_string(), chain.to_string()];
            let offset = sample * num_chains * num_dims + chain * num_dims;
            let row_slice = &flat[offset..offset + num_dims];
            row.extend(row_slice.iter().map(|v| v.to_string()));
            wtr.write_record(&row)?;
        }
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    use csv::Reader;
    use ndarray::arr3;
    use std::fs;
    use tempfile::NamedTempFile;

    // --- CSV Tests ---

    /// Test saving empty data to CSV (zero chains).
    #[test]
    fn test_save_csv_empty_data() {
        let data = arr3::<f32, 0, 0>(&[]);
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_csv(&data, filename);
        assert!(
            result.is_ok(),
            "Saving empty data to CSV failed: {:?}",
            result
        );

        // Verify that the CSV file is created and only has a header row (or is empty).
        let contents = fs::read_to_string(filename).unwrap();
        // The function writes a header even if there's no data.
        // The header should be "chain,sample" only, because num_dimensions=0
        assert_eq!(contents.trim(), "chain,sample");
    }

    /// Test saving a single chain with a single sample (and single dimension) to CSV.
    #[test]
    fn test_save_csv_single_chain_single_sample() {
        let data = arr3(&[[[42.0]]]); // chain=0, sample=0, dim_0=42
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_csv(&data, filename);
        assert!(
            result.is_ok(),
            "Saving single chain/single sample to CSV failed: {:?}",
            result
        );

        let contents = fs::read_to_string(filename).unwrap();
        let expected = "chain,sample,dim_0\n0,0,42";
        assert_eq!(contents.trim(), expected);
    }

    /// Test multiple chains, multiple samples, multiple dimensions to CSV.
    #[test]
    fn test_save_csv_multi_chain() {
        // data[chain][sample][dim]
        let data = arr3(&[[[1, 2], [3, 4]], [[10, 20], [30, 40]]]);
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_csv(&data, filename);
        assert!(result.is_ok());

        let contents = fs::read_to_string(filename).unwrap();
        let expected = "\
chain,sample,dim_0,dim_1
0,0,1,2
0,1,3,4
1,0,10,20
1,1,30,40";
        assert_eq!(contents.trim(), expected);
    }

    #[test]
    fn test_save_csv_tensor_data() -> Result<(), Box<dyn std::error::Error>> {
        // Create a tensor with shape [2, 2, 2]: 2 samples, 2 chains, 2 dimensions.
        let tensor = Tensor::<NdArray, 3>::from_floats(
            [[[1.0, 2.0], [3.0, 4.0]], [[1.1, 2.1], [3.1, 4.1]]],
            &NdArrayDevice::Cpu,
        );
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();
        save_csv_tensor::<NdArray, _, f32>(&tensor, filename)?;
        let contents = fs::read_to_string(filename)?;

        // Use csv::Reader to parse the CSV file.
        let mut rdr = Reader::from_reader(contents.as_bytes());
        let headers = rdr.headers()?;
        assert_eq!(&headers[0], "sample");
        assert_eq!(&headers[1], "chain");
        assert_eq!(&headers[2], "dim_0");
        assert_eq!(&headers[3], "dim_1");

        let records: Vec<_> = rdr.records().collect::<Result<_, _>>()?;
        // There should be 2 samples * 2 chains = 4 records.
        assert_eq!(records.len(), 4);

        // Expected ordering: For each sample, for each chain.
        // Row 0: sample 0, chain 0, dims: [1.0, 2.0]
        // Row 1: sample 0, chain 1, dims: [3.0, 4.0]
        // Row 2: sample 1, chain 0, dims: [1.1, 2.1]
        // Row 3: sample 1, chain 1, dims: [3.1, 4.1]
        let expected = [
            vec!["0", "0", "1", "2"],
            vec!["0", "1", "3", "4"],
            vec!["1", "0", "1.1", "2.1"],
            vec!["1", "1", "3.1", "4.1"],
        ];
        for (record, exp) in records.iter().zip(expected.iter()) {
            for (field, &exp_field) in record.iter().zip(exp.iter()) {
                // Allow small differences in formatting for floating-point numbers.
                assert!(
                    field.contains(exp_field),
                    "Expected field '{}' to contain '{}'",
                    field,
                    exp_field
                );
            }
        }
        Ok(())
    }
}
