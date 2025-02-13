use nalgebra as na;
use std::error::Error;
use std::fs::File;
use std::sync::Arc;

#[cfg(feature = "csv")]
use csv::Writer;

#[cfg(feature = "parquet")]
use parquet::{arrow::ArrowWriter, file::properties::WriterProperties};

#[cfg(feature = "arrow")]
use arrow::{
    array::{ArrayRef, Float64Builder, UInt32Builder},
    datatypes::{DataType, Field, Schema},
    ipc::writer::FileWriter,
    record_batch::RecordBatch,
};

#[cfg(feature = "csv")]
/**
Saves MCMC sample data as a CSV file.

The data is expected to be in a shape of **chain × sample × dimension**,
but each chain’s data is stored in an [`nalgebra::DMatrix<T>`] with:

  - **`nrows()` = number of samples**,
  - **`ncols()` = number of dimensions**.

The outer `&[DMatrix<T>]` slice corresponds to the chain index.

The resulting CSV file will have:
- A header row containing `"chain"`, `"sample"`, and one column per dimension
  named `"dim_0"`, `"dim_1"`, etc.
- Each subsequent row will correspond to a single sample of a specific chain.

If `data` is empty or if the first chain is empty, the CSV file will only
contain the header row.

# Arguments

* `data` - A slice of MCMC data, one `DMatrix<T>` per chain. Each matrix must
  have the same number of columns (dimensions) and (optionally) the same
  number of rows (samples). If not, an error is returned.
* `filename` - The file path where the CSV data will be written.

# Returns

Returns `Ok(())` if successful, or an error if any I/O or CSV formatting
issue occurs.

# Examples

```rust
use mini_mcmc::io::save_csv;
use nalgebra as na;
// A single chain (1 DMatrix) with 2 samples and 4 dimensions.
// This matrix has 2 rows (samples) and 4 columns (dimensions).
let data = vec![na::DMatrix::from_row_slice(2, 4, &[1, 2, 3, 4,
                                                   5, 6, 7, 8])];
save_csv(&data, "/tmp/output.csv")?;
Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_csv<T: std::fmt::Display>(
    data: &[na::DMatrix<T>],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    let mut wtr = Writer::from_writer(File::create(filename)?);

    let num_dimensions = if !data.is_empty() && data[0].nrows() > 0 {
        data[0].ncols()
    } else {
        0
    };
    let mut header: Vec<String> = vec!["chain".to_string(), "sample".to_string()];
    header.extend((0..num_dimensions).map(|i| format!("dim_{}", i)));
    wtr.write_record(&header)?;

    // Flatten and write data
    for (chain_idx, chain) in data.iter().enumerate() {
        for (sample_idx, sample) in chain.row_iter().enumerate() {
            let mut row = vec![chain_idx.to_string(), sample_idx.to_string()];
            row.extend(sample.iter().map(|v| v.to_string()));
            wtr.write_record(&row)?;
        }
    }

    wtr.flush()?;
    Ok(())
}

#[cfg(feature = "arrow")]
/**
Saves MCMC data (chain × sample × dimension) as an Apache Arrow (IPC) file.

# Arguments

* `data` - A slice of MCMC data, where each `DMatrix<T>` in `data` represents
  a single chain. In each matrix, **rows** are samples and **columns** are
  dimensions. If you have `k` chains, you'll pass a `Vec<DMatrix<T>>` of
  length `k`.

  - `data[c].nrows()` => number of samples in chain `c`.
  - `data[c].ncols()` => number of dimensions in chain `c`.

  This function requires that each chain has the same shape (same number of
  samples and same number of dimensions). If not, it returns an error.

* `filename` - The path to the Arrow (IPC) file to create.

# Type Parameters

* `T` - Must implement `Into<f64> + Copy`. Each dimension value will be
  converted to `f64` in the Arrow output.

# Returns

Returns `Ok(())` if the file was successfully written. Otherwise, an error
is returned if any I/O or Arrow‐related error occurs.

# Example

```rust
use mini_mcmc::io::save_arrow;
use nalgebra as na;
// Suppose we have 2 chains, each with 2 samples, and 3 dimensions.
// Each DMatrix is 2 rows × 3 columns (samples × dimensions).
let chain0 = na::DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
                                               4.0, 5.0, 6.0]);
let chain1 = na::DMatrix::from_row_slice(2, 3, &[10.0, 20.0, 30.0,
                                               40.0, 50.0, 60.0]);

let data = vec![chain0, chain1];

save_arrow(&data, "/tmp/output.arrow")?;
Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_arrow<T: Into<f64> + Copy>(
    data: &[na::DMatrix<T>],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    // Compute dimensions (but don't return early if empty)
    let n_chains = data.len();
    let n_samples = if n_chains > 0 { data[0].nrows() } else { 0 };
    let n_dims = if n_chains > 0 && n_samples > 0 {
        data[0].ncols()
    } else {
        0
    };

    // Validate that each chain has the same number of samples,
    // and each sample has the same number of dimensions.
    for chain in data {
        if chain.nrows() != n_samples {
            return Err("Inconsistent number of samples among chains".into());
        }
        for sample in chain.row_iter() {
            if sample.len() != n_dims {
                return Err("Inconsistent sample dimensions among chains".into());
            }
        }
    }

    // Define the schema: chain (UInt32), sample (UInt32), dim_0..dim_n (Float64)
    let mut fields = vec![
        Field::new("chain", DataType::UInt32, false),
        Field::new("sample", DataType::UInt32, false),
    ];
    for dim_idx in 0..n_dims {
        fields.push(Field::new(
            format!("dim_{}", dim_idx),
            DataType::Float64,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Create our Arrow builders for chain & sample (UInt32) + each dim (Float64).
    // Even if no data, we need them to create an empty batch.
    let mut chain_builder = UInt32Builder::new();
    let mut sample_builder = UInt32Builder::new();
    let mut dim_builders: Vec<Float64Builder> =
        (0..n_dims).map(|_| Float64Builder::new()).collect();

    // If there's actual data, fill the builders
    if n_chains > 0 {
        for (chain_idx, chain) in data.iter().enumerate() {
            for (sample_idx, sample) in chain.row_iter().enumerate() {
                println!("{chain_idx}, {sample_idx}");
                chain_builder.append_value(chain_idx as u32);
                sample_builder.append_value(sample_idx as u32);

                for (dim_idx, val) in sample.iter().enumerate() {
                    dim_builders[dim_idx].append_value((*val).into());
                }
            }
        }
    }

    // Convert the builders into Arrow arrays
    let chain_array = Arc::new(chain_builder.finish()) as ArrayRef;
    let sample_array = Arc::new(sample_builder.finish()) as ArrayRef;

    let mut dim_arrays = Vec::with_capacity(n_dims);
    for mut builder in dim_builders {
        dim_arrays.push(Arc::new(builder.finish()) as ArrayRef);
    }

    // Combine into a single RecordBatch
    let mut arrays = vec![chain_array, sample_array];
    arrays.extend(dim_arrays);
    let record_batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // Write the RecordBatch (possibly zero rows) to an Arrow IPC file
    let file = File::create(filename)?;
    let mut writer = FileWriter::try_new(file, &schema)?;
    writer.write(&record_batch)?;
    writer.finish()?;

    Ok(())
}

#[cfg(feature = "parquet")]
/**
Saves MCMC data (chain × sample × dimension) to a Parquet file.

# Arguments

* `data` - A slice of MCMC data, where each `DMatrix<T>` in `data` represents
  a single chain. In each matrix, **rows** are samples and **columns** are
  dimensions.

  - `data[c].nrows()` => number of samples in chain `c`.
  - `data[c].ncols()` => number of dimensions in chain `c`.

  All chains must have the same number of rows and columns. If they differ,
  an error is returned.

* `filename` - The path to the Parquet file to create.

# Type Parameters

* `T` - Must implement `Into<f64> + Copy`. Each dimension value is converted
  to `f64` in the underlying Arrow arrays before Parquet encoding.

# Returns

Returns `Ok(())` if the file was written successfully, otherwise returns an
error wrapped in a `Box<dyn Error>`.

# Example

```rust
use mini_mcmc::io::save_parquet;
use nalgebra as na;
// 1 chain, 2 samples, 3 dimensions => a 2×3 DMatrix.
let single_chain = na::DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0,
                                                    4.0, 5.0, 6.0]);
let data = vec![single_chain];

save_parquet(&data, "/tmp/output.parquet")?;
Ok::<(), Box<dyn std::error::Error>>(())
```
*/
pub fn save_parquet<T: Into<f64> + Copy>(
    data: &[na::DMatrix<T>],
    filename: &str,
) -> Result<(), Box<dyn Error>> {
    // Compute dimensions
    let n_chains = data.len();
    let n_samples = if n_chains > 0 { data[0].nrows() } else { 0 };
    let n_dims = if n_chains > 0 && n_samples > 0 {
        data[0].ncols()
    } else {
        0
    };

    // Validate that each chain has the same number of samples
    // and that each sample has the same number of dimensions
    for chain in data {
        if chain.nrows() != n_samples {
            return Err("Inconsistent number of samples among chains".into());
        }
        for sample in chain.row_iter() {
            if sample.len() != n_dims {
                return Err("Inconsistent sample dimensions among chains".into());
            }
        }
    }

    // Define the Arrow schema: chain (UInt32), sample (UInt32), then dim_0..dim_n (Float64)
    let mut fields = vec![
        Field::new("chain", DataType::UInt32, false),
        Field::new("sample", DataType::UInt32, false),
    ];
    for dim_idx in 0..n_dims {
        fields.push(Field::new(
            format!("dim_{}", dim_idx),
            DataType::Float64,
            false,
        ));
    }
    let schema = Arc::new(Schema::new(fields));

    // Create builders for each column
    let mut chain_builder = UInt32Builder::new();
    let mut sample_builder = UInt32Builder::new();
    let mut dim_builders: Vec<Float64Builder> =
        (0..n_dims).map(|_| Float64Builder::new()).collect();

    // Populate builders
    if n_chains > 0 {
        for (chain_idx, chain) in data.iter().enumerate() {
            for (sample_idx, sample) in chain.row_iter().enumerate() {
                chain_builder.append_value(chain_idx as u32);
                sample_builder.append_value(sample_idx as u32);

                for (dim_idx, val) in sample.iter().enumerate() {
                    dim_builders[dim_idx].append_value((*val).into());
                }
            }
        }
    }

    // Convert builders into Arrow arrays
    let chain_array = Arc::new(chain_builder.finish()) as ArrayRef;
    let sample_array = Arc::new(sample_builder.finish()) as ArrayRef;
    let mut dim_arrays = Vec::with_capacity(n_dims);
    for mut builder in dim_builders {
        dim_arrays.push(Arc::new(builder.finish()) as ArrayRef);
    }

    // Create a single RecordBatch
    let mut arrays = vec![chain_array, sample_array];
    arrays.extend(dim_arrays);
    let record_batch = RecordBatch::try_new(schema.clone(), arrays)?;

    // Create the Parquet writer and write the batch
    let file = File::create(filename)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema.clone(), Some(props))?;

    writer.write(&record_batch)?;
    // Close the writer to ensure metadata is written
    writer.close()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{
        array::{Float64Array, UInt32Array},
        ipc::reader::FileReader,
    };
    use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
    use std::fs;
    use std::{error::Error, fs::File};
    use tempfile::NamedTempFile;

    // --- CSV Tests ---

    /// Test saving empty data to CSV (zero chains).
    #[test]
    fn test_save_csv_empty_data() {
        let data: Vec<na::DMatrix<f32>> = vec![]; // no chains
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
        let data = vec![na::dmatrix![42]]; // chain=0, sample=0, dim_0=42
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
        let data = vec![
            na::DMatrix::<i32>::from_row_slice(2, 2, &[1, 2, 3, 4]),
            na::DMatrix::<i32>::from_row_slice(2, 2, &[10, 20, 30, 40]),
        ];
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

    // --- Arrow Tests ---

    /// Test saving empty data to Arrow (zero chains).
    #[test]
    fn test_save_arrow_empty_data() {
        let data: Vec<na::DMatrix<f32>> = vec![]; // no chains
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_arrow(&data, filename);
        assert!(
            result.is_ok(),
            "Saving empty data to Arrow failed: {:?}",
            result
        );

        // The file should exist, but there's effectively no data in it.
        let metadata = fs::metadata(filename).unwrap();
        assert!(metadata.len() > 0, "Arrow file is unexpectedly empty");

        // (Optional) We can verify that the file indeed has an empty RecordBatch.
        let file = File::open(filename).unwrap();
        let mut reader = FileReader::try_new(file, None).unwrap();
        // Should have exactly one batch, with 0 rows
        if let Some(Ok(batch)) = reader.next() {
            dbg!(&batch);
            assert_eq!(batch.num_rows(), 0);
            assert_eq!(batch.num_columns(), 2);
        } else {
            panic!("Expected an empty batch, found none or an error");
        }
        // No second batch
        assert!(reader.next().is_none());
    }

    /// Test saving a single chain/single sample (with single dimension) to Arrow using `f64`.
    #[test]
    fn test_save_arrow_single_chain_single_sample_f64() -> Result<(), Box<dyn Error>> {
        let data = vec![na::dmatrix![42f64]];
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        // Write Arrow
        save_arrow(&data, filename)?;

        // Read back the file to verify
        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0, "Arrow file is unexpectedly empty");

        let file = File::open(filename)?;
        let mut reader = FileReader::try_new(file, None)?;
        let batch = reader.next().expect("No record batch found")?.clone(); // read first batch
        assert!(reader.next().is_none(), "Expected only one batch");

        // We expect 1 row, 1 dimension => columns = chain(0), sample(0), dim_0(42.0)
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 3);

        // Downcast columns
        let chain_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let sample_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(chain_array.value(0), 0);
        assert_eq!(sample_array.value(0), 0);
        assert!((dim0_array.value(0) - 42.0).abs() < f64::EPSILON);

        Ok(())
    }

    /// Test multiple chains, multiple samples, multiple dimensions with `f32`.
    #[test]
    fn test_save_arrow_multi_chain_f32() -> Result<(), Box<dyn Error>> {
        // 2 chains, 2 samples each, 2 dims => total 4 rows
        // chain=0, sample=0 => dims=[1.0, 2.5]
        // chain=0, sample=1 => dims=[3.0, 4.5]
        // chain=1, sample=0 => dims=[10.0, 20.5]
        // chain=1, sample=1 => dims=[30.0, 40.5]
        let data = vec![
            na::DMatrix::<f32>::from_row_slice(2, 2, &[1f32, 2.5f32, 3f32, 4.5f32]),
            na::DMatrix::<f32>::from_row_slice(2, 2, &[10f32, 20.5f32, 30f32, 40.5f32]),
        ];
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        save_arrow(&data, filename)?;

        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0);

        // Read back Arrow
        let file = File::open(filename)?;
        let mut reader = FileReader::try_new(file, None)?;
        let batch = reader.next().expect("No record batch found")?.clone();
        assert!(reader.next().is_none());

        // Check shape: 4 rows, columns = chain, sample, dim_0, dim_1 => total 4 columns
        dbg!(batch.clone());
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 4);

        let chain_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let sample_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let dim1_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Row 0: chain=0, sample=0, dim0=1.0, dim1=2.5
        assert_eq!(chain_array.value(0), 0);
        assert_eq!(sample_array.value(0), 0);
        assert!((dim0_array.value(0) - 1.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(0) - 2.5).abs() < f64::EPSILON);

        // Row 1: chain=0, sample=1, dim0=3.0, dim1=4.5
        assert_eq!(chain_array.value(1), 0);
        assert_eq!(sample_array.value(1), 1);
        assert!((dim0_array.value(1) - 3.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(1) - 4.5).abs() < f64::EPSILON);

        // Row 2: chain=1, sample=0, dim0=10.0, dim1=20.5
        assert_eq!(chain_array.value(2), 1);
        assert_eq!(sample_array.value(2), 0);
        assert!((dim0_array.value(2) - 10.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(2) - 20.5).abs() < f64::EPSILON);

        // Row 3: chain=1, sample=1, dim0=30.0, dim1=40.5
        assert_eq!(chain_array.value(3), 1);
        assert_eq!(sample_array.value(3), 1);
        assert!((dim0_array.value(3) - 30.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(3) - 40.5).abs() < f64::EPSILON);

        Ok(())
    }

    /// Test saving data with an integer type (i32) to Arrow
    /// to ensure `T: Into<f64> + Copy` is satisfied with different numeric types.
    #[test]
    fn test_save_arrow_integer_data() {
        let data = vec![
            na::dmatrix![100, 200, 300, 400, 500, 600],
            na::dmatrix![700, 800, 900, 1000, 1100, 1200],
        ];
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_arrow(&data, filename);
        assert!(
            result.is_ok(),
            "Saving integer data to Arrow failed: {:?}",
            result
        );

        let metadata = fs::metadata(filename).unwrap();
        assert!(metadata.len() > 0);
        // (Optional) read back and verify numeric values the same way shown above
    }

    /// Test error when chain lengths are inconsistent.
    #[test]
    fn test_save_arrow_inconsistent_chain_lengths() {
        // Chain 0 has 2 samples, chain 1 has 1 sample
        let data = vec![na::dmatrix![1f64, 2f64], na::dmatrix![3f64]];

        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_arrow(&data, filename);
        assert!(
            result.is_err(),
            "Expected an error due to inconsistent samples per chain"
        );
    }

    /// Test error when dimension lengths are inconsistent.
    #[test]
    fn test_save_arrow_inconsistent_dimensions() {
        // The second sample in chain 0 has only 1 dimension,
        // but the first sample had 2 dimensions.
        let data = vec![na::dmatrix![1f64, 2f64], na::dmatrix![3f64]];

        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_arrow(&data, filename);
        assert!(
            result.is_err(),
            "Expected an error due to inconsistent dimension lengths"
        );
    }

    /// Test that save_arrow returns an error when one chain has a different number of samples.
    #[test]
    fn test_save_arrow_inconsistent_chain_rows() {
        // Create two chains:
        // chain0 has 2 samples (rows) and 2 dimensions (columns)
        let chain0 = na::DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        // chain1 has 1 sample (row) and 2 dimensions (columns)
        let chain1 = na::DMatrix::<f64>::from_row_slice(1, 2, &[5.0, 6.0]);
        let data = vec![chain0, chain1];

        let file = NamedTempFile::new().expect("Failed to create temp file");
        let filename = file.path().to_str().unwrap();
        let result = save_arrow(&data, filename);
        assert!(
            result.is_err(),
            "Expected error due to inconsistent number of samples among chains"
        );

        let err_msg = result.err().unwrap().to_string();
        assert_eq!(err_msg, "Inconsistent number of samples among chains");
    }

    /// Test saving empty data to Parquet (zero chains).
    #[test]
    fn test_save_parquet_empty_data() -> Result<(), Box<dyn Error>> {
        let data: Vec<na::DMatrix<f64>> = vec![]; // no chains
                                                  // let file = NamedTempFile::new()?;
        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_parquet(&data, filename);
        assert!(
            result.is_ok(),
            "Saving empty data to Parquet failed: {:?}",
            result
        );

        let metadata = fs::metadata(filename)?;
        // Even though there's no chain data, the file shouldn't be completely empty
        assert!(metadata.len() > 0, "Parquet file is unexpectedly empty");

        // Check emptyness
        let file = File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        assert!(reader.next().is_none());
        Ok(())
    }

    /// Test saving a single chain, single sample (one dimension).
    #[test]
    fn test_save_parquet_single_chain_single_sample() -> Result<(), Box<dyn Error>> {
        let data = vec![na::dmatrix![42f64]]; // chain=0, sample=0, dim_0=42
        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        save_parquet(&data, filename)?;

        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0, "Parquet file is unexpectedly empty");

        // Read back the batch
        let file = File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        let batch = reader.next().expect("Expected a record batch")?.clone();
        assert!(reader.next().is_none(), "Expected only one batch");

        // Should have 1 row and 3 columns (chain, sample, dim_0)
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(batch.num_columns(), 3);

        // Check values
        let chain_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let sample_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        assert_eq!(chain_array.value(0), 0);
        assert_eq!(sample_array.value(0), 0);
        assert!((dim0_array.value(0) - 42.0).abs() < f64::EPSILON);

        Ok(())
    }

    /// Test multiple chains, multiple samples, multiple dimensions to Parquet.
    #[test]
    fn test_save_parquet_multi_chain() -> Result<(), Box<dyn Error>> {
        // data[chain][sample][dim]
        // chain=0 => sample=0 => [1.0, 2.0], sample=1 => [3.0, 4.0]
        // chain=1 => sample=0 => [10.0, 20.0], sample=1 => [30.0, 40.0]
        let data = vec![
            na::DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]),
            na::DMatrix::<f64>::from_row_slice(2, 2, &[10.0, 20.0, 30.0, 40.0]),
        ];

        let file = NamedTempFile::new()?;
        let filename = file.path().to_str().unwrap();

        save_parquet(&data, filename)?;

        let metadata = fs::metadata(filename)?;
        assert!(metadata.len() > 0);

        // Read back
        let file = File::open(filename)?;
        let mut reader = ParquetRecordBatchReader::try_new(file, 1024)?;
        let batch = reader.next().expect("No record batch found")?;
        assert!(reader.next().is_none(), "Expected only one batch");

        // We expect 4 rows total: 2 chains × 2 samples each
        // columns = chain, sample, dim_0, dim_1 => 4 columns
        assert_eq!(batch.num_rows(), 4);
        assert_eq!(batch.num_columns(), 4);

        let chain_array = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let sample_array = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let dim0_array = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let dim1_array = batch
            .column(3)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();

        // Row 0: chain=0, sample=0, (dim_0=1.0, dim_1=2.0)
        assert_eq!(chain_array.value(0), 0);
        assert_eq!(sample_array.value(0), 0);
        assert!((dim0_array.value(0) - 1.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(0) - 2.0).abs() < f64::EPSILON);

        // Row 1: chain=0, sample=1, (dim_0=3.0, dim_1=4.0)
        assert_eq!(chain_array.value(1), 0);
        assert_eq!(sample_array.value(1), 1);
        assert!((dim0_array.value(1) - 3.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(1) - 4.0).abs() < f64::EPSILON);

        // Row 2: chain=1, sample=0, (dim_0=10.0, dim_1=20.0)
        assert_eq!(chain_array.value(2), 1);
        assert_eq!(sample_array.value(2), 0);
        assert!((dim0_array.value(2) - 10.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(2) - 20.0).abs() < f64::EPSILON);

        // Row 3: chain=1, sample=1, (dim_0=30.0, dim_1=40.0)
        assert_eq!(chain_array.value(3), 1);
        assert_eq!(sample_array.value(3), 1);
        assert!((dim0_array.value(3) - 30.0).abs() < f64::EPSILON);
        assert!((dim1_array.value(3) - 40.0).abs() < f64::EPSILON);

        Ok(())
    }

    /// Test error when chain lengths are inconsistent.
    #[test]
    fn test_save_parquet_inconsistent_chain_lengths() {
        // Chain 0 has 2 samples, chain 1 has 1 sample
        let data = vec![na::dmatrix![1f64, 2f64], na::dmatrix![3f64]];

        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_parquet(&data, filename);
        assert!(
            result.is_err(),
            "Expected an error due to inconsistent samples per chain"
        );
    }

    /// Test error when dimension lengths are inconsistent.
    #[test]
    fn test_save_parquet_inconsistent_dimensions() {
        // The second sample in chain 0 has only 1 dimension,
        // but the first sample had 2 dimensions.
        let data = vec![na::dmatrix![1, 2], na::dmatrix![3]];

        let file = NamedTempFile::new().expect("Could not create temp file");
        let filename = file.path().to_str().unwrap();

        let result = save_parquet(&data, filename);
        assert!(
            result.is_err(),
            "Expected an error due to inconsistent dimension lengths"
        );
    }

    /// Test that save_parquet returns an error when one chain has a different number of samples.
    #[test]
    fn test_save_parquet_inconsistent_chain_rows() {
        // Create two chains:
        // chain0 has 2 samples (rows) and 2 dimensions (columns)
        let chain0 = na::DMatrix::<f64>::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        // chain1 has 1 sample (row) and 2 dimensions (columns)
        let chain1 = na::DMatrix::<f64>::from_row_slice(1, 2, &[5.0, 6.0]);
        let data = vec![chain0, chain1];

        let file = NamedTempFile::new().expect("Failed to create temp file");
        let filename = file.path().to_str().unwrap();
        let result = save_parquet(&data, filename);
        assert!(
            result.is_err(),
            "Expected error due to inconsistent number of samples among chains"
        );

        let err_msg = result.err().unwrap().to_string();
        assert_eq!(err_msg, "Inconsistent number of samples among chains");
    }
}
