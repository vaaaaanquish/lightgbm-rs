use libc::{c_char, c_void};
use lightgbm_sys;
use std;
use std::ffi::CString;

#[cfg(feature = "dataframe")]
use polars::prelude::*;

use crate::{Error, Result};

/// Dataset used throughout LightGBM for training.
///
/// # Examples
///
/// ## from mat
///
/// ```
/// use lightgbm::Dataset;
///
/// let data = vec![vec![1.0, 0.1, 0.2, 0.1],
///                vec![0.7, 0.4, 0.5, 0.1],
///                vec![0.9, 0.8, 0.5, 0.1],
///                vec![0.2, 0.2, 0.8, 0.7],
///                vec![0.1, 0.7, 1.0, 0.9]];
/// let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
/// let dataset = Dataset::from_mat(data, label).unwrap();
/// ```
///
/// ## from file
///
/// ```
/// use lightgbm::Dataset;
///
/// let dataset = Dataset::from_file(&"lightgbm-sys/lightgbm/examples/binary_classification/binary.train").unwrap();
/// ```
pub struct Dataset {
    pub(crate) handle: lightgbm_sys::DatasetHandle,
}

impl Dataset {
    fn new(handle: lightgbm_sys::DatasetHandle) -> Self {
        Self { handle }
    }

    /// Create a new `Dataset` from dense array in row-major order.
    ///
    /// Example
    /// ```
    /// use lightgbm::Dataset;
    ///
    /// let data = vec![vec![1.0, 0.1, 0.2, 0.1],
    ///                vec![0.7, 0.4, 0.5, 0.1],
    ///                vec![0.9, 0.8, 0.5, 0.1],
    ///                vec![0.2, 0.2, 0.8, 0.7],
    ///                vec![0.1, 0.7, 1.0, 0.9]];
    /// let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let dataset = Dataset::from_mat(data, label).unwrap();
    /// ```
    pub fn from_mat(data: Vec<Vec<f64>>, label: Vec<f32>) -> Result<Self> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let label_str = CString::new("label").unwrap();
        let reference = std::ptr::null_mut(); // not use
        let mut handle = std::ptr::null_mut();
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

        lgbm_call!(lightgbm_sys::LGBM_DatasetCreateFromMat(
            flat_data.as_ptr() as *const c_void,
            lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
            data_length as i32,
            feature_length as i32,
            1_i32,
            params.as_ptr() as *const c_char,
            reference,
            &mut handle
        ))?;

        lgbm_call!(lightgbm_sys::LGBM_DatasetSetField(
            handle,
            label_str.as_ptr() as *const c_char,
            label.as_ptr() as *const c_void,
            data_length as i32,
            lightgbm_sys::C_API_DTYPE_FLOAT32 as i32
        ))?;

        Ok(Self::new(handle))
    }

    /// Create a new `Dataset` from file.
    ///
    /// file is `tsv`.
    /// ```text
    /// <label>\t<value>\t<value>\t...
    /// ```
    ///
    /// ```text
    /// 2 0.11 0.89 0.2
    /// 3 0.39 0.1 0.4
    /// 0 0.1 0.9 1.0
    /// ```
    ///
    /// Example
    /// ```
    /// use lightgbm::Dataset;
    ///
    /// let dataset = Dataset::from_file(&"lightgbm-sys/lightgbm/examples/binary_classification/binary.train");
    /// ```
    pub fn from_file(file_path: &str) -> Result<Self> {
        let file_path_str = CString::new(file_path).unwrap();
        let params = CString::new("").unwrap();
        let mut handle = std::ptr::null_mut();

        lgbm_call!(lightgbm_sys::LGBM_DatasetCreateFromFile(
            file_path_str.as_ptr() as *const c_char,
            params.as_ptr() as *const c_char,
            std::ptr::null_mut(),
            &mut handle
        ))?;

        Ok(Self::new(handle))
    }

    /// Create a new `Dataset` from a polars DataFrame.
    ///
    /// Note: the feature ```dataframe``` is required for this method
    ///
    /// Example
    ///
    #[cfg_attr(
        feature = "dataframe",
        doc = r##"
    extern crate polars;

    use lightgbm::Dataset;
    use polars::prelude::*;
    use polars::df;

    let df: DataFrame = df![
            "feature_1" => [1.0, 0.7, 0.9, 0.2, 0.1],
            "feature_2" => [0.1, 0.4, 0.8, 0.2, 0.7],
            "feature_3" => [0.2, 0.5, 0.5, 0.1, 0.1],
            "feature_4" => [0.1, 0.1, 0.1, 0.7, 0.9],
            "label" => [0.0, 0.0, 0.0, 1.0, 1.0]
        ].unwrap();
    let dataset = Dataset::from_dataframe(df, String::from("label")).unwrap();
    "##
    )]
    #[cfg(feature = "dataframe")]
    pub fn from_dataframe(mut dataframe: DataFrame, label_column: String) -> Result<Self> {
        let label_col_name = label_column.as_str();

        let (m, n) = dataframe.shape();

        let label_series = &dataframe.select_series([label_col_name])?[0].cast(&DataType::Float32)?;

        if label_series.null_count() != 0 {
            panic!("Cannot create a dataset with null values, encountered nulls when creating the label array")
        }

        let _ = dataframe.drop_in_place(label_col_name)?;

        let mut label_values = Vec::with_capacity(m);

        let label_values_ca = label_series.unpack::<Float32Type>()?;

        label_values_ca
            .into_no_null_iter()
            .enumerate()
            .for_each(|(_row_idx, val)| {
                label_values.push(val);
            });

        let mut feature_values = Vec::with_capacity(m);
        for _i in 0..m {
            feature_values.push(Vec::with_capacity(n));
        }

        for (_col_idx, series) in dataframe.get_columns().iter().enumerate() {
            if series.null_count() != 0 {
                panic!("Cannot create a dataset with null values, encountered nulls when creating the features array")
            }

            let series = series.cast(&DataType::Float64)?;
            let ca = series.unpack::<Float64Type>()?;

            ca.into_no_null_iter()
                .enumerate()
                .for_each(|(row_idx, val)| feature_values[row_idx].push(val));
        }
        Self::from_mat(feature_values, label_values)
    }
}

impl Drop for Dataset {
    fn drop(&mut self) {
        lgbm_call!(lightgbm_sys::LGBM_DatasetFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read_train_file() -> Result<Dataset> {
        Dataset::from_file(&"lightgbm-sys/lightgbm/examples/binary_classification/binary.train")
    }

    #[test]
    fn read_file() {
        assert!(read_train_file().is_ok());
    }

    #[test]
    fn from_mat() {
        let data = vec![
            vec![1.0, 0.1, 0.2, 0.1],
            vec![0.7, 0.4, 0.5, 0.1],
            vec![0.9, 0.8, 0.5, 0.1],
            vec![0.2, 0.2, 0.8, 0.7],
            vec![0.1, 0.7, 1.0, 0.9],
        ];
        let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_mat(data, label);
        assert!(dataset.is_ok());
    }

    #[cfg(feature = "dataframe")]
    #[test]
    fn from_dataframe() {
        use polars::df;
        let df: DataFrame = df![
            "feature_1" => [1.0, 0.7, 0.9, 0.2, 0.1],
            "feature_2" => [0.1, 0.4, 0.8, 0.2, 0.7],
            "feature_3" => [0.2, 0.5, 0.5, 0.1, 0.1],
            "feature_4" => [0.1, 0.1, 0.1, 0.7, 0.9],
            "label" => [0.0, 0.0, 0.0, 1.0, 1.0]
        ]
        .unwrap();

        let df_dataset = Dataset::from_dataframe(df, String::from("label"));
        assert!(df_dataset.is_ok());
    }
}
