use libc::{c_char, c_double, c_longlong, c_void};
use std;
use std::ffi::CString;

use serde_json::Value;

use lightgbm_sys;

use crate::{Dataset, Error, Result};

pub const PREDICT_TYPE_NORMAL: i32 = lightgbm_sys::C_API_PREDICT_NORMAL as i32;
pub const PREDICT_TYPE_LEAF_INDEX: i32 = lightgbm_sys::C_API_PREDICT_LEAF_INDEX as i32;


fn load_pandas_categorical(model_string: &str) -> Option<Vec<Vec<Value>>> {
    let pandas_key = "pandas_categorical:";
    let idx = model_string.rfind(pandas_key);

    if let Some(idx) = idx {
        let idx = idx + pandas_key.len();
        let value = model_string[idx..].trim();
        serde_json::from_str(value).ok()
    } else {
        None
    }
}

/// Core model in LightGBM, containing functions for training, evaluating and predicting.
pub struct Booster {
    handle: lightgbm_sys::BoosterHandle,
    pub pandas_categorical: Option<Vec<Vec<Value>>>,
}

impl Booster {
    fn new(handle: lightgbm_sys::BoosterHandle, pandas_categorical: Option<Vec<Vec<Value>>>) -> Self {
        Booster { handle, pandas_categorical }
    }

    /// Init from model file.
    pub fn from_file(filename: &str) -> Result<Self> {
        let filename_str = CString::new(filename).unwrap();
        let mut out_num_iterations = 0;
        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm_sys::LGBM_BoosterCreateFromModelfile(
            filename_str.as_ptr() as *const c_char,
            &mut out_num_iterations,
            &mut handle
        ))?;
        // pandas_categorical is not implemented
        Ok(Booster::new(handle, None))
    }

    /// Init from model string.
    pub fn from_string(model_string: &str) -> Result<Self> {
        let model_c_str = CString::new(model_string).unwrap();
        let mut out_num_iterations = 0;
        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm_sys::LGBM_BoosterLoadModelFromString(
            model_c_str.as_ptr() as *const c_char,
            &mut out_num_iterations,
            &mut handle
        ))?;

        let pandas_categorical = load_pandas_categorical(model_string);

        Ok(Booster::new(handle, pandas_categorical))
    }

    /// Create a new Booster model with given Dataset and parameters.
    ///
    /// Example
    /// ```
    /// extern crate serde_json;
    /// use lightgbm::{Dataset, Booster};
    /// use serde_json::json;
    ///
    /// let data = vec![vec![1.0, 0.1, 0.2, 0.1],
    ///                vec![0.7, 0.4, 0.5, 0.1],
    ///                vec![0.9, 0.8, 0.5, 0.1],
    ///                vec![0.2, 0.2, 0.8, 0.7],
    ///                vec![0.1, 0.7, 1.0, 0.9]];
    /// let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    /// let dataset = Dataset::from_mat(data, label).unwrap();
    /// let params = json!{
    ///    {
    ///         "num_iterations": 3,
    ///         "objective": "binary",
    ///         "metric": "auc"
    ///     }
    /// };
    /// let bst = Booster::train(dataset, &params).unwrap();
    /// ```
    pub fn train(dataset: Dataset, parameter: &Value) -> Result<Self> {
        // get num_iterations
        let num_iterations: i64 = if parameter["num_iterations"].is_null() {
            100
        } else {
            parameter["num_iterations"].as_i64().unwrap()
        };

        // exchange params {"x": "y", "z": 1} => "x=y z=1"
        let params_string = parameter
            .as_object()
            .unwrap()
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join(" ");
        let params_cstring = CString::new(params_string).unwrap();

        let mut handle = std::ptr::null_mut();
        lgbm_call!(lightgbm_sys::LGBM_BoosterCreate(
            dataset.handle,
            params_cstring.as_ptr() as *const c_char,
            &mut handle
        ))?;

        let mut is_finished: i32 = 0;
        for _ in 1..num_iterations {
            lgbm_call!(lightgbm_sys::LGBM_BoosterUpdateOneIter(
                handle,
                &mut is_finished
            ))?;
        }
        // pandas_categorical is not implemented
        Ok(Booster::new(handle, None))
    }

    /// Predict results for given data.
    ///
    /// Input data example
    /// ```
    /// let data = vec![vec![1.0, 0.1, 0.2],
    ///                vec![0.7, 0.4, 0.5],
    ///                vec![0.1, 0.7, 1.0]];
    /// ```
    ///
    /// Output data example
    /// ```
    /// let output = vec![vec![1.0, 0.109, 0.433]];
    /// ```
    pub fn predict(&self, data: Vec<Vec<f64>>, predict_type: i32) -> Result<Vec<Vec<f64>>> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let mut out_length: c_longlong = 0;
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

        // get num_class
        let mut num_class = 0;
        lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumClasses(
            self.handle,
            &mut num_class
        ))?;

        let out_result: Vec<f64> = vec![Default::default(); data_length * num_class as usize];

        lgbm_call!(lightgbm_sys::LGBM_BoosterPredictForMat(
            self.handle,
            flat_data.as_ptr() as *const c_void,
            lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
            data_length as i32,
            feature_length as i32,
            1_i32,
            predict_type,
            0_i32,
            -1_i32,
            params.as_ptr() as *const c_char,
            &mut out_length,
            out_result.as_ptr() as *mut c_double
        ))?;

        // reshape for multiclass [1,2,3,4,5,6] -> [[1,2,3], [4,5,6]]  # 3 class
        let reshaped_output = if num_class > 1 {
            out_result
                .chunks(num_class as usize)
                .map(|x| x.to_vec())
                .collect()
        } else {
            vec![out_result]
        };
        Ok(reshaped_output)
    }

    /// Get Feature Num.
    pub fn num_feature(&self) -> Result<i32> {
        let mut out_len = 0;
        lgbm_call!(lightgbm_sys::LGBM_BoosterGetNumFeature(
            self.handle,
            &mut out_len
        ))?;
        Ok(out_len)
    }

    /// Get Feature Names.
    pub fn feature_name(&self) -> Result<Vec<String>> {
        let num_feature = self.num_feature()?;
        let mut tmp_out_len = 0;
        let reserved_string_buffer_size: u64 = 255;
        let reserved_string_buffer_size_usize = 255;
        let mut required_string_buffer_size = 0;
        let out_strs = (0..num_feature)
            .map(|_| {
                CString::new(" ".repeat(reserved_string_buffer_size_usize))
                    .unwrap()
                    .into_raw() as *mut c_char
            })
            .collect::<Vec<_>>();
        lgbm_call!(lightgbm_sys::LGBM_BoosterGetFeatureNames(
            self.handle,
            num_feature as i32,
            &mut tmp_out_len,
            reserved_string_buffer_size as u64,
            &mut required_string_buffer_size,
            out_strs.as_ptr() as *mut *mut c_char
        ))?;
        let actual_string_buffer_size = required_string_buffer_size.clone();
        let actual_string_buffer_size_usize= actual_string_buffer_size.clone() as usize;
        let out_strs = if actual_string_buffer_size > reserved_string_buffer_size {
                (0..num_feature)
                .map(|_| {
                    CString::new(" ".repeat(actual_string_buffer_size_usize))
                        .unwrap()
                        .into_raw() as *mut c_char
                })
                .collect::<Vec<_>>()}  else {out_strs};
        if actual_string_buffer_size > reserved_string_buffer_size {
                lgbm_call!(lightgbm_sys::LGBM_BoosterGetFeatureNames(
                    self.handle,
                    num_feature as i32,
                    &mut tmp_out_len,
                    actual_string_buffer_size as u64,
                    &mut required_string_buffer_size,
                    out_strs.as_ptr() as *mut *mut c_char
                ))?;
        };
        let output: Vec<String> = out_strs
            .into_iter()
            .map(|s| unsafe { CString::from_raw(s).into_string().unwrap() })
            .collect();
        Ok(output)
    }

    // Get Feature Importance
    pub fn feature_importance(&self) -> Result<Vec<f64>> {
        let num_feature = self.num_feature()?;
        let out_result: Vec<f64> = vec![Default::default(); num_feature as usize];
        lgbm_call!(lightgbm_sys::LGBM_BoosterFeatureImportance(
            self.handle,
            0_i32,
            0_i32,
            out_result.as_ptr() as *mut c_double
        ))?;
        Ok(out_result)
    }

    /// Save model to file.
    pub fn save_file(&self, filename: &str) -> Result<()> {
        let filename_str = CString::new(filename).unwrap();
        lgbm_call!(lightgbm_sys::LGBM_BoosterSaveModel(
            self.handle,
            0_i32,
            -1_i32,
            0_i32,
            filename_str.as_ptr() as *const c_char
        ))?;
        Ok(())
    }
}

impl Drop for Booster {
    fn drop(&mut self) {
        lgbm_call!(lightgbm_sys::LGBM_BoosterFree(self.handle)).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::fs;
    use std::path::Path;

    fn _read_train_file() -> Result<Dataset> {
        Dataset::from_file(&"lightgbm-sys/lightgbm/examples/binary_classification/binary.train")
    }

    fn _train_booster(params: &Value) -> Booster {
        let dataset = _read_train_file().unwrap();
        Booster::train(dataset, &params).unwrap()
    }

    fn _default_params() -> Value {
        let params = json! {
            {
                "num_iterations": 1,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        params
    }

    #[test]
    fn predict() {
        let params = json! {
            {
                "num_iterations": 10,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        let bst = _train_booster(&params);
        let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];
        let result = bst.predict(feature, PREDICT_TYPE_NORMAL).unwrap();
        let mut normalized_result = Vec::new();
        for r in &result[0] {
            normalized_result.push(if r > &0.5 { 1 } else { 0 });
        }
        assert_eq!(normalized_result, vec![0, 0, 1]);
    }

    #[test]
    fn num_feature() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let num_feature = bst.num_feature().unwrap();
        assert_eq!(num_feature, 28);
    }

    #[test]
    fn feature_importance() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let feature_importance = bst.feature_importance().unwrap();
        assert_eq!(feature_importance, vec![0.0; 28]);
    }

    #[test]
    fn feature_name() {
        let params = _default_params();
        let bst = _train_booster(&params);
        let feature_name = bst.feature_name().unwrap();
        let target = (0..28).map(|i| format!("Column_{}", i)).collect::<Vec<_>>();
        assert_eq!(feature_name, target);
    }

    #[test]
    fn save_file() {
        let params = _default_params();
        let bst = _train_booster(&params);
        assert_eq!(bst.save_file(&"./test/test_save_file.output"), Ok(()));
        assert!(Path::new("./test/test_save_file.output").exists());
        let _ = fs::remove_file("./test/test_save_file.output");
    }

    #[test]
    fn from_file() {
        let _ = Booster::from_file(&"./test/test_from_file.input");
    }

    #[test]
    fn test_load_pandas_categorical() {
        let model_string = "[num_gpu: 1]\n\nend of parameters\n\npandas_categorical:[[1, 2], [\"a\"], [\"c\", \"cc\"], [\"b\", \"bb\"]]\n";
        let pandas_categorical = load_pandas_categorical(model_string).unwrap();
        assert_eq!(pandas_categorical.get(0).unwrap().iter().map(|x| x.as_i64().unwrap()).collect::<Vec<i64>>(), vec![1, 2]);
        assert_eq!(pandas_categorical.get(1).unwrap().iter().map(|x| x.as_str().unwrap()).collect::<Vec<&str>>(), vec!["a"]);
    }
}
