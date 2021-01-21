use libc::{c_char, c_double, c_void, c_long};
use std::ffi::CString;
use std;

use serde_json::Value;

use lightgbm_sys;


use super::{LGBMResult, Dataset, LGBMError};


/// Core model in LightGBM, containing functions for training, evaluating and predicting.
pub struct Booster {
    pub(super) handle: lightgbm_sys::BoosterHandle,
}


impl Booster {
    fn new(handle: lightgbm_sys::BoosterHandle) -> LGBMResult<Self> {
        Ok(Booster{handle})
    }

    /// Init from model file.
    pub fn from_file(filename: String) -> LGBMResult<Self>{
        let filename_str = CString::new(filename).unwrap();
        let mut out_num_iterations = 0;
        let mut handle = std::ptr::null_mut();
        lgbm_call!(
            lightgbm_sys::LGBM_BoosterCreateFromModelfile(
                filename_str.as_ptr() as *const c_char,
                &mut out_num_iterations,
                &mut handle
            )
        ).unwrap();
        Ok(Booster::new(handle)?)
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
    pub fn train(dataset: Dataset, parameter: &Value) -> LGBMResult<Self> {

        // get num_iterations
        let num_iterations: i64;
        if parameter["num_iterations"].is_null(){
            num_iterations = 100;
        } else {
            num_iterations = parameter["num_iterations"].as_i64().unwrap();
        }

        // exchange params {"x": "y", "z": 1} => "x=y z=1"
        let params_string = parameter.as_object().unwrap().iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(" ");
        let params_cstring = CString::new(params_string).unwrap();

        let mut handle = std::ptr::null_mut();
        lgbm_call!(
            lightgbm_sys::LGBM_BoosterCreate(
                dataset.handle,
                params_cstring.as_ptr() as *const c_char,
                &mut handle
            )
        )?;

        let mut is_finished: i32 = 0;
        for _ in 1..num_iterations {
            lgbm_call!(lightgbm_sys::LGBM_BoosterUpdateOneIter(handle, &mut is_finished))?;
        }
        Ok(Booster::new(handle)?)
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
    pub fn predict(&self, data: Vec<Vec<f64>>) -> LGBMResult<Vec<Vec<f64>>> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let mut out_length: c_long = 0;
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

        // get num_class
        let mut num_class = 0;
        lgbm_call!(
            lightgbm_sys::LGBM_BoosterGetNumClasses(
                self.handle,
                &mut num_class
            )
        )?;

        let out_result: Vec<f64> = vec![Default::default(); data_length * num_class as usize];

        lgbm_call!(
            lightgbm_sys::LGBM_BoosterPredictForMat(
                self.handle,
                flat_data.as_ptr() as *const c_void,
                lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
                data_length as i32,
                feature_length as i32,
                1 as i32,
                0 as i32,
                0 as i32,
                -1 as i32,
                params.as_ptr() as *const c_char,
                &mut out_length,
                out_result.as_ptr() as *mut c_double
            )
        )?;

        // reshape for multiclass [1,2,3,4,5,6] -> [[1,2,3], [4,5,6]]  # 3 class
        let reshaped_output;
        if num_class > 1{
            reshaped_output = out_result.chunks(num_class as usize).map(|x| x.to_vec()).collect();
        } else {
            reshaped_output = vec![out_result];
        }
        Ok(reshaped_output)
    }


    /// Save model to file.
    pub fn save_file(&self, filename: String){
        let filename_str = CString::new(filename).unwrap();
        lgbm_call!(
            lightgbm_sys::LGBM_BoosterSaveModel(
                self.handle,
                0 as i32,
                -1 as i32,
                0 as i32,
                filename_str.as_ptr() as *const c_char
            )
        ).unwrap();
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
    use std::path::Path;
    use std::fs;

    fn read_train_file() -> LGBMResult<Dataset> {
        Dataset::from_file("lightgbm-sys/lightgbm/examples/binary_classification/binary.train".to_string())
    }

    #[test]
    fn predict() {
        let dataset = read_train_file().unwrap();
        let params = json!{
            {
                "num_iterations": 10,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        let bst = Booster::train(dataset, &params).unwrap();
        let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];
        let result = bst.predict(feature).unwrap();
        let mut normalized_result = Vec::new();
        for r in &result[0]{
            if r > &0.5{
                normalized_result.push(1);
            } else {
                normalized_result.push(0);
            }
        }
        assert_eq!(normalized_result, vec![0, 0, 1]);
    }

    #[test]
    fn save_file() {
        let dataset = read_train_file().unwrap();
        let params = json!{
            {
                "num_iterations": 1,
                "objective": "binary",
                "metric": "auc",
                "data_random_seed": 0
            }
        };
        let bst = Booster::train(dataset, &params).unwrap();
        bst.save_file("./test/test_save_file.output".to_string());
        assert!(Path::new("./test/test_save_file.output").exists());
        fs::remove_file("./test/test_save_file.output");
    }

    #[test]
    fn from_file(){
        let bst = Booster::from_file("./test/test_from_file.input".to_string());
    }
}
