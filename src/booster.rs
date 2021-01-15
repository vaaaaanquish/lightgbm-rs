use libc::{c_char, c_double, c_void, c_long};
use std::ffi::CString;
use std;


use lightgbm_sys;


use super::{LGBMResult, Dataset, LGBMError};


/// Core model in LightGBM, containing functions for training, evaluating and predicting.
pub struct Booster {
    pub(super) handle: lightgbm_sys::BoosterHandle
}


impl Booster {
    fn new(handle: lightgbm_sys::BoosterHandle) -> LGBMResult<Self> {
        Ok(Booster{handle})
    }

    /// Create a new Booster model with given Dataset and parameters.
    pub fn train(dataset: Dataset, params: String) -> LGBMResult<Self> {
        let params = CString::new(params).unwrap();
        let mut handle = std::ptr::null_mut();
        lgbm_call!(
            lightgbm_sys::LGBM_BoosterCreate(
                dataset.handle,
                params.as_ptr() as *const c_char,
                &mut handle
            )
        )?;

        let mut is_finished: i32 = 0;
        for _ in 1..100 {
            lgbm_call!(lightgbm_sys::LGBM_BoosterUpdateOneIter(handle, &mut is_finished))?;
        }
        Ok(Booster::new(handle)?)
    }

    /// Predict results for given data.
    pub fn predict(&self, data: Vec<Vec<f64>>) -> LGBMResult<Vec<f64>> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let mut out_length: c_long = 0;
        let out_result: Vec<f64> = vec![Default::default(); data.len()];
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

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
        Ok(out_result)
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
    fn read_train_file() -> LGBMResult<Dataset> {
        Dataset::from_file("lightgbm-sys/lightgbm/examples/binary_classification/binary.train".to_string())
    }

    #[test]
    fn predict() {
        let dataset = read_train_file().unwrap();
        let bst = Booster::train(dataset, "objective=binary metric=auc".to_string()).unwrap();
        let feature = vec![vec![0.5; 28], vec![0.0; 28], vec![0.9; 28]];
        let result = bst.predict(feature).unwrap();
        let mut normalized_result = Vec::new();
        for r in result{
            if r > 0.5{
                normalized_result.push(1);
            } else {
                normalized_result.push(0);
            }
        }
        assert_eq!(normalized_result, vec![0, 0, 1]);
    }
}
