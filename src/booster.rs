use lightgbm_sys;

use libc::{c_char, c_double, c_void, c_long};
use std::ffi::CString;
use std;

use super::{LGBMResult, Dataset};


pub struct Booster {
    pub(super) handle: lightgbm_sys::BoosterHandle
}

impl Booster {
    fn new(handle: lightgbm_sys::BoosterHandle) -> LGBMResult<Self> {
        Ok(Booster{handle})
    }

    pub fn train(dataset: Dataset) -> LGBMResult<Self> {
        let params = CString::new("objective=binary metric=auc").unwrap();
        let mut handle = std::ptr::null_mut();
        unsafe {
            lightgbm_sys::LGBM_BoosterCreate(
                dataset.handle,
                params.as_ptr() as *const c_char,
                &mut handle
            );
        }

        // train
        let mut is_finished: i32 = 0;
        unsafe{
            for _ in 1..100 {
                lightgbm_sys::LGBM_BoosterUpdateOneIter(handle, &mut is_finished);
            }
        }
        Ok(Booster::new(handle)?)
    }

    pub fn predict(&self, data: Vec<Vec<f64>>) -> LGBMResult<Vec<f64>> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let mut out_length: c_long = 0;
        let out_result: Vec<f64> = vec![Default::default(); data.len()];
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

        unsafe {
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
            );
        }
        Ok(out_result)
    }
}
