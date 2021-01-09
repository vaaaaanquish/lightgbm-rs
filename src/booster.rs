use lightgbm_sys;

use libc::{c_char, c_int, c_float, c_double, c_long, c_void};
use std::ffi::CString;
use std::convert::TryInto;
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
        let mut handle = std::ptr::null_mut();
        let mut params = CString::new("app=binary metric=auc num_leaves=31").unwrap();
        unsafe {
            lightgbm_sys::LGBM_BoosterCreate(
                dataset.handle,
                params.as_ptr() as *const c_char,
                &mut handle);
        }

        // train
        let mut is_finished: i32 = 0;
        unsafe{
            for n in 1..50 {
                let ret = lightgbm_sys::LGBM_BoosterUpdateOneIter(handle, &mut is_finished);
            }
        }
        Ok(Booster::new(handle)?)
    }

    pub fn predict(&self, data: Vec<Vec<f32>>) -> LGBMResult<Vec<f64>> {
        let data_length = data.len() as i32;
        let feature_length = data[0].len() as i32;
        let mut params = CString::new("").unwrap();
        let mut out_len: c_long = 0;
        // let mut out_result =  Vec::with_capacity(data_length.try_into().unwrap());
        let data_size = data_length.try_into().unwrap();
        let mut out_result: Vec<f64> = vec![Default::default(); data_size];

        unsafe {
            lightgbm_sys::LGBM_BoosterPredictForMat(
                self.handle,
                data.as_ptr() as * mut c_void,
                lightgbm_sys::C_API_DTYPE_FLOAT32.try_into().unwrap(),
                data_length,
                feature_length,
                0,
                0,
                0,
                0,
                params.as_ptr() as *const c_char,
                &mut out_len,
                out_result.as_ptr() as *mut c_double
                );
        }
        Ok(out_result)
    }
}
