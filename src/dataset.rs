use libc::{c_void,c_char};

use std;
use std::ffi::CString;
use lightgbm_sys;

use super::LGBMResult;

pub struct Dataset {
    pub(super) handle: lightgbm_sys::DatasetHandle
}

#[link(name = "c")]
impl Dataset {
    fn new(handle: lightgbm_sys::DatasetHandle) -> LGBMResult<Self> {
        Ok(Dataset{handle})
    }

    pub fn from_mat(data: Vec<Vec<f64>>, label: Vec<f32>) -> LGBMResult<Self> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let label_str = CString::new("label").unwrap();
        let reference = std::ptr::null_mut();  // not use
        let mut handle = std::ptr::null_mut();
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

        unsafe{
            lightgbm_sys::LGBM_DatasetCreateFromMat(
                flat_data.as_ptr() as *const c_void,
                lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
                data_length as i32,
                feature_length as i32,
                1 as i32,
                params.as_ptr() as *const c_char,
                reference,
                &mut handle
            );

            lightgbm_sys::LGBM_DatasetSetField(
                handle,
                label_str.as_ptr() as *const c_char,
                label.as_ptr() as *const c_void,
                data_length as i32,
                lightgbm_sys::C_API_DTYPE_FLOAT32 as i32
            );
        }
        Ok(Dataset::new(handle)?)
    }

    pub fn from_file(file_path: String) -> LGBMResult<Self> {
        let file_path_str = CString::new(file_path).unwrap();
        let params = CString::new("").unwrap();
        let mut handle = std::ptr::null_mut();

        unsafe {
            lightgbm_sys::LGBM_DatasetCreateFromFile(
                file_path_str.as_ptr() as *const c_char,
                params.as_ptr() as *const c_char,
                std::ptr::null_mut(),
                &mut handle
            );
        }
        Ok(Dataset::new(handle)?)
    }
}

