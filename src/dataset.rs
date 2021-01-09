use libc::{c_void,c_char};

use std;
use std::convert::TryInto;
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

    pub fn from_mat(data: Vec<Vec<f32>>, label: Vec<f32>) -> LGBMResult<Self> {
        let mut handle = std::ptr::null_mut();
        let data_length = data.len() as i32;
        let feature_length = data[0].len() as i32;
        let params = CString::new("").unwrap();
        let label_str = CString::new("label").unwrap();

        unsafe{
            lightgbm_sys::LGBM_DatasetCreateFromMat(
                data.as_ptr() as * mut c_void,
                lightgbm_sys::C_API_DTYPE_FLOAT32.try_into().unwrap(),
                data_length,
                feature_length,
                1,
                params.as_ptr() as *const c_char,
                std::ptr::null_mut(),
                &mut handle);

            lightgbm_sys::LGBM_DatasetSetField(
                handle,
                label_str.as_ptr() as *const c_char,
                label.as_ptr() as * mut c_void,
                data_length,
                lightgbm_sys::C_API_DTYPE_FLOAT32.try_into().unwrap());
        }
        Ok(Dataset::new(handle)?)
    }

    pub fn from_file(file_path: String) -> LGBMResult<Self> {
        let mut handle = std::ptr::null_mut();
        let file_path_str = CString::new(file_path).unwrap();
        let params = CString::new("").unwrap();

        unsafe {
            lightgbm_sys::LGBM_DatasetCreateFromFile(
                file_path_str.as_ptr() as * const c_char,
                params.as_ptr() as *const c_char,
                std::ptr::null_mut(),
                &mut handle);
        }
        Ok(Dataset::new(handle)?)
    }
}

