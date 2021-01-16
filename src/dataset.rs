use std;
use std::ffi::CString;
use libc::{c_void,c_char};
use lightgbm_sys;


use super::{LGBMResult, LGBMError};


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
/// let dataset = Dataset::from_file(
///     "lightgbm-sys/lightgbm/examples/binary_classification/binary.train"
///     .to_string()).unwrap();
/// ```
pub struct Dataset {
    pub(super) handle: lightgbm_sys::DatasetHandle
}


#[link(name = "c")]
impl Dataset {
    fn new(handle: lightgbm_sys::DatasetHandle) -> LGBMResult<Self> {
        Ok(Dataset{handle})
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
    pub fn from_mat(data: Vec<Vec<f64>>, label: Vec<f32>) -> LGBMResult<Self> {
        let data_length = data.len();
        let feature_length = data[0].len();
        let params = CString::new("").unwrap();
        let label_str = CString::new("label").unwrap();
        let reference = std::ptr::null_mut();  // not use
        let mut handle = std::ptr::null_mut();
        let flat_data = data.into_iter().flatten().collect::<Vec<_>>();

        lgbm_call!(
            lightgbm_sys::LGBM_DatasetCreateFromMat(
                flat_data.as_ptr() as *const c_void,
                lightgbm_sys::C_API_DTYPE_FLOAT64 as i32,
                data_length as i32,
                feature_length as i32,
                1 as i32,
                params.as_ptr() as *const c_char,
                reference,
                &mut handle
            )
        )?;

        lgbm_call!(
            lightgbm_sys::LGBM_DatasetSetField(
                handle,
                label_str.as_ptr() as *const c_char,
                label.as_ptr() as *const c_void,
                data_length as i32,
                lightgbm_sys::C_API_DTYPE_FLOAT32 as i32
            )
        )?;

        Ok(Dataset::new(handle)?)
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
    /// let dataset = Dataset::from_file("lightgbm-sys/lightgbm/examples/binary_classification/binary.train".to_string());
    /// ```
    pub fn from_file(file_path: String) -> LGBMResult<Self> {
        let file_path_str = CString::new(file_path).unwrap();
        let params = CString::new("").unwrap();
        let mut handle = std::ptr::null_mut();

        lgbm_call!(
            lightgbm_sys::LGBM_DatasetCreateFromFile(
                file_path_str.as_ptr() as *const c_char,
                params.as_ptr() as *const c_char,
                std::ptr::null_mut(),
                &mut handle
            )
        )?;

        Ok(Dataset::new(handle)?)
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
    fn read_train_file() -> LGBMResult<Dataset> {
        Dataset::from_file("lightgbm-sys/lightgbm/examples/binary_classification/binary.train".to_string())
    }

    #[test]
    fn read_file() {
        assert!(read_train_file().is_ok());
    }

    #[test]
    fn from_mat(){
        let data = vec![vec![1.0, 0.1, 0.2, 0.1],
                        vec![0.7, 0.4, 0.5, 0.1],
                        vec![0.9, 0.8, 0.5, 0.1],
                        vec![0.2, 0.2, 0.8, 0.7],
                        vec![0.1, 0.7, 1.0, 0.9]];
        let label = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let dataset = Dataset::from_mat(data, label);
        assert!(dataset.is_ok());
    }
}
