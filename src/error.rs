//! Functionality related to errors and error handling.

use std;
use std::error::Error;
use std::ffi::CStr;
use std::fmt::{self, Display};

use lightgbm_sys;

/// Convenience return type for most operations which can return an `LightGBM`.
pub type LGBMResult<T> = std::result::Result<T, LGBMError>;

/// Wrap errors returned by the LightGBM library.
#[derive(Debug, Eq, PartialEq)]
pub struct LGBMError {
    desc: String,
}

impl LGBMError {
    pub(crate) fn new<S: Into<String>>(desc: S) -> Self {
        LGBMError { desc: desc.into() }
    }

    /// Check the return value from an LightGBM FFI call, and return the last error message on error.
    ///
    /// Return values of 0 are treated as success, returns values of -1 are treated as errors.
    ///
    /// Meaning of any other return values are undefined, and will cause a panic.
    pub(crate) fn check_return_value(ret_val: i32) -> LGBMResult<()> {
        match ret_val {
            0 => Ok(()),
            -1 => Err(LGBMError::from_lightgbm()),
            _ => panic!(format!(
                "unexpected return value '{}', expected 0 or -1",
                ret_val
            )),
        }
    }

    /// Get the last error message from LightGBM.
    fn from_lightgbm() -> Self {
        let c_str = unsafe { CStr::from_ptr(lightgbm_sys::LGBM_GetLastError()) };
        let str_slice = c_str.to_str().unwrap();
        Self::new(str_slice)
    }
}

impl Error for LGBMError {}

impl Display for LGBMError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "LightGBM error: {}", &self.desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn return_value_handling() {
        let result = LGBMError::check_return_value(0);
        assert_eq!(result, Ok(()));

        let result = LGBMError::check_return_value(-1);
        assert_eq!(result, Err(LGBMError::new("Everything is fine")));
    }
}
