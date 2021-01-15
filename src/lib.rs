extern crate lightgbm_sys;
extern crate libc;

#[macro_use]
macro_rules! lgbm_call {
    ($x:expr) => {
        LGBMError::check_return_value(unsafe { $x })
    };
}


mod error;
pub use error::{LGBMError,LGBMResult};

mod dataset;
pub use dataset::Dataset;

mod booster;
pub use booster::Booster;
