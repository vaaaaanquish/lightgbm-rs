extern crate libc;
extern crate lightgbm_sys;
extern crate serde_json;

#[cfg(feature = "dataframe")]
extern crate polars;

#[macro_use]
macro_rules! lgbm_call {
    ($x:expr) => {
        Error::check_return_value(unsafe { $x })
    };
}

mod error;
pub use error::{Error, Result};

mod dataset;
pub use dataset::Dataset;

mod booster;
pub use booster::Booster;
