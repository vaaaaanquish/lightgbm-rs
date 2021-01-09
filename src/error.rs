use std;

pub type LGBMResult<T> = std::result::Result<T, LGBMError>;


#[derive(Debug, Eq, PartialEq)]
pub struct LGBMError {
    desc: String,
}

impl LGBMError {
    pub(crate) fn new<S: Into<String>>(desc: S) -> Self {
        LGBMError { desc: desc.into() }
    }
}
