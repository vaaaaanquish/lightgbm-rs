# lightgbm-rs
LightGBM Rust binding


# Require

You need an environment that can build LightGBM.

```
# linux
apt install -y cmake libclang-dev libc++-dev gcc-multilib

# OS X
brew install cmake
```

Please see below for details.
https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html



# Usage

Please see `./examples`.

|example|link|
|---|---|
|binary classification|[link](https://github.com/vaaaaanquish/lightgbm-rs/blob/main/examples/binary_classification/src/main.rs)|
|multiclass classification|[link](https://github.com/vaaaaanquish/lightgbm-rs/blob/main/examples/multiclass_classification/src/main.rs)|
|regression|[link](https://github.com/vaaaaanquish/lightgbm-rs/blob/main/examples/regression/src/main.rs)|



# Develop

```
git clone --recursive https://github.com/vaaaaanquish/lightgbm-rs
```

```
docker build -t lgbmrs .
docker run -it -v $PWD:/app lgbmrs bash

# cargo build
```


# Thanks

Much reference was made to implementation and documentation. Thanks.

- [microsoft/LightGBM](https://github.com/microsoft/LightGBM)
- [davechallis/rust-xgboost](https://github.com/davechallis/rust-xgboost)
