# lightgbm-rs
LightGBM Rust binding


Now: Done is better than perfect.


# develop

```
git clone --recursive https://github.com/vaaaaanquish/lightgbm-rs
```

```
docker pull rust:1.49.0
docker run -it -v $PWD:/app rust bash
```

```
apt update
apt install -y cmake libclang-dev libc++-dev gcc-multilib
cd app
cargo build
```

LightGBM C API doc
https://lightgbm.readthedocs.io/en/latest/C-API.html
