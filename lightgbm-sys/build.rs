extern crate bindgen;
extern crate cmake;

use cmake::Config;
use std::process::Command;
use std::env;
use std::path::{Path, PathBuf};


fn main() {
    let out_dir = env::var("OUT_DIR").unwrap();
    let lgbm_root = Path::new(&out_dir).join("lightgbm");

    // copy source code
    if !lgbm_root.exists() {
        Command::new("cp")
            .args(&["-r", "lightgbm", lgbm_root.to_str().unwrap()])
            .status()
            .unwrap_or_else(|e| {
                panic!("Failed to copy ./lightgbm to {}: {}", lgbm_root.display(), e);
            });
    }

    // CMake
    let dst = Config::new(&lgbm_root)
        .profile("Release")
        .uses_cxx11()
        .define("BUILD_STATIC_LIB", "ON")
        .build();

    // bindgen build
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-x","c++", "-std=c++11"])
        .clang_arg(format!("-I{}", lgbm_root.join("include").display()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    println!("cargo:rustc-link-search={}", out_path.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=gomp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=_lightgbm");
}
