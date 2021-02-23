extern crate bindgen;
extern crate cmake;

use cmake::Config;
use std::env;
use std::path::PathBuf;

fn main() {
    // CMake
    let dst = Config::new("lightgbm")
        .profile("Release")
        .uses_cxx11()
        .define("BUILD_STATIC_LIB", "ON")
        .build();

    // bindgen build
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-x", "c++", "-std=c++11"])
        .clang_arg(format!("-I{}", out_path.join("include").display()))
        .generate()
        .expect("Unable to generate bindings");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");

    // link to appropriate C++ lib
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=dylib=omp");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=dylib=gomp");
    }

    println!("cargo:rustc-link-search={}", out_path.join("lib").display());
    println!("cargo:rustc-link-search=native={}", dst.display());

    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=static=lib_lightgbm");
    } else {
        println!("cargo:rustc-link-lib=static=_lightgbm");
    }
}
