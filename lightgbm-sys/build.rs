extern crate bindgen;
extern crate cmake;

use cmake::Config;
use std::env;
use std::path::{Path, PathBuf};


fn main() {
    let lgbm_root = Path::new("./lightgbm");

    // CMake
    let _dst = Config::new("lightgbm")
        .profile("Release")
        .uses_cxx11()
        .build();
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_args(&["-x","c++", "-std=c++11"])
        .clang_arg(format!("-I{}", lgbm_root.join("include").display()))
        .clang_arg(format!("-I{}", lgbm_root.join("external_libs/fmt").display()))
        .clang_arg(format!("-I{}", lgbm_root.join("external_libs/fast_double_parser").display()))
        .generate()
        .expect("Unable to generate bindings");
    let out_path = Path::new("./target");
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings.");
}
