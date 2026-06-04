use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() { return; }

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let obj = out.join("kernels.o");
    let lib = out.join("libkernels.a");

    let cuda_home = std::env::var("CUDA_HOME")
        .unwrap_or_else(|_| "/usr/local/cuda-12.8".into());

    let nvcc = format!("{}/bin/nvcc", cuda_home);

    let st = std::process::Command::new(&nvcc)
        .args([
            "-O3",
            "-arch=sm_120",
            "--compiler-options", "-fPIC",
            "-c", "src/cuda_kernels.cu",
            "-o", obj.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|_| panic!("nvcc not found at {}", nvcc));
    assert!(st.success(), "nvcc failed");

    let st = std::process::Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("ar failed");
    assert!(st.success(), "ar failed");

    println!("cargo:rustc-link-lib=static=kernels");
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search={}/lib64", cuda_home);
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");
}
