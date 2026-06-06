use std::path::PathBuf;

fn main() {
    if std::env::var("CARGO_FEATURE_CUDA").is_err() { return; }

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    let cuda_home = std::env::var("CUDA_HOME").unwrap_or_else(|_| {
        if cfg!(windows) {
            "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8".into()
        } else {
            "/usr/local/cuda-12.8".into()
        }
    });
    let nvcc = if cfg!(windows) {
        format!("{}/bin/nvcc.exe", cuda_home)
    } else {
        format!("{}/bin/nvcc", cuda_home)
    };

    // GPU architecture. Local dev defaults to sm_120 (this box's Blackwell card) for fast
    // builds; shipping/CI sets TINYQ4_CUDA_ARCH=all-major to emit SASS for every major arch
    // (Turing→Blackwell) plus PTX, so the bundled binary runs on any NVIDIA GPU.
    let arch = std::env::var("TINYQ4_CUDA_ARCH").unwrap_or_else(|_| "sm_120".into());

    if cfg!(windows) {
        // nvcc -lib produces kernels.lib directly (no separate archiver / -fPIC on MSVC).
        let lib = out.join("kernels.lib");
        let st = std::process::Command::new(&nvcc)
            .args(["-O3", &format!("-arch={arch}"), "-lib",
                   "src/cuda_kernels.cu", "-o", lib.to_str().unwrap()])
            .status()
            .unwrap_or_else(|_| panic!("nvcc not found at {}", nvcc));
        assert!(st.success(), "nvcc failed");
        println!("cargo:rustc-link-search={}/lib/x64", cuda_home);
    } else {
        let obj = out.join("kernels.o");
        let lib = out.join("libkernels.a");
        let st = std::process::Command::new(&nvcc)
            .args(["-O3", &format!("-arch={arch}"), "--compiler-options", "-fPIC",
                   "-c", "src/cuda_kernels.cu", "-o", obj.to_str().unwrap()])
            .status()
            .unwrap_or_else(|_| panic!("nvcc not found at {}", nvcc));
        assert!(st.success(), "nvcc failed");

        let st = std::process::Command::new("ar")
            .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
            .status()
            .expect("ar failed");
        assert!(st.success(), "ar failed");
        println!("cargo:rustc-link-search={}/lib64", cuda_home);
    }

    println!("cargo:rustc-link-lib=static=kernels");
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rerun-if-changed=src/cuda_kernels.cu");
    println!("cargo:rerun-if-env-changed=TINYQ4_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
