use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    if env::var("CARGO_FEATURE_CUDA").is_ok() {
        build_cuda();
    }
    if env::var_os("CARGO_FEATURE_VULKAN").is_some() {
        build_vulkan_shader();
    }
}

fn build_cuda() {
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
    // builds; shipping/CI sets QUARTZ_CUDA_ARCH=all-major to emit SASS for every major arch
    // (Turing→Blackwell) plus PTX, so the bundled binary runs on any NVIDIA GPU.
    let arch = std::env::var("QUARTZ_CUDA_ARCH").unwrap_or_else(|_| "sm_120".into());

    if cfg!(windows) {
        // nvcc -lib produces kernels.lib directly (no separate archiver / -fPIC on MSVC).
        let lib = out.join("kernels.lib");
        let st = std::process::Command::new(&nvcc)
            .args([
                "-O3",
                &format!("-arch={arch}"),
                "-lib",
                "src/cuda_kernels.cu",
                "-o",
                lib.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|_| panic!("nvcc not found at {}", nvcc));
        assert!(st.success(), "nvcc failed");
        println!("cargo:rustc-link-search={}/lib/x64", cuda_home);
    } else {
        let obj = out.join("kernels.o");
        let lib = out.join("libkernels.a");
        let st = std::process::Command::new(&nvcc)
            .args([
                "-O3",
                &format!("-arch={arch}"),
                "--compiler-options",
                "-fPIC",
                "-c",
                "src/cuda_kernels.cu",
                "-o",
                obj.to_str().unwrap(),
            ])
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
    println!("cargo:rerun-if-env-changed=QUARTZ_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}

fn build_vulkan_shader() {
    println!("cargo:rerun-if-changed=shaders/fp16_gemm.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_conv2d.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_attention.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_im2col.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_groupnorm_silu.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_residual_add.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_gemm_heads.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_merge_heads.comp");
    println!("cargo:rerun-if-changed=shaders/fp16_geglu.comp");
    println!("cargo:rerun-if-changed=shaders/f32_elementwise.comp");
    println!("cargo:rerun-if-changed=shaders/f32_channel_rmsnorm.comp");
    println!("cargo:rerun-if-changed=shaders/f32_f16_linear.comp");
    println!("cargo:rerun-if-changed=shaders/f32_layernorm.comp");
    println!("cargo:rerun-if-changed=shaders/f32_rmsnorm.comp");
    println!("cargo:rerun-if-changed=shaders/f32_rope.comp");
    println!("cargo:rerun-if-changed=shaders/f32_attention.comp");
    println!("cargo:rerun-if-changed=shaders/f32_patch_layout.comp");
    println!("cargo:rerun-if-changed=shaders/f32_wan_head_modulate.comp");
    println!("cargo:rerun-if-changed=shaders/f32_f16_conv3d.comp");
    let glslc = find_glslc().unwrap_or_else(|| {
        panic!("Vulkan feature requires GLSLC or an Android NDK containing shader-tools/glslc")
    });
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo supplies OUT_DIR"));
    for (source, output) in [
        ("shaders/fp16_gemm.comp", "fp16_gemm.spv"),
        ("shaders/fp16_conv2d.comp", "fp16_conv2d.spv"),
        ("shaders/fp16_attention.comp", "fp16_attention.spv"),
        ("shaders/fp16_im2col.comp", "fp16_im2col.spv"),
        (
            "shaders/fp16_groupnorm_silu.comp",
            "fp16_groupnorm_silu.spv",
        ),
        ("shaders/fp16_residual_add.comp", "fp16_residual_add.spv"),
        ("shaders/fp16_gemm_heads.comp", "fp16_gemm_heads.spv"),
        ("shaders/fp16_merge_heads.comp", "fp16_merge_heads.spv"),
        ("shaders/fp16_geglu.comp", "fp16_geglu.spv"),
        ("shaders/f32_elementwise.comp", "f32_elementwise.spv"),
        (
            "shaders/f32_channel_rmsnorm.comp",
            "f32_channel_rmsnorm.spv",
        ),
        ("shaders/f32_f16_linear.comp", "f32_f16_linear.spv"),
        ("shaders/f32_layernorm.comp", "f32_layernorm.spv"),
        ("shaders/f32_rmsnorm.comp", "f32_rmsnorm.spv"),
        ("shaders/f32_rope.comp", "f32_rope.spv"),
        ("shaders/f32_attention.comp", "f32_attention.spv"),
        ("shaders/f32_patch_layout.comp", "f32_patch_layout.spv"),
        (
            "shaders/f32_wan_head_modulate.comp",
            "f32_wan_head_modulate.spv",
        ),
        ("shaders/f32_f16_conv3d.comp", "f32_f16_conv3d.spv"),
    ] {
        let status = Command::new(&glslc)
            .args(["--target-env=vulkan1.1", "-O", source, "-o"])
            .arg(out.join(output))
            .status()
            .unwrap_or_else(|error| panic!("cannot run {}: {error}", glslc.display()));
        assert!(status.success(), "GLSL compiler rejected {source}");
    }
}

fn find_glslc() -> Option<PathBuf> {
    if let Some(path) = env::var_os("GLSLC").map(PathBuf::from) {
        if path.is_file() {
            return Some(path);
        }
    }
    let ndk = env::var_os("ANDROID_NDK_HOME")
        .map(PathBuf::from)
        .or_else(latest_sdk_ndk)
        .or_else(|| latest_child(&PathBuf::from("/usr/lib/android-sdk/ndk")))?;
    let path = ndk.join("shader-tools/linux-x86_64/glslc");
    path.is_file().then_some(path)
}

fn latest_sdk_ndk() -> Option<PathBuf> {
    let sdk = env::var_os("ANDROID_SDK_ROOT")
        .or_else(|| env::var_os("ANDROID_HOME"))
        .map(PathBuf::from)?;
    latest_child(&sdk.join("ndk"))
}

fn latest_child(parent: &std::path::Path) -> Option<PathBuf> {
    let mut children = fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect::<Vec<_>>();
    children.sort();
    children.pop()
}
