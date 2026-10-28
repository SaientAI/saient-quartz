//! Experimental LoRA training. Head-only real-GGUF path is an intermediate
//! milestone, not q/v transformer training.
pub mod head;
pub mod lora;
pub mod loss;
pub mod ops;
pub mod qv;
pub mod qv_cli;
pub mod qv_gradient;
pub mod qv_io;
