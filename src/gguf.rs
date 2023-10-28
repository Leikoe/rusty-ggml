use std::ffi::c_char;
use std::ptr;

use anyhow::{ensure, Result};
use ggml_sys_bleedingedge as gg;
use ggml_sys_bleedingedge::{ggml_context, gguf_context};

use crate::context::GContext;

pub struct GGgufContext {
        ggml_ctx: GContext,
        gguf_ctx: *mut gguf_context,
}

impl GGgufContext {
        pub fn init_from_file(file_name: &str, no_alloc: bool) -> Result<GGgufContext> {
                let mut ggml_ctx_raw: *mut ggml_context = ptr::null_mut();
                let gguf_ctx = unsafe {
                        let gguf_params = gg::gguf_init_params {
                                no_alloc,
                                ctx: &mut ggml_ctx_raw,
                        };
                        let gguf_ctx = gg::gguf_init_from_file(file_name.as_ptr() as *const c_char, gguf_params);
                        ensure!(!gguf_ctx.is_null(), "GGUF init failed");

                        gguf_ctx
                };

                Ok(GGgufContext {
                        // TODO: update internal trackers
                        ggml_ctx: GContext::from_pointer(ggml_ctx_raw, 0, false)?,
                        gguf_ctx,
                })
        }

        pub fn get_ggml_context(self) -> GContext {
                self.ggml_ctx
        }
}
