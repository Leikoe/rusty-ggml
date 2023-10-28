use std::ffi::{CStr, CString};
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
        let mut p_ggml_ctx: *mut ggml_context = ptr::null_mut();
        let (gguf_ctx, used_mem, mem_size) = unsafe {
            let gguf_params = gg::gguf_init_params {
                no_alloc,
                ctx: &mut p_ggml_ctx,
            };
            let file_name_cstring = CString::new(file_name).unwrap();
            let gguf_ctx = gg::gguf_init_from_file(file_name_cstring.as_ptr(), gguf_params);
            ensure!(!gguf_ctx.is_null(), "GGUF init failed");

            (gguf_ctx, gg::ggml_used_mem(p_ggml_ctx), gg::ggml_get_mem_size(p_ggml_ctx))
        };
        let ggml_ctx = GContext::from_pointer(p_ggml_ctx, mem_size, false)?;

        // TODO: update internal trackers
        // update used memory
        ggml_ctx.with_icontext_infallible(|mut ictx| {
            ictx.context_used = used_mem;
        })?;

        Ok(GGgufContext {
            ggml_ctx,
            gguf_ctx,
        })
    }

    pub fn get_version(&self) -> i32 {
        unsafe {
            gg::gguf_get_version(self.gguf_ctx)
        }
    }

    pub fn get_alignment(&self) -> usize {
        unsafe {
            gg::gguf_get_alignment(self.gguf_ctx)
        }
    }

    pub fn get_data_offset(&self) -> usize {
        unsafe {
            gg::gguf_get_data_offset(self.gguf_ctx)
        }
    }

    pub fn get_n_kv(&self) -> i32 {
        unsafe {
            gg::gguf_get_n_kv(self.gguf_ctx)
        }
    }

    pub fn find_key(&self, name: &str) -> i32 {
        unsafe {
            let name_cstring = CString::new(name).unwrap();
            gg::gguf_find_key(self.gguf_ctx, name_cstring.as_ptr())
        }
    }

    pub fn get_n_tensors(&self) -> i32 {
        unsafe {
            gg::gguf_get_n_tensors(self.gguf_ctx)
        }
    }

    pub fn find_tensor(&self, name: &str) -> i32 {
        unsafe {
            let name_cstring = CString::new(name).unwrap();
            gg::gguf_find_tensor(self.gguf_ctx, name_cstring.as_ptr())
        }
    }

    pub fn get_tensor_name(&self, i: i32) -> &'static str {
        unsafe {
            CStr::from_ptr(gg::gguf_get_tensor_name(self.gguf_ctx, i)).to_str().unwrap()
        }
    }

    pub fn get_ggml_context(&self) -> GContext {
        self.ggml_ctx.clone()
    }
}

impl Drop for GGgufContext {
    fn drop(&mut self) {
        unsafe {
            gg::gguf_free(self.gguf_ctx);
        }
    }
}
