use anyhow::{bail, Result};

use ggml_sys_bleedingedge as gg;

use crate::{
    context::{GContext, GContextError, IContext},
    util::GType,
};

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum GMemoryRequestType {
    #[default]
    Unknown,
    Tensor {
        typ: GType,
        shape: [usize; gg::GGML_MAX_DIMS as usize],
    },
}

impl GMemoryRequestType {
    pub fn new_tensor_request<const DIMS: usize>(typ: GType, shape: [usize; DIMS]) -> Self {
        let shape = shape
            .into_iter()
            .chain(std::iter::repeat(1))
            .take(gg::GGML_MAX_DIMS as usize)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Impossible: Could not convert to array");
        Self::Tensor { typ, shape }
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct GMemoryRequest {
    pub reqtype: GMemoryRequestType,
    pub required_ctx: usize,
    pub required_scratch: usize,
    pub total_required: usize,
    pub available_ctx: usize,
    pub available_scratch: usize,
    pub current_scratch_buffer: Option<usize>,
    pub fits: bool,
}

impl GMemoryRequest {
    pub(crate) fn estimate_tensor_request_ictx(
        ctx: &GContext,
        ictx: &IContext,
        typ: GType,
        shape: impl AsRef<[usize]>,
    ) -> Self {
        let shape = shape.as_ref();
        let padded_shape = shape
            .iter()
            .copied()
            .chain(std::iter::repeat(1))
            .take(gg::GGML_MAX_DIMS as usize)
            .collect::<Vec<_>>()
            .try_into()
            .expect("Impossible: Could not convert to array");
        let reqtype = GMemoryRequestType::Tensor {
            typ,
            shape: padded_shape,
        };
        let elsize = typ.element_sizef() as f64;
        let elcount = shape.iter().map(|i| *i as f64).product::<f64>();
        // 16 is the worst case for alignment but it probably doesn't hurt to be a little
        // bit conservative here.
        let required_ctx = gg::GGML_OBJECT_SIZE + std::mem::size_of::<gg::ggml_tensor>() + 16;
        let required = (elsize * elcount).round() as usize + typ.block_size();
        let used_ctx = unsafe { gg::ggml_used_mem(ictx.gctx.as_ptr()) };
        let available_ctx = ctx.context_size - used_ctx;

        if let Some(bufid) = ictx.current_scratch_buffer {
            let sbuf = &ictx.scratch_buffers[bufid];
            let available_scratch = sbuf.buf.len() - sbuf.used;
            Self {
                reqtype,
                required_ctx,
                required_scratch: required,
                total_required: required + required_ctx,
                available_scratch,
                available_ctx,
                current_scratch_buffer: ictx.current_scratch_buffer,
                fits: ctx.no_alloc
                    || (required_ctx < available_ctx && required < available_scratch),
            }
        } else {
            let total_required = required_ctx + required;
            Self {
                reqtype,
                required_ctx: total_required,
                total_required,
                available_ctx,
                fits: ctx.no_alloc || required_ctx < available_ctx,
                ..Default::default()
            }
        }
    }

    pub fn fit_or_die(self) -> Result<Self> {
        if self.required_ctx <= self.available_ctx
            && self.required_scratch <= self.available_scratch
        {
            Ok(self)
        } else {
            bail!(GContextError::InsufficientMemory(self))
        }
    }
}
