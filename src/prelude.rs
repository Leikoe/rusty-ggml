/// Re-export of low level GGML binding.
pub use ggml_sys_bleedingedge as ggml_sys;

pub use crate::{context::*, dims::*, gtensor::*, map_binop, map_unop, quantize::*, util::*};

/// Alias for one dimensional tensors.
pub type GTensor1 = GTensor<1>;

/// Alias for two dimensional tensors.
pub type GTensor2 = GTensor<2>;

/// Alias for three dimensional tensors.
pub type GTensor3 = GTensor<3>;

/// Alias for four dimensional tensors.
pub type GTensor4 = GTensor<4>;
