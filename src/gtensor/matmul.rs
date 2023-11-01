use std::ops;

use ggml_sys_bleedingedge as gg;

use crate::{dims::*, validation::*};
use crate::context::GContext;

use super::tensor::*;

// impl<const DIMS: usize> GTensor<DIMS> where Dim<DIMS>: DimValid {}

impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::BitXor<T> for &'a GTensor<DIMS>
    where
        Dim<DIMS>: DimValid,
        GTensor<DIMS>: GMulMatT<DIMS, DIMS>,
{
    type Output = <GTensor<DIMS> as GMulMatT<DIMS, DIMS>>::Output;

    fn bitxor(self, rhs: T) -> Self::Output {
        self.mul_mat(rhs.as_ref())
    }
}

impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::BitXor<T> for GTensor<DIMS>
    where
        Dim<DIMS>: DimValid,
        Self: GMulMatT<DIMS, DIMS>,
{
    type Output = <Self as GMulMatT<DIMS, DIMS>>::Output;

    fn bitxor(self, rhs: T) -> Self::Output {
        &self ^ rhs.as_ref()
    }
}

// FIXME: There should be a better way to do this.
// FIXME: Should be a sealed trait?
/// GGML matrix multiplication
///
/// **Note:: Not intended to be implemented by users.
pub trait GMulMat<const LDIMS: usize, const RDIMS: usize>
    where
        Dim<LDIMS>: DimValid,
        Dim<RDIMS>: DimValid,
{
    type Output;

    /// Matrix multiplication of tensor `A` with tensor `B`.
    /// Returns a new tensor.
    ///
    /// `a.mul_mat(b)` or `a ^ b`
    ///
    /// **Invariants**
    /// 1. `A` and `B` must have the same shape with
    ///    the exception of the second dimension.
    /// 2. The result will have the minimum of the dimensions
    ///    between `A` and `B`.
    ///
    /// **Example** (pseudocode):
    /// ```ignore
    ///
    /// let a =
    ///     [ [1, 1],
    ///       [2, 2] ];
    /// let b = [2, 2];
    /// let expected = [4, 8];
    /// let result = a.mul_mat(b);
    /// assert_eq!(result, expected);
    /// ```
    fn mul_mat<L: AsRef<GTensor<LDIMS>>, R: AsRef<GTensor<RDIMS>>>(&self, lhs: L, rhs: R) -> Self::Output;
}

// FIXME: There should be a better way to do this.
// FIXME: Should be a sealed trait?
/// GGML matrix multiplication
///
/// **Note:: Not intended to be implemented by users.
pub trait GMulMatT<const LDIMS: usize, const RDIMS: usize>
    where
        Dim<LDIMS>: DimValid,
        Dim<RDIMS>: DimValid,
{
    type Output;

    /// Matrix multiplication of tensor `A` with tensor `B`.
    /// Returns a new tensor.
    ///
    /// `a.mul_mat(b)` or `a ^ b`
    ///
    /// **Invariants**
    /// 1. `A` and `B` must have the same shape with
    ///    the exception of the second dimension.
    /// 2. The result will have the minimum of the dimensions
    ///    between `A` and `B`.
    ///
    /// **Example** (pseudocode):
    /// ```ignore
    ///
    /// let a =
    ///     [ [1, 1],
    ///       [2, 2] ];
    /// let b = [2, 2];
    /// let expected = [4, 8];
    /// let result = a.mul_mat(b);
    /// assert_eq!(result, expected);
    /// ```
    fn mul_mat<T: AsRef<GTensor<RDIMS>>>(&self, rhs: T) -> Self::Output;
}

impl<const DIMS: usize> GMulMat<DIMS, DIMS> for GContext
    where
        Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn mul_mat<L: AsRef<GTensor<DIMS>>, R: AsRef<GTensor<DIMS>>>(&self, lhs: L, rhs: R) -> Self::Output {
        let lmd = lhs.as_ref().md.clone();
        let rmd = rhs.as_ref().md.clone();

        let lhs = lhs.as_ref();
        let rhs = rhs.as_ref();
        self.new_binary(lhs, rhs, |ctx, ictx, ltptr, rtptr| {
            if !lmd.can_mul_mat_with(&rmd) {
                Err(GTensorError::InvalidOperation)?;
            }
            let mr =
                GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, lmd.typ, lmd.shape)
                    .fit_or_die()?;
            unsafe { Ok((mr, gg::ggml_mul_mat(ictx.gptr(), ltptr, rtptr))) }
        })
    }
}

impl<const DIMS: usize> GMulMatT<DIMS, DIMS> for GTensor<DIMS>
    where
        Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn mul_mat<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self::Output {
        self.ctx.mul_mat(self, rhs)
    }
}

// This is rather unpleasant.
macro_rules! mk_gmulmatinstances {
    ( $( ($l:literal, $r:literal, $o:literal) ),+ ) => { $(
        impl GMulMat<$l, $r> for GContext {
            type Output = GTensor<$o>;

            fn mul_mat<L: AsRef<GTensor<$l>>, R: AsRef<GTensor<$r>>>(&self, lhs: L, rhs: R) -> Self::Output {
                let lmd = lhs.as_ref().md.clone();
                let rmd = rhs.as_ref().md.clone();

                let lhs = lhs.as_ref();
                let rhs = rhs.as_ref();
                self.new_binary(lhs, rhs, |ctx, ictx, ltptr, rtptr| {
                    if !lmd.can_mul_mat_with(&rmd) {
                        Err(GTensorError::InvalidOperation)?;
                    }
                    let shp = if lmd.shape.len() < rmd.shape.len() {
                        &lmd.shape[..]
                    } else {
                        &rmd.shape[..]
                    };
                    let mr =
                        GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, lmd.typ, shp)
                            .fit_or_die()?;
                    unsafe {
                        let t = gg::ggml_mul_mat(ictx.gptr(), ltptr, rtptr);
                        // FIXME: Horrible hack to pretend mul_mat has the old non-broadcasting behavior.
                        let real_dims = (*t).ne.iter().take_while(|i| **i != 1).collect::<Vec<_>>().len();
                        if real_dims != $o {
                            Err(GTensorError::InvalidOperation)?;
                        }
                        (*t).n_dims = $o;
                        Ok((mr, t))
                    }
                })
            }
        }

        impl GMulMatT<$l, $r> for GTensor<$l> {
            type Output = GTensor<$o>;

            fn mul_mat<R: AsRef<GTensor<$r>>>(&self, rhs: R) -> Self::Output {
                self.ctx.mul_mat(self, rhs)
            }
        }

        impl<'a, 'b> ops::BitXor<&'b GTensor<$r>> for &'a GTensor<$l>
        where
            GTensor<$l>: GMulMatT<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: &'b GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(self, rhs)
            }
        }

        impl<'a> ops::BitXor<GTensor<$r>> for &'a GTensor<$l>
        where
            GTensor<$l>: GMulMatT<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(self, &rhs)
            }
        }

        impl<'a> ops::BitXor<&'a GTensor<$r>> for GTensor<$l>
        where
            GTensor<$l>: GMulMatT<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: &'a GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(&self, rhs)
            }
        }

        impl ops::BitXor<GTensor<$r>> for GTensor<$l>
        where
            GTensor<$l>: GMulMatT<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(&self, &rhs)
            }
        }
    )*};
}

mk_gmulmatinstances!(
    (2, 1, 1),
    (3, 1, 1),
    (1, 2, 1),
    (3, 2, 2),
    (1, 3, 1),
    (2, 3, 2)
);
