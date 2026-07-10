import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.lax import lax as _lax
from jax._src.lax.control_flow import conditionals as _cf

import stablehlo_normalize as normalizer


def f32(*shape):
    return jnp.zeros(shape, jnp.float32)


def i32(*shape):
    return jnp.zeros(shape, jnp.int32)


def bl(*shape):
    return jnp.zeros(shape, jnp.bool_)


CASES = [
    ("identity_vec", lambda x: x, [f32(3)]),
    ("identity_scalar", lambda x: x, [f32()]),
    ("multi_out", lambda x, y: (x, y), [f32(2), i32(3)]),
    ("dup_out", lambda x: (x, x), [f32(2)]),
    ("const_scalar_f32", lambda: jnp.float32(2.0), []),
    ("const_scalar_i32", lambda: jnp.int32(7), []),
    ("const_scalar_bool", lambda: jnp.array(True), []),
    ("const_vec_i32", lambda: jnp.array([1, 2, 3], jnp.int32), []),
    ("const_vec_f32", lambda: jnp.array([1.0, 2.0, 3.0], jnp.float32), []),
    ("const_mat_f32", lambda: jnp.array([[1.0, 2.0], [3.0, 4.0]], jnp.float32), []),
    ("const_splat_i32", lambda: jnp.array([5, 5, 5], jnp.int32), []),
    ("const_splat_f32", lambda: jnp.array([2.0, 2.0], jnp.float32), []),
    ("const_neg_zero", lambda: jnp.array([-0.0, 1.5, -2.25], jnp.float32), []),
    ("const_and_arg", lambda x: (jnp.float32(2.0), x), [f32(2)]),
    ("unary_abs", lax.abs, [f32(3)]),
    ("unary_acos", lax.acos, [f32(3)]),
    ("unary_acosh", lax.acosh, [f32(3)]),
    ("unary_asin", lax.asin, [f32(3)]),
    ("unary_asinh", lax.asinh, [f32(3)]),
    ("unary_atan", lax.atan, [f32(3)]),
    ("unary_atanh", lax.atanh, [f32(3)]),
    ("unary_cbrt", lax.cbrt, [f32(3)]),
    ("unary_ceil", lax.ceil, [f32(3)]),
    ("unary_clz", lax.clz, [i32(3)]),
    ("unary_copy", lambda x: _lax.copy_p.bind(x), [f32(3)]),
    ("unary_cos", lax.cos, [f32(3)]),
    ("unary_cosh", lax.cosh, [f32(3)]),
    ("unary_exp", lax.exp, [f32(3)]),
    ("unary_exp2", lax.exp2, [f32(3)]),
    ("unary_expm1", lax.expm1, [f32(3)]),
    ("unary_floor", lax.floor, [f32(3)]),
    ("unary_integer_pow", lambda x: lax.integer_pow(x, 3), [f32(3)]),
    ("unary_is_finite", lax.is_finite, [f32(3)]),
    ("unary_log", lax.log, [f32(3)]),
    ("unary_log1p", lax.log1p, [f32(3)]),
    ("unary_logistic", lax.logistic, [f32(3)]),
    ("unary_neg", lax.neg, [f32(3)]),
    ("unary_not", lax.bitwise_not, [i32(3)]),
    ("unary_population_count", lax.population_count, [i32(3)]),
    ("unary_round", lax.round, [f32(3)]),
    ("unary_rsqrt", lax.rsqrt, [f32(3)]),
    ("unary_sign", lax.sign, [f32(3)]),
    ("unary_sin", lax.sin, [f32(3)]),
    ("unary_sinh", lax.sinh, [f32(3)]),
    ("unary_sqrt", lax.sqrt, [f32(3)]),
    ("unary_square", lax.square, [f32(3)]),
    ("unary_tan", lax.tan, [f32(3)]),
    ("unary_tanh", lax.tanh, [f32(3)]),
    ("binary_add", lax.add, [f32(3), f32(3)]),
    ("binary_and", lax.bitwise_and, [i32(3), i32(3)]),
    ("binary_atan2", lax.atan2, [f32(3), f32(3)]),
    ("binary_div", lax.div, [f32(3), f32(3)]),
    ("binary_max", lax.max, [f32(3), f32(3)]),
    ("binary_min", lax.min, [f32(3), f32(3)]),
    ("binary_mul", lax.mul, [f32(3), f32(3)]),
    ("binary_mulhi", lax.mulhi, [i32(3), i32(3)]),
    ("binary_nextafter", lax.nextafter, [f32(3), f32(3)]),
    ("binary_or", lax.bitwise_or, [i32(3), i32(3)]),
    ("binary_pow", lax.pow, [f32(3), f32(3)]),
    ("binary_rem", lax.rem, [f32(3), f32(3)]),
    ("binary_shift_left", lax.shift_left, [i32(3), i32(3)]),
    (
        "binary_shift_right_arithmetic",
        lax.shift_right_arithmetic,
        [i32(3), i32(3)],
    ),
    ("binary_shift_right_logical", lax.shift_right_logical, [i32(3), i32(3)]),
    ("binary_sub", lax.sub, [f32(3), f32(3)]),
    ("binary_xor", lax.bitwise_xor, [i32(3), i32(3)]),
    ("compare_eq", lax.eq, [f32(3), f32(3)]),
    ("compare_ne", lax.ne, [f32(3), f32(3)]),
    ("compare_ge", lax.ge, [f32(3), f32(3)]),
    ("compare_gt", lax.gt, [f32(3), f32(3)]),
    ("compare_le", lax.le, [f32(3), f32(3)]),
    ("compare_lt", lax.lt, [f32(3), f32(3)]),
    ("compare_eq_i32", lax.eq, [i32(3), i32(3)]),
    ("compare_eq_bool", lax.eq, [bl(3), bl(3)]),
    ("compare_eq_to", lambda a, b: _lax.eq_to_p.bind(a, b), [f32(3), f32(3)]),
    ("compare_le_to", lambda a, b: _lax.le_to_p.bind(a, b), [f32(3), f32(3)]),
    ("compare_lt_to", lambda a, b: _lax.lt_to_p.bind(a, b), [f32(3), f32(3)]),
    ("clamp", lax.clamp, [f32(3), f32(3), f32(3)]),
    ("select_n2", lambda p, x, y: lax.select_n(p, x, y), [bl(3), f32(3), f32(3)]),
    (
        "select_n3",
        lambda p, x, y, z: lax.select_n(p, x, y, z),
        [i32(3), f32(3), f32(3), f32(3)],
    ),
    ("convert_f32_to_i32", lambda x: lax.convert_element_type(x, jnp.int32), [f32(3)]),
    ("convert_i32_to_f32", lambda x: lax.convert_element_type(x, jnp.float32), [i32(3)]),
    ("convert_bool_to_i32", lambda x: lax.convert_element_type(x, jnp.int32), [bl(3)]),
    ("convert_f32_to_bool", lambda x: lax.convert_element_type(x, jnp.bool_), [f32(3)]),
    ("bitcast_f32_to_i32", lambda x: lax.bitcast_convert_type(x, jnp.int32), [f32(3)]),
    ("optimization_barrier", lambda x: lax.optimization_barrier(x), [f32(3)]),
    (
        "reduce_precision",
        lambda x: lax.reduce_precision(x, exponent_bits=8, mantissa_bits=10),
        [f32(3)],
    ),
    ("tie", lambda x, y: _lax.tie_p.bind(x, y), [f32(3), f32(3)]),
    (
        "empty",
        lambda: _lax.empty_p.bind(
            shape=(3,), dtype=jnp.dtype(jnp.float32), out_sharding=None
        ),
        [],
    ),
    (
        "platform_index",
        lambda: _cf.platform_index_p.bind(platforms=(("cpu",),)),
        [],
    ),
    (
        "shape_broadcast_in_dim",
        lambda x: lax.broadcast_in_dim(x, (2, 3), (1,)),
        [f32(3)],
    ),
    ("shape_concatenate", lambda x, y: lax.concatenate([x, y], 0), [f32(2), f32(3)]),
    ("shape_iota", lambda: lax.broadcasted_iota(jnp.int32, (2, 3), 1), []),
    ("shape_pad", lambda x: lax.pad(x, jnp.float32(0.0), [(1, 2, 0)]), [f32(3)]),
    (
        "shape_pad_interior",
        lambda x: lax.pad(x, jnp.float32(0.0), [(0, 0, 1)]),
        [f32(3)],
    ),
    ("shape_reshape", lambda x: lax.reshape(x, (2, 3)), [f32(6)]),
    ("shape_rev", lambda x: lax.rev(x, [0]), [f32(3)]),
    ("shape_split", lambda x: lax.split(x, [2, 3], 0), [f32(5)]),
    ("shape_squeeze", lambda x: lax.squeeze(x, [0]), [f32(1, 3)]),
    ("shape_stack", lambda x, y: lax.stack([x, y], 0), [f32(3), f32(3)]),
    ("shape_tile", lambda x: lax.tile(x, [2]), [f32(3)]),
    ("shape_transpose", lambda x: lax.transpose(x, (1, 0)), [f32(2, 3)]),
    ("shape_unstack", lambda x: lax.unstack(x, 0), [f32(2, 3)]),
    (
        "reduce_sum",
        lambda x: _lax.reduce_sum_p.bind(x, axes=(0,), out_sharding=None),
        [f32(3)],
    ),
    ("reduce_max", lambda x: _lax.reduce_max_p.bind(x, axes=(0,)), [f32(3)]),
    ("reduce_min", lambda x: _lax.reduce_min_p.bind(x, axes=(0,)), [f32(3)]),
    ("reduce_prod", lambda x: _lax.reduce_prod_p.bind(x, axes=(0,)), [f32(3)]),
    ("reduce_and", lambda x: _lax.reduce_and_p.bind(x, axes=(0,)), [i32(3)]),
    ("reduce_or", lambda x: _lax.reduce_or_p.bind(x, axes=(0,)), [i32(3)]),
    ("reduce_xor", lambda x: _lax.reduce_xor_p.bind(x, axes=(0,)), [i32(3)]),
    ("argmax", lambda x: _lax.argmax(x, 0, jnp.int32), [f32(3)]),
    ("argmin", lambda x: _lax.argmin(x, 0, jnp.int32), [f32(3)]),
    ("cumsum", lambda x: lax.cumsum(x, 0), [f32(3)]),
    ("cumprod", lambda x: lax.cumprod(x, 0), [f32(3)]),
    ("cummax", lambda x: lax.cummax(x, 0), [f32(3)]),
    ("cummin", lambda x: lax.cummin(x, 0), [f32(3)]),
    ("cumlogsumexp", lambda x: lax.cumlogsumexp(x, 0), [f32(3)]),
]


def main():
    cases = []
    for name, fn, args in CASES:
        text = jax.jit(fn).lower(*args).as_text()
        cases.append({"name": name, "text": normalizer.normalize(text)})
    cases.sort(key=lambda c: c["name"])
    out = {"cases": cases}
    json.dump(
        out, sys.stdout, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
