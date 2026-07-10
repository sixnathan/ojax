import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
from jax import lax
from jax._src.lax import lax as _lax

import stablehlo_normalize as normalizer


def f32(*shape):
    return jnp.zeros(shape, jnp.float32)


def i32(*shape):
    return jnp.zeros(shape, jnp.int32)


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
