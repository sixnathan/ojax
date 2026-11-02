import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import argparse
import json
import platform

import numpy as np

import jax
import jax.numpy as jnp
from jax import lax, random

PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def f32_2x3():
    return jnp.arange(6, dtype=jnp.float32).reshape(2, 3)


def text_cases():
    k = random.key
    kd = random.key_dtype
    sub = jnp.issubdtype
    return {
        "key_dtype_default": lambda: str(kd()),
        "key_dtype_threefry2x32": lambda: str(kd("threefry2x32")),
        "key_dtype_rbg": lambda: str(kd("rbg")),
        "key_dtype_unsafe_rbg": lambda: str(kd("unsafe_rbg")),
        "key_dtype_bad": lambda: str(kd("bad")),
        "key_dtype_itemsize_fry": lambda: str(kd().itemsize),
        "key_dtype_itemsize_rbg": lambda: str(kd("rbg").itemsize),
        "key_dtype_of_key": lambda: str(k(0).dtype),
        "key_dtype_of_rbg_key": lambda: str(k(0, impl="rbg").dtype),
        "key_dtype_eq_fry": lambda: str(k(0).dtype == kd()),
        "key_dtype_eq_rbg": lambda: str(k(0, impl="rbg").dtype == kd("rbg")),
        "key_dtype_eq_mixed": lambda: str(k(0, impl="rbg").dtype == kd()),
        "key_repr_0": lambda: repr(k(0)),
        "key_repr_42": lambda: repr(k(42)),
        "key_repr_12345": lambda: repr(k(12345)),
        "key_repr_split3": lambda: repr(random.split(k(0), 3)),
        "key_repr_split_2x2": lambda: repr(random.split(k(0), (2, 2))),
        "key_str_split2": lambda: str(random.split(k(0), 2)),
        "key_repr_fold_in": lambda: repr(random.fold_in(k(0), 7)),
        "key_repr_rbg1": lambda: repr(k(1, impl="rbg")),
        "key_repr_urbg1": lambda: repr(k(1, impl="unsafe_rbg")),
        "key_repr_split_rbg": lambda: repr(random.split(k(3, impl="rbg"), 2)),
        "key_repr_split_urbg": lambda: repr(random.split(k(3, impl="unsafe_rbg"), 2)),
        "key_shape_split3": lambda: str(random.split(k(0), 3).shape),
        "typeof_key": lambda: str(jax.typeof(k(0))),
        "typeof_split3": lambda: str(jax.typeof(random.split(k(0), 3))),
        "typeof_split_2x2": lambda: str(jax.typeof(random.split(k(0), (2, 2)))),
        "typeof_rbg": lambda: str(jax.typeof(k(0, impl="rbg"))),
        "typeof_urbg": lambda: str(jax.typeof(k(0, impl="unsafe_rbg"))),
        "typeof_f32_2x3": lambda: str(jax.typeof(f32_2x3())),
        "typeof_weak_float": lambda: str(jax.typeof(jnp.asarray(2.0))),
        "typeof_weak_int": lambda: str(jax.typeof(jnp.asarray(2))),
        "typeof_weak_complex": lambda: str(jax.typeof(jnp.asarray(1j))),
        "typeof_bool": lambda: str(jax.typeof(jnp.asarray(True))),
        "typeof_bf16_empty": lambda: str(jax.typeof(jnp.zeros((0, 2), jnp.bfloat16))),
        "typeof_zeros_2x0x3": lambda: str(jax.typeof(jnp.zeros((2, 0, 3)))),
        "typeof_arange5": lambda: str(jax.typeof(jnp.arange(5))),
        "typeof_weak_product": lambda: str(jax.typeof(jnp.asarray(2.0) * 2)),
        "typeof_strong_sum": lambda: str(jax.typeof(jnp.ones(3) + 1.0)),
        "typeof_int8_vec": lambda: str(jax.typeof(jnp.arange(4, dtype=jnp.int8))),
        "typeof_key_data": lambda: str(jax.typeof(random.key_data(k(0)))),
        "issub_key_prng_key": lambda: str(sub(k(0).dtype, jax.dtypes.prng_key)),
        "issub_key_extended": lambda: str(sub(k(0).dtype, jax.dtypes.extended)),
        "issub_key_generic": lambda: str(sub(k(0).dtype, jnp.generic)),
        "issub_key_number": lambda: str(sub(k(0).dtype, jnp.number)),
        "issub_key_uint32": lambda: str(sub(k(0).dtype, jnp.uint32)),
        "issub_key_same": lambda: str(sub(kd(), kd())),
        "issub_key_rbg_fry": lambda: str(sub(kd("rbg"), kd())),
        "issub_key_rbg_prng_key": lambda: str(sub(kd("rbg"), jax.dtypes.prng_key)),
        "issub_key_urbg_extended": lambda: str(sub(kd("unsafe_rbg"), jax.dtypes.extended)),
        "issub_float32_prng_key": lambda: str(sub(jnp.float32, jax.dtypes.prng_key)),
        "issub_uint32_prng_key": lambda: str(sub(jnp.uint32, jax.dtypes.prng_key)),
        "issub_prng_key_extended": lambda: str(sub(jax.dtypes.prng_key, jax.dtypes.extended)),
        "issub_prng_key_key": lambda: str(sub(jax.dtypes.prng_key, kd())),
        "wrap_repr_rbg": lambda: repr(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="rbg")),
        "wrap_repr_urbg_2x4": lambda: repr(random.wrap_key_data(jnp.arange(8, dtype=jnp.uint32).reshape(2, 4), impl="unsafe_rbg")),
        "wrap_repr_fry": lambda: repr(random.wrap_key_data(jnp.arange(2, dtype=jnp.uint32))),
        "wrap_repr_fry_3x2": lambda: repr(random.wrap_key_data(jnp.arange(6, dtype=jnp.uint32).reshape(3, 2), impl="threefry2x32")),
        "wrap_impl_rbg": lambda: str(random.key_impl(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="rbg"))),
        "wrap_impl_urbg": lambda: str(random.key_impl(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="unsafe_rbg"))),
        "wrap_dtype_urbg": lambda: str(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="unsafe_rbg").dtype),
        "wrap_typeof_rbg": lambda: str(jax.typeof(random.wrap_key_data(jnp.arange(8, dtype=jnp.uint32).reshape(2, 4), impl="rbg"))),
        "wrap_split_urbg": lambda: repr(random.split(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="unsafe_rbg"))),
        "wrap_split_rbg": lambda: repr(random.split(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="rbg"))),
        "wrap_err_ndim": lambda: repr(random.wrap_key_data(jnp.uint32(3), impl="rbg")),
        "wrap_err_int32": lambda: repr(random.wrap_key_data(jnp.arange(4, dtype=jnp.int32), impl="rbg")),
        "wrap_err_float32": lambda: repr(random.wrap_key_data(jnp.arange(4.0, dtype=jnp.float32), impl="unsafe_rbg")),
        "wrap_err_shape_rbg": lambda: repr(random.wrap_key_data(jnp.arange(2, dtype=jnp.uint32), impl="rbg")),
        "wrap_err_shape_fry": lambda: repr(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32))),
        "repr_weak_float": lambda: repr(jnp.asarray(2.0)),
        "repr_weak_int": lambda: repr(jnp.asarray(3)),
        "repr_weak_neg_zero": lambda: repr(jnp.asarray(-0.0)),
        "repr_weak_complex": lambda: repr(jnp.asarray(1.5 + 2j)),
        "repr_bool": lambda: repr(jnp.asarray(True)),
        "repr_strong_float": lambda: repr(jnp.float32(2.0)),
        "repr_weak_product": lambda: repr(jnp.asarray(2.0) * 3),
        "repr_weak_bcast12": lambda: repr(lax.broadcast(jnp.asarray(1.5), (12,))),
        "repr_weak_bcast30": lambda: repr(lax.broadcast(jnp.asarray(1.5), (30,))),
        "repr_weak_bcast_int_2x3": lambda: repr(lax.broadcast(jnp.asarray(2), (2, 3))),
        "repr_weak_bcast_int17": lambda: repr(lax.broadcast(jnp.asarray(123456), (17,))),
        "repr_weak_empty": lambda: repr(lax.broadcast(jnp.asarray(2.0), (0,))),
        "repr_strong_vec": lambda: repr(jnp.asarray([1.0, 2.0])),
        "repr_bf16_special": lambda: repr(jnp.asarray([1.0, -0.0, np.nan, np.inf, 1234567.0, 1e-7, 3.140625], dtype=jnp.bfloat16)),
        "repr_bf16_neg_nan": lambda: repr(jnp.asarray([-np.nan, -np.inf], dtype=jnp.bfloat16)),
        "repr_bf16_wrap": lambda: repr(jnp.arange(40, dtype=jnp.bfloat16).reshape(2, 20) / 3),
        "repr_f8_e4m3fn": lambda: repr(jnp.asarray([1.0, 0.5, -448], dtype=jnp.float8_e4m3fn)),
        "repr_f8_e5m2": lambda: repr(jnp.asarray([1.0, 0.1, -3.0], dtype=jnp.float8_e5m2)),
        "repr_int4": lambda: repr(jnp.asarray([1, -3, 7], dtype=jnp.int4)),
        "repr_uint4_2x2": lambda: repr(jnp.asarray([[1, 15], [10, 0]], dtype=jnp.uint4)),
        "repr_bf16_scalar": lambda: repr(jnp.asarray(2.5, dtype=jnp.bfloat16)),
        "repr_bf16_third": lambda: repr(jnp.asarray(1.0, dtype=jnp.bfloat16) / 3),
        "str_bf16_vec": lambda: str(jnp.asarray([2.5, 1.0], dtype=jnp.bfloat16)),
        "str_bf16_scalar_third": lambda: str(jnp.asarray(1.0, dtype=jnp.bfloat16) / 3),
        "tolist_bf16": lambda: repr(jnp.asarray([2.5, 1.0 / 3], dtype=jnp.bfloat16).tolist()),
        "item_f32_flat1": lambda: repr(f32_2x3().item(1)),
        "item_f32_neg1": lambda: repr(f32_2x3().item(-1)),
        "item_f32_1_2": lambda: repr(f32_2x3().item(1, 2)),
        "item_f32_0_neg1": lambda: repr(f32_2x3().item(0, -1)),
        "item_err_size": lambda: repr(f32_2x3().item()),
        "item_err_flat_oob": lambda: repr(f32_2x3().item(10)),
        "item_err_axis_oob": lambda: repr(f32_2x3().item(2, 0)),
        "item_err_count": lambda: repr(f32_2x3().item(1, 2, 0)),
        "item_f32_tenth": lambda: repr(jnp.asarray(0.1, dtype=jnp.float32).item()),
        "item_weak_float": lambda: repr(jnp.asarray(0.1).item()),
        "item_int": lambda: repr(jnp.asarray(3).item()),
        "item_bool": lambda: repr(jnp.asarray(False).item()),
        "item_complex": lambda: repr(jnp.asarray(1 + 2j).item()),
        "item_complex_neg_zero": lambda: repr(jnp.asarray(complex(1.5, -0.0), dtype=jnp.complex64).item()),
        "item_complex_pure_imag": lambda: repr(jnp.asarray(2.5j, dtype=jnp.complex64).item()),
        "item_uint32_max": lambda: repr(jnp.asarray(4294967295, dtype=jnp.uint32).item()),
        "item_int8_neg": lambda: repr(jnp.asarray(-7, dtype=jnp.int8).item()),
        "item_f32_big": lambda: repr(jnp.asarray(1e20, dtype=jnp.float32).item()),
        "item_f32_small": lambda: repr(jnp.asarray(1e-5, dtype=jnp.float32).item()),
        "item_f32_1e16": lambda: repr(jnp.asarray(1e16, dtype=jnp.float32).item()),
        "item_f32_123456789": lambda: repr(jnp.asarray(123456789.0, dtype=jnp.float32).item()),
        "item_f32_inf": lambda: repr(jnp.asarray(-np.inf, dtype=jnp.float32).item()),
        "item_f32_nan": lambda: repr(jnp.asarray(np.nan, dtype=jnp.float32).item()),
        "item_f32_neg_zero": lambda: repr(jnp.asarray(-0.0, dtype=jnp.float32).item()),
        "item_f16": lambda: repr(jnp.asarray(0.1, dtype=jnp.float16).item()),
        "item_bf16": lambda: repr(jnp.asarray(0.1, dtype=jnp.bfloat16).item()),
        "item_f64": lambda: repr(jnp.asarray(0.1, dtype=jnp.float64).item()),
        "item_int64_big": lambda: repr(jnp.asarray(2**40, dtype=jnp.int64).item()),
        "tolist_f32_2x3": lambda: repr(f32_2x3().tolist()),
        "tolist_int_vec": lambda: repr(jnp.arange(4).tolist()),
        "tolist_bool": lambda: repr(jnp.asarray([True, False]).tolist()),
        "tolist_complex": lambda: repr(jnp.asarray([1 + 2j, -3j, 0.5]).tolist()),
        "tolist_scalar": lambda: repr(jnp.asarray(2.5).tolist()),
        "tolist_empty": lambda: repr(jnp.zeros((0,)).tolist()),
        "tolist_empty_2x0": lambda: repr(jnp.zeros((2, 0)).tolist()),
        "tolist_f32_tenths": lambda: repr((jnp.arange(3, dtype=jnp.float32) / 10).tolist()),
        "tolist_f16": lambda: repr(jnp.asarray([0.1, 2.0], dtype=jnp.float16).tolist()),
        "tolist_uint8": lambda: repr(jnp.asarray([0, 255], dtype=jnp.uint8).tolist()),
        "tobytes_f32_2x3": lambda: f32_2x3().tobytes().hex(),
        "tobytes_int8": lambda: jnp.asarray([-1, 2, 127], dtype=jnp.int8).tobytes().hex(),
        "tobytes_bool": lambda: jnp.asarray([True, False, True]).tobytes().hex(),
        "tobytes_complex64": lambda: jnp.asarray([1 + 2j, -0.5j], dtype=jnp.complex64).tobytes().hex(),
        "tobytes_uint16": lambda: jnp.arange(3, dtype=jnp.uint16).tobytes().hex(),
        "tobytes_scalar": lambda: jnp.asarray(1.5).tobytes().hex(),
        "tobytes_empty": lambda: jnp.zeros((0, 3)).tobytes().hex(),
        "itemsize_f32": lambda: str(f32_2x3().itemsize),
        "itemsize_bool": lambda: str(jnp.asarray(True).itemsize),
        "itemsize_int8": lambda: str(jnp.arange(3, dtype=jnp.int8).itemsize),
        "itemsize_complex64": lambda: str(jnp.asarray(1j, dtype=jnp.complex64).itemsize),
        "itemsize_f16": lambda: str(jnp.zeros(2, jnp.float16).itemsize),
        "itemsize_bf16": lambda: str(jnp.zeros(2, jnp.bfloat16).itemsize),
        "itemsize_default_int": lambda: str(jnp.arange(3).itemsize),
        "itemsize_weak_float": lambda: str(jnp.asarray(1.0).itemsize),
        "nbytes_f32": lambda: str(f32_2x3().nbytes),
        "nbytes_bool": lambda: str(jnp.asarray([True, False, True]).nbytes),
        "nbytes_complex64": lambda: str(jnp.zeros((2, 2), jnp.complex64).nbytes),
        "nbytes_empty": lambda: str(jnp.zeros((0, 3)).nbytes),
        "nbytes_scalar": lambda: str(jnp.asarray(1.0).nbytes),
        "nbytes_default_int": lambda: str(jnp.arange(5).nbytes),
        "flat_err": lambda: repr(f32_2x3().flat),
        "view_err_0d": lambda: repr(jnp.asarray(2.0, dtype=jnp.float32).view(jnp.int16)),
        "view_err_divisor": lambda: repr(jnp.arange(3, dtype=jnp.int8).view(jnp.int32)),
        "view_err_complex": lambda: repr(f32_2x3().view(jnp.complex64)),
    }


def array_cases():
    i8 = jnp.arange(4, dtype=jnp.int8)
    c64 = jnp.asarray([1 + 2j, 3 - 4j], dtype=jnp.complex64)
    return {
        "view_f32_int32": lambda: f32_2x3().view(jnp.int32),
        "view_f32_uint32": lambda: f32_2x3().view(jnp.uint32),
        "view_f32_uint8": lambda: f32_2x3().view(jnp.uint8),
        "view_f32_uint16": lambda: f32_2x3().view(jnp.uint16),
        "view_f32_float16": lambda: f32_2x3().view(jnp.float16),
        "view_f32_int16": lambda: f32_2x3().view(jnp.int16),
        "view_f32_bool": lambda: f32_2x3().view(jnp.bool_),
        "view_f32_same": lambda: f32_2x3().view(jnp.float32),
        "view_int8_bool": lambda: i8.view(jnp.bool_),
        "view_int8_int16": lambda: i8.view(jnp.int16),
        "view_int8_int32": lambda: i8.view(jnp.int32),
        "view_int8_float32": lambda: jnp.asarray([0, 0, -128, 63], dtype=jnp.int8).view(jnp.float32),
        "view_uint8_float32": lambda: jnp.asarray([0, 0, 128, 63, 0, 0, 0, 64], dtype=jnp.uint8).view(jnp.float32),
        "view_bool_uint8": lambda: jnp.asarray([True, False, True, False]).view(jnp.uint8),
        "view_bool_int16": lambda: jnp.asarray([True, False, True, False]).view(jnp.int16),
        "view_c64_float32": lambda: c64.view(jnp.float32),
        "view_c64_int32": lambda: c64.view(jnp.int32),
        "view_c64_uint8": lambda: c64.view(jnp.uint8),
        "view_f32_c64": lambda: jnp.arange(4, dtype=jnp.float32).view(jnp.complex64),
        "view_int32_c64": lambda: jnp.arange(4, dtype=jnp.int32).view(jnp.complex64),
        "view_f32_c64_special": lambda: jnp.asarray([1.0, np.inf, -0.0, np.nan], dtype=jnp.float32).view(jnp.complex64),
        "view_scalar_int32": lambda: jnp.asarray(2.0, dtype=jnp.float32).view(jnp.int32),
        "view_scalar_uint32_weak": lambda: jnp.asarray(2.0).view(jnp.uint32),
        "view_f64_int64": lambda: jnp.arange(3, dtype=jnp.float64).view(jnp.int64),
        "view_bf16_uint16": lambda: jnp.arange(4, dtype=jnp.bfloat16).view(jnp.uint16),
        "view_bf16_int8": lambda: jnp.arange(4, dtype=jnp.bfloat16).view(jnp.int8),
        "view_f16_bf16": lambda: jnp.asarray([1.0, -2.5], dtype=jnp.float16).view(jnp.bfloat16),
        "view_3d_int16_int8": lambda: jnp.arange(12, dtype=jnp.int16).reshape(1, 2, 6).view(jnp.int8),
        "view_3d_int8_int16": lambda: jnp.arange(8, dtype=jnp.int8).reshape(1, 2, 4).view(jnp.int16),
        "byteswap_f32": lambda: f32_2x3().byteswap(),
        "byteswap_uint16": lambda: jnp.arange(4, dtype=jnp.uint16).byteswap(),
        "byteswap_int32": lambda: jnp.arange(5, dtype=jnp.int32).byteswap(),
        "byteswap_bool": lambda: jnp.asarray([True, False]).byteswap(),
        "byteswap_int8": lambda: jnp.asarray([-1, 2, 3], dtype=jnp.int8).byteswap(),
        "byteswap_c64": lambda: c64.byteswap(),
        "byteswap_f16": lambda: jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.float16).byteswap(),
        "byteswap_bf16": lambda: jnp.asarray([1.0, -2.0, 0.5], dtype=jnp.bfloat16).byteswap(),
        "byteswap_f64": lambda: jnp.asarray([1.0, -2.0], dtype=jnp.float64).byteswap(),
        "byteswap_scalar": lambda: jnp.asarray(7, dtype=jnp.int32).byteswap(),
        "byteswap_twice": lambda: jnp.arange(5, dtype=jnp.int32).byteswap().byteswap(),
        "wrap_bits_rbg": lambda: random.bits(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="rbg"), (3,)),
        "wrap_bits_urbg": lambda: random.bits(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="unsafe_rbg"), (3,)),
        "wrap_key_data_rbg": lambda: random.key_data(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32), impl="rbg")),
        "wrap_uniform_urbg": lambda: random.uniform(random.wrap_key_data(jnp.arange(4, dtype=jnp.uint32) + 9, impl="unsafe_rbg"), (4,)),
        "key_data_split3": lambda: random.key_data(random.split(random.key(0), 3)),
    }


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    rows = []
    for name, f in text_cases().items():
        row = {"id": name, "x64": x64}
        try:
            row["text"] = f()
        except Exception as e:
            row["error_class"] = type(e).__name__
            row["message"] = str(e)
        rows.append(row)
    for name, f in array_cases().items():
        row = {"id": name, "x64": x64}
        try:
            a = f()
            h = np.asarray(a)
            row["dtype"] = str(h.dtype)
            row["shape"] = list(h.shape)
            row["hex"] = h.tobytes().hex()
            row["repr"] = repr(a)
        except Exception as e:
            row["error_class"] = type(e).__name__
            row["message"] = str(e)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "keys_arrays"))
    args = parser.parse_args()
    if platform.python_version() != PY_VERSION or jax.__version__ != JAX_VERSION:
        raise SystemExit("needs python " + PY_VERSION + " and jax " + JAX_VERSION)
    rows = run(False) + run(True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cases.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
