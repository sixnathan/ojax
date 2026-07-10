import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["JAX_NUM_GENERATED_CASES"] = "10"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import functools
import hashlib
import json
import platform
import shutil
import sys
import zlib

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import dtypes as jax_dtypes

import emit_common as ec

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"


def preflight():
    if platform.python_version() != PY_VERSION:
        raise SystemExit("python " + platform.python_version() + " != " + PY_VERSION)
    if jax.__version__ != JAX_VERSION:
        raise SystemExit("jax " + jax.__version__ + " != " + JAX_VERSION)


def load_tolerances():
    with open(os.path.join(ROOT, "tools", "tolerances.json"), encoding="utf-8") as fh:
        return json.load(fh)


TOLERANCES = load_tolerances()


def resolve_tol(dtype_name):
    if ec.is_exact_dtype(dtype_name):
        return "exact", {"atol": 0, "rtol": 0}
    tol = TOLERANCES["default"][dtype_name]
    return "allclose", {"atol": tol, "rtol": tol}


def np_dtype(dtype_name):
    return np.dtype(dtype_name)


def _rand_dtype(rand, shape, dtype, scale=1.0, post=None):
    shape = tuple(shape)
    vals = np.asarray(scale * rand(*shape)).astype(dtype)
    if post is not None:
        vals = np.asarray(post(vals), dtype).astype(dtype)
    return vals.reshape(shape)


def rand_default(rng, shape, dtype):
    return _rand_dtype(rng.randn, shape, dtype, scale=3.0)


def rand_small(rng, shape, dtype):
    return _rand_dtype(rng.randn, shape, dtype, scale=1e-3)


def rand_positive(rng, shape, dtype):
    return _rand_dtype(rng.rand, shape, dtype, scale=2.0, post=lambda x: x + 1)


def rand_nonzero(rng, shape, dtype):
    post = lambda x: np.where(x == 0, np.array(1, dtype=x.dtype), x)
    return _rand_dtype(rng.randn, shape, dtype, scale=3.0, post=post)


def rand_int(rng, shape, dtype):
    high = np.iinfo(dtype).max
    return rng.randint(0, high, size=tuple(shape), dtype=dtype)


def rand_int_small(rng, shape, dtype):
    return rng.randint(0, 10, size=tuple(shape), dtype=dtype)


def rand_int_small_nz(rng, shape, dtype):
    return rng.randint(1, 10, size=tuple(shape), dtype=dtype)


def rand_int_float(rng, shape, dtype):
    return rng.randint(0, 10, size=tuple(shape)).astype(dtype)


def rand_seg(rng, shape, dtype):
    return rng.randint(0, 4, size=tuple(shape), dtype=dtype)


def rand_bool(rng, shape, dtype):
    return np.asarray(rng.rand(*tuple(shape)) < 0.5, dtype=dtype)


def rand_index_unique(rng, shape, dtype):
    n = int(np.prod(shape)) if len(shape) else 1
    return rng.choice(10, size=n, replace=False).astype(dtype).reshape(tuple(shape))


def rand_uniform(rng, shape, dtype):
    return _rand_dtype(rng.rand, shape, dtype, scale=1.0)


def rand_gt_one(rng, shape, dtype):
    return _rand_dtype(rng.rand, shape, dtype, scale=2.0, post=lambda x: x + 1.5)


def rand_poly_order(rng, shape, dtype):
    return rng.randint(1, 3, size=tuple(shape)).astype(dtype)


def rand_sorted(rng, shape, dtype):
    return np.sort(_rand_dtype(rng.randn, shape, dtype, scale=3.0), axis=-1)


def rand_sorted_desc(rng, shape, dtype):
    return np.sort(_rand_dtype(rng.randn, shape, dtype, scale=3.0), axis=-1)[
        ..., ::-1
    ]


def rand_simplex(rng, shape, dtype):
    x = rng.rand(*tuple(shape)).astype(dtype) + np.array(0.1, dtype=dtype)
    return (x / x.sum(axis=0, keepdims=True)).astype(dtype)


def rand_complex(rng, shape, dtype):
    shape = tuple(shape)
    re = 3.0 * rng.randn(*shape)
    im = 3.0 * rng.randn(*shape)
    return (re + 1j * im).astype(dtype).reshape(shape)


def rand_complex_small(rng, shape, dtype):
    shape = tuple(shape)
    re = 0.5 * rng.randn(*shape)
    im = 0.5 * rng.randn(*shape)
    return (re + 1j * im).astype(dtype).reshape(shape)


RNG_FACTORIES = {
    "rand_complex": rand_complex,
    "rand_complex_small": rand_complex_small,
    "rand_default": rand_default,
    "rand_small": rand_small,
    "rand_positive": rand_positive,
    "rand_nonzero": rand_nonzero,
    "rand_int": rand_int,
    "rand_int_small": rand_int_small,
    "rand_int_small_nz": rand_int_small_nz,
    "rand_int_float": rand_int_float,
    "rand_seg": rand_seg,
    "rand_bool": rand_bool,
    "rand_index_unique": rand_index_unique,
    "rand_uniform": rand_uniform,
    "rand_gt_one": rand_gt_one,
    "rand_poly_order": rand_poly_order,
    "rand_sorted": rand_sorted,
    "rand_sorted_desc": rand_sorted_desc,
    "rand_simplex": rand_simplex,
}


def draw(rng_name, seed, shape, dtype_name):
    rng = np.random.RandomState(seed)
    return RNG_FACTORIES[rng_name](rng, shape, np_dtype(dtype_name))


def weak_scalar(dtype_name, seed):
    rng = np.random.RandomState(seed)
    if dtype_name.startswith("complex"):
        return complex(rng.randn(), rng.randn())
    if dtype_name.startswith("float") or dtype_name.startswith("bfloat"):
        return float(rng.randn())
    if dtype_name == "bool":
        return bool(rng.rand() < 0.5)
    return int(rng.randint(0, 10))


LAX = jax.lax
from jax._src.lax import lax as LAX_INTERNAL
from jax._src.lax import windowed_reductions as WR


def _unary(fn):
    return lambda params: (lambda x: fn(x))


def _binary(fn):
    return lambda params: (lambda a, b: fn(a, b))


def lax_convert(params):
    dt = np.dtype(params["new_dtype"])
    return lambda x: LAX.convert_element_type(x, dt)


def lax_broadcast_in_dim(params):
    shape = tuple(params["shape"])
    dims = tuple(params["dims"])
    return lambda x: LAX.broadcast_in_dim(x, shape, dims)


def lax_reshape(params):
    new_sizes = tuple(params["new_sizes"])
    return lambda x: LAX.reshape(x, new_sizes)


def lax_reduce_sum(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_sum(x, axes=axes)


def lax_dot_general(params):
    dn = params["dimension_numbers"]
    dimension_numbers = (
        (tuple(dn[0][0]), tuple(dn[0][1])),
        (tuple(dn[1][0]), tuple(dn[1][1])),
    )
    return lambda a, b: LAX.dot_general(a, b, dimension_numbers)


def lax_select_n(params):
    return lambda which, *cases: LAX.select_n(which, *cases)


def lax_integer_pow(params):
    y = int(params["y"])
    return lambda x: LAX.integer_pow(x, y)


def lax_concatenate(params):
    dimension = int(params["dimension"])
    return lambda *xs: LAX.concatenate(list(xs), dimension)


def lax_pad(params):
    cfg = [tuple(t) for t in params["padding_config"]]
    return lambda operand, pv: LAX.pad(operand, pv, cfg)


def lax_rev(params):
    dims = tuple(params["dimensions"])
    return lambda x: LAX.rev(x, dims)


def lax_split(params):
    sizes = tuple(params["sizes"])
    axis = int(params["axis"])
    return lambda x: LAX.split(x, sizes, axis)


def lax_squeeze(params):
    dims = tuple(params["dimensions"])
    return lambda x: LAX.squeeze(x, dims)


def lax_stack(params):
    axis = int(params["axis"])
    return lambda *xs: LAX.stack(list(xs), axis)


def lax_tile(params):
    reps = tuple(params["reps"])
    return lambda x: LAX.tile(x, reps)


def lax_transpose(params):
    perm = tuple(params["permutation"])
    return lambda x: LAX.transpose(x, perm)


def lax_unstack(params):
    axis = int(params["axis"])
    return lambda x: LAX.unstack(x, axis)


def lax_reduce_max(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_max(x, axes=axes)


def lax_reduce_min(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_min(x, axes=axes)


def lax_reduce_prod(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_prod(x, axes=axes)


def lax_reduce_and(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_and(x, axes=axes)


def lax_reduce_or(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_or(x, axes=axes)


def lax_reduce_xor(params):
    axes = tuple(params["axes"])
    return lambda x: LAX.reduce_xor(x, axes=axes)


def lax_argmax(params):
    axis = int(params["axis"])
    index_dtype = np.dtype(params["index_dtype"])
    return lambda x: LAX.argmax(x, axis, index_dtype)


def lax_argmin(params):
    axis = int(params["axis"])
    index_dtype = np.dtype(params["index_dtype"])
    return lambda x: LAX.argmin(x, axis, index_dtype)


def lax_reduce(params):
    dimensions = tuple(params["dimensions"])
    return lambda operand, init: LAX.reduce(operand, init, LAX.add, dimensions)


def lax_clamp(params):
    return lambda mn, x, mx: LAX.clamp(mn, x, mx)


def lax_bitcast_convert_type(params):
    dt = np.dtype(params["new_dtype"])
    return lambda x: LAX.bitcast_convert_type(x, dt)


def lax_iota(params):
    dt = np.dtype(params["dtype"])
    shape = tuple(params["shape"])
    dimension = int(params["dimension"])
    return lambda: LAX.broadcasted_iota(dt, shape, dimension)


def lax_optimization_barrier(params):
    return lambda *xs: LAX.optimization_barrier(tuple(xs))


def lax_reduce_precision(params):
    exponent_bits = int(params["exponent_bits"])
    mantissa_bits = int(params["mantissa_bits"])
    return lambda x: LAX.reduce_precision(
        x, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits
    )


def lax_sort(params):
    dimension = int(params["dimension"])
    is_stable = bool(params["is_stable"])
    num_keys = int(params["num_keys"])
    return lambda x: LAX.sort(
        x, dimension=dimension, is_stable=is_stable, num_keys=num_keys
    )


def lax_tie(params):
    return lambda x, y: LAX_INTERNAL.tie_p.bind(x, y)


def lax_top_k(params):
    k = int(params["k"])
    axis = int(params["axis"])
    return lambda x: LAX.top_k(x, k, axis=axis)


def lax_slice(params):
    start = tuple(params["start_indices"])
    limit = tuple(params["limit_indices"])
    strides = params.get("strides")
    strides = None if strides is None else tuple(strides)
    return lambda x: LAX.slice(x, start, limit, strides)


def lax_dynamic_slice(params):
    slice_sizes = tuple(params["slice_sizes"])
    return lambda operand, *starts: LAX.dynamic_slice(
        operand, list(starts), slice_sizes
    )


def lax_dynamic_update_slice(params):
    return lambda operand, update, *starts: LAX.dynamic_update_slice(
        operand, update, list(starts)
    )


def _gather_dnums(params):
    return LAX.GatherDimensionNumbers(
        offset_dims=tuple(params["offset_dims"]),
        collapsed_slice_dims=tuple(params["collapsed_slice_dims"]),
        start_index_map=tuple(params["start_index_map"]),
        operand_batching_dims=tuple(params.get("operand_batching_dims", [])),
        start_indices_batching_dims=tuple(
            params.get("start_indices_batching_dims", [])
        ),
    )


def _scatter_dnums(params):
    return LAX.ScatterDimensionNumbers(
        update_window_dims=tuple(params["update_window_dims"]),
        inserted_window_dims=tuple(params["inserted_window_dims"]),
        scatter_dims_to_operand_dims=tuple(params["scatter_dims_to_operand_dims"]),
        operand_batching_dims=tuple(params.get("operand_batching_dims", [])),
        scatter_indices_batching_dims=tuple(
            params.get("scatter_indices_batching_dims", [])
        ),
    )


MODE = LAX.GatherScatterMode.CLIP


def lax_gather(params):
    dnums = _gather_dnums(params)
    slice_sizes = tuple(params["slice_sizes"])
    return lambda operand, indices: LAX.gather(
        operand, indices, dnums, slice_sizes, mode=MODE
    )


def lax_scatter(params):
    dnums = _scatter_dnums(params)
    unique = bool(params.get("unique_indices", True))
    return lambda operand, indices, updates: LAX.scatter(
        operand, indices, updates, dnums, unique_indices=unique, mode=MODE
    )


def lax_scatter_add(params):
    dnums = _scatter_dnums(params)
    return lambda operand, indices, updates: LAX.scatter_add(
        operand, indices, updates, dnums, mode=MODE
    )


def lax_scatter_sub(params):
    dnums = _scatter_dnums(params)
    return lambda operand, indices, updates: LAX.scatter_sub(
        operand, indices, updates, dnums, mode=MODE
    )


def lax_scatter_mul(params):
    dnums = _scatter_dnums(params)
    unique = bool(params.get("unique_indices", True))
    return lambda operand, indices, updates: LAX.scatter_mul(
        operand, indices, updates, dnums, unique_indices=unique, mode=MODE
    )


def lax_scatter_min(params):
    dnums = _scatter_dnums(params)
    return lambda operand, indices, updates: LAX.scatter_min(
        operand, indices, updates, dnums, mode=MODE
    )


def lax_scatter_max(params):
    dnums = _scatter_dnums(params)
    return lambda operand, indices, updates: LAX.scatter_max(
        operand, indices, updates, dnums, mode=MODE
    )


def lax_conv_general_dilated(params):
    dn = LAX.ConvDimensionNumbers(
        lhs_spec=tuple(params["dimension_numbers"][0]),
        rhs_spec=tuple(params["dimension_numbers"][1]),
        out_spec=tuple(params["dimension_numbers"][2]),
    )
    ws = tuple(params["window_strides"])
    padding = tuple((int(lo), int(hi)) for lo, hi in params["padding"])
    ld = tuple(params["lhs_dilation"])
    rd = tuple(params["rhs_dilation"])
    fgc = int(params["feature_group_count"])
    bgc = int(params["batch_group_count"])
    return lambda lhs, rhs: LAX.conv_general_dilated(
        lhs,
        rhs,
        ws,
        padding,
        lhs_dilation=ld,
        rhs_dilation=rd,
        dimension_numbers=dn,
        feature_group_count=fgc,
        batch_group_count=bgc,
    )


def _window_geometry(params):
    wd = tuple(params["window_dimensions"])
    ws = tuple(params["window_strides"])
    padding = tuple((int(lo), int(hi)) for lo, hi in params["padding"])
    bd = tuple(params["base_dilation"])
    wdil = tuple(params["window_dilation"])
    return wd, ws, padding, bd, wdil


def _window_select(params):
    return LAX_INTERNAL.ge_p if params["select"] == "ge" else LAX_INTERNAL.le_p


def lax_reduce_window_sum(params):
    wd, ws, padding, bd, wdil = _window_geometry(params)
    return lambda x: LAX.reduce_window(
        x, LAX_INTERNAL._get_sum_identity(x.dtype), LAX.add, wd, ws, padding, bd, wdil
    )


def lax_reduce_window_max(params):
    wd, ws, padding, bd, wdil = _window_geometry(params)
    return lambda x: LAX.reduce_window(
        x, LAX_INTERNAL._get_max_identity(x.dtype), LAX.max, wd, ws, padding, bd, wdil
    )


def lax_reduce_window_min(params):
    wd, ws, padding, bd, wdil = _window_geometry(params)
    return lambda x: LAX.reduce_window(
        x, LAX_INTERNAL._get_min_identity(x.dtype), LAX.min, wd, ws, padding, bd, wdil
    )


def lax_reduce_window(params):
    wd, ws, padding, bd, wdil = _window_geometry(params)
    return lambda x, init: LAX.reduce_window(
        x, init, LAX.mul, wd, ws, padding, bd, wdil
    )


def lax_select_and_gather_add(params):
    sel = _window_select(params)
    wd, ws, padding, bd, wdil = _window_geometry(params)
    return lambda t, x: WR._select_and_gather_add(
        t, x, sel, wd, ws, padding, bd, wdil
    )


def lax_select_and_scatter_add(params):
    sel = _window_select(params)
    wd, ws, padding, _bd, _wdil = _window_geometry(params)
    return lambda source, operand: WR._select_and_scatter_add(
        source, operand, sel, wd, ws, padding
    )


LAX_BUILDERS = {
    "neg": _unary(LAX.neg),
    "sin": _unary(LAX.sin),
    "cos": _unary(LAX.cos),
    "exp": _unary(LAX.exp),
    "log": _unary(LAX.log),
    "tanh": _unary(LAX.tanh),
    "abs": _unary(LAX.abs),
    "sign": _unary(LAX.sign),
    "add": _binary(LAX.add),
    "sub": _binary(LAX.sub),
    "mul": _binary(LAX.mul),
    "div": _binary(LAX.div),
    "max": _binary(LAX.max),
    "min": _binary(LAX.min),
    "pow": _binary(LAX.pow),
    "eq": _binary(LAX.eq),
    "lt": _binary(LAX.lt),
    "gt": _binary(LAX.gt),
    "select_n": lax_select_n,
    "convert_element_type": lax_convert,
    "broadcast_in_dim": lax_broadcast_in_dim,
    "reshape": lax_reshape,
    "reduce_sum": lax_reduce_sum,
    "dot_general": lax_dot_general,
    "acos": _unary(LAX.acos),
    "acosh": _unary(LAX.acosh),
    "asin": _unary(LAX.asin),
    "asinh": _unary(LAX.asinh),
    "atan": _unary(LAX.atan),
    "atanh": _unary(LAX.atanh),
    "cbrt": _unary(LAX.cbrt),
    "ceil": _unary(LAX.ceil),
    "clz": _unary(LAX.clz),
    "copy": _unary(LAX.copy_p.bind),
    "conj": _unary(LAX.conj),
    "real": _unary(LAX.real),
    "imag": _unary(LAX.imag),
    "complex": _binary(LAX.complex),
    "cosh": _unary(LAX.cosh),
    "exp2": _unary(LAX.exp2),
    "expm1": _unary(LAX.expm1),
    "floor": _unary(LAX.floor),
    "integer_pow": lax_integer_pow,
    "is_finite": _unary(LAX.is_finite),
    "log1p": _unary(LAX.log1p),
    "logistic": _unary(LAX.logistic),
    "not": _unary(LAX.bitwise_not),
    "population_count": _unary(LAX.population_count),
    "round": _unary(LAX.round),
    "rsqrt": _unary(LAX.rsqrt),
    "sinh": _unary(LAX.sinh),
    "sqrt": _unary(LAX.sqrt),
    "square": _unary(LAX.square),
    "tan": _unary(LAX.tan),
    "and": _binary(LAX.and_p.bind),
    "atan2": _binary(LAX.atan2),
    "eq_to": _binary(LAX.eq_to_p.bind),
    "ge": _binary(LAX.ge),
    "le": _binary(LAX.le),
    "le_to": _binary(LAX.le_to_p.bind),
    "lt_to": _binary(LAX.lt_to_p.bind),
    "mulhi": _binary(LAX.mulhi),
    "ne": _binary(LAX.ne),
    "nextafter": _binary(LAX.nextafter),
    "or": _binary(LAX.or_p.bind),
    "rem": _binary(LAX.rem),
    "shift_left": _binary(LAX.shift_left),
    "shift_right_arithmetic": _binary(LAX.shift_right_arithmetic),
    "shift_right_logical": _binary(LAX.shift_right_logical),
    "xor": _binary(LAX.xor_p.bind),
    "concatenate": lax_concatenate,
    "pad": lax_pad,
    "rev": lax_rev,
    "split": lax_split,
    "squeeze": lax_squeeze,
    "stack": lax_stack,
    "tile": lax_tile,
    "transpose": lax_transpose,
    "unstack": lax_unstack,
    "reduce_max": lax_reduce_max,
    "reduce_min": lax_reduce_min,
    "reduce_prod": lax_reduce_prod,
    "reduce_and": lax_reduce_and,
    "reduce_or": lax_reduce_or,
    "reduce_xor": lax_reduce_xor,
    "argmax": lax_argmax,
    "argmin": lax_argmin,
    "reduce": lax_reduce,
    "clamp": lax_clamp,
    "bitcast_convert_type": lax_bitcast_convert_type,
    "iota": lax_iota,
    "optimization_barrier": lax_optimization_barrier,
    "reduce_precision": lax_reduce_precision,
    "sort": lax_sort,
    "tie": lax_tie,
    "top_k": lax_top_k,
    "slice": lax_slice,
    "dynamic_slice": lax_dynamic_slice,
    "dynamic_update_slice": lax_dynamic_update_slice,
    "gather": lax_gather,
    "scatter": lax_scatter,
    "scatter_add": lax_scatter_add,
    "scatter_sub": lax_scatter_sub,
    "scatter_mul": lax_scatter_mul,
    "scatter_min": lax_scatter_min,
    "scatter_max": lax_scatter_max,
    "conv_general_dilated": lax_conv_general_dilated,
    "reduce_window_sum": lax_reduce_window_sum,
    "reduce_window_max": lax_reduce_window_max,
    "reduce_window_min": lax_reduce_window_min,
    "reduce_window": lax_reduce_window,
    "select_and_gather_add": lax_select_and_gather_add,
    "select_and_scatter_add": lax_select_and_scatter_add,
    "bessel_i0e": _unary(LAX.bessel_i0e),
    "bessel_i1e": _unary(LAX.bessel_i1e),
    "digamma": _unary(LAX.digamma),
    "erf": _unary(LAX.erf),
    "erf_inv": _unary(LAX.erf_inv),
    "erfc": _unary(LAX.erfc),
    "lgamma": _unary(LAX.lgamma),
    "igamma": _binary(LAX.igamma),
    "igamma_grad_a": _binary(LAX.igamma_grad_a),
    "igammac": _binary(LAX.igammac),
    "polygamma": _binary(LAX.polygamma),
    "zeta": _binary(LAX.zeta),
    "regularized_incomplete_beta": lambda params: (
        lambda a, b, x: LAX.betainc(a, b, x)
    ),
}


def _opt_axes(params, key):
    v = params.get(key)
    return None if v is None else tuple(v)


def np_transpose(params):
    axes = _opt_axes(params, "axes")
    return lambda x: jnp.transpose(x, axes)


def np_permute_dims(params):
    axes = tuple(params["axes"])
    return lambda x: jnp.permute_dims(x, axes)


def np_matrix_transpose(params):
    return lambda x: jnp.matrix_transpose(x)


def np_flip(params):
    axis = _opt_axes(params, "axis")
    return lambda x: jnp.flip(x, axis)


def np_fliplr(params):
    return lambda x: jnp.fliplr(x)


def np_flipud(params):
    return lambda x: jnp.flipud(x)


def np_rot90(params):
    k = int(params["k"])
    axes = tuple(params["axes"])
    return lambda x: jnp.rot90(x, k=k, axes=axes)


def np_reshape(params):
    shape = tuple(params["shape"])
    return lambda x: jnp.reshape(x, shape)


def np_ravel(params):
    return lambda x: jnp.ravel(x)


def np_trunc(params):
    return lambda x: jnp.trunc(x)


def np_fmax(params):
    return lambda a, b: jnp.fmax(a, b)


def np_fmin(params):
    return lambda a, b: jnp.fmin(a, b)


def np_diff(params):
    n = int(params["n"])
    axis = int(params["axis"])
    return lambda x: jnp.diff(x, n=n, axis=axis)


def np_ediff1d(params):
    return lambda x: jnp.ediff1d(x)


def np_angle(params):
    deg = bool(params["deg"])
    return lambda x: jnp.angle(x, deg=deg)


def np_conjugate(params):
    return lambda x: jnp.conjugate(x)


def np_imag(params):
    return lambda x: jnp.imag(x)


def np_real(params):
    return lambda x: jnp.real(x)


def np_convolve(params):
    mode = params["mode"]
    return lambda a, b: jnp.convolve(a, b, mode=mode)


def np_correlate(params):
    mode = params["mode"]
    return lambda a, b: jnp.correlate(a, b, mode=mode)


def np_iscomplex(params):
    return lambda x: jnp.iscomplex(x)


def np_isreal(params):
    return lambda x: jnp.isreal(x)


def np_allclose(params):
    return lambda a, b: jnp.allclose(a, b)


def np_isclose(params):
    return lambda a, b: jnp.isclose(a, b)


def np_clip(params):
    mn = params.get("min")
    mx = params.get("max")
    return lambda x: jnp.clip(x, mn, mx)


def np_round(params):
    d = int(params["decimals"])
    return lambda x: jnp.round(x, decimals=d)


def np_around(params):
    d = int(params["decimals"])
    return lambda x: jnp.around(x, decimals=d)


def np_nan_to_num(params):
    return lambda x: jnp.nan_to_num(x)


def np_expand_dims(params):
    axis = tuple(params["axis"])
    return lambda x: jnp.expand_dims(x, axis)


def np_squeeze(params):
    axis = params.get("axis")
    axis = None if axis is None else tuple(axis)
    return lambda x: jnp.squeeze(x, axis)


def np_swapaxes(params):
    a1 = int(params["axis1"])
    a2 = int(params["axis2"])
    return lambda x: jnp.swapaxes(x, a1, a2)


def np_moveaxis(params):
    source = tuple(params["source"])
    destination = tuple(params["destination"])
    return lambda x: jnp.moveaxis(x, source, destination)


def np_broadcast_to(params):
    shape = tuple(params["shape"])
    return lambda x: jnp.broadcast_to(x, shape)


def np_broadcast_arrays(params):
    return lambda *xs: jnp.broadcast_arrays(*xs)


def np_resize(params):
    new_shape = tuple(params["new_shape"])
    return lambda x: jnp.resize(x, new_shape)


def np_unravel_index(params):
    shape = tuple(params["shape"])
    return lambda x: jnp.unravel_index(x, shape)


def np_unwrap(params):
    axis = int(params.get("axis", -1))
    return lambda x: jnp.unwrap(x, axis=axis)


def np_where(params):
    return lambda c, x, y: jnp.where(c, x, y)


def np_select(params):
    k = int(params["n"])
    default = params.get("default", 0)
    return lambda *xs: jnp.select(list(xs[:k]), list(xs[k:]), default)


def _ios(params):
    idx = params.get("indices")
    if idx is not None:
        return list(idx)
    return int(params["sections"])


def np_split(params):
    axis = int(params.get("axis", 0))
    ios = _ios(params)
    return lambda x: jnp.split(x, ios, axis=axis)


def np_array_split(params):
    axis = int(params.get("axis", 0))
    ios = _ios(params)
    return lambda x: jnp.array_split(x, ios, axis=axis)


def np_vsplit(params):
    ios = _ios(params)
    return lambda x: jnp.vsplit(x, ios)


def np_hsplit(params):
    ios = _ios(params)
    return lambda x: jnp.hsplit(x, ios)


def np_dsplit(params):
    ios = _ios(params)
    return lambda x: jnp.dsplit(x, ios)


def np_astype(params):
    dt = np.dtype(params["dtype"])
    return lambda x: jnp.astype(x, dt)


def np_copy(params):
    return lambda x: jnp.copy(x)


def np_atleast_1d(params):
    return lambda x: jnp.atleast_1d(x)


def np_atleast_2d(params):
    return lambda x: jnp.atleast_2d(x)


def np_atleast_3d(params):
    return lambda x: jnp.atleast_3d(x)


def np_concatenate(params):
    axis = int(params.get("axis", 0))
    return lambda *xs: jnp.concatenate(list(xs), axis=axis)


def np_concat(params):
    axis = int(params.get("axis", 0))
    return lambda *xs: jnp.concat(list(xs), axis=axis)


def np_stack(params):
    axis = int(params.get("axis", 0))
    return lambda *xs: jnp.stack(list(xs), axis=axis)


def np_unstack(params):
    axis = int(params.get("axis", 0))
    return lambda x: jnp.unstack(x, axis=axis)


def np_vstack(params):
    return lambda *xs: jnp.vstack(list(xs))


def np_hstack(params):
    return lambda *xs: jnp.hstack(list(xs))


def np_dstack(params):
    return lambda *xs: jnp.dstack(list(xs))


def np_column_stack(params):
    return lambda *xs: jnp.column_stack(list(xs))


def np_tile(params):
    reps = tuple(params["reps"])
    return lambda x: jnp.tile(x, reps)


def np_pad(params):
    pad_width = [tuple(t) for t in params["pad_width"]]
    cval = params.get("constant_values", 0)
    return lambda x: jnp.pad(x, pad_width, constant_values=cval)


def np_i0(params):
    return lambda x: jnp.i0(x)


def np_array_equal(params):
    equal_nan = bool(params.get("equal_nan", False))
    return lambda a, b: jnp.array_equal(a, b, equal_nan=equal_nan)


def np_array_equiv(params):
    return lambda a, b: jnp.array_equiv(a, b)


def np_arange(params):
    start = params.get("start", 0)
    step = params.get("step", 1)
    stop = params["stop"]
    dt = np.dtype(params["dtype"])
    return lambda: jnp.arange(start, stop, step, dtype=dt)


def np_eye(params):
    n = int(params["n"])
    m = params.get("m")
    k = int(params.get("k", 0))
    dt = np.dtype(params["dtype"])
    return lambda: jnp.eye(n, m, k=k, dtype=dt)


def np_identity(params):
    n = int(params["n"])
    dt = np.dtype(params["dtype"])
    return lambda: jnp.identity(n, dtype=dt)


def np_indices(params):
    dims = tuple(params["dimensions"])
    dt = np.dtype(params["dtype"])
    return lambda: jnp.indices(dims, dtype=dt)


def np_meshgrid(params):
    indexing = params.get("indexing", "xy")
    sparse = bool(params.get("sparse", False))
    return lambda *xs: jnp.meshgrid(*xs, indexing=indexing, sparse=sparse)


def np_ix_(params):
    return lambda *xs: jnp.ix_(*xs)


def _opt_dtype(params):
    d = params.get("dtype")
    return np.dtype(d) if d is not None else None


def _opt_shape(params):
    s = params.get("shape")
    return tuple(s) if s is not None else None


def np_zeros(params):
    return lambda: jnp.zeros(tuple(params["shape"]), dtype=_opt_dtype(params))


def np_ones(params):
    return lambda: jnp.ones(tuple(params["shape"]), dtype=_opt_dtype(params))


def np_empty(params):
    return lambda: jnp.empty(tuple(params["shape"]), dtype=_opt_dtype(params))


def np_full(params):
    return lambda: jnp.full(
        tuple(params["shape"]), params["fill_value"], dtype=_opt_dtype(params)
    )


def np_zeros_like(params):
    return lambda x: jnp.zeros_like(
        x, dtype=_opt_dtype(params), shape=_opt_shape(params)
    )


def np_ones_like(params):
    return lambda x: jnp.ones_like(
        x, dtype=_opt_dtype(params), shape=_opt_shape(params)
    )


def np_empty_like(params):
    return lambda x: jnp.empty_like(
        x, dtype=_opt_dtype(params), shape=_opt_shape(params)
    )


def np_full_like(params):
    return lambda x: jnp.full_like(
        x, params["fill_value"], dtype=_opt_dtype(params), shape=_opt_shape(params)
    )


def np_linspace(params):
    num = int(params.get("num", 50))
    endpoint = bool(params.get("endpoint", True))
    return lambda: jnp.linspace(
        params["start"], params["stop"], num, endpoint=endpoint,
        dtype=_opt_dtype(params)
    )


def np_logspace(params):
    num = int(params.get("num", 50))
    endpoint = bool(params.get("endpoint", True))
    base = params.get("base", 10.0)
    return lambda: jnp.logspace(
        params["start"], params["stop"], num, endpoint=endpoint, base=base,
        dtype=_opt_dtype(params)
    )


def np_geomspace(params):
    num = int(params.get("num", 50))
    endpoint = bool(params.get("endpoint", True))
    return lambda: jnp.geomspace(
        params["start"], params["stop"], num, endpoint=endpoint,
        dtype=_opt_dtype(params)
    )


def np_array(params):
    ndmin = int(params.get("ndmin", 0))
    return lambda x: jnp.array(x, dtype=_opt_dtype(params), ndmin=ndmin)


def np_asarray(params):
    return lambda x: jnp.asarray(x, dtype=_opt_dtype(params))


def _scalar_type(name):
    def build(params):
        return lambda x: getattr(jnp, name)(x)

    return build


def np_append(params):
    axis = params.get("axis")
    return lambda a, b: jnp.append(a, b, axis=axis)


def np_argmax(params):
    axis = params.get("axis")
    keepdims = bool(params.get("keepdims", False))
    return lambda x: jnp.argmax(x, axis=axis, keepdims=keepdims)


def np_cross(params):
    axisa = int(params.get("axisa", -1))
    axisb = int(params.get("axisb", -1))
    axisc = int(params.get("axisc", -1))
    axis = params.get("axis")
    return lambda a, b: jnp.cross(a, b, axisa=axisa, axisb=axisb, axisc=axisc, axis=axis)


def np_diag(params):
    k = int(params.get("k", 0))
    return lambda x: jnp.diag(x, k=k)


def np_diagflat(params):
    k = int(params.get("k", 0))
    return lambda x: jnp.diagflat(x, k=k)


def np_diagonal(params):
    offset = int(params.get("offset", 0))
    axis1 = int(params.get("axis1", 0))
    axis2 = int(params.get("axis2", 1))
    return lambda x: jnp.diagonal(x, offset=offset, axis1=axis1, axis2=axis2)


def np_diag_indices(params):
    n = int(params["n"])
    ndim = int(params.get("ndim", 2))
    return lambda: jnp.diag_indices(n, ndim=ndim)


def np_diag_indices_from(params):
    return lambda x: jnp.diag_indices_from(x)


def np_kron(params):
    return lambda a, b: jnp.kron(a, b)


def np_repeat(params):
    repeats = int(params["repeats"])
    axis = params.get("axis")
    return lambda x: jnp.repeat(x, repeats, axis=axis)


def np_trace(params):
    offset = int(params.get("offset", 0))
    axis1 = int(params.get("axis1", 0))
    axis2 = int(params.get("axis2", 1))
    return lambda x: jnp.trace(x, offset=offset, axis1=axis1, axis2=axis2)


def np_trapezoid(params):
    dx = float(params.get("dx", 1.0))
    axis = int(params.get("axis", -1))
    has_x = bool(params.get("has_x", False))
    if has_x:
        return lambda y, x: jnp.trapezoid(y, x=x, axis=axis)
    return lambda y: jnp.trapezoid(y, dx=dx, axis=axis)


def np_tri(params):
    n = int(params["n"])
    m = params.get("m")
    k = int(params.get("k", 0))
    dt = np.dtype(params["dtype"])
    return lambda: jnp.tri(n, m, k=k, dtype=dt)


def np_tril(params):
    k = int(params.get("k", 0))
    return lambda x: jnp.tril(x, k=k)


def np_triu(params):
    k = int(params.get("k", 0))
    return lambda x: jnp.triu(x, k=k)


def np_vander(params):
    n = params.get("N")
    increasing = bool(params.get("increasing", False))
    return lambda x: jnp.vander(x, N=n, increasing=increasing)


def np_argmin(params):
    axis = params.get("axis")
    keepdims = bool(params.get("keepdims", False))
    return lambda x: jnp.argmin(x, axis=axis, keepdims=keepdims)


def np_nanargmax(params):
    axis = params.get("axis")
    keepdims = bool(params.get("keepdims", False))
    return lambda x: jnp.nanargmax(x, axis=axis, keepdims=keepdims)


def np_nanargmin(params):
    axis = params.get("axis")
    keepdims = bool(params.get("keepdims", False))
    return lambda x: jnp.nanargmin(x, axis=axis, keepdims=keepdims)


def np_roll(params):
    shift = params["shift"]
    axis = params.get("axis")
    return lambda x: jnp.roll(x, shift, axis=axis)


def np_rollaxis(params):
    axis = int(params["axis"])
    start = int(params.get("start", 0))
    return lambda x: jnp.rollaxis(x, axis, start=start)


def np_gcd(params):
    return lambda a, b: jnp.gcd(a, b)


def np_lcm(params):
    return lambda a, b: jnp.lcm(a, b)


def np_searchsorted(params):
    side = params.get("side", "left")
    return lambda a, v: jnp.searchsorted(a, v, side=side)


def np_digitize(params):
    right = bool(params.get("right", False))
    return lambda x, bins: jnp.digitize(x, bins, right=right)


def np_cov(params):
    rowvar = bool(params.get("rowvar", True))
    bias = bool(params.get("bias", False))
    ddof = params.get("ddof")
    return lambda *xs: jnp.cov(
        xs[0], y=(xs[1] if len(xs) > 1 else None), rowvar=rowvar, bias=bias, ddof=ddof
    )


def np_corrcoef(params):
    rowvar = bool(params.get("rowvar", True))
    return lambda *xs: jnp.corrcoef(
        xs[0], y=(xs[1] if len(xs) > 1 else None), rowvar=rowvar
    )


def np_take(params):
    axis = params.get("axis")
    mode = params.get("mode")
    return lambda a, ind: jnp.take(a, ind, axis=axis, mode=mode)


def np_take_along_axis(params):
    axis = params.get("axis", -1)
    return lambda a, ind: jnp.take_along_axis(a, ind, axis=axis)


def np_put(params):
    mode = params.get("mode")
    return lambda a, ind, v: jnp.put(a, ind, v, mode=mode, inplace=False)


def np_put_along_axis(params):
    axis = params.get("axis")
    return lambda arr, ind, v: jnp.put_along_axis(
        arr, ind, v, axis, inplace=False
    )


NUMPY_BUILDERS = {
    "take": np_take,
    "take_along_axis": np_take_along_axis,
    "put": np_put,
    "put_along_axis": np_put_along_axis,
    "argmin": np_argmin,
    "nanargmax": np_nanargmax,
    "nanargmin": np_nanargmin,
    "roll": np_roll,
    "rollaxis": np_rollaxis,
    "gcd": np_gcd,
    "lcm": np_lcm,
    "searchsorted": np_searchsorted,
    "digitize": np_digitize,
    "cov": np_cov,
    "corrcoef": np_corrcoef,
    "astype": np_astype,
    "copy": np_copy,
    "atleast_1d": np_atleast_1d,
    "atleast_2d": np_atleast_2d,
    "atleast_3d": np_atleast_3d,
    "concatenate": np_concatenate,
    "concat": np_concat,
    "stack": np_stack,
    "unstack": np_unstack,
    "vstack": np_vstack,
    "hstack": np_hstack,
    "dstack": np_dstack,
    "column_stack": np_column_stack,
    "tile": np_tile,
    "pad": np_pad,
    "i0": np_i0,
    "array_equal": np_array_equal,
    "array_equiv": np_array_equiv,
    "arange": np_arange,
    "eye": np_eye,
    "identity": np_identity,
    "indices": np_indices,
    "meshgrid": np_meshgrid,
    "ix_": np_ix_,
    "zeros": np_zeros,
    "ones": np_ones,
    "empty": np_empty,
    "full": np_full,
    "zeros_like": np_zeros_like,
    "ones_like": np_ones_like,
    "empty_like": np_empty_like,
    "full_like": np_full_like,
    "linspace": np_linspace,
    "logspace": np_logspace,
    "geomspace": np_geomspace,
    "array": np_array,
    "asarray": np_asarray,
    "transpose": np_transpose,
    "permute_dims": np_permute_dims,
    "matrix_transpose": np_matrix_transpose,
    "flip": np_flip,
    "fliplr": np_fliplr,
    "flipud": np_flipud,
    "rot90": np_rot90,
    "reshape": np_reshape,
    "ravel": np_ravel,
    "trunc": np_trunc,
    "fmax": np_fmax,
    "fmin": np_fmin,
    "diff": np_diff,
    "ediff1d": np_ediff1d,
    "angle": np_angle,
    "convolve": np_convolve,
    "correlate": np_correlate,
    "iscomplex": np_iscomplex,
    "isreal": np_isreal,
    "conjugate": np_conjugate,
    "imag": np_imag,
    "real": np_real,
    "allclose": np_allclose,
    "isclose": np_isclose,
    "clip": np_clip,
    "round": np_round,
    "around": np_around,
    "nan_to_num": np_nan_to_num,
    "expand_dims": np_expand_dims,
    "squeeze": np_squeeze,
    "swapaxes": np_swapaxes,
    "moveaxis": np_moveaxis,
    "broadcast_to": np_broadcast_to,
    "broadcast_arrays": np_broadcast_arrays,
    "resize": np_resize,
    "unravel_index": np_unravel_index,
    "unwrap": np_unwrap,
    "where": np_where,
    "select": np_select,
    "split": np_split,
    "array_split": np_array_split,
    "vsplit": np_vsplit,
    "hsplit": np_hsplit,
    "dsplit": np_dsplit,
    "append": np_append,
    "argmax": np_argmax,
    "cross": np_cross,
    "diag": np_diag,
    "diagflat": np_diagflat,
    "diagonal": np_diagonal,
    "diag_indices": np_diag_indices,
    "diag_indices_from": np_diag_indices_from,
    "kron": np_kron,
    "repeat": np_repeat,
    "trace": np_trace,
    "trapezoid": np_trapezoid,
    "tri": np_tri,
    "tril": np_tril,
    "triu": np_triu,
    "vander": np_vander,
}

for _sname in ["bool_", "int32", "int64", "float32", "float64"]:
    NUMPY_BUILDERS[_sname] = _scalar_type(_sname)


_UFUNC_UNARY = [
    "negative",
    "positive",
    "sign",
    "fabs",
    "floor",
    "ceil",
    "exp",
    "expm1",
    "log",
    "log1p",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "arcsinh",
    "arccosh",
    "tanh",
    "arctanh",
    "sqrt",
    "cbrt",
    "bitwise_not",
    "bitwise_invert",
    "invert",
    "logical_not",
    "spacing",
    "abs",
    "absolute",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "deg2rad",
    "degrees",
    "exp2",
    "isfinite",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "log10",
    "log2",
    "rad2deg",
    "radians",
    "reciprocal",
    "rint",
    "signbit",
    "sinc",
    "square",
]

_UFUNC_BINARY = [
    "add",
    "subtract",
    "multiply",
    "maximum",
    "minimum",
    "bitwise_and",
    "bitwise_or",
    "bitwise_xor",
    "left_shift",
    "bitwise_left_shift",
    "logical_and",
    "logical_or",
    "logical_xor",
    "equal",
    "not_equal",
    "greater",
    "greater_equal",
    "arctan2",
    "float_power",
    "nextafter",
    "atan2",
    "bitwise_right_shift",
    "copysign",
    "divide",
    "floor_divide",
    "fmod",
    "heaviside",
    "hypot",
    "less",
    "less_equal",
    "logaddexp",
    "logaddexp2",
    "mod",
    "power",
    "pow",
    "remainder",
    "right_shift",
    "true_divide",
]

for _name in _UFUNC_UNARY:
    NUMPY_BUILDERS[_name] = _unary(getattr(jnp, _name))
for _name in _UFUNC_BINARY:
    NUMPY_BUILDERS[_name] = _binary(getattr(jnp, _name))


def np_divmod(params):
    return lambda a, b: jnp.divmod(a, b)


def np_modf(params):
    return lambda x: jnp.modf(x)


NUMPY_BUILDERS["divmod"] = np_divmod
NUMPY_BUILDERS["modf"] = np_modf


def np_dot(params):
    return lambda a, b: jnp.dot(a, b)


def np_matmul(params):
    return lambda a, b: jnp.matmul(a, b)


def np_matvec(params):
    return lambda a, b: jnp.matvec(a, b)


def np_vecmat(params):
    return lambda a, b: jnp.vecmat(a, b)


def np_vdot(params):
    return lambda a, b: jnp.vdot(a, b)


def np_vecdot(params):
    axis = int(params.get("axis", -1))
    return lambda a, b: jnp.vecdot(a, b, axis=axis)


def np_inner(params):
    return lambda a, b: jnp.inner(a, b)


def np_outer(params):
    return lambda a, b: jnp.outer(a, b)


def np_tensordot(params):
    axes = params["axes"]
    if isinstance(axes, list):
        axes = (tuple(axes[0]), tuple(axes[1]))
    return lambda a, b: jnp.tensordot(a, b, axes=axes)


def np_einsum(params):
    subscripts = params["subscripts"]
    return lambda *xs: jnp.einsum(subscripts, *xs)


NUMPY_BUILDERS["dot"] = np_dot
NUMPY_BUILDERS["matmul"] = np_matmul
NUMPY_BUILDERS["matvec"] = np_matvec
NUMPY_BUILDERS["vecmat"] = np_vecmat
NUMPY_BUILDERS["vdot"] = np_vdot
NUMPY_BUILDERS["vecdot"] = np_vecdot
NUMPY_BUILDERS["inner"] = np_inner
NUMPY_BUILDERS["outer"] = np_outer
NUMPY_BUILDERS["tensordot"] = np_tensordot
NUMPY_BUILDERS["einsum"] = np_einsum


def _red_axis(params):
    ax = params.get("axis")
    return None if ax is None else tuple(ax)


_REDUCTION_SIMPLE = [
    "sum",
    "prod",
    "max",
    "min",
    "amax",
    "amin",
    "all",
    "any",
    "mean",
    "ptp",
    "count_nonzero",
    "nansum",
    "nanprod",
    "nanmax",
    "nanmin",
    "nanmean",
]

_REDUCTION_DDOF = ["var", "std", "nanvar", "nanstd"]


def _reduction_simple(name):
    fn = getattr(jnp, name)

    def build(params):
        ax = _red_axis(params)
        kd = params.get("keepdims", False)
        return lambda x: fn(x, axis=ax, keepdims=kd)

    return build


def _reduction_ddof(name):
    fn = getattr(jnp, name)

    def build(params):
        ax = _red_axis(params)
        kd = params.get("keepdims", False)
        ddof = params.get("ddof", 0)
        return lambda x: fn(x, axis=ax, keepdims=kd, ddof=ddof)

    return build


for _name in _REDUCTION_SIMPLE:
    NUMPY_BUILDERS[_name] = _reduction_simple(_name)
for _name in _REDUCTION_DDOF:
    NUMPY_BUILDERS[_name] = _reduction_ddof(_name)


def np_cumsum(params):
    ax = params.get("axis")
    return lambda x: jnp.cumsum(x, axis=ax)


def np_average(params):
    ax = _red_axis(params)
    kd = params.get("keepdims", False)

    def build(*inputs):
        if len(inputs) == 2:
            return jnp.average(inputs[0], axis=ax, weights=inputs[1], keepdims=kd)
        return jnp.average(inputs[0], axis=ax, keepdims=kd)

    return build


NUMPY_BUILDERS["cumsum"] = np_cumsum
NUMPY_BUILDERS["average"] = np_average


def np_blackman(params):
    m = int(params["M"])
    return lambda: jnp.blackman(m)


def np_bartlett(params):
    m = int(params["M"])
    return lambda: jnp.bartlett(m)


def np_hamming(params):
    m = int(params["M"])
    return lambda: jnp.hamming(m)


def np_hanning(params):
    m = int(params["M"])
    return lambda: jnp.hanning(m)


def np_kaiser(params):
    m = int(params["M"])
    beta = float(params["beta"])
    return lambda: jnp.kaiser(m, beta)


def np_sort(params):
    ax = params.get("axis", -1)
    stable = params.get("stable", True)
    descending = params.get("descending", False)
    return lambda x: jnp.sort(x, axis=ax, stable=stable, descending=descending)


def np_argsort(params):
    ax = params.get("axis", -1)
    stable = params.get("stable", True)
    descending = params.get("descending", False)
    return lambda x: jnp.argsort(
        x, axis=ax, stable=stable, descending=descending
    )


def np_lexsort(params):
    ax = int(params.get("axis", -1))
    return lambda *xs: jnp.lexsort(list(xs), axis=ax)


def np_partition(params):
    kth = int(params["kth"])
    ax = int(params.get("axis", -1))
    return lambda x: jnp.partition(x, kth, axis=ax)


def np_isin(params):
    invert = params.get("invert", False)
    return lambda a, b: jnp.isin(a, b, invert=invert)


NUMPY_BUILDERS["blackman"] = np_blackman
NUMPY_BUILDERS["bartlett"] = np_bartlett
NUMPY_BUILDERS["hamming"] = np_hamming
NUMPY_BUILDERS["hanning"] = np_hanning
NUMPY_BUILDERS["kaiser"] = np_kaiser
NUMPY_BUILDERS["sort"] = np_sort
NUMPY_BUILDERS["argsort"] = np_argsort
NUMPY_BUILDERS["lexsort"] = np_lexsort
NUMPY_BUILDERS["partition"] = np_partition
NUMPY_BUILDERS["isin"] = np_isin


def _poly_binary(name):
    fn = getattr(jnp, name)

    def build(params):
        return lambda a, b: fn(a, b)

    return build


def _poly_unary(name):
    fn = getattr(jnp, name)

    def build(params):
        return lambda a: fn(a)

    return build


def _polyint(params):
    m = params.get("m", 1)
    return lambda p: jnp.polyint(p, m=m)


def _polyder(params):
    m = params.get("m", 1)
    return lambda p: jnp.polyder(p, m=m)


for _name in ["polyval", "polyadd", "polysub", "polymul"]:
    NUMPY_BUILDERS[_name] = _poly_binary(_name)
NUMPY_BUILDERS["poly"] = _poly_unary("poly")
NUMPY_BUILDERS["polyint"] = _polyint
NUMPY_BUILDERS["polyder"] = _polyder


def _cum_simple(name):
    fn = getattr(jnp, name)

    def build(params):
        ax = params.get("axis")
        return lambda x: fn(x, axis=ax)

    return build


def _cum_api(name):
    fn = getattr(jnp, name)

    def build(params):
        ax = params.get("axis")
        ii = params.get("include_initial", False)
        return lambda x: fn(x, axis=ax, include_initial=ii)

    return build


def _median(name):
    fn = getattr(jnp, name)

    def build(params):
        ax = params.get("axis")
        kd = params.get("keepdims", False)
        return lambda x: fn(x, axis=ax, keepdims=kd)

    return build


def _quantile(name):
    fn = getattr(jnp, name)

    def build(params):
        q = jnp.array(params["q"])
        ax = params.get("axis")
        kd = params.get("keepdims", False)
        method = params.get("method", "linear")
        return lambda x: fn(x, q, axis=ax, method=method, keepdims=kd)

    return build


for _name in ["cumprod", "nancumsum", "nancumprod"]:
    NUMPY_BUILDERS[_name] = _cum_simple(_name)
for _name in ["cumulative_sum", "cumulative_prod"]:
    NUMPY_BUILDERS[_name] = _cum_api(_name)
for _name in ["median", "nanmedian"]:
    NUMPY_BUILDERS[_name] = _median(_name)
for _name in ["quantile", "nanquantile", "percentile", "nanpercentile"]:
    NUMPY_BUILDERS[_name] = _quantile(_name)


SHORT_DTYPE = {
    "float32": "f32",
    "float64": "f64",
    "int32": "i32",
    "int64": "i64",
    "bool": "bool",
}


def jaxpr_short_dtype(name):
    return SHORT_DTYPE[name]


def jaxpr_aval_short(aval):
    shape = "[" + ",".join(str(int(d)) for d in aval.shape) + "]"
    return jaxpr_short_dtype(aval.dtype.name) + shape


def jaxpr_int_tuple(xs):
    return "(" + ",".join(str(int(x)) for x in xs) + ")"


def jaxpr_dot_dims(dn):
    (lc, rc), (lb, rb) = dn
    return (
        "(("
        + jaxpr_int_tuple(lc)
        + ","
        + jaxpr_int_tuple(rc)
        + "),("
        + jaxpr_int_tuple(lb)
        + ","
        + jaxpr_int_tuple(rb)
        + "))"
    )


def jaxpr_prim_params(name, params):
    if name == "convert_element_type":
        return "[new_dtype=" + jaxpr_short_dtype(np.dtype(params["new_dtype"]).name) + "]"
    if name == "broadcast_in_dim":
        return (
            "[broadcast_dimensions="
            + jaxpr_int_tuple(params["broadcast_dimensions"])
            + " shape="
            + jaxpr_int_tuple(params["shape"])
            + "]"
        )
    if name == "reshape":
        return "[new_sizes=" + jaxpr_int_tuple(params["new_sizes"]) + "]"
    if name == "reduce_sum":
        return "[axes=" + jaxpr_int_tuple(params["axes"]) + "]"
    if name == "dot_general":
        return "[dimension_numbers=" + jaxpr_dot_dims(params["dimension_numbers"]) + "]"
    return ""


def jaxpr_encode_var(n):
    s = ""
    while True:
        s = chr(97 + (n % 26)) + s
        n = n // 26 - 1
        if n < 0:
            break
    return s


def jaxpr_lit_str(a):
    name = a.aval.dtype.name
    if name == "bool":
        val = "True" if bool(a.val) else "False"
    elif name.startswith("int"):
        val = str(int(a.val))
    else:
        val = repr(float(a.val))
    return val + ":" + jaxpr_aval_short(a.aval)


def jaxpr_render(cj):
    jx = cj.jaxpr
    names = {}
    counter = [0]

    def bind(v):
        if id(v) not in names:
            names[id(v)] = jaxpr_encode_var(counter[0])
            counter[0] += 1

    binders = list(jx.constvars) + list(jx.invars)
    for v in binders:
        bind(v)
    for e in jx.eqns:
        for v in e.outvars:
            bind(v)

    def is_lit(a):
        return type(a).__name__ == "Literal"

    def atom(a):
        if is_lit(a):
            return jaxpr_lit_str(a)
        return names[id(a)]

    def binder(v):
        return names[id(v)] + ":" + jaxpr_aval_short(v.aval)

    def eqn(e):
        lhs = " ".join(binder(v) for v in e.outvars)
        rhs = (
            e.primitive.name
            + jaxpr_prim_params(e.primitive.name, dict(e.params))
            + " "
            + " ".join(atom(a) for a in e.invars)
        )
        return lhs + " = " + rhs

    bstr = ", ".join(binder(v) for v in binders)
    estr = " ; ".join(eqn(e) for e in jx.eqns)
    ostr = ", ".join(atom(a) for a in jx.outvars)
    return "{ lambda " + bstr + " . let " + estr + " in ( " + ostr + " ) }"


JAXPR_FNS = {
    "sin": lambda x: LAX.sin(x),
    "sin_mul": lambda x, y: LAX.mul(LAX.sin(x), y),
    "chain": lambda x: LAX.exp(LAX.neg(x)),
    "reduce": lambda x: LAX.reduce_sum(x, axes=(0,)),
    "dot": lambda x, y: LAX.dot_general(x, y, (((1,), (0,)), ((), ()))),
    "reshape": lambda x: LAX.reshape(x, (6,)),
    "broadcast": lambda x: LAX.broadcast_in_dim(x, (2, 3), (1,)),
    "convert": lambda x: LAX.convert_element_type(x, np.int32),
    "compare": lambda x, y: LAX.lt(x, y),
    "select": lambda p, a, b: LAX.select_n(p, a, b),
    "lit_mul": lambda: LAX.mul(np.float32(2.0), np.float32(3.0)),
    "nested": lambda x, y: LAX.add(LAX.mul(x, y), LAX.sin(x)),
}


def gen_jaxpr_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir)
    manifest_cases = []
    for c in cases:
        fn = JAXPR_FNS[c["fn"]]
        args = [
            np.zeros(tuple(a["shape"]), np.dtype(a["dtype"])) for a in c["in_avals"]
        ]
        cj = jax.make_jaxpr(fn)(*args)
        manifest_cases.append({"case_id": c["case_id"], "text": jaxpr_render(cj)})
    manifest_cases.sort(key=lambda c: c["case_id"])
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_jaxpr(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_jaxpr_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_jaxpr_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


AD_FNS = {
    "sin": lambda x: LAX.sin(x),
    "sin_mul": lambda x, y: LAX.mul(LAX.sin(x), y),
    "exp_neg": lambda x: LAX.exp(LAX.neg(x)),
    "tanh": lambda x: LAX.tanh(x),
    "cubic": lambda x: LAX.sub(LAX.mul(x, LAX.mul(x, x)), x),
    "div2": lambda x, y: LAX.div(x, y),
    "dot": lambda x, y: LAX.dot_general(x, y, (((1,), (0,)), ((), ()))),
    "max2": lambda x, y: LAX.max(x, y),
    "reduce": lambda x: LAX.reduce_sum(x, axes=(0,)),
    "bcast": lambda x: LAX.broadcast_in_dim(x, (2, 3), (1,)),
    "reshape_fn": lambda x: LAX.reshape(x, (6,)),
    "sum_sin": lambda x: LAX.reduce_sum(LAX.sin(x), axes=(0,)),
    "sum_cubic": lambda x: LAX.reduce_sum(
        LAX.sub(LAX.mul(x, LAX.mul(x, x)), x), axes=(0,)
    ),
    "sum_max": lambda x, y: LAX.reduce_sum(LAX.max(x, y), axes=(0,)),
    "bcast_sum": lambda x: LAX.reduce_sum(
        LAX.broadcast_in_dim(x, (2, 3), (1,)), axes=(0, 1)
    ),
    "reshape_sum": lambda x: LAX.reduce_sum(LAX.reshape(x, (6,)), axes=(0,)),
}


def run_ad_case(c, seed):
    fn = AD_FNS[c["fn"]]
    mode = c["mode"]
    avals = c["in_avals"]
    primals = [
        draw(a["rng"], seed + i, a["shape"], a["dtype"]) for i, a in enumerate(avals)
    ]
    if mode == "jvp":
        tangents = [
            draw(a["rng"], seed + 1000 + i, a["shape"], a["dtype"])
            for i, a in enumerate(avals)
        ]
        po, to = jax.jvp(fn, tuple(primals), tuple(tangents))
        return primals, tangents, [np.asarray(po), np.asarray(to)]
    if mode == "grad":
        g = jax.grad(fn)(*primals)
        return primals, None, [np.asarray(g)]
    if mode == "grad2":
        g = jax.grad(jax.grad(fn))(*primals)
        return primals, None, [np.asarray(g)]
    raise SystemExit("unknown ad mode " + mode)


def gen_ad_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and any(ec.is_wide64(a["dtype"]) for a in c["in_avals"]):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        primals, tangents, outputs = run_ad_case(c, seed)
        in_arrays = {"arg" + str(i): np.asarray(v) for i, v in enumerate(primals)}
        tan_meta = []
        if tangents is not None:
            for i, v in enumerate(tangents):
                in_arrays["tan" + str(i)] = np.asarray(v)
                tv = np.asarray(v)
                tan_meta.append(
                    {
                        "name": "tan" + str(i),
                        "shape": [int(d) for d in tv.shape],
                        "dtype": tv.dtype.name,
                    }
                )
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": "arg" + str(i),
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
            }
            for i, (a, v) in enumerate(zip(c["in_avals"], primals))
        ]
        outs_meta = [
            {
                "name": "out" + str(i),
                "shape": [int(d) for d in np.asarray(o).shape],
                "dtype": np.asarray(o).dtype.name,
            }
            for i, o in enumerate(outputs)
        ]
        compare, tol = resolve_tol(np.asarray(outputs[0]).dtype.name)
        manifest_cases.append(
            {
                "case_id": case_id,
                "fn": c["fn"],
                "mode": c["mode"],
                "args": args_meta,
                "tangents": tan_meta,
                "outputs": outs_meta,
                "compare": compare,
                "tol": tol,
            }
        )
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_ad(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_ad_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_ad_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _cubic(z):
    return LAX.sub(LAX.mul(z, LAX.mul(z, z)), z)


def _sum_sin(z):
    return LAX.reduce_sum(LAX.sin(z), axes=(0,))


BATCH_FNS = {
    "sin": lambda x: LAX.sin(x),
    "neg": lambda x: LAX.neg(x),
    "exp": lambda x: LAX.exp(x),
    "tanh": lambda x: LAX.tanh(x),
    "add": lambda x, y: LAX.add(x, y),
    "mul": lambda x, y: LAX.mul(x, y),
    "sub": lambda x, y: LAX.sub(x, y),
    "div": lambda x, y: LAX.div(x, y),
    "max": lambda x, y: LAX.max(x, y),
    "min": lambda x, y: LAX.min(x, y),
    "sum_sin": _sum_sin,
    "bcast": lambda x: LAX.broadcast_in_dim(x, (2, 3), (1,)),
    "reshape_fn": lambda x: LAX.reshape(x, (6,)),
    "convert": lambda x: LAX.convert_element_type(x, np.int32),
    "select": lambda p, a, b: LAX.select_n(p, a, b),
    "jvp_sin": lambda x, t: jax.jvp(lambda z: LAX.sin(z), (x,), (t,))[1],
    "jvp_cubic": lambda x, t: jax.jvp(_cubic, (x,), (t,))[1],
    "jvp_sum_sin": lambda x, t: jax.jvp(_sum_sin, (x,), (t,))[1],
}


def _in_axes_tuple(in_axes):
    return tuple(None if a is None else int(a) for a in in_axes)


def run_batch_case(c, seed):
    fn = BATCH_FNS[c["fn"]]
    in_axes = _in_axes_tuple(c["in_axes"])
    avals = c["in_avals"]
    inputs = [
        draw(a["rng"], seed + i, a["shape"], a["dtype"]) for i, a in enumerate(avals)
    ]
    out = jax.vmap(fn, in_axes=in_axes)(*inputs)
    return inputs, [np.asarray(out)]


def gen_batch_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and any(ec.is_wide64(a["dtype"]) for a in c["in_avals"]):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        inputs, outputs = run_batch_case(c, seed)
        in_arrays = {"arg" + str(i): np.asarray(v) for i, v in enumerate(inputs)}
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": "arg" + str(i),
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
                "in_axis": c["in_axes"][i],
            }
            for i, (a, v) in enumerate(zip(c["in_avals"], inputs))
        ]
        outs_meta = [
            {
                "name": "out" + str(i),
                "shape": [int(d) for d in np.asarray(o).shape],
                "dtype": np.asarray(o).dtype.name,
            }
            for i, o in enumerate(outputs)
        ]
        compare, tol = resolve_tol(np.asarray(outputs[0]).dtype.name)
        manifest_cases.append(
            {
                "case_id": case_id,
                "fn": c["fn"],
                "in_axes": c["in_axes"],
                "args": args_meta,
                "outputs": outs_meta,
                "compare": compare,
                "tol": tol,
            }
        )
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_batching(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_batch_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_batch_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


API_FNS = {
    "sin": lambda x: LAX.sin(x),
    "cubic": lambda x: LAX.sub(LAX.mul(x, LAX.mul(x, x)), x),
    "exp_neg": lambda x: LAX.exp(LAX.neg(x)),
    "tanh": lambda x: LAX.tanh(x),
    "sin_mul": lambda x, y: LAX.mul(LAX.sin(x), y),
    "sum_sin": lambda x: LAX.reduce_sum(LAX.sin(x), axes=(0,)),
}


def api_fun_with_nested_calls_2(x):
    one = np.asarray(1.0, x.dtype)

    def bar(y):
        def baz(w):
            q = y
            q = q + y
            q = q + (w + y)
            inner = LAX.mul(LAX.sin(x), y)
            return inner + q

        p, t = jax.jvp(baz, (x + one,), (y,))
        return t + (x * p)

    return bar(x)


def run_api_case(c, seed):
    mode = c["mode"]
    avals = c["in_avals"]
    primals = [
        draw(a["rng"], seed + i, a["shape"], a["dtype"]) for i, a in enumerate(avals)
    ]
    if mode == "jit":
        out = jax.jit(API_FNS[c["fn"]])(*primals)
        return primals, None, [np.asarray(out)]
    if mode == "jvp_jit":
        tangents = [
            draw(a["rng"], seed + 1000 + i, a["shape"], a["dtype"])
            for i, a in enumerate(avals)
        ]
        po, to = jax.jvp(
            jax.jit(API_FNS[c["fn"]]), tuple(primals), tuple(tangents)
        )
        return primals, tangents, [np.asarray(po), np.asarray(to)]
    if mode == "vmap_jit":
        in_axes = tuple(None if a is None else int(a) for a in c["in_axes"])
        out = jax.vmap(jax.jit(API_FNS[c["fn"]]), in_axes=in_axes)(*primals)
        return primals, None, [np.asarray(out)]
    if mode == "grad_jit":
        g = jax.grad(jax.jit(API_FNS[c["fn"]]))(*primals)
        return primals, None, [np.asarray(g)]
    if mode == "nested2_jit":
        out = jax.jit(api_fun_with_nested_calls_2)(*primals)
        return primals, None, [np.asarray(out)]
    if mode == "nested2_vmap":
        out = jax.vmap(api_fun_with_nested_calls_2)(*primals)
        return primals, None, [np.asarray(out)]
    raise SystemExit("unknown api mode " + mode)


def gen_api_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and any(ec.is_wide64(a["dtype"]) for a in c["in_avals"]):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        primals, tangents, outputs = run_api_case(c, seed)
        in_arrays = {"arg" + str(i): np.asarray(v) for i, v in enumerate(primals)}
        tan_meta = []
        if tangents is not None:
            for i, v in enumerate(tangents):
                in_arrays["tan" + str(i)] = np.asarray(v)
                tv = np.asarray(v)
                tan_meta.append(
                    {
                        "name": "tan" + str(i),
                        "shape": [int(d) for d in tv.shape],
                        "dtype": tv.dtype.name,
                    }
                )
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": "arg" + str(i),
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
            }
            for i, (a, v) in enumerate(zip(c["in_avals"], primals))
        ]
        outs_meta = [
            {
                "name": "out" + str(i),
                "shape": [int(d) for d in np.asarray(o).shape],
                "dtype": np.asarray(o).dtype.name,
            }
            for i, o in enumerate(outputs)
        ]
        compare, tol = resolve_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "fn": c["fn"],
            "mode": c["mode"],
            "args": args_meta,
            "tangents": tan_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if "in_axes" in c:
            entry["in_axes"] = c["in_axes"]
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_api(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_api_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_api_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


from jax._src.lax.control_flow.conditionals import platform_index_p

COND_BRANCHES = {
    "sin_cos": (lambda x: LAX.sin(x), lambda x: LAX.cos(x)),
    "mul_add": (lambda x, y: LAX.mul(x, y), lambda x, y: LAX.add(x, y)),
    "sq_neg": (lambda x: LAX.mul(x, x), lambda x: LAX.neg(x)),
}


def cond_fn(name, pred_val):
    tf, ff = COND_BRANCHES[name]
    return lambda *ops: LAX.cond(bool(pred_val), tf, ff, *ops)


def _cond_platforms(spec):
    return tuple(tuple(p) if p is not None else None for p in spec)


def run_conditionals_case(c, seed):
    mode = c["mode"]
    if mode == "platform":
        out = platform_index_p.bind(platforms=_cond_platforms(c["platforms"]))
        return [], None, [np.asarray(out)]
    avals = c["in_avals"]
    fn = cond_fn(c["fn"], c["pred"])
    primals = [
        draw(a["rng"], seed + i, a["shape"], a["dtype"]) for i, a in enumerate(avals)
    ]
    if mode == "eval":
        out = jax.jit(fn)(*primals)
        return primals, None, [np.asarray(out)]
    if mode == "jvp":
        tangents = [
            draw(a["rng"], seed + 1000 + i, a["shape"], a["dtype"])
            for i, a in enumerate(avals)
        ]
        po, to = jax.jvp(fn, tuple(primals), tuple(tangents))
        return primals, tangents, [np.asarray(po), np.asarray(to)]
    if mode == "grad":
        g = jax.grad(fn)(*primals)
        return primals, None, [np.asarray(g)]
    if mode == "vmap":
        in_axes = tuple(None if a is None else int(a) for a in c["in_axes"])
        out = jax.vmap(fn, in_axes=in_axes)(*primals)
        return primals, None, [np.asarray(out)]
    raise SystemExit("unknown conditionals mode " + mode)


def gen_conditionals_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and any(ec.is_wide64(a["dtype"]) for a in c["in_avals"]):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        primals, tangents, outputs = run_conditionals_case(c, seed)
        in_arrays = {"arg" + str(i): np.asarray(v) for i, v in enumerate(primals)}
        tan_meta = []
        if tangents is not None:
            for i, v in enumerate(tangents):
                in_arrays["tan" + str(i)] = np.asarray(v)
                tv = np.asarray(v)
                tan_meta.append(
                    {
                        "name": "tan" + str(i),
                        "shape": [int(d) for d in tv.shape],
                        "dtype": tv.dtype.name,
                    }
                )
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": "arg" + str(i),
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
            }
            for i, (a, v) in enumerate(zip(c["in_avals"], primals))
        ]
        outs_meta = [
            {
                "name": "out" + str(i),
                "shape": [int(d) for d in np.asarray(o).shape],
                "dtype": np.asarray(o).dtype.name,
            }
            for i, o in enumerate(outputs)
        ]
        compare, tol = resolve_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "fn": c["fn"],
            "mode": c["mode"],
            "pred": c.get("pred"),
            "platforms": c.get("platforms"),
            "args": args_meta,
            "tangents": tan_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if "in_axes" in c:
            entry["in_axes"] = c["in_axes"]
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_conditionals(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_conditionals_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_conditionals_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _scan_pyf(name):
    if name == "cumsum":
        return lambda c, x: (c + x, c + x)
    if name == "cumprod":
        return lambda c, x: (c * x, c)
    if name == "lin":
        return lambda c, x: (c + x, jnp.sin(c))
    if name == "twocarry":
        return lambda c, x: ((c[0] + x, c[1] * x), c[0])
    raise SystemExit("loops: unknown fn " + name)


def loops_fn(name, reverse, num_carry):
    f = _scan_pyf(name)

    def g(*flat):
        init_arrays = list(flat[:num_carry])
        xs_arrays = list(flat[num_carry:])
        init = init_arrays[0] if num_carry == 1 else tuple(init_arrays)
        xs = xs_arrays[0]
        carry, ys = jax.lax.scan(f, init, xs, reverse=reverse)
        return jax.tree_util.tree_leaves(carry) + jax.tree_util.tree_leaves(ys)

    return g


def _while_pyf(name):
    if name == "wdouble":
        return (lambda v: v < 8.0, lambda v: v + v)
    if name == "wtwo":
        return (lambda s: s[0] < 20.0, lambda s: (s[0] + s[1], s[1]))
    raise SystemExit("loops: unknown while fn " + name)


def while_fn(name, num_carry):
    cond_fun, body_fun = _while_pyf(name)

    def g(*flat):
        init_arrays = list(flat[:num_carry])
        init = init_arrays[0] if num_carry == 1 else tuple(init_arrays)
        out = jax.lax.while_loop(cond_fun, body_fun, init)
        return jax.tree_util.tree_leaves(out)

    return g


def cumulative_fn(name, axis, reverse):
    prim = {
        "cumsum": jax.lax.cumsum,
        "cumprod": jax.lax.cumprod,
        "cummax": jax.lax.cummax,
        "cummin": jax.lax.cummin,
        "cumlogsumexp": jax.lax.cumlogsumexp,
    }[name]

    def g(*flat):
        (x,) = flat
        return [prim(x, axis=axis, reverse=reverse)]

    return g


def run_loops_case(c, seed):
    mode = c["mode"]
    kind = c.get("kind", "scan")
    if kind == "cumulative":
        avals = [c["operand"]]
        g = cumulative_fn(c["fn"], c["axis"], c["reverse"])
    elif kind == "while":
        num_carry = len(c["init"])
        avals = c["init"]
        g = while_fn(c["fn"], num_carry)
    else:
        num_carry = len(c["init"])
        avals = c["init"] + c.get("xs", [])
        g = loops_fn(c["fn"], c["reverse"], num_carry)
    primals = [
        draw(a["rng"], seed + i, a["shape"], a["dtype"]) for i, a in enumerate(avals)
    ]
    if mode == "eval":
        out = jax.jit(g)(*primals)
        return primals, None, [np.asarray(o) for o in out]
    if mode == "jvp":
        tangents = [
            draw(a["rng"], seed + 1000 + i, a["shape"], a["dtype"])
            for i, a in enumerate(avals)
        ]
        po, to = jax.jvp(g, tuple(primals), tuple(tangents))
        return primals, tangents, [np.asarray(o) for o in po] + [
            np.asarray(o) for o in to
        ]
    if mode == "vmap":
        in_axes = tuple(None if a is None else int(a) for a in c["in_axes"])
        out = jax.vmap(g, in_axes=in_axes)(*primals)
        return primals, None, [np.asarray(o) for o in out]
    raise SystemExit("unknown loops mode " + mode)


def gen_loops_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if c.get("kind", "scan") == "cumulative":
            avals = [c["operand"]]
        else:
            avals = c["init"] + c.get("xs", [])
        if not x64 and any(ec.is_wide64(a["dtype"]) for a in avals):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        primals, tangents, outputs = run_loops_case(c, seed)
        in_arrays = {"arg" + str(i): np.asarray(v) for i, v in enumerate(primals)}
        tan_meta = []
        if tangents is not None:
            for i, v in enumerate(tangents):
                in_arrays["tan" + str(i)] = np.asarray(v)
                tv = np.asarray(v)
                tan_meta.append(
                    {
                        "name": "tan" + str(i),
                        "shape": [int(d) for d in tv.shape],
                        "dtype": tv.dtype.name,
                    }
                )
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": "arg" + str(i),
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
            }
            for i, (a, v) in enumerate(zip(avals, primals))
        ]
        outs_meta = [
            {
                "name": "out" + str(i),
                "shape": [int(d) for d in np.asarray(o).shape],
                "dtype": np.asarray(o).dtype.name,
            }
            for i, o in enumerate(outputs)
        ]
        compare, tol = resolve_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "fn": c["fn"],
            "kind": c.get("kind", "scan"),
            "mode": c["mode"],
            "reverse": c["reverse"],
            "num_carry": len(c.get("init", [])),
            "axis": c.get("axis", 0),
            "args": args_meta,
            "tangents": tan_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if "in_axes" in c:
            entry["in_axes"] = c["in_axes"]
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_loops(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_loops_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_loops_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _solves_call(d, symmetric, has_ts):
    matvec = lambda x: jnp.asarray(d) * x
    solve = lambda mv, bb: bb / jnp.asarray(d)
    ts = (lambda mv, bb: bb / jnp.asarray(d)) if has_ts else None
    return lambda bb: jax.lax.custom_linear_solve(
        matvec, bb, solve, ts, symmetric=symmetric
    )


def run_solves_case(c, seed):
    mode = c["mode"]
    dspec = c["const"]
    bspec = c["operand"]
    d = draw(dspec["rng"], seed, dspec["shape"], dspec["dtype"])
    b = draw(bspec["rng"], seed + 1, bspec["shape"], bspec["dtype"])
    call = _solves_call(d, c["symmetric"], c["has_ts"])
    if mode == "eval":
        out = jax.jit(call)(b)
        return [d, b], None, [np.asarray(out)]
    if mode == "jvp":
        bdot = draw(bspec["rng"], seed + 1000, bspec["shape"], bspec["dtype"])
        po, to = jax.jvp(call, (b,), (bdot,))
        return [d, b], [bdot], [np.asarray(po), np.asarray(to)]
    if mode == "grad":
        g = jax.grad(lambda bb: jnp.sum(call(bb)))(b)
        return [d, b], None, [np.asarray(g)]
    raise SystemExit("unknown solves mode " + mode)


def gen_solves_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        avals = [c["const"], c["operand"]]
        if not x64 and any(ec.is_wide64(a["dtype"]) for a in avals):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        primals, tangents, outputs = run_solves_case(c, seed)
        in_arrays = {"arg" + str(i): np.asarray(v) for i, v in enumerate(primals)}
        tan_meta = []
        if tangents is not None:
            for i, v in enumerate(tangents):
                in_arrays["tan" + str(i)] = np.asarray(v)
                tv = np.asarray(v)
                tan_meta.append(
                    {
                        "name": "tan" + str(i),
                        "shape": [int(d) for d in tv.shape],
                        "dtype": tv.dtype.name,
                    }
                )
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        rngs = [c["const"]["rng"], c["operand"]["rng"]]
        args_meta = [
            {
                "name": "arg" + str(i),
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": rngs[i],
            }
            for i, v in enumerate(primals)
        ]
        outs_meta = [
            {
                "name": "out" + str(i),
                "shape": [int(d) for d in np.asarray(o).shape],
                "dtype": np.asarray(o).dtype.name,
            }
            for i, o in enumerate(outputs)
        ]
        compare, tol = resolve_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "fn": c["fn"],
            "mode": c["mode"],
            "symmetric": c["symmetric"],
            "has_ts": c["has_ts"],
            "args": args_meta,
            "tangents": tan_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_solves(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_solves_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_solves_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _add_all(*xs):
    return functools.reduce(jnp.add, xs)


def _am_reduce(name):
    def build(params):
        ax = _red_axis(params)
        kd = params.get("keepdims", False)
        return lambda x: getattr(x, name)(axis=ax, keepdims=kd)

    return build


def _am_reduce_ddof(name):
    def build(params):
        ax = _red_axis(params)
        kd = params.get("keepdims", False)
        ddof = params.get("ddof", 0)
        return lambda x: getattr(x, name)(axis=ax, keepdims=kd, ddof=ddof)

    return build


def _am_cum(name):
    def build(params):
        ax = params.get("axis")
        return lambda x: getattr(x, name)(axis=ax)

    return build


def _am_arg(name):
    def build(params):
        ax = params.get("axis")
        return lambda x: getattr(x, name)(axis=ax)

    return build


def _am_call0(name):
    return lambda params: (lambda x: getattr(x, name)())


def _am_prop(name):
    return lambda params: (lambda x: getattr(x, name))


def _am_reshape(params):
    shape = tuple(params["shape"])
    return lambda x: x.reshape(shape)


def _am_transpose(params):
    axes = params.get("axes")
    if axes is None:
        return lambda x: x.transpose()
    return lambda x: x.transpose(tuple(axes))


def _am_squeeze(params):
    axis = params.get("axis")
    axis = None if axis is None else tuple(axis)
    return lambda x: x.squeeze(axis=axis)


def _am_swapaxes(params):
    a1 = params["axis1"]
    a2 = params["axis2"]
    return lambda x: x.swapaxes(a1, a2)


def _am_repeat(params):
    n = params["repeats"]
    axis = params.get("axis")
    return lambda x: x.repeat(n, axis=axis)


def _am_astype(params):
    dt = np.dtype(params["dtype"])
    return lambda x: x.astype(dt)


def _am_clip(params):
    lo = params.get("min")
    hi = params.get("max")
    return lambda x: x.clip(min=lo, max=hi)


def _am_round(params):
    d = params.get("decimals", 0)
    return lambda x: x.round(decimals=d)


def _am_diagonal(params):
    off = params.get("offset", 0)
    a1 = params.get("axis1", 0)
    a2 = params.get("axis2", 1)
    return lambda x: x.diagonal(offset=off, axis1=a1, axis2=a2)


def _am_trace(params):
    off = params.get("offset", 0)
    a1 = params.get("axis1", 0)
    a2 = params.get("axis2", 1)
    return lambda x: x.trace(offset=off, axis1=a1, axis2=a2)


def _am_searchsorted(params):
    side = params.get("side", "left")
    return lambda a, v: a.searchsorted(v, side=side)


def _am_take(params):
    axis = params.get("axis")
    mode = params.get("mode")
    return lambda a, ind: a.take(ind, axis=axis, mode=mode)


ARRAY_METHODS_BUILDERS = {
    "reshape": _am_reshape,
    "ravel": _am_call0("ravel"),
    "flatten": _am_call0("flatten"),
    "copy": _am_call0("copy"),
    "conj": _am_call0("conj"),
    "conjugate": _am_call0("conjugate"),
    "transpose": _am_transpose,
    "squeeze": _am_squeeze,
    "swapaxes": _am_swapaxes,
    "repeat": _am_repeat,
    "astype": _am_astype,
    "clip": _am_clip,
    "round": _am_round,
    "diagonal": _am_diagonal,
    "trace": _am_trace,
    "searchsorted": _am_searchsorted,
    "take": _am_take,
    "cumsum": _am_cum("cumsum"),
    "cumprod": _am_cum("cumprod"),
    "argmax": _am_arg("argmax"),
    "argmin": _am_arg("argmin"),
    "T": _am_prop("T"),
    "mT": _am_prop("mT"),
    "real": _am_prop("real"),
    "imag": _am_prop("imag"),
    "add": lambda params: (lambda a, b: a + b),
    "sub": lambda params: (lambda a, b: a - b),
    "mul": lambda params: (lambda a, b: a * b),
    "truediv": lambda params: (lambda a, b: a / b),
    "floordiv": lambda params: (lambda a, b: a // b),
    "mod": lambda params: (lambda a, b: a % b),
    "pow": lambda params: (lambda a, b: a**b),
    "eq": lambda params: (lambda a, b: a == b),
    "ne": lambda params: (lambda a, b: a != b),
    "lt": lambda params: (lambda a, b: a < b),
    "le": lambda params: (lambda a, b: a <= b),
    "gt": lambda params: (lambda a, b: a > b),
    "ge": lambda params: (lambda a, b: a >= b),
    "and": lambda params: (lambda a, b: a & b),
    "or": lambda params: (lambda a, b: a | b),
    "xor": lambda params: (lambda a, b: a ^ b),
    "lshift": lambda params: (lambda a, b: a << b),
    "rshift": lambda params: (lambda a, b: a >> b),
    "neg": lambda params: (lambda x: -x),
    "pos": lambda params: (lambda x: +x),
    "abs": lambda params: (lambda x: abs(x)),
    "invert": lambda params: (lambda x: ~x),
}
for _name in ["sum", "prod", "max", "min", "mean", "all", "any", "ptp"]:
    ARRAY_METHODS_BUILDERS[_name] = _am_reduce(_name)
for _name in ["var", "std"]:
    ARRAY_METHODS_BUILDERS[_name] = _am_reduce_ddof(_name)


from jax._src.random import prng as PRNG_MOD
from jax._src.random import threefry2x32 as TF_MOD
from jax._src.random.threefry2x32 import threefry_prng_impl as PRNG_IMPL


def _prng_wrap(a):
    return PRNG_MOD.random_wrap(a, impl=PRNG_IMPL)


def _prng_unwrap(x):
    return PRNG_MOD.random_unwrap(x)


def prng_threefry_seed(params):
    return lambda s: TF_MOD.threefry_seed(s)


def prng_threefry_2x32(params):
    return lambda k, c: TF_MOD.threefry_2x32(k, c)


def prng_threefry_split(params):
    shape = tuple(params["shape"])
    return lambda k: TF_MOD.threefry_split(k, shape)


def prng_threefry_fold_in(params):
    return lambda k, d: TF_MOD.threefry_fold_in(k, d)


def prng_threefry_random_bits(params):
    shape = tuple(params["shape"])
    bw = int(params["bit_width"])
    return lambda k: TF_MOD.threefry_random_bits(k, bw, shape)


def prng_iota_2x32_shape(params):
    shape = tuple(params["shape"])
    return lambda: list(PRNG_MOD.iota_2x32_shape(shape))


def prng_random_seed(params):
    return lambda s: _prng_unwrap(PRNG_MOD.random_seed(s, impl=PRNG_IMPL))


def prng_random_split(params):
    shape = tuple(params["shape"])
    return lambda k: _prng_unwrap(PRNG_MOD.random_split(_prng_wrap(k), shape))


def prng_random_fold_in(params):
    return lambda k, m: _prng_unwrap(PRNG_MOD.random_fold_in(_prng_wrap(k), m))


def prng_random_bits(params):
    shape = tuple(params["shape"])
    bw = int(params["bit_width"])
    return lambda k: PRNG_MOD.random_bits(_prng_wrap(k), bw, shape)


def prng_random_wrap(params):
    return lambda k: _prng_unwrap(_prng_wrap(k))


PRNG_BUILDERS = {
    "threefry_seed": prng_threefry_seed,
    "threefry_2x32": prng_threefry_2x32,
    "threefry_split": prng_threefry_split,
    "threefry_fold_in": prng_threefry_fold_in,
    "threefry_random_bits": prng_threefry_random_bits,
    "iota_2x32_shape": prng_iota_2x32_shape,
    "random_seed": prng_random_seed,
    "random_split": prng_random_split,
    "random_fold_in": prng_random_fold_in,
    "random_bits": prng_random_bits,
    "random_wrap": prng_random_wrap,
    "random_unwrap": prng_random_wrap,
}


def rc_wrap(raw):
    return PRNG_MOD.random_wrap(raw, impl=PRNG_IMPL)


def rc_key(params):
    return lambda s: jax.random.key_data(jax.random.key(s))


def rc_key_data(params):
    return lambda raw: jax.random.key_data(rc_wrap(raw))


def rc_wrap_key_data(params):
    return lambda raw: jax.random.key_data(
        jax.random.wrap_key_data(raw, impl="threefry2x32")
    )


def rc_clone(params):
    return lambda raw: jax.random.key_data(jax.random.clone(rc_wrap(raw)))


def rc_fold_in(params):
    return lambda raw, d: jax.random.key_data(jax.random.fold_in(rc_wrap(raw), d))


def rc_split(params):
    num = int(params["num"])
    return lambda raw: jax.random.key_data(jax.random.split(rc_wrap(raw), num))


def rc_bits(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.bits(rc_wrap(raw), shape, dtype=jnp.uint32)


def rc_randint(params):
    shape = tuple(params["shape"])
    minval = int(params["minval"])
    maxval = int(params["maxval"])
    return lambda raw: jax.random.randint(
        rc_wrap(raw), shape, minval, maxval, dtype=jnp.int32
    )


def rc_uniform(params):
    shape = tuple(params["shape"])
    minval = float(params["minval"])
    maxval = float(params["maxval"])
    return lambda raw: jax.random.uniform(
        rc_wrap(raw), shape, dtype=jnp.float32, minval=minval, maxval=maxval
    )


def rc_normal(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.normal(rc_wrap(raw), shape, dtype=jnp.float32)


def rc_truncated_normal(params):
    shape = tuple(params["shape"])
    lower = float(params["lower"])
    upper = float(params["upper"])
    return lambda raw: jax.random.truncated_normal(
        rc_wrap(raw), lower, upper, shape, dtype=jnp.float32
    )


def rc_permutation(params):
    n = int(params["n"])
    return lambda raw: jax.random.permutation(rc_wrap(raw), n)


def rc_choice(params):
    n = int(params["n"])
    shape = tuple(params["shape"])
    replace = bool(params["replace"])
    return lambda raw: jax.random.choice(
        rc_wrap(raw), n, shape=shape, replace=replace
    )


def rc_exponential(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.exponential(rc_wrap(raw), shape, dtype=jnp.float32)


def rc_cauchy(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.cauchy(rc_wrap(raw), shape, dtype=jnp.float32)


def rc_laplace(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.laplace(rc_wrap(raw), shape, dtype=jnp.float32)


def rc_logistic(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.logistic(rc_wrap(raw), shape, dtype=jnp.float32)


def rc_gumbel(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.gumbel(rc_wrap(raw), shape, dtype=jnp.float32)


def rc_pareto(params):
    shape = tuple(params["shape"])
    b = float(params["b"])
    return lambda raw: jax.random.pareto(rc_wrap(raw), b, shape, dtype=jnp.float32)


def rc_rayleigh(params):
    shape = tuple(params["shape"])
    scale = float(params["scale"])
    return lambda raw: jax.random.rayleigh(
        rc_wrap(raw), scale, shape, dtype=jnp.float32
    )


def rc_weibull_min(params):
    shape = tuple(params["shape"])
    scale = float(params["scale"])
    concentration = float(params["concentration"])
    return lambda raw: jax.random.weibull_min(
        rc_wrap(raw), scale, concentration, shape, dtype=jnp.float32
    )


def rc_lognormal(params):
    shape = tuple(params["shape"])
    sigma = float(params["sigma"])
    return lambda raw: jax.random.lognormal(
        rc_wrap(raw), sigma, shape, dtype=jnp.float32
    )


def rc_triangular(params):
    shape = tuple(params["shape"])
    left = float(params["left"])
    mode = float(params["mode"])
    right = float(params["right"])
    return lambda raw: jax.random.triangular(
        rc_wrap(raw), left, mode, right, shape, dtype=jnp.float32
    )


def rc_wald(params):
    shape = tuple(params["shape"])
    mean = float(params["mean"])
    return lambda raw: jax.random.wald(rc_wrap(raw), mean, shape, dtype=jnp.float32)


def rc_geometric(params):
    shape = tuple(params["shape"])
    p = np.float32(params["p"])
    return lambda raw: jax.random.geometric(rc_wrap(raw), p, shape)


def rc_bernoulli(params):
    shape = tuple(params["shape"])
    p = np.float32(params["p"])
    return lambda raw: jax.random.bernoulli(rc_wrap(raw), p, shape)


def rc_rademacher(params):
    shape = tuple(params["shape"])
    return lambda raw: jax.random.rademacher(rc_wrap(raw), shape)


def rc_categorical(params):
    axis = int(params["axis"])
    return lambda raw, logits: jax.random.categorical(
        rc_wrap(raw), logits, axis=axis
    )


def rc_gamma(params):
    shape = tuple(params["shape"])
    a = float(params["a"])
    return lambda raw: jax.random.gamma(rc_wrap(raw), a, shape, dtype=jnp.float32)


def rc_loggamma(params):
    shape = tuple(params["shape"])
    a = float(params["a"])
    return lambda raw: jax.random.loggamma(rc_wrap(raw), a, shape, dtype=jnp.float32)


def rc_beta(params):
    shape = tuple(params["shape"])
    a = float(params["a"])
    b = float(params["b"])
    return lambda raw: jax.random.beta(rc_wrap(raw), a, b, shape, dtype=jnp.float32)


def rc_chisquare(params):
    shape = tuple(params["shape"])
    df = float(params["df"])
    return lambda raw: jax.random.chisquare(
        rc_wrap(raw), df, shape, dtype=jnp.float32
    )


def rc_t(params):
    shape = tuple(params["shape"])
    df = float(params["df"])
    return lambda raw: jax.random.t(rc_wrap(raw), df, shape, dtype=jnp.float32)


def rc_f(params):
    shape = tuple(params["shape"])
    dfnum = float(params["dfnum"])
    dfden = float(params["dfden"])
    return lambda raw: jax.random.f(
        rc_wrap(raw), dfnum, dfden, shape, dtype=jnp.float32
    )


def rc_generalized_normal(params):
    shape = tuple(params["shape"])
    p = float(params["p"])
    return lambda raw: jax.random.generalized_normal(
        rc_wrap(raw), p, shape, dtype=jnp.float32
    )


def rc_dirichlet(params):
    return lambda raw, alpha: jax.random.dirichlet(
        rc_wrap(raw), alpha, dtype=jnp.float32
    )


def rc_poisson(params):
    shape = tuple(params["shape"])
    lam = float(params["lam"])
    return lambda raw: jax.random.poisson(rc_wrap(raw), lam, shape)


def rc_binomial(params):
    shape = tuple(params["shape"])
    n = np.float32(params["n"])
    p = np.float32(params["p"])
    return lambda raw: jax.random.binomial(
        rc_wrap(raw), n, p, shape, dtype=jnp.float32
    )


def rc_multinomial(params):
    n = float(params["n"])
    return lambda raw, p: jax.random.multinomial(
        rc_wrap(raw), n, p, dtype=jnp.float32
    )


RANDOM_CORE_BUILDERS = {
    "key": rc_key,
    "key_data": rc_key_data,
    "wrap_key_data": rc_wrap_key_data,
    "clone": rc_clone,
    "fold_in": rc_fold_in,
    "split": rc_split,
    "bits": rc_bits,
    "randint": rc_randint,
    "uniform": rc_uniform,
    "normal": rc_normal,
    "truncated_normal": rc_truncated_normal,
    "permutation": rc_permutation,
    "choice": rc_choice,
    "exponential": rc_exponential,
    "cauchy": rc_cauchy,
    "laplace": rc_laplace,
    "logistic": rc_logistic,
    "gumbel": rc_gumbel,
    "pareto": rc_pareto,
    "rayleigh": rc_rayleigh,
    "weibull_min": rc_weibull_min,
    "lognormal": rc_lognormal,
    "triangular": rc_triangular,
    "wald": rc_wald,
    "geometric": rc_geometric,
    "bernoulli": rc_bernoulli,
    "rademacher": rc_rademacher,
    "categorical": rc_categorical,
    "gamma": rc_gamma,
    "loggamma": rc_loggamma,
    "beta": rc_beta,
    "chisquare": rc_chisquare,
    "t": rc_t,
    "f": rc_f,
    "generalized_normal": rc_generalized_normal,
    "dirichlet": rc_dirichlet,
    "poisson": rc_poisson,
    "binomial": rc_binomial,
    "multinomial": rc_multinomial,
}

import jax.nn as jnn


def nn_unary(name):
    fn = getattr(jnn, name)

    def build(params):
        return lambda x: fn(x)

    return build


def nn_elu(params):
    alpha = float(params["alpha"])
    return lambda x: jnn.elu(x, alpha)


def nn_celu(params):
    alpha = float(params["alpha"])
    return lambda x: jnn.celu(x, alpha)


def nn_leaky_relu(params):
    slope = float(params["negative_slope"])
    return lambda x: jnn.leaky_relu(x, slope)


def nn_squareplus(params):
    b = float(params["b"])
    return lambda x: jnn.squareplus(x, b)


def nn_gelu(params):
    approximate = bool(params["approximate"])
    return lambda x: jnn.gelu(x, approximate=approximate)


def nn_glu(params):
    axis = int(params["axis"])
    return lambda x: jnn.glu(x, axis=axis)


def nn_softmax(params):
    axis = int(params["axis"])
    return lambda x: jnn.softmax(x, axis=axis)


def nn_log_softmax(params):
    axis = int(params["axis"])
    return lambda x: jnn.log_softmax(x, axis=axis)


def nn_standardize(params):
    axis = int(params["axis"])
    epsilon = float(params["epsilon"])
    return lambda x: jnn.standardize(x, axis=axis, epsilon=epsilon)


def nn_logmeanexp(params):
    keepdims = bool(params.get("keepdims", False))
    if "axis" in params:
        axis = int(params["axis"])
        return lambda x: jnn.logmeanexp(x, axis=axis, keepdims=keepdims)
    return lambda x: jnn.logmeanexp(x, keepdims=keepdims)


def nn_one_hot(params):
    num_classes = int(params["num_classes"])
    axis = int(params["axis"])
    return lambda x: jnn.one_hot(x, num_classes, axis=axis)


def nn_scaled_dot_general(params):
    lc = tuple(int(d) for d in params["lhs_contract"])
    rc = tuple(int(d) for d in params["rhs_contract"])
    lb = tuple(int(d) for d in params["lhs_batch"])
    rb = tuple(int(d) for d in params["rhs_batch"])
    dn = ((lc, rc), (lb, rb))
    return lambda a, b: jnn.scaled_dot_general(a, b, dn)


NN_BUILDERS = {
    "elu": nn_elu,
    "celu": nn_celu,
    "leaky_relu": nn_leaky_relu,
    "squareplus": nn_squareplus,
    "gelu": nn_gelu,
    "glu": nn_glu,
    "softmax": nn_softmax,
    "log_softmax": nn_log_softmax,
    "standardize": nn_standardize,
    "logmeanexp": nn_logmeanexp,
    "one_hot": nn_one_hot,
    "scaled_dot_general": nn_scaled_dot_general,
}

for _nn_name in [
    "identity",
    "relu",
    "relu6",
    "softplus",
    "sparse_plus",
    "soft_sign",
    "sigmoid",
    "sparse_sigmoid",
    "silu",
    "mish",
    "log_sigmoid",
    "hard_tanh",
    "hard_sigmoid",
    "hard_silu",
    "selu",
    "log1mexp",
]:
    NN_BUILDERS[_nn_name] = nn_unary(_nn_name)


import jax.nn.initializers as jinit


def init_zeros(params):
    shape = tuple(params["shape"])
    return lambda raw: jinit.zeros(rc_wrap(raw), shape, jnp.float32)


def init_ones(params):
    shape = tuple(params["shape"])
    return lambda raw: jinit.ones(rc_wrap(raw), shape, jnp.float32)


def init_constant(params):
    shape = tuple(params["shape"])
    value = float(params["value"])
    init = jinit.constant(value)
    return lambda raw: init(rc_wrap(raw), shape, jnp.float32)


def init_uniform(params):
    shape = tuple(params["shape"])
    scale = float(params["scale"])
    init = jinit.uniform(scale)
    return lambda raw: init(rc_wrap(raw), shape, jnp.float32)


def init_normal(params):
    shape = tuple(params["shape"])
    stddev = float(params["stddev"])
    init = jinit.normal(stddev)
    return lambda raw: init(rc_wrap(raw), shape, jnp.float32)


def init_truncated_normal(params):
    shape = tuple(params["shape"])
    stddev = float(params["stddev"])
    lower = float(params["lower"])
    upper = float(params["upper"])
    init = jinit.truncated_normal(stddev, lower=lower, upper=upper)
    return lambda raw: init(rc_wrap(raw), shape, jnp.float32)


def init_variance_scaling(params):
    shape = tuple(params["shape"])
    scale = float(params["scale"])
    mode = str(params["mode"])
    distribution = str(params["distribution"])
    init = jinit.variance_scaling(scale, mode, distribution)
    return lambda raw: init(rc_wrap(raw), shape, jnp.float32)


def init_fan(name):
    fn = getattr(jinit, name)

    def build(params):
        shape = tuple(params["shape"])
        init = fn()
        return lambda raw: init(rc_wrap(raw), shape, jnp.float32)

    return build


INIT_BUILDERS = {
    "zeros": init_zeros,
    "ones": init_ones,
    "constant": init_constant,
    "uniform": init_uniform,
    "normal": init_normal,
    "truncated_normal": init_truncated_normal,
    "variance_scaling": init_variance_scaling,
}

for _init_name in [
    "glorot_uniform",
    "glorot_normal",
    "lecun_uniform",
    "lecun_normal",
    "he_uniform",
    "he_normal",
]:
    INIT_BUILDERS[_init_name] = init_fan(_init_name)


import jax.ops as jops
from jax.scipy.special import logsumexp as jlogsumexp


def ops_segment(fn):
    def build(params):
        ns = params.get("num_segments")
        ns = None if ns is None else int(ns)
        return lambda data, seg: fn(data, seg, num_segments=ns)

    return build


def ops_logsumexp(params):
    axis = params.get("axis")
    axis = None if axis is None else tuple(int(x) for x in axis)
    keepdims = bool(params.get("keepdims", False))
    has_b = bool(params.get("has_b", False))
    if has_b:
        return lambda a, b: jlogsumexp(a, axis=axis, b=b, keepdims=keepdims)
    return lambda a: jlogsumexp(a, axis=axis, keepdims=keepdims)


OPS_BUILDERS = {
    "segment_sum": ops_segment(jops.segment_sum),
    "segment_prod": ops_segment(jops.segment_prod),
    "segment_max": ops_segment(jops.segment_max),
    "segment_min": ops_segment(jops.segment_min),
    "logsumexp": ops_logsumexp,
}


from jax._src.image import scale as jimage_scale


def image_resize(params):
    shape = tuple(params["shape"])
    method = params["method"]
    antialias = bool(params.get("antialias", True))
    return lambda image: jax.image.resize(
        image, shape, method, antialias=antialias
    )


def image_scale_and_translate(params):
    shape = tuple(params["shape"])
    spatial_dims = tuple(params["spatial_dims"])
    scale = np.asarray(params["scale"], dtype=np.float64)
    translation = np.asarray(params["translation"], dtype=np.float64)
    method = params["method"]
    antialias = bool(params.get("antialias", True))
    return lambda image: jax.image.scale_and_translate(
        image, shape, spatial_dims, scale, translation, method, antialias=antialias
    )


def image_compute_weight_mat(params):
    input_size = int(params["input_size"])
    output_size = int(params["output_size"])
    scale = float(params["scale"])
    translation = float(params["translation"])
    method = jimage_scale.ResizeMethod.from_string(params["method"])
    antialias = bool(params.get("antialias", True))
    radius, kernel = jimage_scale._kernels[method]
    return lambda: jimage_scale.compute_weight_mat(
        input_size,
        output_size,
        scale,
        translation,
        kernel,
        antialias,
        edge_padding=False,
        radius=radius,
    )


IMAGE_BUILDERS = {
    "resize": image_resize,
    "scale_and_translate": image_scale_and_translate,
    "compute_weight_mat": image_compute_weight_mat,
}


import jax.scipy.special as jsp


def sp_unary(name):
    fn = getattr(jsp, name)

    def build(params):
        return lambda x: fn(x)

    return build


def sp_binary(name):
    fn = getattr(jsp, name)

    def build(params):
        return lambda a, b: fn(a, b)

    return build


def sp_ternary(name):
    fn = getattr(jsp, name)

    def build(params):
        return lambda a, b, c: fn(a, b, c)

    return build


def sp_multigammaln(params):
    d = int(params["d"])
    return lambda a: jsp.multigammaln(a, d)


def sp_softmax(params):
    axis = params.get("axis")
    return lambda x: jsp.softmax(x, axis=axis)


def sp_log_softmax(params):
    axis = params.get("axis")
    return lambda x: jsp.log_softmax(x, axis=axis)


def sp_bernoulli(params):
    n = int(params["n"])
    return lambda: jsp.bernoulli(n)


def sp_bessel_jn(params):
    v = int(params["v"])
    n_iter = int(params.get("n_iter", 50))
    return lambda z: jsp.bessel_jn(z, v=v, n_iter=n_iter)


SCIPY_SPECIAL_BUILDERS = {
    "multigammaln": sp_multigammaln,
    "softmax": sp_softmax,
    "log_softmax": sp_log_softmax,
    "bernoulli": sp_bernoulli,
    "bessel_jn": sp_bessel_jn,
}

for _sp_name in [
    "gammaln",
    "gammasgn",
    "loggamma",
    "gamma",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "erfcx",
    "dawsn",
    "expit",
    "logit",
    "entr",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "ndtr",
    "ndtri",
    "log_ndtr",
    "factorial",
    "spence",
    "exp1",
    "sici",
]:
    SCIPY_SPECIAL_BUILDERS[_sp_name] = sp_unary(_sp_name)

for _sp_name in [
    "betaln",
    "beta",
    "comb",
    "gammainc",
    "gammaincc",
    "xlogy",
    "xlog1py",
    "boxcox",
    "boxcox1p",
    "rel_entr",
    "kl_div",
    "zeta",
    "polygamma",
    "poch",
    "owens_t",
    "expn",
]:
    SCIPY_SPECIAL_BUILDERS[_sp_name] = sp_binary(_sp_name)

SCIPY_SPECIAL_BUILDERS["betainc"] = sp_ternary("betainc")
SCIPY_SPECIAL_BUILDERS["hyp1f1"] = sp_ternary("hyp1f1")


import jax.scipy.stats as jstats


def stats_fn(dist, name):
    fn = getattr(getattr(jstats, dist), name)

    def build(params):
        return lambda *inputs: fn(*inputs)

    return build


STATS_BUILDERS = {}
for _dist, _names in {
    "bernoulli": ["cdf", "logpmf", "pmf", "ppf"],
    "beta": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "sf"],
    "betabinom": ["logpmf", "pmf"],
    "binom": ["logpmf", "pmf"],
    "cauchy": ["cdf", "isf", "logcdf", "logpdf", "logsf", "pdf", "ppf", "sf"],
    "chi2": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "sf"],
    "dirichlet": ["logpdf", "pdf"],
    "expon": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "ppf", "sf"],
    "gamma": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "sf"],
    "gennorm": ["cdf", "logpdf", "pdf"],
    "geom": ["logpmf", "pmf"],
    "gumbel_l": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "ppf", "sf"],
    "gumbel_r": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "ppf", "sf"],
    "laplace": ["cdf", "logpdf", "pdf"],
    "logistic": ["cdf", "isf", "logpdf", "pdf", "ppf", "sf"],
    "multinomial": ["logpmf", "pmf"],
    "nbinom": ["logpmf", "pmf"],
    "norm": ["cdf", "isf", "logcdf", "logpdf", "logsf", "pdf", "ppf", "sf"],
    "pareto": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "ppf", "sf"],
    "poisson": ["cdf", "entropy", "logpmf", "pmf"],
    "t": ["logpdf", "pdf"],
    "truncnorm": ["cdf", "logcdf", "logpdf", "logsf", "pdf", "sf"],
    "uniform": ["cdf", "logpdf", "pdf", "ppf"],
    "vonmises": ["logpdf", "pdf"],
    "wrapcauchy": ["logpdf", "pdf"],
}.items():
    for _fname in _names:
        STATS_BUILDERS[_dist + "." + _fname] = stats_fn(_dist, _fname)

import jax._src.scipy.stats._core as _stats_core


def core_fn(fn):
    def build(params):
        return lambda *inputs: fn(*inputs)

    return build


STATS_BUILDERS["core.sem"] = core_fn(jstats.sem)
STATS_BUILDERS["core.invert_permutation"] = core_fn(_stats_core.invert_permutation)


import jax.scipy.integrate as jsintegrate
import jax.scipy.ndimage as jsndimage


def integrate_trapezoid(params):
    dx = float(params.get("dx", 1.0))
    axis = int(params.get("axis", -1))
    if params.get("has_x", False):
        return lambda y, x: jsintegrate.trapezoid(y, x, dx, axis)
    return lambda y: jsintegrate.trapezoid(y, None, dx, axis)


INTEGRATE_BUILDERS = {"trapezoid": integrate_trapezoid}


def ndimage_map_coordinates(params):
    order = int(params["order"])
    mode = params.get("mode", "constant")
    cval = float(params.get("cval", 0.0))

    def fn(inp, *coords):
        return jsndimage.map_coordinates(inp, list(coords), order, mode=mode, cval=cval)

    return fn


NDIMAGE_BUILDERS = {"map_coordinates": ndimage_map_coordinates}


def run_case(c, seed):
    if c["primitive"].startswith("scipy.integrate.") and c["op"] in INTEGRATE_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(INTEGRATE_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("scipy.ndimage.") and c["op"] in NDIMAGE_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(NDIMAGE_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("scipy.stats.") and c["op"] in STATS_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        if c["op"].startswith("multinomial."):
            x = inputs[0]
            n = np.sum(x, dtype=x.dtype)
            p = inputs[2]
            p = p / np.sum(p)
            inputs = [x, n, p]
        out = jax.jit(STATS_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("scipy.special.") and c["op"] in SCIPY_SPECIAL_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(SCIPY_SPECIAL_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("image.") and c["op"] in IMAGE_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(IMAGE_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("ops.") and c["op"] in OPS_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = OPS_BUILDERS[c["op"]](c["params"])(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("initializers.") and c["op"] in INIT_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(INIT_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("nn.") and c["op"] in NN_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = NN_BUILDERS[c["op"]](c["params"])(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("prng.") and c["op"] in PRNG_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(PRNG_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if c["primitive"].startswith("random.") and c["op"] in RANDOM_CORE_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(RANDOM_CORE_BUILDERS[c["op"]](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    return run_case_std(c, seed)


def run_case_std(c, seed):
    op = c["op"]
    is_am = c["primitive"].startswith("arr.") and op in ARRAY_METHODS_BUILDERS
    is_numpy = c["primitive"].startswith("jnp.") and op in NUMPY_BUILDERS
    if is_am or is_numpy or op in LAX_BUILDERS:
        builders = (
            ARRAY_METHODS_BUILDERS
            if is_am
            else (NUMPY_BUILDERS if is_numpy else LAX_BUILDERS)
        )
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(builders[op](c["params"]))(*inputs)
        if isinstance(out, (tuple, list)):
            outs = [np.asarray(o) for o in out]
        else:
            outs = [np.asarray(out)]
        return inputs, outs, [None] * len(outs)
    if op == "result_type":
        operands = []
        stored = []
        for i, a in enumerate(c["args"]):
            if a["weak"]:
                v = weak_scalar(a["dtype"], seed + i)
                operands.append(v)
                canon = jax_dtypes.canonicalize_dtype(np.asarray(v).dtype)
                stored.append(np.asarray(v).astype(canon))
            else:
                arr = draw(a["rng"], seed + i, a["shape"], a["dtype"])
                operands.append(arr)
                stored.append(arr)
        aval = jax.eval_shape(_add_all, *operands)
        out = jax.jit(_add_all)(*operands)
        return stored, [np.asarray(out)], [bool(aval.weak_type)]
    raise SystemExit("unknown op " + op)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sha256sums(outdir):
    entries = []
    for base, dirs, files in os.walk(outdir):
        dirs.sort()
        for f in sorted(files):
            if f == "SHA256SUMS":
                continue
            path = os.path.join(base, f)
            rel = os.path.relpath(path, outdir)
            entries.append((rel, sha256_file(path)))
    entries.sort()
    lines = "".join(digest + "  " + rel + "\n" for rel, digest in entries)
    with open(os.path.join(outdir, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write(lines)


def gen_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and any(
            (not a["weak"]) and ec.is_wide64(a["dtype"]) for a in c["args"]
        ):
            continue
        if x64 and c.get("x64_off_only"):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        stored, outputs, out_weak = run_case(c, seed)
        in_arrays = {a["name"]: np.asarray(v) for a, v in zip(c["args"], stored)}
        out_arrays = {"out" + str(i): np.asarray(o) for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **in_arrays)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        out0 = np.asarray(outputs[0])
        compare, tol = resolve_tol(out0.dtype.name)
        widen = c.get("tol_widen")

        def widen_tol(dtype_name, base_compare, base_tol):
            if widen is None or base_compare != "allclose":
                return base_tol, None
            factor = float(widen["factor"])
            val = TOLERANCES["default"][dtype_name] * factor
            return {"atol": val, "rtol": val}, widen["reason"]

        tol, case_reason = widen_tol(out0.dtype.name, compare, tol)
        args_meta = [
            {
                "name": a["name"],
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
                "rng": a["rng"],
                "weak": a["weak"],
            }
            for a, v in zip(c["args"], stored)
        ]
        outs_meta = []
        for i, o in enumerate(outputs):
            oa = np.asarray(o)
            ocompare, otol = resolve_tol(oa.dtype.name)
            otol, oreason = widen_tol(oa.dtype.name, ocompare, otol)
            entry = {
                "name": "out" + str(i),
                "shape": [int(d) for d in oa.shape],
                "dtype": oa.dtype.name,
                "compare": ocompare,
                "tol": otol,
            }
            if oreason is not None:
                entry["tol_reason"] = oreason
            if out_weak[i] is not None:
                entry["weak"] = out_weak[i]
            outs_meta.append(entry)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "nargs": c["nargs"],
            "args": args_meta,
            "params": c["params"],
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
            "grads": None,
        }
        if case_reason is not None:
            entry["tol_reason"] = case_reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def load_spec(module):
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    cases = list(doc["cases"])
    cases.sort(key=ec.sort_key)
    return cases


def generate(module):
    preflight()
    cases = load_spec(module)
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


LINALG_TOL_FACTOR = 10
LINALG_TOL_REASON = "matmul_reassociation"


def _linalg_tol(dtype_name):
    if ec.is_exact_dtype(dtype_name):
        return "exact", {"atol": 0, "rtol": 0}, None
    base = TOLERANCES["default"][dtype_name]
    v = base * LINALG_TOL_FACTOR
    return "allclose", {"atol": v, "rtol": v}, LINALG_TOL_REASON


def _linalg_out_meta(name, arr):
    a = np.asarray(arr)
    compare, tol, reason = _linalg_tol(a.dtype.name)
    entry = {
        "name": name,
        "shape": [int(d) for d in a.shape],
        "dtype": a.dtype.name,
        "compare": compare,
        "tol": tol,
    }
    if reason is not None:
        entry["tol_reason"] = reason
    return entry


def _linalg_inputs(op, params, npdt, rng):
    jla = jax.lax.linalg
    from jax._src.lax import linalg as jla_src
    if op == "cholesky":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = (m @ m.T + n * np.eye(n)).astype(npdt)
        return {"a": a}, jla.cholesky(a)
    if op == "lu":
        m = params["m"]
        n = params["n"]
        a = rng.standard_normal((m, n)).astype(npdt)
        return {"a": a}, jla.lu(a)
    if op == "qr":
        m = params["m"]
        n = params["n"]
        fm = params["full_matrices"]
        a = rng.standard_normal((m, n)).astype(npdt)
        return {"a": a}, jla.qr(a, full_matrices=fm)
    if op == "householder_product":
        m = params["m"]
        n = params["n"]
        a0 = rng.standard_normal((m, n)).astype(npdt)
        packed, taus = jla_src.geqrf(a0)
        packed = np.asarray(packed)
        taus = np.asarray(taus)
        return {"a": packed, "taus": taus}, jla.householder_product(packed, taus)
    if op == "lu_pivots_to_permutation":
        m = params["m"]
        n = params["n"]
        ps = params["permutation_size"]
        a = rng.standard_normal((m, n)).astype(npdt)
        _, piv, _ = jla.lu(a)
        piv = np.asarray(piv)
        return {"pivots": piv}, jla.lu_pivots_to_permutation(piv, ps)
    if op == "triangular_solve":
        m = params["m"]
        k = params["k"]
        ls = params["left_side"]
        lo = params["lower"]
        ta = params["transpose_a"]
        ud = params["unit_diagonal"]
        a = (rng.standard_normal((m, m)) + m * np.eye(m)).astype(npdt)
        bshape = (m, k) if ls else (k, m)
        b = rng.standard_normal(bshape).astype(npdt)
        out = jla.triangular_solve(
            a,
            b,
            left_side=ls,
            lower=lo,
            transpose_a=ta,
            conjugate_a=False,
            unit_diagonal=ud,
        )
        return {"a": a, "b": b}, out
    if op == "eigh":
        n = params["n"]
        lower = params["lower"]
        m = rng.standard_normal((n, n))
        a = (m + m.T).astype(npdt)
        v, w = jla.eigh(a, lower=lower, symmetrize_input=False)
        return {"a": a}, (v, w)
    if op == "eig":
        n = params["n"]
        cl = params["compute_left"]
        cr = params["compute_right"]
        a = rng.standard_normal((n, n)).astype(npdt)
        out = jla.eig(
            a, compute_left_eigenvectors=cl, compute_right_eigenvectors=cr
        )
        return {"a": a}, out
    if op == "hessenberg":
        n = params["n"]
        a = rng.standard_normal((n, n)).astype(npdt)
        return {"a": a}, jla.hessenberg(a)
    if op == "schur":
        n = params["n"]
        cv = params["compute_schur_vectors"]
        a = rng.standard_normal((n, n)).astype(npdt)
        return {"a": a}, jla.schur(a, compute_schur_vectors=cv)
    if op == "svd":
        m = params["m"]
        n = params["n"]
        fm = params["full_matrices"]
        uv = params["compute_uv"]
        a = rng.standard_normal((m, n)).astype(npdt)
        return {"a": a}, jla.svd(a, full_matrices=fm, compute_uv=uv)
    if op == "tridiagonal":
        n = params["n"]
        lower = params["lower"]
        m = rng.standard_normal((n, n))
        a = (m + m.T).astype(npdt)
        return {"a": a}, jla.tridiagonal(a, lower=lower)
    if op == "tridiagonal_solve":
        m = params["m"]
        k = params["k"]
        d = rng.uniform(4.0, 5.0, m).astype(npdt)
        dl = (rng.standard_normal(m) * 0.5).astype(npdt)
        dl[0] = 0
        du = (rng.standard_normal(m) * 0.5).astype(npdt)
        du[m - 1] = 0
        b = rng.standard_normal((m, k)).astype(npdt)
        out = jla.tridiagonal_solve(dl, d, du, b)
        return {"dl": dl, "d": d, "du": du, "b": b}, out
    raise SystemExit("unknown linalg op " + op)


def run_linalg_case(c):
    seed = zlib.adler32(c["case_id"].encode("utf-8"))
    rng = np.random.RandomState(seed)
    npdt = np.dtype(c["dtype"])
    in_arrays, out = _linalg_inputs(c["op"], c["params"], npdt, rng)
    if isinstance(out, (tuple, list)):
        outputs = [np.asarray(o) for o in out]
    else:
        outputs = [np.asarray(out)]
    return in_arrays, outputs


def gen_linalg_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        case_id = c["case_id"]
        in_arrays, outputs = run_linalg_case(c)
        stored_in = {k: np.asarray(v) for k, v in in_arrays.items()}
        out_arrays = {"out" + str(i): o for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **stored_in)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": k,
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
            }
            for k, v in sorted(stored_in.items())
        ]
        outs_meta = [
            _linalg_out_meta("out" + str(i), o) for i, o in enumerate(outputs)
        ]
        compare, tol, reason = _linalg_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": args_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_linalg(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_linalg_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_linalg_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _ord_val(s):
    return {
        "none": None,
        "fro": "fro",
        "nuc": "nuc",
        "0": 0,
        "1": 1,
        "-1": -1,
        "2": 2,
        "-2": -2,
        "3": 3,
        "inf": np.inf,
        "-inf": -np.inf,
    }[s]


def _norm_axis_val(s):
    return {"none": None, "int0": 0, "matrix": (-2, -1)}[s]


def _well_conditioned(rng, n, npdt):
    return (rng.standard_normal((n, n)) + n * np.eye(n)).astype(npdt)


def _numpy_linalg_inputs(op, params, npdt, rng):
    jla = jnp.linalg
    if op == "cholesky":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = (m @ m.T + n * np.eye(n)).astype(npdt)
        return {"a": a}, jla.cholesky(a, upper=params["upper"])
    if op == "svd":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        out = jla.svd(
            a, full_matrices=params["full_matrices"], compute_uv=params["compute_uv"]
        )
        return {"a": a}, out
    if op == "svdvals":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        return {"a": a}, jla.svdvals(a)
    if op == "solve":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        if params["b_vector"]:
            b = rng.standard_normal(n).astype(npdt)
        else:
            b = rng.standard_normal((n, params["nrhs"])).astype(npdt)
        return {"a": a, "b": b}, jla.solve(a, b)
    if op == "inv":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        return {"a": a}, jla.inv(a)
    if op == "slogdet":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        sign, logabsdet = jla.slogdet(a)
        return {"a": a}, (sign, logabsdet)
    if op == "det":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        return {"a": a}, jla.det(a)
    if op == "eig":
        n = params["n"]
        a = rng.standard_normal((n, n)).astype(npdt)
        w, v = jla.eig(a)
        return {"a": a}, (w, v)
    if op == "eigvals":
        n = params["n"]
        a = rng.standard_normal((n, n)).astype(npdt)
        return {"a": a}, jla.eigvals(a)
    if op == "eigh":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = (m + m.T).astype(npdt)
        w, v = jla.eigh(a, UPLO=params["uplo"])
        return {"a": a}, (w, v)
    if op == "eigvalsh":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = (m + m.T).astype(npdt)
        return {"a": a}, jla.eigvalsh(a, UPLO=params["uplo"])
    if op == "pinv":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        return {"a": a}, jla.pinv(a)
    if op == "matrix_power":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        return {"a": a}, jla.matrix_power(a, params["p"])
    if op == "matrix_rank":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        return {"a": a}, jla.matrix_rank(a)
    if op == "vector_norm":
        x = rng.standard_normal(tuple(params["shape"])).astype(npdt)
        out = jla.vector_norm(
            x,
            axis=_norm_axis_val(params["axis"]),
            keepdims=params["keepdims"],
            ord=_ord_val(params["ord"]),
        )
        return {"x": x}, out
    if op == "norm":
        x = rng.standard_normal(tuple(params["shape"])).astype(npdt)
        out = jla.norm(
            x,
            ord=_ord_val(params["ord"]),
            axis=_norm_axis_val(params["axis"]),
            keepdims=params["keepdims"],
        )
        return {"x": x}, out
    if op == "matrix_norm":
        x = rng.standard_normal(tuple(params["shape"])).astype(npdt)
        out = jla.matrix_norm(x, ord=_ord_val(params["ord"]), keepdims=params["keepdims"])
        return {"x": x}, out
    if op == "matrix_transpose":
        x = rng.standard_normal(tuple(params["shape"])).astype(npdt)
        return {"x": x}, jla.matrix_transpose(x)
    if op == "qr":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        out = jla.qr(a, mode=params["mode"])
        if params["mode"] == "r":
            return {"a": a}, out
        return {"a": a}, (out.Q, out.R)
    if op == "lstsq":
        m = params["m"]
        n = params["n"]
        a = rng.standard_normal((m, n)).astype(npdt)
        if params["b_vector"]:
            b = rng.standard_normal(m).astype(npdt)
        else:
            b = rng.standard_normal((m, params["nrhs"])).astype(npdt)
        x, resid, rank, s = jla.lstsq(a, b)
        return {"a": a, "b": b}, (x, resid, rank, s)
    if op == "cross":
        x1 = rng.standard_normal(3).astype(npdt)
        x2 = rng.standard_normal(3).astype(npdt)
        return {"x1": x1, "x2": x2}, jla.cross(x1, x2)
    if op == "outer":
        x1 = rng.standard_normal(params["m"]).astype(npdt)
        x2 = rng.standard_normal(params["n"]).astype(npdt)
        return {"x1": x1, "x2": x2}, jla.outer(x1, x2)
    if op == "matmul":
        x1 = rng.standard_normal(tuple(params["s1"])).astype(npdt)
        x2 = rng.standard_normal(tuple(params["s2"])).astype(npdt)
        return {"x1": x1, "x2": x2}, jla.matmul(x1, x2)
    if op == "vecdot":
        x1 = rng.standard_normal(params["k"]).astype(npdt)
        x2 = rng.standard_normal(params["k"]).astype(npdt)
        return {"x1": x1, "x2": x2}, jla.vecdot(x1, x2)
    if op == "tensordot":
        x1 = rng.standard_normal(tuple(params["s1"])).astype(npdt)
        x2 = rng.standard_normal(tuple(params["s2"])).astype(npdt)
        return {"x1": x1, "x2": x2}, jla.tensordot(x1, x2, axes=params["axes"])
    if op == "diagonal":
        x = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        return {"x": x}, jla.diagonal(x, offset=params["offset"])
    if op == "trace":
        x = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        return {"x": x}, jla.trace(x, offset=params["offset"])
    if op == "tensorinv":
        a4 = _well_conditioned(rng, 4, npdt)
        a = a4.reshape(2, 2, 4)
        return {"a": a}, jla.tensorinv(a, ind=params["ind"])
    if op == "tensorsolve":
        a4 = _well_conditioned(rng, 4, npdt)
        a = a4.reshape(2, 2, 4)
        b = rng.standard_normal((2, 2)).astype(npdt)
        return {"a": a, "b": b}, jla.tensorsolve(a, b)
    if op == "multi_dot":
        m1 = rng.standard_normal((2, 3)).astype(npdt)
        m2 = rng.standard_normal((3, 4)).astype(npdt)
        m3 = rng.standard_normal((4, 2)).astype(npdt)
        return {"m0": m1, "m1": m2, "m2": m3}, jla.multi_dot([m1, m2, m3])
    if op == "cond":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        return {"a": a}, jla.cond(a, p=_ord_val(params["p"]))
    raise SystemExit("unknown numpy_linalg op " + op)


def run_numpy_linalg_case(c):
    seed = zlib.adler32(c["case_id"].encode("utf-8"))
    rng = np.random.RandomState(seed)
    npdt = np.dtype(c["dtype"])
    in_arrays, out = _numpy_linalg_inputs(c["op"], c["params"], npdt, rng)
    if isinstance(out, (tuple, list)):
        outputs = [np.asarray(o) for o in out]
    else:
        outputs = [np.asarray(out)]
    return in_arrays, outputs


def gen_numpy_linalg_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        case_id = c["case_id"]
        in_arrays, outputs = run_numpy_linalg_case(c)
        stored_in = {k: np.asarray(v) for k, v in in_arrays.items()}
        out_arrays = {"out" + str(i): o for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **stored_in)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": k,
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
            }
            for k, v in sorted(stored_in.items())
        ]
        outs_meta = [
            _linalg_out_meta("out" + str(i), o) for i, o in enumerate(outputs)
        ]
        compare, tol, reason = _linalg_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": args_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_numpy_linalg(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_numpy_linalg_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_numpy_linalg_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _scipy_linalg_inputs(op, params, npdt, rng):
    jsl = jax.scipy.linalg
    if op == "cholesky":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = (m @ m.T + n * np.eye(n)).astype(npdt)
        return {"a": a}, jsl.cholesky(a, lower=params["lower"])
    if op == "cho_factor":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = (m @ m.T + n * np.eye(n)).astype(npdt)
        c, _ = jsl.cho_factor(a, lower=params["lower"])
        return {"a": a}, c
    if op == "cho_solve":
        n = params["n"]
        lower = params["lower"]
        m = rng.standard_normal((n, n))
        a = (m @ m.T + n * np.eye(n)).astype(npdt)
        c, _ = jsl.cho_factor(a, lower=lower)
        c = np.asarray(c).astype(npdt)
        if params["b_vector"]:
            b = rng.standard_normal(n).astype(npdt)
        else:
            b = rng.standard_normal((n, params["nrhs"])).astype(npdt)
        return {"b": b, "c": c}, jsl.cho_solve((c, lower), b)
    if op == "det":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        return {"a": a}, jsl.det(a)
    if op == "inv":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        return {"a": a}, jsl.inv(a)
    if op == "lu":
        m = params["m"]
        n = params["n"]
        a = rng.standard_normal((m, n)).astype(npdt)
        return {"a": a}, jsl.lu(a, permute_l=params["permute_l"])
    if op == "lu_factor":
        n = params["n"]
        a = _well_conditioned(rng, n, npdt)
        lu, piv = jsl.lu_factor(a)
        return {"a": a}, (lu, piv)
    if op == "lu_solve":
        n = params["n"]
        trans = params["trans"]
        a = _well_conditioned(rng, n, npdt)
        lu, piv = jsl.lu_factor(a)
        lu = np.asarray(lu).astype(npdt)
        piv = np.asarray(piv)
        if params["b_vector"]:
            b = rng.standard_normal(n).astype(npdt)
        else:
            b = rng.standard_normal((n, params["nrhs"])).astype(npdt)
        return {"b": b, "lu": lu, "piv": piv}, jsl.lu_solve((lu, piv), b, trans=trans)
    if op == "qr":
        m = params["m"]
        n = params["n"]
        a = rng.standard_normal((m, n)).astype(npdt)
        return {"a": a}, jsl.qr(a, mode=params["mode"], pivoting=False)
    if op == "solve":
        n = params["n"]
        assume_a = params["assume_a"]
        if assume_a == "pos":
            m = rng.standard_normal((n, n))
            a = (m @ m.T + n * np.eye(n)).astype(npdt)
        else:
            a = _well_conditioned(rng, n, npdt)
        if params["b_vector"]:
            b = rng.standard_normal(n).astype(npdt)
        else:
            b = rng.standard_normal((n, params["nrhs"])).astype(npdt)
        return {"a": a, "b": b}, jsl.solve(
            a, b, lower=params["lower"], assume_a=assume_a
        )
    if op == "solve_triangular":
        n = params["n"]
        lower = params["lower"]
        full = (rng.standard_normal((n, n)) + n * np.eye(n)).astype(npdt)
        a = (np.tril(full) if lower else np.triu(full)).astype(npdt)
        if params["b_vector"]:
            b = rng.standard_normal(n).astype(npdt)
        else:
            b = rng.standard_normal((n, params["nrhs"])).astype(npdt)
        out = jsl.solve_triangular(
            a,
            b,
            trans=params["trans"],
            lower=lower,
            unit_diagonal=params["unit_diagonal"],
        )
        return {"a": a, "b": b}, out
    if op == "svd":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        out = jsl.svd(
            a, full_matrices=params["full_matrices"], compute_uv=params["compute_uv"]
        )
        return {"a": a}, out
    if op == "eigh":
        n = params["n"]
        m = rng.standard_normal((n, n))
        a = ((m + m.T) / 2).astype(npdt)
        out = jsl.eigh(a, lower=params["lower"], eigvals_only=params["eigvals_only"])
        return {"a": a}, out
    if op == "schur":
        n = params["n"]
        a = rng.standard_normal((n, n)).astype(npdt)
        return {"a": a}, jsl.schur(a, output=params["output"])
    if op == "block_diag":
        arrs = [rng.standard_normal(tuple(s)).astype(npdt) for s in params["shapes"]]
        inp = {"a" + str(i): v for i, v in enumerate(arrs)}
        return inp, jsl.block_diag(*arrs)
    if op == "toeplitz":
        c = rng.standard_normal(params["m"]).astype(npdt)
        if params["has_r"]:
            r = rng.standard_normal(params["n"]).astype(npdt)
            return {"c": c, "r": r}, jsl.toeplitz(c, r)
        return {"c": c}, jsl.toeplitz(c)
    if op == "hessenberg":
        n = params["n"]
        a = rng.standard_normal((n, n)).astype(npdt)
        return {"a": a}, jsl.hessenberg(a, calc_q=params["calc_q"])
    if op == "expm":
        n = params["n"]
        a = (0.5 * rng.standard_normal((n, n))).astype(npdt)
        if params["upper_triangular"]:
            a = np.triu(a).astype(npdt)
        return {"a": a}, jsl.expm(a, upper_triangular=params["upper_triangular"])
    if op == "polar":
        a = rng.standard_normal((params["m"], params["n"])).astype(npdt)
        out = jsl.polar(a, side=params["side"], method="svd")
        return {"a": a}, out
    if op == "eigh_tridiagonal":
        n = params["n"]
        d = rng.standard_normal(n).astype(npdt)
        e = rng.standard_normal(n - 1).astype(npdt)
        return {"d": d, "e": e}, jsl.eigh_tridiagonal(d, e, eigvals_only=True)
    raise SystemExit("unknown scipy_linalg op " + op)


def run_scipy_linalg_case(c):
    seed = zlib.adler32(c["case_id"].encode("utf-8"))
    rng = np.random.RandomState(seed)
    npdt = np.dtype(c["dtype"])
    in_arrays, out = _scipy_linalg_inputs(c["op"], c["params"], npdt, rng)
    if isinstance(out, (tuple, list)):
        outputs = [np.asarray(o) for o in out]
    else:
        outputs = [np.asarray(out)]
    return in_arrays, outputs


def gen_scipy_linalg_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        case_id = c["case_id"]
        in_arrays, outputs = run_scipy_linalg_case(c)
        stored_in = {k: np.asarray(v) for k, v in in_arrays.items()}
        out_arrays = {"out" + str(i): o for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **stored_in)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": k,
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
            }
            for k, v in sorted(stored_in.items())
        ]
        outs_meta = [
            _linalg_out_meta("out" + str(i), o) for i, o in enumerate(outputs)
        ]
        compare, tol, reason = _linalg_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": args_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_scipy_linalg(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_scipy_linalg_set(module, cases, False, os.path.join(base, "x64_off"))
    n_on = gen_scipy_linalg_set(module, cases, True, os.path.join(base, "x64_on"))
    return n_off, n_on


def _scipy_sparse_linalg_inputs(op, params, npdt, rng):
    jssl = jax.scipy.sparse.linalg
    n = params["n"]
    tol = params.get("tol", 1e-5)
    atol = params.get("atol", 0.0)
    maxiter = params.get("maxiter", None)
    if op == "cg":
        m = rng.standard_normal((n, n))
        a = (m @ m.T + n * np.eye(n)).astype(npdt)
        b = rng.standard_normal(n).astype(npdt)
        stored = {"a": a, "b": b}
        precond = params.get("precond", False)
        mmat = None
        if precond:
            mmat = np.diag(1.0 / np.diag(a)).astype(npdt)
            stored["m"] = mmat
        x, _ = jssl.cg(a, b, tol=tol, atol=atol, maxiter=maxiter, M=mmat)
        return stored, x
    if op == "bicgstab":
        a = (rng.standard_normal((n, n)) + n * np.eye(n)).astype(npdt)
        b = rng.standard_normal(n).astype(npdt)
        x, _ = jssl.bicgstab(a, b, tol=tol, atol=atol, maxiter=maxiter)
        return {"a": a, "b": b}, x
    if op == "gmres":
        a = (rng.standard_normal((n, n)) + n * np.eye(n)).astype(npdt)
        b = rng.standard_normal(n).astype(npdt)
        restart = params.get("restart", 20)
        method = params.get("solve_method", "batched")
        x, _ = jssl.gmres(
            a, b, tol=tol, atol=atol, restart=restart, maxiter=maxiter,
            solve_method=method,
        )
        return {"a": a, "b": b}, x
    raise SystemExit("unknown scipy_sparse_linalg op " + op)


def run_scipy_sparse_linalg_case(c):
    seed = zlib.adler32(c["case_id"].encode("utf-8"))
    rng = np.random.RandomState(seed)
    npdt = np.dtype(c["dtype"])
    in_arrays, out = _scipy_sparse_linalg_inputs(c["op"], c["params"], npdt, rng)
    return in_arrays, [np.asarray(out)]


def gen_scipy_sparse_linalg_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        case_id = c["case_id"]
        in_arrays, outputs = run_scipy_sparse_linalg_case(c)
        stored_in = {k: np.asarray(v) for k, v in in_arrays.items()}
        out_arrays = {"out" + str(i): o for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **stored_in)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": k,
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
            }
            for k, v in sorted(stored_in.items())
        ]
        outs_meta = [
            _linalg_out_meta("out" + str(i), o) for i, o in enumerate(outputs)
        ]
        compare, tol, reason = _linalg_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": args_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_scipy_sparse_linalg(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_scipy_sparse_linalg_set(
        module, cases, False, os.path.join(base, "x64_off")
    )
    n_on = gen_scipy_sparse_linalg_set(
        module, cases, True, os.path.join(base, "x64_on")
    )
    return n_off, n_on


def _scipy_cluster_vq_inputs(op, params, npdt, rng):
    jvq = jax.scipy.cluster.vq
    m = params["m"]
    n = params["n"]
    k = params["k"]
    ndim = params.get("ndim", 2)
    if ndim == 1:
        obs = rng.standard_normal(m).astype(npdt)
        cb = rng.standard_normal(k).astype(npdt)
    else:
        obs = rng.standard_normal((m, n)).astype(npdt)
        cb = rng.standard_normal((k, n)).astype(npdt)
    code, dist = jvq.vq(obs, cb)
    return {"obs": obs, "cb": cb}, [code, dist]


def run_scipy_cluster_vq_case(c):
    seed = zlib.adler32(c["case_id"].encode("utf-8"))
    rng = np.random.RandomState(seed)
    npdt = np.dtype(c["dtype"])
    in_arrays, out = _scipy_cluster_vq_inputs(c["op"], c["params"], npdt, rng)
    return in_arrays, [np.asarray(o) for o in out]


def gen_scipy_cluster_vq_set(module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        case_id = c["case_id"]
        in_arrays, outputs = run_scipy_cluster_vq_case(c)
        stored_in = {k: np.asarray(v) for k, v in in_arrays.items()}
        out_arrays = {"out" + str(i): o for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **stored_in)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": k,
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
            }
            for k, v in sorted(stored_in.items())
        ]
        outs_meta = [
            _linalg_out_meta("out" + str(i), o) for i, o in enumerate(outputs)
        ]
        compare, tol, reason = _linalg_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": args_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_scipy_cluster_vq(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = gen_scipy_cluster_vq_set(
        module, cases, False, os.path.join(base, "x64_off")
    )
    n_on = gen_scipy_cluster_vq_set(
        module, cases, True, os.path.join(base, "x64_on")
    )
    return n_off, n_on


def _vectorize_pyfunc(op):
    if op == "matvec":
        return jnp.vectorize(lambda m, x: m @ x, signature="(n,m),(m)->(n)")
    if op == "cross3":
        return jnp.vectorize(
            lambda a, b: jnp.array(
                [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            ),
            signature="(k),(k)->(k)",
        )
    if op == "add":
        return jnp.vectorize(lambda a, b: a + b)
    raise SystemExit("unknown vectorize op " + op)


def _vectorize_inputs(op, params, npdt, rng):
    f = _vectorize_pyfunc(op)
    a = rng.standard_normal(tuple(params["ashape"])).astype(npdt)
    b = rng.standard_normal(tuple(params["bshape"])).astype(npdt)
    return {"a": a, "b": b}, [np.asarray(f(a, b))]


def _rand_quat(rng, npdt, n):
    if n == 0:
        q = rng.standard_normal(4)
        q = q / np.linalg.norm(q)
        return q.astype(npdt)
    q = rng.standard_normal((n, 4))
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    return q.astype(npdt)


def _spatial_transform_inputs(op, params, npdt, rng):
    import jax.scipy.spatial.transform as _st

    rot = _st.Rotation
    slerp = _st.Slerp
    n = params.get("n", 0)
    deg = params.get("degrees", False)
    if op == "as_quat":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).as_quat()]
    if op == "as_quat_canonical":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).as_quat(canonical=True)]
    if op == "as_matrix":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).as_matrix()]
    if op == "as_rotvec":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).as_rotvec(degrees=deg)]
    if op == "as_mrp":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).as_mrp()]
    if op == "as_euler":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).as_euler(params["seq"], degrees=deg)]
    if op == "magnitude":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).magnitude()]
    if op == "inv":
        q = _rand_quat(rng, npdt, n)
        return {"quat": q}, [rot.from_quat(q).inv().as_quat()]
    if op == "from_rotvec":
        shp = (3,) if n == 0 else (n, 3)
        rv = (0.5 * rng.standard_normal(shp)).astype(npdt)
        return {"rotvec": rv}, [rot.from_rotvec(rv, degrees=deg).as_quat()]
    if op == "from_matrix":
        q = _rand_quat(rng, npdt, n)
        m = np.asarray(rot.from_quat(q).as_matrix()).astype(npdt)
        return {"matrix": m}, [rot.from_matrix(m).as_quat()]
    if op == "from_mrp":
        shp = (3,) if n == 0 else (n, 3)
        mrp = (0.3 * rng.standard_normal(shp)).astype(npdt)
        return {"mrp": mrp}, [rot.from_mrp(mrp).as_quat()]
    if op == "from_euler":
        seq = params["seq"]
        k = len(seq)
        shp = (k,) if n == 0 else (n, k)
        scale = 30.0 if deg else 0.5
        angles = (scale * rng.standard_normal(shp)).astype(npdt)
        return {"angles": angles}, [
            rot.from_euler(seq, angles, degrees=deg).as_quat()
        ]
    if op == "apply":
        q = _rand_quat(rng, npdt, n)
        shp = (3,) if n == 0 else (n, 3)
        vec = rng.standard_normal(shp).astype(npdt)
        return {"quat": q, "vectors": vec}, [
            rot.from_quat(q).apply(vec, inverse=params.get("inverse", False))
        ]
    if op == "compose":
        p = _rand_quat(rng, npdt, n)
        qq = _rand_quat(rng, npdt, n)
        return {"p": p, "q": qq}, [(rot.from_quat(p) * rot.from_quat(qq)).as_quat()]
    if op == "mean":
        q = _rand_quat(rng, npdt, params["n"])
        return {"quat": q}, [rot.from_quat(q).mean().as_quat()]
    if op == "slerp":
        seq = params["seq"]
        m = params["n"]
        angles = (20.0 * rng.standard_normal((m, len(seq)))).astype(npdt)
        kr = rot.from_euler(seq, angles, degrees=True)
        times = np.arange(m).astype(npdt)
        query = np.linspace(0.0, float(m - 1), params["t"]).astype(npdt)
        sl = slerp.init(times, kr)
        out = sl(query).as_euler(seq)
        return {"angles": angles, "times": times, "query": query}, [np.asarray(out)]
    raise SystemExit("unknown spatial_transform op " + op)


def _gen_composed_set(inputs_fn, module, cases, x64, outdir):
    jax.config.update("jax_enable_x64", x64)
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "inputs"))
    os.makedirs(os.path.join(outdir, "outputs"))
    manifest_cases = []
    for c in cases:
        if not x64 and ec.is_wide64(c["dtype"]):
            continue
        case_id = c["case_id"]
        seed = zlib.adler32(case_id.encode("utf-8"))
        rng = np.random.RandomState(seed)
        npdt = np.dtype(c["dtype"])
        in_arrays, out = inputs_fn(c["op"], c["params"], npdt, rng)
        outputs = [np.asarray(o) for o in out]
        stored_in = {k: np.asarray(v) for k, v in in_arrays.items()}
        out_arrays = {"out" + str(i): o for i, o in enumerate(outputs)}
        np.savez(os.path.join(outdir, "inputs", case_id + ".npz"), **stored_in)
        np.savez(os.path.join(outdir, "outputs", case_id + ".npz"), **out_arrays)
        args_meta = [
            {
                "name": k,
                "shape": [int(d) for d in np.asarray(v).shape],
                "dtype": np.asarray(v).dtype.name,
            }
            for k, v in sorted(stored_in.items())
        ]
        outs_meta = [
            _linalg_out_meta("out" + str(i), o) for i, o in enumerate(outputs)
        ]
        compare, tol, reason = _linalg_tol(np.asarray(outputs[0]).dtype.name)
        entry = {
            "case_id": case_id,
            "op": c["op"],
            "primitive": c["primitive"],
            "params": c["params"],
            "args": args_meta,
            "outputs": outs_meta,
            "compare": compare,
            "tol": tol,
        }
        if reason is not None:
            entry["tol_reason"] = reason
        manifest_cases.append(entry)
    manifest = {
        "schema_version": 1,
        "module": module,
        "jax_version": JAX_VERSION,
        "x64": x64,
        "cases": manifest_cases,
    }
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    write_sha256sums(outdir)
    return len(manifest_cases)


def generate_vectorize(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = _gen_composed_set(
        _vectorize_inputs, module, cases, False, os.path.join(base, "x64_off")
    )
    n_on = _gen_composed_set(
        _vectorize_inputs, module, cases, True, os.path.join(base, "x64_on")
    )
    return n_off, n_on


def generate_spatial_transform(module):
    preflight()
    path = os.path.join(ROOT, "spec", module + ".cases.json")
    with open(path, encoding="utf-8") as fh:
        cases = list(json.load(fh)["cases"])
    cases.sort(key=lambda c: c["case_id"])
    base = os.path.join(ROOT, "goldens", module)
    n_off = _gen_composed_set(
        _spatial_transform_inputs, module, cases, False, os.path.join(base, "x64_off")
    )
    n_on = _gen_composed_set(
        _spatial_transform_inputs, module, cases, True, os.path.join(base, "x64_on")
    )
    return n_off, n_on


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_goldens.py <module>")
    if sys.argv[1] == "jaxpr":
        n_off, n_on = generate_jaxpr(sys.argv[1])
    elif sys.argv[1] == "ad":
        n_off, n_on = generate_ad(sys.argv[1])
    elif sys.argv[1] == "batching":
        n_off, n_on = generate_batching(sys.argv[1])
    elif sys.argv[1] == "api":
        n_off, n_on = generate_api(sys.argv[1])
    elif sys.argv[1] == "conditionals":
        n_off, n_on = generate_conditionals(sys.argv[1])
    elif sys.argv[1] == "loops":
        n_off, n_on = generate_loops(sys.argv[1])
    elif sys.argv[1] == "solves":
        n_off, n_on = generate_solves(sys.argv[1])
    elif sys.argv[1] == "linalg":
        n_off, n_on = generate_linalg(sys.argv[1])
    elif sys.argv[1] == "numpy_linalg":
        n_off, n_on = generate_numpy_linalg(sys.argv[1])
    elif sys.argv[1] == "scipy_linalg":
        n_off, n_on = generate_scipy_linalg(sys.argv[1])
    elif sys.argv[1] == "scipy_sparse_linalg":
        n_off, n_on = generate_scipy_sparse_linalg(sys.argv[1])
    elif sys.argv[1] == "scipy_cluster_vq":
        n_off, n_on = generate_scipy_cluster_vq(sys.argv[1])
    elif sys.argv[1] == "vectorize":
        n_off, n_on = generate_vectorize(sys.argv[1])
    elif sys.argv[1] == "spatial_transform":
        n_off, n_on = generate_spatial_transform(sys.argv[1])
    else:
        n_off, n_on = generate(sys.argv[1])
    sys.stdout.write(sys.argv[1] + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
