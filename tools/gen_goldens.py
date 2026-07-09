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


RNG_FACTORIES = {
    "rand_default": rand_default,
    "rand_small": rand_small,
    "rand_positive": rand_positive,
    "rand_nonzero": rand_nonzero,
    "rand_int": rand_int,
    "rand_int_small": rand_int_small,
    "rand_int_small_nz": rand_int_small_nz,
    "rand_bool": rand_bool,
    "rand_index_unique": rand_index_unique,
    "rand_uniform": rand_uniform,
    "rand_gt_one": rand_gt_one,
    "rand_poly_order": rand_poly_order,
    "rand_sorted": rand_sorted,
    "rand_sorted_desc": rand_sorted_desc,
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


NUMPY_BUILDERS = {
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


def run_case(c, seed):
    op = c["op"]
    is_numpy = c["primitive"].startswith("jnp.") and op in NUMPY_BUILDERS
    if is_numpy or op in LAX_BUILDERS:
        builders = NUMPY_BUILDERS if is_numpy else LAX_BUILDERS
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
    else:
        n_off, n_on = generate(sys.argv[1])
    sys.stdout.write(sys.argv[1] + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
