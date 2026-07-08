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


RNG_FACTORIES = {
    "rand_default": rand_default,
    "rand_small": rand_small,
    "rand_positive": rand_positive,
    "rand_nonzero": rand_nonzero,
    "rand_int": rand_int,
    "rand_int_small": rand_int_small,
    "rand_int_small_nz": rand_int_small_nz,
    "rand_bool": rand_bool,
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


def _add_all(*xs):
    return functools.reduce(jnp.add, xs)


def run_case(c, seed):
    op = c["op"]
    if op in LAX_BUILDERS:
        inputs = [
            draw(a["rng"], seed + i, a["shape"], a["dtype"])
            for i, a in enumerate(c["args"])
        ]
        out = jax.jit(LAX_BUILDERS[op](c["params"]))(*inputs)
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
            entry = {
                "name": "out" + str(i),
                "shape": [int(d) for d in oa.shape],
                "dtype": oa.dtype.name,
            }
            if out_weak[i] is not None:
                entry["weak"] = out_weak[i]
            outs_meta.append(entry)
        manifest_cases.append(
            {
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
    else:
        n_off, n_on = generate(sys.argv[1])
    sys.stdout.write(sys.argv[1] + " x64_off " + str(n_off) + " x64_on " + str(n_on) + "\n")


if __name__ == "__main__":
    main()
