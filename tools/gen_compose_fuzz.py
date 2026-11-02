import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
import warnings
import zlib

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import compare_text
import emit_common as ec
import gen_goldens as gg
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src import core as jax_core

warnings.filterwarnings("ignore")

MODULE = "compose_fuzz"
DEFAULT_SEED = 20261102
SETS = (("x64_off", False), ("x64_on", True))
CASE_TIMEOUT = 90
COMPOSE_CASES = 1000
PRNG_CASES = 60

UNARY = {
    "sin": jnp.sin,
    "cos": jnp.cos,
    "tanh": jnp.tanh,
    "exp": jnp.exp,
    "expm1": jnp.expm1,
    "sinh": jnp.sinh,
    "cosh": jnp.cosh,
    "arctan": jnp.arctan,
    "square": jnp.square,
    "abs": jnp.abs,
    "sum": jnp.sum,
    "mean": jnp.mean,
    "prod": jnp.prod,
    "max": jnp.max,
    "min": jnp.min,
    "var": jnp.var,
    "cumsum": jnp.cumsum,
    "cumprod": jnp.cumprod,
    "sort": jnp.sort,
    "log1p_sq": lambda x: jnp.log1p(jnp.square(x)),
    "sqrt_abs": lambda x: jnp.sqrt(jnp.abs(x)),
    "softmax": jax.nn.softmax,
    "logsumexp": jax.scipy.special.logsumexp,
    "sigmoid": jax.nn.sigmoid,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "softplus": jax.nn.softplus,
    "erf": jax.scipy.special.erf,
    "gammaln": jax.scipy.special.gammaln,
    "where_sq": lambda x: jnp.where(jnp.greater(x, 0.0), jnp.square(x), x),
    "clip": lambda x: jnp.clip(x, -1.0, 1.0),
    "flip": jnp.flip,
    "ravel": jnp.ravel,
    "transpose": jnp.transpose,
    "concat": lambda x: jnp.concatenate([x, x]),
    "stack": lambda x: jnp.stack([x, x]),
    "norm": jnp.linalg.norm,
    "self_dot": lambda x: jnp.dot(x, x),
    "self_matmul": lambda x: jnp.matmul(x, x),
    "det": jnp.linalg.det,
    "inv": jnp.linalg.inv,
    "cholesky": lambda x: jnp.linalg.cholesky(jnp.matmul(x, jnp.transpose(x))),
}

BINARY = {
    "add": jnp.add,
    "subtract": jnp.subtract,
    "multiply": jnp.multiply,
    "divide": jnp.divide,
    "power": lambda x, y: jnp.power(jnp.abs(x), y),
    "arctan2": jnp.arctan2,
    "hypot": jnp.hypot,
    "logaddexp": jnp.logaddexp,
    "maximum": jnp.maximum,
    "minimum": jnp.minimum,
    "dot": jnp.dot,
    "matmul": jnp.matmul,
    "where_gt": lambda x, y: jnp.where(jnp.greater(x, y), x, y),
}

TRANSFORMS = ("jit", "vmap", "grad", "jvp", "vjp", "jacfwd", "jacrev", "hessian", "value_and_grad", "checkpoint", "custom_jvp")
CONTROL = ("scan", "while_loop", "cond", "switch")
ARG_SHAPES = ((), (3,), (2,), (2, 3), (3, 3), (3, 2), (4,))

PROMOTE_OPS = ("add", "subtract", "multiply", "true_divide", "floor_divide", "power", "maximum", "less", "arctan2", "logaddexp", "remainder", "bitwise_and")
STRONG = {
    "x64_off": ("bool", "uint8", "uint16", "uint32", "int8", "int16", "int32", "float16", "bfloat16", "float32", "complex64"),
    "x64_on": ("bool", "uint8", "uint16", "uint32", "uint64", "int8", "int16", "int32", "int64", "float16", "bfloat16", "float32", "float64", "complex64", "complex128"),
}
PYTHON_SCALARS = (("bool", True), ("int", 2), ("float", 1.5), ("complex", [1.5, -0.5]))
IMPLS = ("threefry2x32", "rbg", "unsafe_rbg")
DRAWS = ("uniform", "normal", "bits", "randint", "bernoulli", "permutation")


def digest(text):
    return zlib.adler32(text.encode("utf-8"))


def named(f):
    f.__name__ = "g"
    return f


def tree_sum(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    total = jnp.sum(leaves[0])
    for leaf in leaves[1:]:
        total = jnp.add(total, jnp.sum(leaf))
    return total


def scalarized(f):
    def g(*a):
        return tree_sum(f(*a))

    return named(g)


def step_value(f, c, rest):
    return jnp.add(c, jnp.multiply(0.125, tree_sum(f(c, *rest))))


def apply_layer(layer, f):
    t = layer["t"]
    if t == "jit":
        return named(jax.jit(f))
    if t == "vmap":
        return named(jax.vmap(f, in_axes=tuple(layer["in_axes"]), out_axes=layer["out_axes"]))
    if t == "grad":
        return named(jax.grad(scalarized(f), argnums=layer["argnum"]))
    if t == "value_and_grad":
        return named(jax.value_and_grad(scalarized(f), argnums=layer["argnum"]))
    if t == "jacfwd":
        return named(jax.jacfwd(f, argnums=layer["argnum"]))
    if t == "jacrev":
        return named(jax.jacrev(f, argnums=layer["argnum"]))
    if t == "hessian":
        return named(jax.hessian(scalarized(f), argnums=layer["argnum"]))
    if t == "checkpoint":
        return named(jax.checkpoint(f))
    if t == "jvp":

        def g(*a):
            return jax.jvp(f, a, tuple(jnp.full_like(x, 0.5) for x in a))

        return named(g)
    if t == "vjp":

        def g(*a):
            y, pull = jax.vjp(f, *a)
            return (y, pull(jax.tree_util.tree_map(jnp.ones_like, y)))

        return named(g)
    if t == "custom_jvp":
        wrapped = jax.custom_jvp(f)

        def rule(primals, tangents):
            y, dy = jax.jvp(f, primals, tangents)
            return y, jax.tree_util.tree_map(lambda v: jnp.multiply(v, 1.5), dy)

        wrapped.defjvp(rule)
        wrapped.__name__ = "g"
        return wrapped
    if t == "scan":

        def g(*a):
            def body(c, _):
                s = step_value(f, c, a[1:])
                return s, jnp.sum(s)

            return lax.scan(body, a[0], None, length=layer["length"])

        return named(g)
    if t == "while_loop":

        def g(*a):
            def test(state):
                return jnp.less(state[0], layer["length"])

            def body(state):
                return (jnp.add(state[0], 1), step_value(f, state[1], a[1:]))

            return lax.while_loop(test, body, (jnp.asarray(0), a[0]))[1]

        return named(g)
    if t == "cond":

        def g(*a):
            return lax.cond(jnp.greater(jnp.sum(a[0]), 0.0), scalarized(f), lambda *x: jnp.sum(x[0]), *a)

        return named(g)
    if t == "switch":

        def g(*a):
            total = jnp.sum(a[0])
            index = jnp.add(jnp.greater(total, 0.0).astype(jnp.int32), jnp.greater(total, 1.0).astype(jnp.int32))
            return lax.switch(index, [scalarized(f), lambda *x: jnp.sum(x[0]), lambda *x: jnp.sum(jnp.sin(x[0]))], *a)

        return named(g)
    raise SystemExit("unknown layer " + t)


def compose_program(params):
    base = params["base"]
    f = UNARY[base] if base in UNARY else BINARY[base]

    def inner(*a):
        return f(*a)

    g = named(inner)
    for layer in params["layers"]:
        g = apply_layer(layer, g)
    return g


def promote_operand(spec, args, position):
    if "arg" in spec:
        return args[position]
    if spec["py"] == "complex":
        return complex(spec["v"][0], spec["v"][1])
    return spec["v"]


def promote_program(params):
    op = getattr(jnp, params["binop"])

    def g(*a):
        right = 1 if "arg" in params["lhs"] else 0
        return op(promote_operand(params["lhs"], a, 0), promote_operand(params["rhs"], a, right))

    return named(g)


def prng_program(params):
    impl = params["impl"]

    def g(seed):
        key = jax.random.key(seed, impl=impl)
        outs = []
        for step in params["steps"]:
            if step["op"] == "fold_in":
                key = jax.random.fold_in(key, step["data"])
            elif step["op"] == "split":
                key = jax.random.split(key, step["num"])[step["index"]]
            elif step["op"] == "vmap_fold_in":
                data = step["data"]
                key = jax.vmap(lambda k, d=data: jax.random.fold_in(k, d))(jax.random.split(key, step["num"]))[step["index"]]
            outs.append(jax.random.key_data(key))
        for draw in params["draws"]:
            name = draw["op"]
            shape = tuple(draw["shape"])
            if name == "uniform":
                outs.append(jax.random.uniform(key, shape))
            elif name == "normal":
                outs.append(jax.random.normal(key, shape))
            elif name == "bits":
                outs.append(jax.random.bits(key, shape))
            elif name == "randint":
                outs.append(jax.random.randint(key, shape, -5, 100))
            elif name == "bernoulli":
                outs.append(jax.random.bernoulli(key, 0.25, shape))
            elif name == "permutation":
                outs.append(jax.random.permutation(key, 5))
        return tuple(outs)

    return named(g)


def program(case):
    return {"compose": compose_program, "promote": promote_program, "prng": prng_program}[case["op"]](case["params"])


def random_layer(rng, nargs, rank):
    t = str(rng.choice(TRANSFORMS))
    if t == "vmap":
        pick = rng.random()
        pool = [0, 0, 0, -1, 1, None]
        axes = [pool[int(rng.integers(0, len(pool)))] for _ in range(nargs)]
        if pick < 0.9 and all(a is None for a in axes):
            axes[int(rng.integers(0, nargs))] = 0
        return {"t": t, "in_axes": axes, "out_axes": [0, 0, -1, 1][int(rng.integers(0, 4))]}
    if t in ("grad", "value_and_grad", "jacfwd", "jacrev", "hessian"):
        return {"t": t, "argnum": int(rng.integers(0, nargs))}
    return {"t": t}


def control_layer(rng):
    t = str(rng.choice(CONTROL))
    if t in ("scan", "while_loop"):
        return {"t": t, "length": int(rng.integers(1, 4))}
    return {"t": t}


def compose_cases(seed, set_name, dtype):
    rng = np.random.default_rng([seed, digest("compose" + set_name + dtype)])
    unary = sorted(UNARY)
    binary = sorted(BINARY)
    out = []
    for i in range(COMPOSE_CASES):
        nargs = 1 if rng.random() < 0.6 else 2
        base = (unary if nargs == 1 else binary)[int(rng.integers(0, len(unary if nargs == 1 else binary)))]
        shape = ARG_SHAPES[int(rng.integers(0, len(ARG_SHAPES)))]
        other = shape if rng.random() < 0.85 else ARG_SHAPES[int(rng.integers(0, len(ARG_SHAPES)))]
        shapes = [list(shape), list(other)][:nargs]
        depth = int(rng.integers(1, 4))
        layers = [random_layer(rng, nargs, len(shape)) for _ in range(depth)]
        if rng.random() < 0.3:
            layers.insert(int(rng.integers(0, len(layers) + 1)), control_layer(rng))
        variant = "edge" if rng.random() < 0.15 else "normal"
        out.append(
            {
                "case_id": "compose_" + format(i, "05d") + "_" + dtype,
                "op": "compose",
                "dtype": dtype,
                "variant": variant,
                "params": {"base": base, "layers": layers},
                "shapes": shapes,
            }
        )
    return out


def promote_cases(set_name):
    out = []
    strong_left = [{"arg": 0, "dtype": d, "shape": [3]} for d in STRONG[set_name]]
    strong_right = [{"arg": 1, "dtype": d, "shape": []} for d in STRONG[set_name]]
    python = [{"py": n, "v": v} for n, v in PYTHON_SCALARS]

    def tag(spec):
        return spec.get("dtype", "py" + spec.get("py", "")) + ("s" if spec.get("shape") == [] else "")

    for binop in PROMOTE_OPS:
        for lhs in strong_left + python:
            for rhs in strong_right + python:
                out.append(
                    {
                        "case_id": "promote_" + binop + "_" + tag(lhs) + "_" + tag(rhs),
                        "op": "promote",
                        "dtype": "mixed",
                        "variant": "promote",
                        "params": {"binop": binop, "lhs": lhs, "rhs": rhs},
                    }
                )
    return out


def prng_cases(seed, set_name):
    rng = np.random.default_rng([seed, digest("prng" + set_name)])
    out = []
    for impl in IMPLS:
        for i in range(PRNG_CASES):
            steps = []
            for _ in range(int(rng.integers(1, 5))):
                kind = str(rng.choice(("fold_in", "split", "vmap_fold_in")))
                if kind == "fold_in":
                    steps.append({"op": kind, "data": int(rng.integers(0, 2**31 - 1))})
                elif kind == "split":
                    num = int(rng.integers(2, 5))
                    steps.append({"op": kind, "num": num, "index": int(rng.integers(0, num))})
                else:
                    num = int(rng.integers(2, 4))
                    steps.append({"op": kind, "num": num, "index": int(rng.integers(0, num)), "data": int(rng.integers(0, 1000))})
            draws = []
            for name in DRAWS:
                if rng.random() < 0.6:
                    shape = [[], [3], [2, 2]][int(rng.integers(0, 3))]
                    draws.append({"op": name, "shape": shape})
            out.append(
                {
                    "case_id": "prng_" + impl + "_" + format(i, "03d"),
                    "op": "prng",
                    "dtype": "int32",
                    "variant": "prng",
                    "params": {"impl": impl, "steps": steps, "draws": draws},
                    "seed_value": int(rng.integers(0, 2**31 - 1)),
                }
            )
    return out


def expand(seed):
    cases = {}
    for set_name, x64 in SETS:
        dtype = "float64" if x64 else "float32"
        cases[set_name] = compose_cases(seed, set_name, dtype) + prng_cases(seed, set_name) + promote_cases(set_name)
        cases[set_name].sort(key=lambda c: c["case_id"])
    return cases


def stored_cases(work, seed):
    path = os.path.join(work, "cases.json")
    seed_path = os.path.join(work, "seed")
    if os.path.exists(seed_path):
        with open(seed_path, encoding="utf-8") as fh:
            if int(fh.read()) != seed:
                raise SystemExit("work directory " + work + " was made with another seed")
    else:
        with open(seed_path, "w", encoding="utf-8") as fh:
            fh.write(str(seed))
    if not os.path.exists(path):
        with open(path + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(expand(seed), fh)
        os.replace(path + ".tmp", path)
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def float_values(rng, shape, name, variant):
    normal = rng.standard_normal(shape).astype(name)
    if variant != "edge":
        return normal
    info = np.finfo(name)
    pool = np.array([0.0, -0.0, np.inf, -np.inf, np.nan, float(info.tiny), float(info.max), 1.0, -1.0], dtype=name)
    return np.where(rng.random(shape) < 0.4, pool[rng.integers(0, len(pool), shape)], normal)


def strong_values(dtype, shape):
    base = np.array([1, 2, 3])[: (shape[0] if shape else 1)]
    if not shape:
        base = np.array(3)
    if dtype == "bool":
        return (base % 2 == 1).astype("bool")
    if dtype == "bfloat16":
        return np.asarray(jnp.asarray(base.astype("float32"), dtype=jnp.bfloat16))
    return base.astype(dtype)


def make_inputs(case, seed):
    rng = np.random.default_rng([seed, digest(case["case_id"]), 1])
    if case["op"] == "compose":
        return {"a" + str(i): float_values(rng, tuple(s), case["dtype"], case["variant"]) for i, s in enumerate(case["shapes"])}
    if case["op"] == "prng":
        return {"a0": np.asarray(case["seed_value"], dtype="int32")}
    arrays = {}
    for side in ("lhs", "rhs"):
        spec = case["params"][side]
        if "arg" in spec:
            arrays["a" + str(spec["arg"])] = strong_values(spec["dtype"], spec["shape"])
    return arrays


def to_outputs(result):
    return {"out" + str(i): np.asarray(o, order="C") for i, o in enumerate(jax.tree_util.tree_leaves(result))}


def meta(arrays):
    return [{"name": k, "dtype": v.dtype.name, "shape": [int(d) for d in v.shape]} for k, v in arrays.items()]


def run_case(case, arrays):
    names = sorted(arrays)
    values = [arrays[k] for k in names]
    entry = {"case_id": case["case_id"], "op": case["op"], "dtype": case["dtype"], "variant": case["variant"], "params": case["params"], "args": meta({k: arrays[k] for k in names})}
    saved = {"in_" + k: arrays[k] for k in names}
    fn = program(case)
    try:
        eager = to_outputs(fn(*[jnp.asarray(v) for v in values]))
        entry["eager"] = meta(eager)
        saved.update({"eager_" + k: v for k, v in eager.items()})
    except Exception as error:
        entry["eager_error"] = type(error).__name__
    try:
        jitted = to_outputs(jax.jit(program(case))(*values))
        entry["jit"] = meta(jitted)
        saved.update({"jit_" + k: v for k, v in jitted.items()})
    except Exception as error:
        entry["jit_error"] = type(error).__name__
    row = compare_text._reference_case(program(case), values, jax, jax_core)
    row.update({"case": case["case_id"]})
    return entry, saved, row


def take_core():
    spec = os.environ.get("OJAX_SLOT_CORES", "2-17")
    low, _, high = spec.partition("-")
    directory = os.environ.get("OJAX_SLOT_DIR", "/tmp/ojax_slots")
    os.makedirs(directory, exist_ok=True)
    while True:
        for core in range(int(low), int(high or low) + 1):
            fd = os.open(os.path.join(directory, str(core)), os.O_CREAT | os.O_RDWR, 0o666)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return core, fd
            except OSError:
                os.close(fd)
        time.sleep(0.2)


def worker(set_name, shard, shards, work, seed):
    core, _fd = take_core()
    os.sched_setaffinity(0, {core})
    jax.config.update("jax_enable_x64", set_name == "x64_on")
    cases = stored_cases(work, seed)[set_name]
    directory = os.path.join(work, set_name)
    current = os.path.join(directory, "current." + str(shard))
    count = 0
    for position, case in enumerate(cases):
        if position % shards != shard:
            continue
        stem = os.path.join(directory, case["case_id"])
        if os.path.exists(stem + ".json"):
            continue
        with open(current, "w", encoding="utf-8") as fh:
            fh.write(case["case_id"])
        arrays = make_inputs(case, seed)
        entry, saved, row = run_case(case, arrays)
        np.savez(stem + ".tmp.npz", **saved)
        os.replace(stem + ".tmp.npz", stem + ".npz")
        with open(stem + ".text.tmp", "w", encoding="utf-8") as fh:
            json.dump(row, fh, sort_keys=True)
        os.replace(stem + ".text.tmp", stem + ".text")
        with open(stem + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        os.replace(stem + ".tmp", stem + ".json")
        count += 1
        if count % 50 == 0:
            jax.clear_caches()
    with open(current, "w", encoding="utf-8") as fh:
        fh.write("")


def launch(set_name, shard, shards, work, seed):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.Popen(["nice", "-n", "10", sys.executable, os.path.abspath(__file__), "--worker", set_name, str(shard), str(shards), work, str(seed)], env=env)


def supervise(set_name, shards, work, seed, limit):
    directory = os.path.join(work, set_name)
    os.makedirs(directory, exist_ok=True)
    running = {i: launch(set_name, i, shards, work, seed) for i in range(shards)}
    started = time.time()
    while running:
        time.sleep(1)
        if time.time() - started > limit:
            for proc in running.values():
                proc.kill()
                proc.wait()
            return False
        for shard, proc in list(running.items()):
            path = os.path.join(directory, "current." + str(shard))
            status = proc.poll()
            if status == 0:
                del running[shard]
                continue
            stalled = status is None and os.path.exists(path) and time.time() - os.path.getmtime(path) > CASE_TIMEOUT
            if status is None and not stalled:
                continue
            if stalled:
                proc.kill()
                proc.wait()
            name = open(path, encoding="utf-8").read().strip() if os.path.exists(path) else ""
            if name:
                reason = "timeout" if stalled else "crash " + str(status)
                sys.stdout.write("unrunnable " + set_name + " " + name + " " + reason + "\n")
                with open(os.path.join(directory, name + ".json"), "w", encoding="utf-8") as fh:
                    json.dump({"case_id": name, "unrunnable": reason}, fh)
                os.remove(path)
            running[shard] = launch(set_name, shard, shards, work, seed)
    return True


def assemble(set_name, x64, work, seed, outdir, sink):
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(os.path.join(outdir, "ops"))
    cases = stored_cases(work, seed)[set_name]
    directory = os.path.join(work, set_name)
    entries = []
    unrunnable = []
    by_op = {}
    for case in cases:
        path = os.path.join(directory, case["case_id"])
        with open(path + ".json", encoding="utf-8") as fh:
            entry = json.load(fh)
        if "unrunnable" in entry:
            unrunnable.append({"case_id": case["case_id"], "reason": entry["unrunnable"]})
            continue
        entries.append(entry)
        by_op.setdefault(case["op"], []).append(case["case_id"])
        with open(path + ".text", encoding="utf-8") as fh:
            row = json.load(fh)
        row["set"] = MODULE + "/" + set_name
        sink.write(json.dumps(row, sort_keys=True) + "\n")
    for op, ids in by_op.items():
        bundle = {}
        for case_id in ids:
            with np.load(os.path.join(directory, case_id + ".npz")) as stored:
                for member in stored.files:
                    bundle[case_id + "/" + member] = stored[member]
        np.savez(os.path.join(outdir, "ops", op + ".npz"), **bundle)
    manifest = {"schema_version": 1, "module": MODULE, "seed": seed, "jax_version": gg.JAX_VERSION, "x64": x64, "unrunnable": unrunnable, "cases": entries}
    with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as fh:
        fh.write(ec.canonical_dumps(manifest))
    gg.write_sha256sums(outdir)
    return len(entries), len(unrunnable)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6]))
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--work", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--limit", type=int, default=560)
    parser.add_argument("--output-name", default=MODULE)
    args = parser.parse_args()
    gg.preflight()
    os.makedirs(args.work, exist_ok=True)
    stored_cases(args.work, args.seed)
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), args.output_name)
    started = time.time()
    for set_name, _ in SETS:
        if not supervise(set_name, args.jobs, args.work, args.seed, args.limit - (time.time() - started)):
            sys.stdout.write("time limit reached, run again to continue\n")
            sys.exit(3)
    os.makedirs(os.path.join(base, "text"), exist_ok=True)
    with open(os.path.join(base, "text", "reference.jsonl"), "w", encoding="utf-8") as sink:
        for set_name, x64 in SETS:
            count, unrunnable = assemble(set_name, x64, args.work, args.seed, os.path.join(base, set_name), sink)
            sys.stdout.write(MODULE + " " + set_name + " " + str(count) + " cases, " + str(unrunnable) + " unrunnable\n")


if __name__ == "__main__":
    main()
