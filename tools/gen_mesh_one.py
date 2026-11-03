import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["XLA_FLAGS"] = (
    "--xla_cpu_enable_fast_math=false"
    " --xla_cpu_use_thunk_runtime=true"
    " --xla_force_host_platform_device_count=1"
)

import argparse
import seeded_inputs as si
import json
import platform

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

PY_VERSION = "3.13.5"
JAX_VERSION = "0.10.2"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cases(mesh):
    row = NamedSharding(mesh, P("x", None))
    col = NamedSharding(mesh, P(None, "x"))

    def body(x):
        return jnp.sin(x) * 2.0 + x.sum()

    def two(x, y):
        return x * y + 1.0

    def constraint(x):
        return jax.lax.with_sharding_constraint(x * 2.0, col)

    def smap(collective):
        return lambda x: jax.shard_map(
            lambda y: jnp.sin(y) + collective(y, "x"), mesh=mesh, in_specs=P("x"), out_specs=P("x")
        )(x)

    return [
        ("sin_sum", body, ["row"], {}),
        ("out_sharded", body, ["row"], {"out_shardings": row}),
        ("out_col", body, ["row"], {"out_shardings": col}),
        ("two_placed", two, ["row", "col"], {}),
        ("one_placed", two, ["row", "host"], {}),
        ("constraint", constraint, ["row"], {}),
        ("shard_map_psum", smap(jax.lax.psum), ["row"], {}),
        ("shard_map_pmax", smap(jax.lax.pmax), ["row"], {}),
        ("shard_map_pmin", smap(jax.lax.pmin), ["row"], {}),
    ]


def hexes(leaves):
    out = []
    for a in leaves:
        h = np.asarray(a)
        out.append({"dtype": str(h.dtype), "shape": list(h.shape), "hex": h.tobytes().hex()})
    return out


def run(x64):
    jax.config.update("jax_enable_x64", x64)
    mesh = Mesh(np.array(jax.devices()[:1]), ("x",))
    base = np.arange(16.0, dtype=np.float32).reshape(4, 4)
    if si.seeded():
        base = si.draw(16, np.float32, [41]).reshape(4, 4)
    args = {
        "row": jax.device_put(base, NamedSharding(mesh, P("x", None))),
        "col": jax.device_put(base[::-1] * 0.5, NamedSharding(mesh, P(None, "x"))),
        "host": base + 3.0,
    }
    rows = []
    for name, fn, kinds, options in cases(mesh):
        def f(*xs, _fn=fn):
            return _fn(*xs)

        f.__name__ = "f"
        xs = [args[k] for k in kinds]
        row = {"id": name, "x64": x64, "args": kinds}
        if si.seeded():
            row["base"] = [float(v) for v in base.reshape(-1)]
        try:
            jitted = jax.jit(f, **options)
            row["hlo"] = jitted.lower(*xs).as_text()
            row["jaxpr"] = str(jax.make_jaxpr(f)(*xs))
            row["outs"] = hexes(jax.tree_util.tree_leaves(jitted(*xs)))
        except Exception as e:
            row["error_class"] = type(e).__name__
            row["message"] = str(e)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(ROOT, "goldens", "mesh_one"))
    args = si.parse(parser)
    if platform.python_version() != PY_VERSION or jax.__version__ != JAX_VERSION:
        raise SystemExit("needs python " + PY_VERSION + " and jax " + JAX_VERSION)
    rows = run(False) + run(True)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "cases.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
