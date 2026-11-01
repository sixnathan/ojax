import json
import os
import re
import subprocess
import sys

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ.setdefault("JAX_PLATFORMS", "cpu")


def reference():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    devices = jax.devices()
    mesh22 = Mesh(np.array(devices[:4]).reshape(2, 2), ("a", "b"))
    mesh2 = Mesh(np.array(devices[:2]), ("x",))
    mesh4 = Mesh(np.array(devices[:4]), ("i",))
    ab = jax.device_put(np.arange(16.0, dtype=np.float32).reshape(4, 4), NamedSharding(mesh22, P("a", "b")))
    x_only = jax.device_put(np.arange(8.0, dtype=np.float32).reshape(4, 2), NamedSharding(mesh2, P("x", None)))
    i_only = jax.device_put(np.arange(8.0, dtype=np.float32), NamedSharding(mesh4, P("i")))

    def body(x):
        return jnp.sin(x) * 2.0 + x.sum()

    def two_results(x, y):
        return jnp.dot(x, y.T), jnp.exp(y)

    def unused_argument(x, y):
        return x * 2.0

    def constraint(x):
        return jax.lax.with_sharding_constraint(x * 2.0, NamedSharding(mesh22, P(None, "a")))

    def shard_map(x):
        return jax.shard_map(lambda y: jnp.sin(y) + jax.lax.psum(y, "i"), mesh=mesh4, in_specs=P("i"), out_specs=P("i"))(x)

    def scan_in_shard_map(x):
        def inner(y):
            def body(carry, elt):
                return carry + elt, jax.lax.psum(elt, "i")

            return jax.lax.scan(body, jax.lax.pcast(jnp.float32(0.0), ("i",), to="varying"), y)[1]

        return jax.shard_map(inner, mesh=mesh4, in_specs=P("i"), out_specs=P())(x)

    cases = [
        ("sin_sum", body, [ab], {}),
        ("out_sharded", body, [ab], {"out_shardings": NamedSharding(mesh22, P("a", None))}),
        ("two_results", two_results, [x_only, x_only], {}),
        ("unused_argument", unused_argument, [ab, ab], {}),
        ("constraint", constraint, [ab], {}),
        ("shard_map", shard_map, [i_only], {}),
        ("scan_in_shard_map", scan_in_shard_map, [i_only], {}),
    ]
    rows = {}
    for name, fn, args, options in cases:
        fn = lambda *xs, _fn=fn: _fn(*xs)
        fn.__name__ = "f"
        jitted = jax.jit(fn, **options)
        leaves = jax.tree_util.tree_leaves(jitted(*args))
        rows[name] = {
            "text": jitted.lower(*args).as_text(),
            "values": [[str(int(np.float64(v).view(np.int64))) for v in np.asarray(leaf).ravel()] for leaf in leaves],
        }
    return rows


def main():
    probe = sys.argv[1]
    out = sys.argv[2]
    subprocess.run([probe, out], check=True, env=dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=8"))
    ours = {row["case"]: row for row in map(json.loads, open(out, encoding="utf-8"))}
    rows = reference()
    same_text = same_values = 0
    for name, ref in rows.items():
        mine = ours.get(name, {})
        t = mine.get("text") == ref["text"]
        v = mine.get("values") == ref["values"]
        same_text += t
        same_values += v
        print(name, "text", "identical" if t else "DIFFERENT", "values", "identical" if v else "DIFFERENT")
        if not t and "text" in mine:
            for a, b in zip(ref["text"].split("\n"), mine["text"].split("\n")):
                if a != b:
                    print(" JAX ", a[:300])
                    print(" OJAX", b[:300])
                    break
        if "error" in mine:
            print(" error", mine["error"][:300])
    print("text", same_text, "of", len(rows), "values", same_values, "of", len(rows))


if __name__ == "__main__":
    main()
