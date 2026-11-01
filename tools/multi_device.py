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

    mesh_tail = Mesh(np.array(devices[4:8]), ("i",))
    tail_only = jax.device_put(np.arange(8.0, dtype=np.float32), NamedSharding(mesh_tail, P("i")))

    def callback_pure(x):
        y = jax.pure_callback(lambda a: np.asarray(a) * 2, jax.ShapeDtypeStruct((8,), jnp.float32), x)
        return y + 1.0

    seen = []

    def callback_debug(x):
        jax.debug.callback(lambda a: seen.append(np.asarray(a)), x)
        return x * 2.0

    def shard_unused_input(x, y):
        return jax.shard_map(lambda a, b: jnp.sin(a), mesh=mesh4, in_specs=(P("i"), P("i")), out_specs=P("i"))(x, y)

    def shard_unused_output(x):
        return jax.shard_map(lambda a: (jnp.sin(a), jnp.cos(a)), mesh=mesh4, in_specs=P("i"), out_specs=(P("i"), P("i")))(x)[1]

    def shard_dead_eqns(x):
        def inner(a):
            jnp.cos(a)
            return jnp.sin(a)

        return jax.shard_map(inner, mesh=mesh4, in_specs=P("i"), out_specs=P("i"))(x)

    def shard_all_dead(x):
        jax.shard_map(jnp.sin, mesh=mesh4, in_specs=P("i"), out_specs=P("i"))(x)
        return jnp.cos(x)

    def _callback_quiet(x):
        jax.debug.callback(lambda a: None, x)
        return x * 2.0

    cases = [
        ("shard_unused_input", shard_unused_input, [i_only, i_only], {}),
        ("shard_unused_output", shard_unused_output, [i_only], {}),
        ("shard_dead_eqns", shard_dead_eqns, [i_only], {}),
        ("shard_all_dead", shard_all_dead, [i_only], {}),
        ("callback_pure", callback_pure, [i_only], {}),
        ("callback_debug", callback_debug, [i_only], {}),
        ("callback_tail_mesh", lambda x: _callback_quiet(x), [tail_only], {}),
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
    jax.effects_barrier()
    rows["callback_seen"] = {
        "text": "",
        "values": [[[str(int(np.float64(v).view(np.int64))) for v in a.ravel()]] for a in seen],
    }
    from jax import export

    def staged(x, y):
        a = jnp.sin(x) * 2.0 + 2.0 + jnp.where(x > 0, x, 1.0)

        def body(c, _):
            return c * 3.0 + 1.0, None

        return jax.lax.scan(body, a, None, length=2)[0]

    zeros = np.zeros(3, np.float32)
    lam = lambda x, y: staged(x, y)
    rows["stages_as_text"] = {"text": jax.jit(lam).lower(zeros, zeros).as_text(), "values": []}
    staged.__name__ = "f"
    exported = export.export(jax.jit(staged))(zeros, zeros)
    rows["export_scan"] = {"text": _strip_locations(exported.mlir_module()), "values": []}

    def staged_where(x, y):
        return jnp.sin(x) * 2.0 + 2.0 + jnp.where(x > 0, x, 1.0)

    lam_where = lambda x, y: staged_where(x, y)
    rows["stages_as_text_where"] = {"text": jax.jit(lam_where).lower(zeros, zeros).as_text(), "values": []}
    staged_where.__name__ = "f"
    rows["export_where"] = {
        "text": _strip_locations(export.export(jax.jit(staged_where))(zeros, zeros).mlir_module()),
        "values": [],
    }
    rows["export_kept"] = {"text": ",".join(map(str, exported.module_kept_var_idx)), "values": []}

    platforms = ["cpu", "cuda"]

    def h(x):
        return jnp.sin(x)

    def h2(x):
        return jax.lax.platform_dependent(x, cpu=lambda a: a * 2.0, default=lambda a: a * 3.0)

    z3 = np.zeros(3, np.float32)
    rows["export_multi_sin"] = {"text": _strip_locations(export.export(jax.jit(h), platforms=platforms)(z3).mlir_module()), "values": []}
    rows["export_multi_dependent"] = {
        "text": _strip_locations(export.export(jax.jit(h2), platforms=platforms)(z3).mlir_module()),
        "values": [],
    }
    rows["export_single_dependent"] = {
        "text": _strip_locations(export.export(jax.jit(h2), platforms=["cpu"])(z3).mlir_module()),
        "values": [],
    }
    rows["lower_dependent"] = {"text": jax.jit(h2).lower(z3).as_text(), "values": []}
    multi = export.export(jax.jit(staged), platforms=platforms)(zeros, zeros)
    rows["export_multi_scan"] = {"text": _strip_locations(multi.mlir_module()), "values": []}
    def nested_scan(x):
        def body(c, e):
            return jax.lax.scan(lambda d, _: (d + e, None), c, None, length=2)[0], e * 3.0

        return jax.lax.scan(body, jnp.float32(0.0), x)

    def scan_cond(x):
        def body(c, e):
            return c + jax.lax.cond(e > 0.0, jnp.sin, jnp.cos, e), None

        return jax.lax.scan(body, jnp.float32(0.0), x)[0]

    def while_body(x):
        return jax.lax.while_loop(lambda s: s[0] < 3, lambda s: (s[0] + 1, jnp.sin(s[1])), (0, x))[1]

    for name, fn in (("nested_scan", nested_scan), ("scan_cond", scan_cond), ("while", while_body)):
        lam = lambda x, _fn=fn: _fn(x)
        rows["stages_" + name] = {"text": jax.jit(lam).lower(z3).as_text(), "values": []}
        fn.__name__ = "f"
        rows["export_" + name] = {"text": _strip_locations(export.export(jax.jit(fn))(z3).mlir_module()), "values": []}
    scan_cond.__name__ = "f"
    while_body.__name__ = "f"
    rows["export_multi_scan_cond"] = {"text": _strip_locations(export.export(jax.jit(scan_cond), platforms=platforms)(z3).mlir_module()), "values": []}
    rows["export_multi_while"] = {"text": _strip_locations(export.export(jax.jit(while_body), platforms=platforms)(z3).mlir_module()), "values": []}

    def tree_out(x):
        return {"a": jnp.sin(x), "b": (jnp.cos(x), x)}

    tree_out.__name__ = "f"

    def named_out(x):
        return jnp.linalg.qr(x)

    named_out.__name__ = "f"
    rows["stages_namedtuple_paths"] = {
        "text": jax.jit(named_out).lower(np.array([[2.0, 1.0], [1.0, 3.0]], np.float32)).as_text(),
        "values": [],
    }
    rows["stages_tree_paths"] = {"text": jax.jit(tree_out).lower(z3).as_text(), "values": []}
    v3 = np.array([1.0, 2.0, -3.0], np.float32)
    rows["export_multi_call"] = {
        "text": "",
        "values": [[str(int(np.float64(v).view(np.int64))) for v in np.asarray(multi.call(v3, v3)).ravel()]],
    }
    return rows


def _strip_locations(text):
    lines = [line for line in text.split("\n") if not line.startswith("#loc")]
    text = "\n".join(lines)
    text = re.sub(r" loc\((?:[^()]|\([^()]*\))*\)", "", text)
    return text.rstrip("\n") + "\n"


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
