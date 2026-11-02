import json
import subprocess
import sys


def reference():
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    device = jax.devices()[0]
    mesh = Mesh(np.array([device]), ("x",))
    spec = NamedSharding(mesh, P("x", None))
    placed = jax.device_put(np.arange(16.0, dtype=np.float32).reshape(4, 4), spec)

    def body(x):
        return jnp.sin(x) * 2.0 + x.sum()

    def constraint(x):
        return jax.lax.with_sharding_constraint(x * 2.0, NamedSharding(mesh, P(None, "x")))

    def shard_map(x):
        return jax.shard_map(lambda y: jnp.sin(y) + jax.lax.psum(y, "x"), mesh=mesh, in_specs=P("x"), out_specs=P("x"))(x)

    rows = {"platform": {"text": device.platform + ":" + str(jax.device_count()), "values": []}}
    for name, fn, options in (
        ("sin_sum", body, {}),
        ("out_sharded", body, {"out_shardings": spec}),
        ("constraint", constraint, {}),
        ("shard_map", shard_map, {}),
    ):
        fn = lambda *xs, _fn=fn: _fn(*xs)
        fn.__name__ = "f"
        jitted = jax.jit(fn, **options)
        leaves = jax.tree_util.tree_leaves(jitted(placed))
        rows[name] = {
            "text": jitted.lower(placed).as_text(),
            "values": [[str(int(np.float64(v).view(np.int64))) for v in np.asarray(leaf).ravel()] for leaf in leaves],
        }
    return rows


def main():
    probe, out = sys.argv[1], sys.argv[2]
    subprocess.run([probe, out], check=True)
    ours = {row["case"]: row for row in map(json.loads, open(out, encoding="utf-8"))}
    rows = reference()
    print("platform", rows.pop("platform")["text"])
    same = 0
    for name, want in rows.items():
        got = ours.get(name, {})
        text = got.get("text") == want["text"]
        values = got.get("values") == want["values"]
        same += text and values
        print(name, "text", "same" if text else "DIFFERENT", "values", "same" if values else "DIFFERENT", got.get("error", "")[:200])
        if not text and "text" in got:
            print("--- jax\n" + want["text"][:1400] + "\n--- ojax\n" + got["text"][:1400])
    print(f"{same}/{len(rows)} identical")


if __name__ == "__main__":
    main()
