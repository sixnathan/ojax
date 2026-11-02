import os
import subprocess
import sys

import gen_compose_fuzz as cf
import jax
import jax.numpy as jnp
import numpy as np
from jax import ad_checkpoint

MODULE = "custom_ad"
DEFAULT_SEED = 20261103
CASES = 300
VMAP_CASES = 60
CUSTOM = (
    "custom_vjp",
    "custom_vjp_nondiff",
    "custom_jvp_nondiff",
    "custom_jvp_symbolic",
    "custom_gradient",
    "closure_convert",
    "checkpoint_policy",
    "checkpoint_name",
)
OUTER = ("jit", "vmap", "grad", "jvp", "vjp", "jacfwd", "jacrev", "value_and_grad")
POLICIES = (
    "everything_saveable",
    "nothing_saveable",
    "dots_saveable",
    "dots_with_no_batch_dims_saveable",
    "save_anything_except_these_names",
    "save_any_names_but_these",
    "save_only_these_names",
    "save_from_both_policies",
    "offload_dot_with_no_batch_dims",
    "save_and_offload_only_these_names",
)
ORIGINAL_APPLY = cf.apply_layer


def policy_of(name):
    cp = jax.checkpoint_policies
    if name == "save_anything_except_these_names":
        return cp.save_anything_except_these_names("n")
    if name == "save_any_names_but_these":
        return cp.save_any_names_but_these("n")
    if name == "save_only_these_names":
        return cp.save_only_these_names("n")
    if name == "save_from_both_policies":
        return cp.save_from_both_policies(cp.dots_saveable, cp.save_only_these_names("n"))
    if name == "offload_dot_with_no_batch_dims":
        return cp.offload_dot_with_no_batch_dims("device", "pinned_host")
    if name == "save_and_offload_only_these_names":
        return cp.save_and_offload_only_these_names(names_which_can_be_saved=[], names_which_can_be_offloaded=["n"], offload_src="device", offload_dst="pinned_host")
    return getattr(cp, name)


def scale(tree, factor):
    return jax.tree_util.tree_map(lambda v: jnp.multiply(v, factor), tree)


def apply_layer(layer, f):
    t = layer["t"]
    if t == "custom_vjp":
        wrapped = jax.custom_vjp(f)

        def fwd(*a):
            return f(*a), a

        def bwd(res, ct):
            _, pull = jax.vjp(f, *res)
            return scale(pull(ct), 1.5)

        wrapped.defvjp(fwd, bwd)
        wrapped.__name__ = "g"
        return wrapped
    if t == "custom_vjp_nondiff":
        inner = jax.custom_vjp(lambda k, *a: f(*a), nondiff_argnums=(0,))

        def fwd(k, *a):
            return f(*a), a

        def bwd(k, res, ct):
            _, pull = jax.vjp(f, *res)
            return scale(pull(ct), k)

        inner.defvjp(fwd, bwd)

        def g(*a):
            return inner(layer["k"], *a)

        return cf.named(g)
    if t == "custom_jvp_nondiff":
        inner = jax.custom_jvp(lambda k, *a: f(*a), nondiff_argnums=(0,))

        def rule(k, primals, tangents):
            y, dy = jax.jvp(f, primals, tangents)
            return y, scale(dy, k)

        inner.defjvp(rule)

        def g(*a):
            return inner(layer["k"], *a)

        return cf.named(g)
    if t == "custom_jvp_symbolic":
        wrapped = jax.custom_jvp(f)

        def rule(primals, tangents):
            tangents = tuple(jnp.zeros(t.shape, t.dtype) if isinstance(t, jax.custom_derivatives.SymbolicZero) else t for t in tangents)
            y, dy = jax.jvp(f, primals, tangents)
            return y, scale(dy, 1.5)

        wrapped.defjvp(rule, symbolic_zeros=True)
        wrapped.__name__ = "g"
        return wrapped
    if t == "custom_gradient":

        @jax.custom_gradient
        def g(*a):
            y, pull = jax.vjp(f, *a)
            return y, lambda ct: scale(pull(ct), 1.5)

        return cf.named(g)
    if t == "closure_convert":

        def g(*a):
            converted, consts = jax.closure_convert(f, *a)
            return converted(*a, *consts)

        return cf.named(g)
    if t == "checkpoint_policy":
        return cf.named(jax.checkpoint(f, policy=policy_of(layer["policy"]), prevent_cse=layer["prevent_cse"]))
    if t == "custom_vmap":
        wrapped = jax.custom_batching.custom_vmap(f)

        @wrapped.def_vmap
        def rule(axis_size, in_batched, *xs):
            out = jax.vmap(f, in_axes=tuple(0 if b else None for b in in_batched))(*xs)
            return scale(out, 1.5), jax.tree_util.tree_map(lambda _: True, out)

        return wrapped
    if t == "sequential_vmap":
        return jax.custom_batching.sequential_vmap(f)
    if t == "checkpoint_name":

        def g(*a):
            return jax.tree_util.tree_map(lambda v: ad_checkpoint.checkpoint_name(v, "n"), f(*a))

        return cf.named(g)
    return ORIGINAL_APPLY(layer, f)


def custom_layer(rng):
    t = str(rng.choice(CUSTOM))
    if t in ("custom_vjp_nondiff", "custom_jvp_nondiff"):
        return {"t": t, "k": [0.5, 1.5, 2.0][int(rng.integers(0, 3))]}
    if t == "checkpoint_policy":
        return {"t": t, "policy": POLICIES[int(rng.integers(0, len(POLICIES)))], "prevent_cse": bool(rng.random() < 0.7)}
    return {"t": t}


def outer_layer(rng, nargs):
    layer = cf.random_layer(rng, nargs, 0)
    while layer["t"] not in OUTER:
        layer = cf.random_layer(rng, nargs, 0)
    return layer


def custom_cases(seed, set_name, dtype):
    rng = np.random.default_rng([seed, cf.digest(MODULE + set_name + dtype)])
    unary = sorted(cf.UNARY)
    binary = sorted(cf.BINARY)
    out = []
    for i in range(CASES):
        nargs = 1 if rng.random() < 0.6 else 2
        pool = unary if nargs == 1 else binary
        base = pool[int(rng.integers(0, len(pool)))]
        shape = cf.ARG_SHAPES[int(rng.integers(0, len(cf.ARG_SHAPES)))]
        shapes = [list(shape), list(shape)][:nargs]
        layers = [custom_layer(rng)]
        if rng.random() < 0.4:
            layers.append(custom_layer(rng))
        for _ in range(int(rng.integers(1, 3))):
            layers.append(outer_layer(rng, nargs))
        variant = "edge" if rng.random() < 0.1 else "normal"
        out.append(
            {
                "case_id": "custom_" + format(i, "05d") + "_" + dtype,
                "op": "compose",
                "dtype": dtype,
                "variant": variant,
                "params": {"base": base, "layers": layers},
                "shapes": shapes,
            }
        )
    vrng = np.random.default_rng([seed, cf.digest(MODULE + "vmap" + set_name + dtype)])
    for i in range(VMAP_CASES):
        nargs = 1 if vrng.random() < 0.6 else 2
        pool = unary if nargs == 1 else binary
        base = pool[int(vrng.integers(0, len(pool)))]
        shape = cf.ARG_SHAPES[int(vrng.integers(0, len(cf.ARG_SHAPES)))]
        shapes = [list(shape), list(shape)][:nargs]
        layers = [{"t": ("custom_vmap", "sequential_vmap")[int(vrng.integers(0, 2))]}]
        for _ in range(int(vrng.integers(1, 3))):
            layer = cf.random_layer(vrng, nargs, 0)
            while layer["t"] not in ("jit", "vmap"):
                layer = cf.random_layer(vrng, nargs, 0)
            layers.append(layer)
        out.append(
            {
                "case_id": "cvmap_" + format(i, "05d") + "_" + dtype,
                "op": "compose",
                "dtype": dtype,
                "variant": "normal",
                "params": {"base": base, "layers": layers},
                "shapes": shapes,
            }
        )
    return out


def expand(seed):
    cases = {}
    for set_name, x64 in cf.SETS:
        dtype = "float64" if x64 else "float32"
        cases[set_name] = custom_cases(seed, set_name, dtype)
        cases[set_name].sort(key=lambda c: c["case_id"])
    return cases


def launch(set_name, shard, shards, work, seed):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    return subprocess.Popen(["nice", "-n", "10", sys.executable, os.path.abspath(__file__), "--worker", set_name, str(shard), str(shards), work, str(seed)], env=env)


cf.MODULE = MODULE
cf.DEFAULT_SEED = DEFAULT_SEED
cf.apply_layer = apply_layer
cf.expand = expand
cf.launch = launch

if __name__ == "__main__":
    if "--output-name" not in sys.argv and not (len(sys.argv) > 1 and sys.argv[1] == "--worker"):
        sys.argv += ["--output-name", MODULE]
    cf.main()
