import argparse
import json
import os

import gen_docs_p2 as gd
import gen_goldens as gg
import jax
import jax.numpy as jnp

MODULE = "ordered_lowering"
CASES = ["cond_ordered", "vmap_ordered", "nested_jit", "fori", "scan", "while",
         "vmap", "grad", "cond", "jvp", "remat"]


class _Lowered:
    def __init__(self, fn, args):
        self.text = jax.jit(fn).lower(*args).as_text()

    def __str__(self):
        return self.text


def lowered_text(name):
    saved = jax.make_jaxpr
    jax.make_jaxpr = lambda fn: (lambda *args: _Lowered(fn, args))
    try:
        return gd.run_case(jax, gd.xform_thunk(jax, jnp, name, "lowered"))
    finally:
        jax.make_jaxpr = saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=gd.DEFAULT_SEED)
    args = ap.parse_args()
    gd.set_base(args.seed)
    gg.preflight()
    base = os.path.join(args.root or os.path.join(gg.ROOT, "goldens"), MODULE)
    os.makedirs(base, exist_ok=True)
    for x64 in [False, True]:
        jax.config.update("jax_enable_x64", x64)
        rows = []
        for name in CASES:
            r = lowered_text(name)
            r.update({"id": f"lowered_{name}", "case": name})
            rows.append(gd.with_base(r))
        with open(os.path.join(base, ("x64_on" if x64 else "x64_off") + ".json"), "w") as fh:
            json.dump({"jax": jax.__version__, "x64": x64, "rows": rows}, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
