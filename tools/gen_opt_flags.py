import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

FLAGS = {
    "base": [],
    "disable_most_optimizations": [("jax_disable_most_optimizations", True)],
    "no_remat_pass": [("jax_compiler_enable_remat_pass", False)],
    "optimization_level_O0": [("jax_optimization_level", "O0")],
    "optimization_level_O3": [("jax_optimization_level", "O3")],
    "memory_fitting_level_O0": [("jax_memory_fitting_level", "O0")],
    "exec_time_effort_high": [("jax_exec_time_optimization_effort", 1.0)],
    "memory_fitting_effort_low": [("jax_memory_fitting_effort", -1.0)],
}

BAD = [("jax_optimization_level", "O4"), ("jax_memory_fitting_level", "fast")]

OPS = ["exp", "tanh", "log_abs", "expm1", "log1p_abs", "sin", "cos", "logistic", "gelu", "softmax",
       "logsumexp", "erf", "sum", "mean", "cumsum", "pow", "sqrt_abs", "dot"]


def vector():
    return (np.arange(4097, dtype=np.float64) / 256.0 - 8.0).astype(np.float32)


def matrix():
    return ((np.arange(4096, dtype=np.float64) % 17) / 8.0 - 1.0).astype(np.float32).reshape(64, 64)


def op_fn(jax, jnp, op):
    return {
        "exp": jnp.exp,
        "tanh": jnp.tanh,
        "log_abs": lambda a: jnp.log(jnp.abs(a)),
        "expm1": jnp.expm1,
        "log1p_abs": lambda a: jnp.log1p(jnp.abs(a)),
        "sin": jnp.sin,
        "cos": jnp.cos,
        "logistic": jax.nn.sigmoid,
        "gelu": jax.nn.gelu,
        "softmax": jax.nn.softmax,
        "logsumexp": jax.nn.logsumexp,
        "erf": jax.scipy.special.erf,
        "sum": jnp.sum,
        "mean": jnp.mean,
        "cumsum": jnp.cumsum,
        "pow": lambda a: jnp.power(jnp.abs(a), 1.7),
        "sqrt_abs": lambda a: jnp.sqrt(jnp.abs(a)),
        "dot": lambda a: jnp.dot(jnp.dot(a, a), a),
    }[op]


def text_of(out):
    a = np.asarray(out)
    return f"{a.dtype} {list(a.shape)} {a.tobytes().hex()}"


def run_flag(flag, x64, out):
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", x64)
    for name, value in FLAGS[flag]:
        jax.config.update(name, value)
    rows = []
    for op in OPS:
        arg = matrix() if op == "dot" else vector()
        fn = op_fn(jax, jnp, op)
        for mode in ("jit", "eager"):
            f = jax.jit(fn) if mode == "jit" else fn
            x = jnp.asarray(arg)
            jax.clear_caches()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    row = {"kind": "ok", "text": text_of(f(x))}
                except Exception as e:
                    row = {"kind": "raise", "class": type(e).__name__, "message": str(e)}
            row["warnings"] = [[w.category.__name__, str(w.message)] for w in caught]
            row.update({"id": f"{flag}_{op}_{mode}", "flag": flag, "op": op, "mode": mode})
            rows.append(row)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=None)
    parser.add_argument("--flag", default=None)
    parser.add_argument("--x64", type=int, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.flag:
        run_flag(args.flag, bool(args.x64), args.out)
        return
    here = os.path.dirname(os.path.abspath(__file__))
    base = os.path.join(args.root or os.path.join(here, "..", "goldens"), "opt_flags")
    os.makedirs(base, exist_ok=True)
    import jax
    for x64 in (False, True):
        tag = "x64_on" if x64 else "x64_off"
        rows = []
        for flag in FLAGS:
            part = os.path.join(base, f"{tag}.{flag}.part.json")
            subprocess.run([sys.executable, os.path.abspath(__file__), "--flag", flag, "--x64", str(int(x64)),
                            "--out", part], check=True)
            with open(part, encoding="utf-8") as fh:
                rows.extend(json.load(fh))
            os.remove(part)
        for name, value in BAD:
            try:
                jax.config.update(name, value)
                row = {"kind": "ok", "text": ""}
            except Exception as e:
                row = {"kind": "raise", "class": type(e).__name__, "message": str(e), "text": ""}
            row.update({"id": f"config_{name}_{value}", "flag": "config", "op": name, "mode": value,
                        "warnings": []})
            rows.append(row)
        with open(os.path.join(base, tag + ".json"), "w", encoding="utf-8") as fh:
            json.dump({"jax": jax.__version__, "x64": x64, "rows": rows}, fh, indent=1, sort_keys=True)
        sys.stdout.write(f"opt_flags {tag} {len(rows)} rows\n")


if __name__ == "__main__":
    main()
