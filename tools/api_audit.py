import argparse
import collections
import glob
import importlib
import inspect
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAP = {
    "jax": ["api", "core", "ad_util", "custom_derivatives", "checkpoint", "tree_util", "flatten_util", "config", "sharding", "backend", "debugging", "basearray", "dtypes", "stages", "random/core", "profiler", "monitoring"],
    "jax.numpy": ["numpy/*"],
    "jax.numpy.fft": ["numpy/fft"],
    "jax.numpy.linalg": ["numpy/linalg", "linalg_extras"],
    "jax.lax": ["lax_api", "lax_extras", "lax_other", "lax/*", "lax/control_flow/*", "lax_config", "api"],
    "jax.lax.linalg": ["lax/linalg", "qdwh", "linalg_extras"],
    "jax.nn": ["nn/functions"],
    "jax.nn.initializers": ["nn/initializers"],
    "jax.random": ["random/*", "random_extras"],
    "jax.tree_util": ["tree_util", "tree"],
    "jax.tree": ["tree", "tree_util"],
    "jax.ops": ["ops/*"],
    "jax.image": ["image/*"],
    "jax.experimental": ["*", "export"],
    "jax.scipy.cluster.vq": ["scipy/cluster/vq"],
    "jax.scipy.fft": ["scipy/fft"],
    "jax.scipy.integrate": ["scipy/integrate"],
    "jax.scipy.interpolate": ["scipy/interpolate"],
    "jax.scipy.linalg": ["scipy/linalg", "scipy/linalg_special", "scipy/dense_solvers", "linalg_extras"],
    "jax.scipy.ndimage": ["scipy/ndimage"],
    "jax.scipy.optimize": ["scipy/optimize/*"],
    "jax.scipy.signal": ["scipy/signal"],
    "jax.scipy.sparse.linalg": ["scipy/sparse/linalg"],
    "jax.scipy.spatial.transform": ["scipy/spatial/transform"],
    "jax.scipy.special": ["scipy/special", "lax/special"],
}
for name in "bernoulli beta betabinom binom cauchy chi2 dirichlet expon gamma gennorm geom gumbel_l gumbel_r laplace logistic multinomial nbinom norm pareto poisson t truncnorm uniform vonmises wrapcauchy".split():
    MAP["jax.scipy.stats." + name] = ["scipy/stats/" + name]
MAP["jax.scipy.stats"] = ["scipy/stats/*"]

VAL = re.compile(r"^\s*(?:val|external)\s+(?:\(\s*)?([a-zA-Z_][a-zA-Z0-9_']*)", re.M)
TYPE = re.compile(r"^\s*(?:type|module)\s+(?:\([^)]*\)\s+|'[a-z]+\s+)?([a-zA-Z_][a-zA-Z0-9_']*)", re.M)
CTOR = re.compile(r"\|\s*([A-Z][A-Za-z0-9_]*)")


def public_names(module):
    mod = importlib.import_module(module)
    names = getattr(mod, "__all__", None)
    if names is None:
        names = [n for n in dir(mod) if not n.startswith("_")]
    out = []
    for name in sorted(set(names)):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name, None)
        if inspect.ismodule(obj):
            continue
        out.append(name)
    return out


def mli_files(patterns):
    files = []
    for pat in patterns:
        files += glob.glob(os.path.join(ROOT, "src", pat + ".mli"))
    return sorted(set(files))


def symbols(files):
    vals, others = set(), set()
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        vals.update(VAL.findall(text))
        others.update(TYPE.findall(text))
        others.update(CTOR.findall(text))
    return vals, others


def norm(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def signature_params(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    return [
        (p.name, None if p.default is inspect.Parameter.empty else repr(p.default))
        for p in sig.parameters.values()
        if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
    ]


def val_labels(files):
    labels = collections.defaultdict(set)
    decl = re.compile(r"^\s*val\s+([a-zA-Z_][a-zA-Z0-9_']*)\s*:(.*?)(?=^\s*(?:val|type|module|external|exception|open)\b|\Z)", re.M | re.S)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in decl.finditer(text):
            labels[m.group(1)].update(re.findall(r"\??([a-z_][a-zA-Z0-9_']*)\s*:", m.group(2)))
            labels[m.group(1)].add("")
    return labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--options", action="store_true")
    args = parser.parse_args()
    all_files = mli_files(["*", "*/*", "*/*/*"])
    loose_vals, loose_others = symbols(all_files)
    loose = {norm(n) for n in loose_vals | loose_others}
    report = {}
    tot = strict_n = loose_n = 0
    for module, patterns in MAP.items():
        names = public_names(module)
        vals, others = symbols(mli_files(patterns))
        known = vals | {n.lower() for n in others}
        lowered = {n.lower() for n in others} | others
        missing_strict = []
        missing_loose = []
        for name in names:
            candidates = {name, name.lower(), name + "_", name.lower() + "_", name + "'"}
            if not candidates & (known | lowered):
                missing_strict.append(name)
                if norm(name) not in loose:
                    missing_loose.append(name)
        tot += len(names)
        strict_n += len(names) - len(missing_strict)
        loose_n += len(names) - len(missing_loose)
        report[module] = {"total": len(names), "missing_strict": missing_strict, "missing_loose": missing_loose}
        print(f"{module}: {len(names)} strict_missing {len(missing_strict)} loose_missing {len(missing_loose)}")
        if missing_strict:
            print("  strict: " + " ".join(missing_strict))
        if missing_loose:
            print("  loose: " + " ".join(missing_loose))
    print(f"TOTAL {tot} strict {strict_n} loose {loose_n}")
    if args.options:
        rows = []
        for module, patterns in MAP.items():
            mod = importlib.import_module(module)
            labels = val_labels(mli_files(patterns))
            for name in public_names(module):
                params = signature_params(getattr(mod, name, None))
                if params is None:
                    continue
                key = name if name in labels else name + "_" if name + "_" in labels else name.lower()
                if key not in labels:
                    continue
                have = {norm(l) for l in labels[key]}
                lacking = [p for p, d in params if d is not None and norm(p) not in have]
                if lacking:
                    rows.append((module, name, lacking))
        report["_options"] = rows
        print("OPTIONS missing:", sum(len(r[2]) for r in rows), "across", len(rows))
        for module, name, lacking in rows:
            print(f"  {module}.{name}: {' '.join(lacking)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
