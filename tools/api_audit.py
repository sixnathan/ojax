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
    "jax": ["spmd", "parallel", "ref", "shape_dtype_struct", "api", "core", "ad_util", "custom_derivatives", "checkpoint", "tree_util", "flatten_util", "config", "sharding", "backend", "debugging", "basearray", "dtypes", "stages", "random/core", "profiler", "monitoring"],
    "jax.numpy": ["numpy/*", "dtypes"],
    "jax.numpy.fft": ["numpy/fft"],
    "jax.numpy.linalg": ["numpy/linalg", "linalg_extras"],
    "jax.lax": ["spmd", "parallel", "lax_api", "lax_extras", "lax_other", "lax/*", "lax/control_flow/*", "lax_config", "api"],
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


NO_RESULT_EFFECT = {
    "device",
    "out_sharding",
    "sharding",
    "copy",
    "order",
    "overwrite_a",
    "overwrite_b",
    "overwrite_c",
    "overwrite_input",
    "overwrite_data",
    "check_finite",
    "debug",
    "lwork",
    "use_magma",
    "dynamic_shape",
    "allow_pickle",
    "kind",
    "donate",
    "may_alias",
    "donate_argnums",
    "donate_argnames",
    "src",
    "backend",
    "keep_unused",
    "inline",
    "compiler_options",
    "reduce_axes",
    "turbo",
}


def rejected_by_jax(module, name, obj, param):
    if param == "out":
        return True
    if isinstance(obj, jax_ufunc()) and param in ("out", "where"):
        return True
    try:
        source = inspect.getsource(inspect.unwrap(obj))
    except (TypeError, OSError):
        return False
    pattern = r"if\s+(?:not\s+)?%s\b[^\n]*:\s*\n\s*raise\s+(?:NotImplementedError|ValueError|TypeError)" % re.escape(param)
    return re.search(pattern, source) is not None


def jax_ufunc():
    import jax.numpy as jnp

    return jnp.ufunc


def val_labels(files):
    labels = collections.defaultdict(set)
    decl = re.compile(r"^\s*val\s+([a-zA-Z_][a-zA-Z0-9_']*)\s*:(.*?)(?=^\s*(?:val|type|module|external|exception|open)\b|\Z)", re.M | re.S)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in decl.finditer(text):
            found = set(re.findall(r"\??([a-z_][a-zA-Z0-9_']*)\s*:", m.group(2)))
            if "Dtype.t" in m.group(2):
                found.add("dtype")
            labels[m.group(1)].update(found)
            labels[m.group(1)].add("")
    return labels


DECL = re.compile(r"^[ \t]*(?:val|external)\s+\(?\s*([a-zA-Z_][a-zA-Z0-9_']*)\s*\)?\s*:(.*?)(?=^[ \t]*(?:val|type|module|external|exception|open|end|include)\b|\Z)", re.M | re.S)


def split_arrows(text):
    parts, depth, cur, i = [], 0, "", 0
    while i < len(text):
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        if depth == 0 and text.startswith("->", i):
            parts.append(cur.strip())
            cur = ""
            i += 2
            continue
        cur += c
        i += 1
    parts.append(cur.strip())
    return parts


def parse_val(text):
    text = re.sub(r"\[@@?[^\]]*\]", "", text)
    if '"' in text:
        text = text[: text.index('"')]
        text = text.rsplit("=", 1)[0]
    parts = split_arrows(text)
    labels, positional = {}, []
    for part in parts[:-1]:
        m = re.match(r"^(\??)([a-z_][a-zA-Z0-9_']*)\s*:(.*)$", part, re.S)
        if m:
            labels[norm(m.group(2))] = m.group(3).strip()
        elif part != "unit":
            positional.append(part)
    return labels, positional, parts[-1]


def declarations(files):
    decls = collections.defaultdict(list)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in DECL.finditer(text):
            decls[m.group(1)].append(parse_val(m.group(2)))
    return decls


def type_expansion(files):
    out = {}
    pat = re.compile(r"^[ \t]*type\s+(?:'[a-z]+\s+)?([a-z_][a-zA-Z0-9_']*)\s*=\s*(.*?)(?=^[ \t]*(?:val|type|module|external|exception|open|end|include)\b|\Z)", re.M | re.S)
    for path in files:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        for m in pat.finditer(text):
            if "->" in m.group(2) and not m.group(2).lstrip().startswith(("|", "{")):
                out[m.group(1)] = m.group(2).strip()
    return out


def jax_parameters(obj):
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    out = []
    for p in sig.parameters.values():
        if p.kind == p.VAR_KEYWORD:
            continue
        if p.kind == p.VAR_POSITIONAL:
            out.append((p.name, "var"))
        elif p.default is inspect.Parameter.empty:
            out.append((p.name, "required"))
        else:
            out.append((p.name, "optional"))
    return out


def ocaml_key(name, decls):
    for key in (name, name + "_", name.lower(), name.lower() + "_", name + "'"):
        if key in decls:
            return key
    return None


def signature_gaps(module, name, obj, decls, types):
    params = jax_parameters(obj)
    if params is None:
        return None
    key = ocaml_key(name, decls)
    if key is None:
        return None
    variants = [d for k, ds in decls.items() if k == key or k.startswith(key + "_") or k.startswith(key.rstrip("_") + "_") for d in ds]
    exact = decls[key]
    labels = set()
    for lab, _, _ in variants:
        labels |= set(lab)
    positional = max(len(pos) for _, pos, _ in exact)
    for _, pos, res in exact:
        for arg in pos + [res]:
            if "flag" in arg:
                return []
        base = re.sub(r"^.*\.", "", res.split()[-1]) if res.split() else ""
        if base in types:
            lab, pos2, _ = parse_val(types[base])
            labels |= set(lab)
            positional = max(positional, len(pos) + len(pos2))
    if any("Dtype.t" in p for _, pos, _ in exact for p in pos):
        labels.add("dtype")
    if isinstance(obj, jax_ufunc()) or "Primitive" in type(obj).__name__:
        return []
    gaps = []
    for pname, kind in params:
        if norm(pname) in labels:
            continue
        if kind in ("required", "var"):
            if positional > 0:
                positional -= 1
                continue
            if kind == "required":
                gaps.append(pname)
        else:
            if pname.startswith("_") or pname in NO_RESULT_EFFECT or rejected_by_jax(module, name, obj, pname):
                continue
            if positional > 0:
                positional -= 1
                continue
            gaps.append(pname)
    return gaps


def signature_audit(report):
    total = ok = no_sig = 0
    rows = []
    for module, patterns in MAP.items():
        mod = importlib.import_module(module)
        files = mli_files(patterns)
        decls = declarations(files)
        types = type_expansion(files)
        missing = set(report[module]["missing_strict"])
        for name in public_names(module):
            if name in missing:
                continue
            total += 1
            gaps = signature_gaps(module, name, getattr(mod, name, None), decls, types)
            if gaps is None:
                no_sig += 1
                ok += 1
            elif gaps:
                rows.append((module, name, gaps))
            else:
                ok += 1
    return total, ok, no_sig, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=None)
    parser.add_argument("--options", action="store_true")
    parser.add_argument("--signatures", action="store_true")
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
        known_norm = {norm(n) for n in known | lowered}
        missing_strict = []
        missing_loose = []
        for name in names:
            candidates = {name, name.lower(), name + "_", name.lower() + "_", name + "'"}
            if not candidates & (known | lowered) and norm(name) not in known_norm:
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
    primitive = sum(1 for m in report.values() for n in m["missing_strict"] if n.endswith("_p"))
    print(f"primitive objects among strict gaps (ojax has variant constructors for them): {primitive}")
    if args.signatures:
        checked, sig_ok, no_sig, sig_rows = signature_audit(report)
        report["_signatures"] = sig_rows
        print(f"SIGNATURES names_present {checked} signature_ok {sig_ok} (of which {no_sig} have no comparable signature) signature_gaps {len(sig_rows)} params_missing {sum(len(r[2]) for r in sig_rows)}")
        print(f"STRICT_WITH_SIGNATURES {sig_ok} of {tot}")
        for module, name, gaps in sig_rows:
            print(f"  {module}.{name}: {' '.join(gaps)}")
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
                have = set()
                for label_key, found in labels.items():
                    if label_key == key or label_key.startswith(key + "_"):
                        have |= {norm(l) for l in found}
                if "preferred" in have:
                    have.add("preferredelementtype")
                if key + "_aux" in labels or key + "_returned" in labels:
                    have |= {"hasaux", "returned"}
                if key + "_holomorphic" in labels:
                    have.add("holomorphic")
                obj = getattr(mod, name, None)
                lacking = [p for p, d in params if d is not None and norm(p) not in have and not p.startswith("_")]
                real = [p for p in lacking if p not in NO_RESULT_EFFECT and not rejected_by_jax(module, name, obj, p)]
                inert = [p for p in lacking if p in NO_RESULT_EFFECT]
                rejected = [p for p in lacking if p not in NO_RESULT_EFFECT and p not in real]
                if lacking:
                    rows.append((module, name, real, inert, rejected))
        report["_options"] = rows
        print("OPTIONS result-changing missing:", sum(len(r[2]) for r in rows), "placement or no-result-effect:", sum(len(r[3]) for r in rows), "JAX rejects:", sum(len(r[4]) for r in rows))
        for module, name, real, _, _ in rows:
            if real:
                print(f"  {module}.{name}: {' '.join(real)}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
