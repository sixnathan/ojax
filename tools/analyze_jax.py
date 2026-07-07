import ast
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JAX_DIR = os.path.join(ROOT, "reference", "jax")
OUT = os.path.join(ROOT, "notes", "research", "module_graph.json")


def module_name(path):
    rel = os.path.relpath(path, os.path.dirname(JAX_DIR))
    rel = rel[:-3] if rel.endswith(".py") else rel
    parts = rel.split(os.sep)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def iter_py():
    out = []
    for base, dirs, files in os.walk(JAX_DIR):
        dirs.sort()
        for f in sorted(files):
            if f.endswith(".py"):
                out.append(os.path.join(base, f))
    return sorted(out)


def loc(text):
    return len(text.splitlines())


def internal_imports(text, modname):
    found = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    pkg_parts = modname.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "jax" or a.name.startswith("jax."):
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            level = node.level or 0
            if level == 0:
                mod = node.module or ""
                if mod == "jax" or mod.startswith("jax."):
                    found.add(mod)
            else:
                base = pkg_parts[:len(pkg_parts) - level]
                mod = node.module
                target = base + (mod.split(".") if mod else [])
                resolved = ".".join(target)
                if resolved == "jax" or resolved.startswith("jax."):
                    found.add(resolved)
    return sorted(found)


PRIM_FUNCS = (
    "Primitive",
    "core.Primitive",
    "standard_primitive",
    "standard_unop",
    "standard_naryop",
    "standard_abstract_eval",
    "unop",
    "binop",
    "naryop",
)
PRIM_RE = re.compile(
    r"^(?P<lhs>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<fn>[A-Za-z_.]+)\s*\(",
)
STR_RE = re.compile(r"""['"]([A-Za-z0-9_.\-]+)['"]""")


def collect_primitives():
    prims = []
    lax_dir = os.path.join(JAX_DIR, "_src", "lax")
    for path in iter_py():
        if not path.startswith(lax_dir + os.sep):
            continue
        rel = os.path.relpath(path, JAX_DIR)
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        buf = ""
        start = 0
        depth = 0
        active = False
        for i, line in enumerate(lines):
            if not active:
                m = PRIM_RE.match(line)
                if m and m.group("fn").split(".")[-1] in [
                    p.split(".")[-1] for p in PRIM_FUNCS
                ]:
                    fn = m.group("fn")
                    if fn in PRIM_FUNCS or fn.split(".")[-1] in PRIM_FUNCS:
                        active = True
                        buf = line
                        start = i
                        depth = line.count("(") - line.count(")")
                        lhs = m.group("lhs")
                        fnname = fn
                        if depth <= 0:
                            _emit(prims, rel, start, lhs, fnname, buf)
                            active = False
                    continue
            else:
                buf += line
                depth += line.count("(") - line.count(")")
                if depth <= 0:
                    _emit(prims, rel, start, lhs, fnname, buf)
                    active = False
    prims.sort(key=lambda d: (d["file"], d["line"], d["var"]))
    return prims


def _emit(prims, rel, start, lhs, fnname, buf):
    strs = STR_RE.findall(buf.split("(", 1)[1])
    name = strs[-1] if strs else None
    prims.append(
        {
            "var": lhs,
            "name": name,
            "constructor": fnname,
            "file": rel,
            "line": start + 1,
        }
    )


def public_exports(relpath):
    path = os.path.join(JAX_DIR, relpath)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    names = set()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for a in node.names:
                nm = a.asname or a.name
                if not nm.startswith("_"):
                    names.add(nm)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "__all__":
                    if isinstance(node.value, (ast.List, ast.Tuple)):
                        for e in node.value.elts:
                            if isinstance(e, ast.Constant) and isinstance(
                                e.value, str
                            ):
                                names.add(e.value)
    return sorted(names)


def main():
    modules = {}
    for path in iter_py():
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        modname = module_name(path)
        modules[modname] = {
            "path": os.path.relpath(path, ROOT),
            "loc": loc(text),
            "internal_imports": internal_imports(text, modname),
        }
    data = {
        "modules": dict(sorted(modules.items())),
        "primitives": collect_primitives(),
        "exports": {
            "jax": public_exports("__init__.py"),
            "jax.numpy": public_exports(os.path.join("numpy", "__init__.py")),
        },
        "totals": {
            "module_count": len(modules),
            "total_loc": sum(m["loc"] for m in modules.values()),
        },
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
        fh.write("\n")


if __name__ == "__main__":
    main()
