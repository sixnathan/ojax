import glob
import inspect
import os
import re
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import api_audit
import option_registry as registry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEARCH = {
    "jnp": ["numpy/reductions", "numpy/sorting", "numpy/setops", "numpy/indexing", "numpy/lax_numpy", "numpy/array_creation", "numpy/array_constructors", "numpy/tensor_contractions", "numpy/polynomial", "numpy/extras", "numpy/window_functions", "numpy/ufuncs", "numpy/einsum", "numpy/grids", "numpy/array_methods"],
    "linalg": ["numpy/linalg", "linalg_extras"],
    "fft": ["numpy/fft"],
    "lax": ["lax_api", "lax_extras", "lax_other", "lax/lax", "lax/slicing", "lax/control_flow/loops"],
    "lax_linalg": ["lax/linalg", "qdwh", "linalg_extras"],
    "nn": ["nn/functions"],
    "random": ["random/core", "random_extras"],
    "ops": ["ops/*"],
    "image": ["image/scale"],
    "sp_fft": ["scipy/fft"],
    "sp_linalg": ["scipy/linalg", "scipy/linalg_special", "scipy/dense_solvers", "linalg_extras"],
    "sp_signal": ["scipy/signal"],
    "sp_special": ["scipy/special", "lax/special"],
    "sp_ndimage": ["scipy/ndimage"],
    "sp_cluster": ["scipy/cluster/vq"],
    "sp_integrate": ["scipy/integrate"],
}
for stat in "norm t gamma beta expon cauchy laplace logistic uniform".split():
    SEARCH["sp_stats_" + stat] = ["scipy/stats/" + stat]


def module_path(path):
    rel = os.path.relpath(path, os.path.join(ROOT, "src"))[:-4]
    return "Ojax." + ".".join(p.capitalize() for p in rel.split("/"))


def files_for(short):
    out = []
    for pattern in SEARCH[short]:
        out += sorted(glob.glob(os.path.join(ROOT, "src", pattern + ".mli")))
    return out


def normalise(ty):
    ty = re.sub(r"\s+", " ", ty).strip()
    ty = re.sub(r"\b(?:Ojax\.)?(?:Types|T|Lax_api|L)\.value\b", "value", ty)
    ty = re.sub(r"\bTypes\.", "", ty)
    ty = re.sub(r"\bString\.t\b", "string", ty)
    return ty


DECODERS = {
    "value": "(A.value arr)",
    "value list": "(A.value_list arr)",
    "int": "A.int_",
    "bool": "A.bool_",
    "float": "A.float_",
    "string": "A.string_",
    "int array": "A.int_array",
    "int list": "A.int_list",
    "float array": "A.float_array",
    "int option": "A.int_option",
    "float option": "A.float_option",
    "bool option": "A.bool_option",
    "string option": "A.string_option",
    "int array option": "A.int_array_option",
    "(int * int) array": "A.int_pairs",
    "int * int": "A.int_pair",
    "string * string": "A.string_pair",
    "Dtype.t": "A.dtype",
    "padding": "A.padding",
    "Lax_config.rounding_method": "A.rounding_method",
    "rounding_method": "A.rounding_method",
    "sections": "A.sections",
    "Tensor_contractions.td_axes": "A.td_axes",
    "ord": "A.ord",
    "norm_axis": "A.norm_axis",
    "td_axes": "A.td_axes",
    "Tensor_contractions.td_axes": "A.td_axes",
}


RESULTS = {
    "value": "[ {} ]",
    "value list": "{}",
    "value * value": "(let a, b = {} in [ a; b ])",
    "value * value * value": "(let a, b, c = {} in [ a; b; c ])",
    "value * value * value * value": "(let a, b, c, d = {} in [ a; b; c; d ])",
}


def declarations(path):
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    out = {}
    for m in api_audit.DECL.finditer(text):
        body = re.sub(r"\[@@?[^\]]*\]", "", m.group(2))
        if '"' in body:
            continue
        out.setdefault(m.group(1), api_audit.split_arrows(body))
    return out


def json_literal(value):
    if value is None:
        return "`Null"
    if hasattr(value, "item") and not isinstance(value, (bool, int, float, str)):
        value = value.item()
    if isinstance(value, bool):
        return "(`Bool " + ("true" if value else "false") + ")"
    if isinstance(value, int):
        return "(`Int (" + str(value) + "))"
    if isinstance(value, float) and value == value and abs(value) != float("inf"):
        return "(`Float (" + repr(value) + "))"
    if isinstance(value, str):
        return '(`String "' + value + '")'
    if isinstance(value, (tuple, list)):
        return "(`List [" + "; ".join(json_literal(v) for v in value) + "])"
    raise ValueError(value)


def arm(spec, parts, module):
    fn = registry.lookup(spec["module"], spec["name"])
    params = [p for p in inspect.signature(fn).parameters.values() if p.kind not in (p.VAR_KEYWORD,)]
    result = normalise(parts[-1])
    if result not in RESULTS:
        return None
    required_index = {name: i for i, (name, _) in enumerate(spec["required"])}
    params = [p for p in params if p.kind != p.VAR_POSITIONAL or p.name in required_index]
    labelled = {}
    positional = []
    for part in parts[:-1]:
        m = re.match(r"^(\??)([a-z_][a-zA-Z0-9_']*)\s*:(.*)$", part, re.S)
        if m:
            labelled[api_audit.norm(m.group(2))] = (m.group(1) == "?", m.group(2), normalise(m.group(3)))
        else:
            positional.append(normalise(part))
    alias = {"preferred": "preferredelementtype"}

    def param_source(param):
        if param.name in required_index:
            return "(positional.(" + str(required_index[param.name]) + "))"
        default = None if param.default is inspect.Parameter.empty else param.default
        try:
            literal = json_literal(default)
        except ValueError:
            literal = "`Null"
        return '(A.get options "' + param.name + '" ' + literal + ")"

    args = []
    used = set()
    for key, (optional, label, ty) in labelled.items():
        target = None
        for param in params:
            if api_audit.norm(param.name) in (key, alias.get(key)):
                target = param
        if target is None:
            if not optional:
                return None
            continue
        used.add(target.name)
        decoder = DECODERS.get(ty)
        if decoder is None:
            if target.name in spec["options"]:
                args.append(("refuse", target.name))
            if not optional:
                return None
            continue
        if ty == "Dtype.t" and not optional and spec["name"] in ("eye", "tri", "identity", "indices", "full"):
            default = "A.default_int" if spec["name"] == "indices" else "A.default_float"
            args.append("~" + label + ":(" + default + " " + param_source(target) + ")")
        elif target.name == "trans" and ty == "int":
            args.append("?" + label + ':(A.opt options "trans" A.trans)')
        elif not optional:
            args.append("~" + label + ":(" + decoder + " " + param_source(target) + ")")
        elif target.name in required_index:
            args.append("?" + label + ":(Some (" + decoder + " " + param_source(target) + "))")
        elif ty.endswith(" option"):
            args.append("?" + label + ':(A.opt_keep options "' + target.name + '" ' + decoder + ")")
        else:
            args.append("?" + label + ':(A.opt options "' + target.name + '" ' + decoder + ")")
    remaining = [p for p in params if p.name not in used]
    for ty in positional:
        if ty == "unit":
            args.append("()")
            continue
        decoder = DECODERS.get(ty)
        if decoder is None or not remaining:
            return None
        param = remaining.pop(0)
        used.add(param.name)
        args.append("(" + decoder + " " + param_source(param) + ")")
    for param in remaining:
        if param.name in required_index:
            return None
    refused = sorted({a[1] for a in args if isinstance(a, tuple)} | {p.name for p in remaining if p.name in spec["options"]})
    call_args = " ".join(a for a in args if isinstance(a, str))
    call = module + "." + ocaml_name(spec, parts) + " " + call_args
    body = RESULTS[result].format("(" + call + ")")
    head = ""
    if refused:
        head = "A.refuse options [" + "; ".join('"' + r + '"' for r in refused) + "];\n"
    return '| "' + spec["module"] + "." + spec["name"] + '" -> ' + head + body


def ocaml_name(spec, parts):
    return spec["_key"]


def main():
    out = [
        "module A = Option_args",
        "",
        "let run op positional options arr =",
        "  ignore positional;",
        "  ignore options;",
        "  ignore arr;",
        "  match op with",
    ]
    arms = 0
    missing = []
    for spec in registry.all_specs():
        found = None
        for path in files_for(spec["module"]):
            decls = declarations(path)
            key = None
            names = (spec["name"], spec["name"] + "_", spec["name"] + "'", spec["name"].lower())
            variant = registry.VARIANT_NAMES.get((spec["module"], spec["name"]))
            if variant:
                names = (variant,) + names
            for candidate in names:
                if candidate in decls:
                    key = candidate
                    break
            if key is None:
                continue
            spec["_key"] = key
            found = arm(spec, decls[key], module_path(path))
            if found:
                break
        if found:
            out.append(found)
            arms += 1
        else:
            missing.append(spec["module"] + "." + spec["name"])
    out.append('  | other -> A.no ("function:" ^ other)')
    target = os.path.join(ROOT, "test", "golden", "option_cases.ml")
    with open(target, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    sys.stdout.write("arms " + str(arms) + " without a bindable signature " + str(len(missing)) + "\n")
    sys.stdout.write(" ".join(missing) + "\n")


if __name__ == "__main__":
    main()
