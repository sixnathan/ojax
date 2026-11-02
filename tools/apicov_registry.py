import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax import tree_util as tu
from jax.experimental import io_callback, primal_tangent_dtype

SINGLE_PRECISION_ONLY = set()
LOOSE_TOLERANCE = {}


def compares_at_single_precision(op, arrays):
    return False


def _text(s):
    return np.frombuffer(str(s).encode("utf-8"), dtype=np.uint8).copy()


def _ints(values):
    return np.asarray(list(values), dtype=np.int32).reshape(-1)


def _flag(b):
    return np.asarray(bool(b))


class Node:
    def __init__(self, x, y, label):
        self.x, self.y, self.label = x, y, label


tu.register_pytree_node(Node, lambda n: ((n.x, n.y), n.label), lambda aux, ch: Node(ch[0], ch[1], aux))


class KeyedNode:
    def __init__(self, x, y):
        self.x, self.y = x, y


tu.register_pytree_with_keys(
    KeyedNode,
    lambda n: (((tu.GetAttrKey("x"), n.x), (tu.DictKey("why"), n.y)), None),
    lambda aux, ch: KeyedNode(ch[0], ch[1]),
)


@tu.register_pytree_node_class
class ClassNode:
    def __init__(self, x):
        self.x = x

    def tree_flatten(self):
        return (self.x,), "cn"

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])


@tu.register_pytree_with_keys_class
class ClassKeyedNode:
    def __init__(self, x):
        self.x = x

    def tree_flatten_with_keys(self):
        return ((tu.GetAttrKey("inner"), self.x),), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(children[0])


@tu.register_dataclass
@dataclasses.dataclass
class Data:
    x: object
    y: object
    op: str = jax.tree.static(default="add")


@tu.register_static
@dataclasses.dataclass(frozen=True)
class Stat:
    tag: str = "st"


def _tree(a):
    return {"q": [a * 2.0, None], "p": (a, a * 3.0)}


def _is_tuple(x):
    return isinstance(x, tuple)


def _leaf_sum(x):
    return x[0] + x[1] if isinstance(x, tuple) else x


def _ops():
    ops = {}

    def logsumexp(a, p, b=None):
        where = a > p["where"] if "where" in p else None
        return jax.nn.logsumexp(a, axis=p.get("axis"), b=b, keepdims=p.get("keepdims", False), return_sign=p.get("return_sign", False), where=where)

    ops["ap_logsumexp"] = lambda a, p: logsumexp(a, p)
    ops["ap_logsumexp_b"] = lambda a, b, p: logsumexp(a, p, b=b)
    ops["ap_special_logsumexp"] = lambda a, p: jax.scipy.special.logsumexp(a, axis=p.get("axis"), return_sign=p.get("return_sign", False))
    ops["ap_shape"] = lambda a, p: _ints(jnp.shape(a))
    ops["ap_broadcast_shapes"] = lambda a, p: _ints(jnp.broadcast_shapes(*[tuple(s) for s in p["shapes"]]))
    ops["ap_iterable"] = lambda a, p: _flag(jnp.iterable(a))
    ops["ap_complex_"] = lambda a, p: jnp.complex_(a)
    ops["ap_np_dtype"] = lambda a, p: _text(jnp.dtype(p["name"]))
    cls = {"flexible": jnp.flexible, "character": jnp.character, "object": jnp.object_, "generic": jnp.generic, "number": jnp.number}
    ops["ap_issubdtype"] = lambda a, p: _flag(jnp.issubdtype(cls[p["left"]], cls[p["right"]]))
    ops["ap_issubdtype_named"] = lambda a, p: _flag(jnp.issubdtype(jnp.dtype(p["left"]), cls[p["right"]]))
    ops["ap_lax_dtype"] = lambda a, p: _text(lax.dtype(a))

    def ds_args(p):
        start = jnp.asarray(p["start"]) if "start" in p else None
        return start

    def ds_take(a, p):
        a = jnp.asarray(a)
        if p.get("full"):
            return a[jax.ds(None)]
        if "size" in p:
            return a[jax.ds(ds_args(p), p["size"], p.get("stride"))]
        return a[jax.ds(p["start"])]

    ops["ap_ds"] = ds_take

    def dce(a, p):
        lax.dce_sink(a, prevent_mlir_dce=p["prevent"])
        return a

    ops["ap_dce_sink"] = dce

    def io(a, p):
        return io_callback(lambda y: np.asarray(y) * 2, jax.ShapeDtypeStruct(a.shape, a.dtype), a, ordered=p.get("ordered", False))

    ops["ap_io_callback"] = io

    def flatten_is_leaf(a, p):
        leaves, td = tu.tree_flatten(_tree(a), is_leaf=_is_tuple)
        return [_leaf_sum(x) for x in leaves] + [_text(str(td).replace("None", "None")), _ints([td.num_leaves, td.num_nodes])]

    ops["ap_tree_flatten_is_leaf"] = flatten_is_leaf

    def paths(a, p):
        tree = _tree(a)
        if p["mode"] == "plain":
            found = tu.tree_leaves_with_path(tree)
        elif p["mode"] == "is_leaf":
            found = tu.tree_leaves_with_path(tree, is_leaf=_is_tuple)
        else:
            found = tu.tree_flatten_with_path(tree, is_leaf=lambda path, x: len(path) >= 2, is_leaf_takes_path=True)[0]
        return [_text(tu.keystr(k)) for k, _ in found]

    ops["ap_tree_paths"] = paths
    ops["ap_tree_map_path"] = lambda a, p: tu.tree_leaves(tu.tree_map_with_path(lambda path, x: _leaf_sum(x) * float(len(path)), _tree(a), is_leaf=_is_tuple))
    ops["ap_tree_reduce"] = lambda a, p: tu.tree_reduce(lambda acc, x: acc + _leaf_sum(x), _tree(a), 1.0, is_leaf=_is_tuple)
    ops["ap_tree_reduce_noinit"] = lambda a, p: tu.tree_reduce(lambda acc, x: acc * x, (a, a + 1.0, a + 2.0))
    ops["ap_tree_assoc"] = lambda a, p: tu.tree_reduce_associative(lambda u, v: u - v, (a, a * 2.0, a * 3.0, a * 5.0, a * 7.0))
    ops["ap_tree_all"] = lambda a, p: _flag(tu.tree_all((True, [True, False]), is_leaf=lambda x: isinstance(x, list)))

    def broadcast(a, p):
        return tu.tree_leaves(tu.tree_broadcast((a, a * 2.0), ([1, 2], {"k": 3, "j": (4, 5)})))

    ops["ap_tree_broadcast"] = broadcast
    ops["ap_all_leaves"] = lambda a, p: _ints([tu.all_leaves([1, 2]), tu.all_leaves([(1,), 2]), tu.all_leaves([(1,), 2], is_leaf=_is_tuple)])
    trees = {"dict": {"b": 1, "a": [2, (3, None)]}, "one": (1,), "leaf": 1, "none": None, "empty": ((), [], {})}
    ops["ap_treedef_str"] = lambda a, p: [_text(jax.tree.structure(trees[p["tree"]])), _ints([jax.tree.structure(trees[p["tree"]]).num_leaves, jax.tree.structure(trees[p["tree"]]).num_nodes])]

    def node(a, p):
        n = Node(a, a * 2.0, "tag")
        leaves, td = tu.tree_flatten(n)
        back = tu.tree_unflatten(td, [x * 3.0 for x in leaves])
        mapped = tu.tree_map(lambda x: x + 1.0, n)
        return [back.x, back.y, mapped.x, mapped.y, _text(back.label), _text(td)] + [_text(tu.keystr(k)) for k, _ in tu.tree_leaves_with_path(n)]

    ops["ap_node"] = node

    def keyed(a, p):
        n = KeyedNode(a, a * 2.0)
        found = tu.tree_leaves_with_path(n)
        return [_text(tu.keystr(k)) for k, _ in found] + [v for _, v in found] + [tu.tree_map(lambda x: x * 2.0, n).y]

    ops["ap_keyed_node"] = keyed

    def class_node(a, p):
        c = ClassNode(a)
        k = ClassKeyedNode(a * 2.0)
        return [tu.tree_map(lambda x: x * 2.0, c).x, _text(jax.tree.structure(c)), tu.tree_leaves(k)[0], _text(tu.keystr(tu.tree_leaves_with_path(k)[0][0]))]

    ops["ap_class_node"] = class_node

    def data(a, p):
        d = Data(a, a * 2.0)
        leaves, td = tu.tree_flatten(d)
        back = tu.tree_unflatten(td, [x + 1.0 for x in leaves])
        return [back.x, back.y, _text(back.op), _text(tu.keystr(tu.tree_leaves_with_path(d)[1][0]))]

    ops["ap_dataclass"] = data
    ops["ap_static"] = lambda a, p: [_ints([jax.tree.structure(Stat()).num_leaves]), _text(tu.tree_unflatten(jax.tree.structure(Stat()), []).tag)]

    def partial(a, p):
        pf = tu.Partial(lambda x, y: x * y, a)
        leaves = tu.tree_leaves(pf)
        mapped = tu.tree_map(lambda x: x + 1.0, pf)
        return [pf(a * 2.0), leaves[0], mapped(a * 2.0)]

    ops["ap_partial"] = partial
    ops["ap_primal_tangent"] = lambda a, p: _text(primal_tangent_dtype(p["primal"], p["tangent"], name=p.get("name")))

    def partition(a, p):
        items = [tuple(x) if isinstance(x, list) else x for x in p["items"]]
        kw = {k: set(p[k]) for k in ("unreduced", "reduced") if k in p}
        return _text(repr(jax.P(*items, **kw)))

    ops["ap_partition"] = partition
    ops["ap_config_flags"] = lambda a, p: _ints([jax.auto_pcast.value, jax.remove_size_one_mesh_axis_from_type.value, jax.config.jax_auto_pcast, jax.config.jax_threefry_partitionable, jax.config.read("jax_enable_x64") == jax.config.jax_enable_x64])
    ops["ap_memory"] = lambda a, p: [_text(str(jax.memory.Space.Device)), _text(str(jax.memory.Space.Host)), _text(str(jax.memory.Space.Any))]
    ops.update(_transform_ops())
    ops.update(_numpy_ops())
    return ops


def _transform_ops():
    ops = {}
    two = lambda x, y: jnp.sum(jnp.sin(x) * y * y)
    ops["ap_grad_argnums"] = lambda a, b, p: jax.grad(two, argnums=p["argnums"])(a, b)
    ops["ap_value_and_grad_argnums"] = lambda a, b, p: jax.value_and_grad(two, argnums=p["argnums"])(a, b)
    ops["ap_vmap_out_axes"] = lambda a, p: jax.vmap(lambda x: x * 2.0, in_axes=(0,), out_axes=p["out_axes"])(a)
    ops["ap_vmap_axis_name"] = lambda a, p: jax.vmap(lambda x: x - lax.psum(x, "i"), in_axes=(0,), axis_name="i")(a)
    ops["ap_vmap_axis_size"] = lambda a, p: jax.vmap(lambda x: x * 2.0, in_axes=(None,), axis_size=p["size"])(a)

    def static(a, p):
        f = jax.jit(lambda x, n: x * n, static_argnums=(1,))
        return [f(a, 2.0), f(a, 3.0)]

    ops["ap_jit_static"] = static
    ops["ap_jacfwd_holo"] = lambda a, p: jax.jacfwd(lambda x: x * x + 1j * x, holomorphic=True)(a)
    ops["ap_jacrev_holo"] = lambda a, p: jax.jacrev(lambda x: x * x + 1j * x, holomorphic=True)(a)
    ops["ap_hessian_holo"] = lambda a, p: jax.hessian(lambda x: jnp.sum(x * x * x), holomorphic=True)(a)
    ops["ap_jacobian_argnums"] = lambda a, b, p: jax.jacobian(lambda x, y: x * y, argnums=1)(a, b)

    def return_shape(a, p):
        _, shape = jax.make_jaxpr(lambda x: (x[0], x * 2.0), return_shape=True)(a)
        return [_ints(s.shape) for s in shape] + [_text(s.dtype) for s in shape]

    ops["ap_make_jaxpr_shape"] = return_shape
    return ops


def _numpy_ops():
    ops = {}
    from_list = lambda p, k: jnp.asarray(p[k], dtype=p.get("input_dtype", "float32"))

    def unique(a, p):
        out = jnp.unique(from_list(p, "values"), equal_nan=p["equal_nan"], return_counts=p.get("counts", False), sorted=True)
        return list(out) if isinstance(out, tuple) else out

    ops["ap_unique"] = unique

    def intersect(a, p):
        return list(jnp.intersect1d(from_list(p, "x"), from_list(p, "y"), assume_unique=p.get("assume_unique", False), return_indices=True))

    ops["ap_intersect_indices"] = intersect

    def delete(a, p):
        return jnp.delete(a, jnp.asarray(p["obj"]), axis=p.get("axis"), assume_unique_indices=p["unique"])

    ops["ap_delete"] = delete
    ops["ap_bin_edges"] = lambda a, p: jnp.histogram_bin_edges(a, bins=p["bins"], weights=a)
    ops["ap_polyfit_full"] = lambda a, b, p: list(jnp.polyfit(a, b, p["deg"], full=True))
    ops["ap_polyfit_cov"] = lambda a, b, p: list(jnp.polyfit(a, b, p["deg"], cov=p["cov"]))
    ops["ap_linspace_retstep"] = lambda a, p: list(jnp.linspace(a[0], a[1], p["num"], endpoint=p.get("endpoint", True), retstep=True))
    ops["ap_indices_sparse"] = lambda a, p: list(jnp.indices(tuple(p["dims"]), sparse=True))
    ops["ap_array_str"] = lambda a, p: [_text(jnp.array_str(jnp.asarray(a), max_line_width=p.get("width"), precision=p.get("precision"), suppress_small=p.get("suppress"))), _text(jnp.array_repr(jnp.asarray(a), max_line_width=p.get("width"), precision=p.get("precision"), suppress_small=p.get("suppress")))]
    for name in "quantile nanquantile percentile nanpercentile".split():
        ops["ap_" + name + "_w"] = lambda a, b, p, _n=name: getattr(jnp, _n)(a, jnp.asarray(p["q"], dtype=a.dtype), axis=p.get("axis"), method="inverted_cdf", keepdims=p.get("keepdims", False), weights=jnp.abs(b))
        ops["ap_" + name + "_cdf"] = lambda a, p, _n=name: getattr(jnp, _n)(a, jnp.asarray(p["q"], dtype=a.dtype), axis=p.get("axis"), method="inverted_cdf", keepdims=p.get("keepdims", False))
    ops["ap_ravel_dtype"] = lambda a, p: jnp.ravel_multi_index((jnp.asarray(p["i"]), jnp.asarray(p["j"])), tuple(p["dims"]), mode="clip", dtype=p["dtype"])
    return ops


OPS = _ops()


def _r(shape):
    return {"a": {"kind": "real", "shape": shape}}


def _gap_cases():
    one = _r([1])
    out = []
    for params in [{}, {"axis": [0]}, {"axis": [1], "keepdims": True}, {"keepdims": True}, {"where": 0.0}, {"axis": [1], "where": -0.5}, {"axis": [0], "keepdims": True, "where": 0.0}, {"return_sign": True}, {"axis": [1], "return_sign": True}, {"axis": [0], "return_sign": True, "where": 0.0}, {"axis": [1], "keepdims": True, "return_sign": True}]:
        out.append(("ap_logsumexp", _r([3, 4]), params))
    for params in [{}, {"axis": [1]}, {"return_sign": True}, {"axis": [1], "return_sign": True, "keepdims": True}, {"axis": [0], "where": 0.0}]:
        out.append(("ap_logsumexp_b", {"a": {"kind": "real", "shape": [3, 4]}, "b": {"kind": "real", "shape": [3, 4]}}, params))
    out.append(("ap_logsumexp", {"a": {"kind": "complex", "shape": [2, 3]}}, {"axis": [1]}))
    out.append(("ap_logsumexp", {"a": {"kind": "complex", "shape": [2, 3]}}, {"axis": [1], "return_sign": True}))
    out.append(("ap_special_logsumexp", _r([3, 4]), {}))
    out.append(("ap_special_logsumexp", _r([3, 4]), {"axis": [0], "return_sign": True}))
    for shape in [[1], [2, 3], [4, 1, 2]]:
        out.append(("ap_shape", _r(shape), {}))
    out.append(("ap_iterable", _r([2]), {}))
    out.append(("ap_iterable", _r([]), {}))
    out.append(("ap_complex_", _r([3]), {}))
    for shapes in [[], [[1], [4]], [[3, 1], [4]], [[3, 1], [1, 4], [5, 1, 1]], [[]], [[2, 0], [1]]]:
        out.append(("ap_broadcast_shapes", one, {"shapes": shapes}))
    for name in "float32 float64 int8 uint16 complex128 bool".split():
        out.append(("ap_np_dtype", one, {"name": name}))
        out.append(("ap_issubdtype_named", one, {"left": name, "right": "flexible"}))
        out.append(("ap_issubdtype_named", one, {"left": name, "right": "object"}))
    for left, right in [("character", "flexible"), ("flexible", "character"), ("flexible", "generic"), ("character", "generic"), ("object", "generic"), ("object", "flexible"), ("number", "flexible")]:
        out.append(("ap_issubdtype", one, {"left": left, "right": right}))
    out.append(("ap_lax_dtype", _r([2]), {}))
    out.append(("ap_lax_dtype", {"a": {"kind": "int", "shape": [2]}}, {}))
    for params in [{"start": 1, "size": 2}, {"start": 3}, {"full": True}, {"start": -2, "size": 2}, {"start": 1, "size": 2, "stride": 2}]:
        out.append(("ap_ds", _r([6, 2]), params))
    for prevent in [False, True]:
        out.append(("ap_dce_sink", _r([2, 3]), {"prevent": prevent}))
        out.append(("ap_dce_sink", _r([]), {"prevent": prevent}))
    out.append(("ap_io_callback", _r([2]), {}))
    out.append(("ap_io_callback", _r([]), {}))
    out.append(("ap_io_callback", _r([2, 2]), {"ordered": True}))
    for op in "ap_tree_flatten_is_leaf ap_tree_map_path ap_tree_reduce ap_tree_reduce_noinit ap_tree_assoc ap_tree_all ap_tree_broadcast ap_all_leaves ap_node ap_keyed_node ap_class_node ap_dataclass ap_static ap_partial ap_config_flags ap_memory".split():
        out.append((op, _r([2]), {}))
    for mode in ["plain", "is_leaf", "takes_path"]:
        out.append(("ap_tree_paths", _r([2]), {"mode": mode}))
    for tree in ["dict", "one", "leaf", "none", "empty"]:
        out.append(("ap_treedef_str", one, {"tree": tree}))
    out.append(("ap_primal_tangent", one, {"primal": "float32", "tangent": "int8"}))
    out.append(("ap_primal_tangent", one, {"primal": "complex64", "tangent": "uint16"}))
    out.append(("ap_primal_tangent", one, {"primal": "float16", "tangent": "float16", "name": "custom"}))
    for params in [{"items": ["i"]}, {"items": ["i", None, ["a", "b"]], "unreduced": ["c"], "reduced": ["d"]}, {"items": []}, {"items": [None, "x"]}]:
        out.append(("ap_partition", one, params))
    return out


def _ab(a, b):
    return {"a": {"kind": "real", "shape": a}, "b": {"kind": "real", "shape": b}}


def _option_cases():
    one = _r([1])
    out = []
    for n in [0, 1]:
        out.append(("ap_grad_argnums", _ab([3], [3]), {"argnums": n}))
        out.append(("ap_value_and_grad_argnums", _ab([3], [3]), {"argnums": n}))
    for axis in [0, 1, -1]:
        out.append(("ap_vmap_out_axes", _r([2, 3]), {"out_axes": axis}))
    out.append(("ap_vmap_axis_name", _r([3]), {}))
    out.append(("ap_vmap_axis_size", _r([2]), {"size": 3}))
    out.append(("ap_jit_static", _r([3]), {}))
    cplx = {"a": {"kind": "complex", "shape": [3]}}
    out.append(("ap_jacfwd_holo", cplx, {}))
    out.append(("ap_jacrev_holo", cplx, {}))
    out.append(("ap_hessian_holo", cplx, {}))
    out.append(("ap_jacobian_argnums", _ab([2], [2]), {}))
    out.append(("ap_make_jaxpr_shape", _r([2, 3]), {}))
    nan = float("nan")
    for equal_nan in [True, False]:
        for counts in [False, True]:
            out.append(("ap_unique", one, {"values": [1.0, nan, 2.0, nan, 1.0], "equal_nan": equal_nan, "counts": counts}))
    out.append(("ap_intersect_indices", one, {"x": [3.0, 1.0, 2.0, 1.0], "y": [2.0, 5.0, 3.0, 2.0]}))
    out.append(("ap_intersect_indices", one, {"x": [3.0, 1.0, 2.0], "y": [2.0, 5.0, 3.0], "assume_unique": True}))
    out.append(("ap_intersect_indices", one, {"x": [1.0], "y": [2.0]}))
    for unique in [False, True]:
        out.append(("ap_delete", _r([6]), {"obj": [1, 4], "unique": unique}))
        out.append(("ap_delete", _r([4, 3]), {"obj": [0, 2], "axis": 0, "unique": unique}))
        out.append(("ap_delete", _r([4, 3]), {"obj": [-1], "axis": 1, "unique": unique}))
    out.append(("ap_bin_edges", _r([5]), {"bins": 4}))
    out.append(("ap_polyfit_full", _ab([6], [6]), {"deg": 2}))
    out.append(("ap_polyfit_cov", _ab([6], [6]), {"deg": 2, "cov": True}))
    out.append(("ap_polyfit_cov", _ab([6], [6]), {"deg": 2, "cov": "unscaled"}))
    out.append(("ap_polyfit_cov", _ab([6], [6, 2]), {"deg": 1, "cov": True}))
    for params in [{"num": 5}, {"num": 5, "endpoint": False}, {"num": 1}, {"num": 1, "endpoint": False}, {"num": 0}]:
        out.append(("ap_linspace_retstep", _r([2]), params))
    for dims in [[2, 3], [4], [2, 1, 3]]:
        out.append(("ap_indices_sparse", one, {"dims": dims}))
    for params in [{}, {"precision": 2}, {"width": 20}, {"suppress": True}, {"precision": 1, "width": 30, "suppress": True}]:
        out.append(("ap_array_str", _r([6]), params))
        out.append(("ap_array_str", _r([2, 3]), params))
    for name, qs in [("quantile", [0.5, [0.1, 0.9]]), ("nanquantile", [0.5, [0.1, 0.9]]), ("percentile", [50.0, [10.0, 90.0]]), ("nanpercentile", [50.0, [10.0, 90.0]])]:
        for q in qs:
            out.append(("ap_" + name + "_w", _ab([6], [6]), {"q": q}))
            out.append(("ap_" + name + "_w", _ab([3, 4], [3, 4]), {"q": q, "axis": 0}))
            out.append(("ap_" + name + "_w", _ab([3, 4], [4]), {"q": q, "axis": 1, "keepdims": True}))
            out.append(("ap_" + name + "_w", _ab([3, 4], [3, 4]), {"q": q, "keepdims": True}))
            out.append(("ap_" + name + "_cdf", _r([3, 4]), {"q": q, "axis": 0}))
            out.append(("ap_" + name + "_cdf", _r([5]), {"q": q}))
    out.append(("ap_ravel_dtype", one, {"i": [0, 1], "j": [2, 7], "dims": [2, 3], "dtype": "int16"}))
    out.append(("ap_ravel_dtype", one, {"i": [0, 1], "j": [2, 7], "dims": [2, 3], "dtype": "uint8"}))
    return out


def all_cases():
    return _gap_cases() + _option_cases()
