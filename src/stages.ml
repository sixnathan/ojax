open Types

type lowered = {
  closed : closed_jaxpr;
  in_avals : aval list;
  out_avals : aval list;
}

type compiled = { run : value list -> value list; source : lowered }
type compiler_option = Pjrt.Executable.compiler_option
type compiler_options = (string * compiler_option) list
type arg_info = { arg_aval : aval; donated : bool }

type traced = {
  jaxpr : closed_jaxpr;
  fun_name : string;
  args_info : arg_info Tree_util.t;
  in_tree : Tree_util.treedef;
  out_tree : Tree_util.treedef;
  out_avals : aval Tree_util.t;
  call : value Tree_util.t list -> value Tree_util.t;
  trace_lower : unit -> lowered;
}

type wrapped = {
  apply : value Tree_util.t list -> value Tree_util.t;
  trace : value Tree_util.t list -> traced;
  wrapped_lower : value Tree_util.t list -> lowered;
}

let arg_info_shape info = info.arg_aval.shape
let arg_info_dtype info = info.arg_aval.dtype

let closed_out_avals (closed : closed_jaxpr) =
  List.map
    (function
      | A_var v -> v.vaval
      | A_lit nd ->
          {
            shape = Ndarray.shape nd;
            dtype = Ndarray.dtype nd;
            weak_type = false;
          }
      | DropVar a -> a)
    closed.jaxpr.outs

let lower f avals =
  let closed = Jaxpr.make_jaxpr avals f in
  { closed; in_avals = avals; out_avals = closed_out_avals closed }

let as_text lowered = Stablehlo.Emit.emit_closed_jaxpr lowered.closed
let jaxpr_text lowered = Pretty_printer.closed_jaxpr_to_string lowered.closed

let as_jax_text ?(name = "_lambda") lowered =
  Stablehlo.Emit.emit_jax_text ~name lowered.closed

let jax_jaxpr_text lowered =
  Pretty_printer.closed_jaxpr_to_jax_string lowered.closed

let compile ?compiler_options lowered =
  if Config.value Config.enable_checks then
    Jaxpr.typecheck_jaxpr lowered.closed.jaxpr;
  {
    run = Backend.executor ?options:compiler_options lowered.closed;
    source = lowered;
  }

let call compiled args =
  let actual = List.map Core.get_aval args in
  List.iter2
    (fun (a : aval) (e : aval) ->
      if a.shape <> e.shape || a.dtype <> e.dtype then
        invalid_arg
          "Argument types differ from the types the function was compiled with")
    actual compiled.source.in_avals;
  compiled.run args

let cost_analysis _ = []

let trace_flat ~name f args =
  let flat_args, in_tree = Tree_util.tree_flatten (Tree_util.Tuple args) in
  let in_avals = List.map Core.get_aval flat_args in
  let flat_fun, out_tree_get = Api_util.flatten_fun f in_tree in
  let closed = Jaxpr.make_jaxpr in_avals (Linear_util.call_wrapped flat_fun) in
  let out_tree = out_tree_get () in
  let out_flat = closed_out_avals closed in
  let call_args call_args =
    let flat, _ = Tree_util.tree_flatten (Tree_util.Tuple call_args) in
    List.iter2
      (fun v (e : aval) ->
        let got = Core.get_aval v in
        if got.shape <> e.shape || got.dtype <> e.dtype then
          invalid_arg
            "Argument types differ from the types the function was traced with")
      flat in_avals;
    Tree_util.tree_unflatten out_tree (Jaxpr.eval_closed_jaxpr closed flat)
  in
  {
    jaxpr = closed;
    fun_name = name;
    args_info =
      Tree_util.tree_unflatten in_tree
        (List.map (fun a -> { arg_aval = a; donated = false }) in_avals);
    in_tree;
    out_tree;
    out_avals = Tree_util.tree_unflatten out_tree out_flat;
    call = call_args;
    trace_lower = (fun () -> { closed; in_avals; out_avals = out_flat });
  }

let jit ?(name = "<fun>") f =
  let trace args = trace_flat ~name f args in
  {
    apply = Api.jit f;
    trace;
    wrapped_lower = (fun args -> (trace args).trace_lower ());
  }
