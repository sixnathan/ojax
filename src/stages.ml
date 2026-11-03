open Types

type lowered = {
  closed : closed_jaxpr;
  in_avals : aval list;
  out_avals : aval list;
  module_name : string;
  result_names : string list option;
  keep_unused : bool;
  donated : bool list;
}

type compiled = {
  run : value list -> value list;
  source : lowered;
  exec : Pjrt.Executable.t Lazy.t;
  options : (string * Pjrt.Executable.compiler_option) list;
}

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

let result_names_of_tree out_tree =
  let rec count = function
    | Tree_util.Leaf_def -> 1
    | Null_def -> 0
    | List_def ds | Tuple_def ds | Custom_def (_, _, ds) -> total ds
    | Dict_def kvs -> total (List.map snd kvs)
  and total ds = List.fold_left (fun acc d -> acc + count d) 0 ds in
  let tree =
    Tree_util.tree_unflatten out_tree (List.init (count out_tree) Fun.id)
  in
  let clean = function
    | Tree_util.Flattened_index_key i -> Tree_util.Sequence_key i
    | key -> key
  in
  List.map
    (fun (path, _) -> "result" ^ Tree_util.keystr (List.map clean path))
    (Tree_util.tree_leaves_with_path tree)

let lower f avals =
  let closed = Jaxpr.make_jaxpr avals f in
  {
    closed;
    in_avals = avals;
    out_avals = closed_out_avals closed;
    module_name = "_lambda";
    result_names = None;
    keep_unused = false;
    donated = [];
  }

let raw_text lowered = Stablehlo.Emit.emit_closed_jaxpr lowered.closed
let jaxpr_text lowered = Pretty_printer.closed_jaxpr_to_string lowered.closed

let as_jax_text ?name ?result_names lowered =
  let name = Option.value name ~default:lowered.module_name in
  let result_names =
    match result_names with
    | Some _ -> result_names
    | None -> lowered.result_names
  in
  Stablehlo.Emit.emit_jax_text ~keep_unused:lowered.keep_unused
    ~donated:lowered.donated ?result_names ~name lowered.closed

let as_text lowered = as_jax_text lowered

let jax_jaxpr_text lowered =
  Pretty_printer.closed_jaxpr_to_jax_string lowered.closed

let compile ?compiler_options lowered =
  if Config.value Config.enable_checks then
    Jaxpr.typecheck_jaxpr lowered.closed.jaxpr;
  {
    run = Backend.executor ?options:compiler_options lowered.closed;
    source = lowered;
    exec = lazy (Backend.Xla.compile lowered.closed);
    options = Option.value ~default:[] compiler_options;
  }

let compiled_as_text compiled =
  Pjrt.Executable.optimized_text ~options:compiled.options
    (Backend.xla_client ())
    (as_jax_text compiled.source)

let call compiled args =
  let actual = List.map Core.get_aval args in
  List.iter2
    (fun (a : aval) (e : aval) ->
      if a.shape <> e.shape || a.dtype <> e.dtype then
        invalid_arg
          "Argument types differ from the types the function was compiled with")
    actual compiled.source.in_avals;
  compiled.run args

let cost_analysis compiled =
  Pjrt.Executable.cost_analysis (Lazy.force compiled.exec)

let memory_analysis compiled =
  Pjrt.Executable.memory_stats (Lazy.force compiled.exec)

let donated_leaves donate_argnums args =
  List.concat
    (List.mapi
       (fun i arg ->
         List.map
           (fun _ -> List.mem i donate_argnums)
           (Tree_util.tree_leaves arg))
       args)

let trace_flat ?(donate_argnums = []) ?(keep_unused = false) ~name f args =
  let donated = donated_leaves donate_argnums args in
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
        (List.map2 (fun a d -> { arg_aval = a; donated = d }) in_avals donated);
    in_tree;
    out_tree;
    out_avals = Tree_util.tree_unflatten out_tree out_flat;
    call = call_args;
    trace_lower =
      (fun () ->
        Stablehlo.Emit.warn_unusable_donations ~keep_unused ~donated closed;
        {
          closed;
          in_avals;
          out_avals = out_flat;
          module_name = (if name = "<fun>" then "_lambda" else name);
          result_names = Some (result_names_of_tree out_tree);
          keep_unused;
          donated;
        });
  }

let jit ?(name = "<fun>") ?donate_argnums ?keep_unused ?inline ?compiler_options
    f =
  let trace args = trace_flat ?donate_argnums ?keep_unused ~name f args in
  {
    apply = Api.jit ?donate_argnums ?keep_unused ?inline ?compiler_options f;
    trace;
    wrapped_lower = (fun args -> (trace args).trace_lower ());
  }
