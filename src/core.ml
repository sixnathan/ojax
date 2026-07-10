open Types

let fresh_id =
  let counter = ref 0 in
  fun () ->
    incr counter;
    !counter

type prim_rules = {
  mutable impl : primitive -> Ndarray.t list -> Ndarray.t list;
  mutable abstract_eval : primitive -> aval list -> aval list;
}

let rules =
  {
    impl = (fun _ _ -> failwith "core: impl rules not installed");
    abstract_eval =
      (fun _ _ -> failwith "core: abstract_eval rules not installed");
  }

type interpreter = {
  i_pure : trace -> value -> value;
  i_lift : trace -> value -> value;
  i_full_lower : value -> value;
  i_process_primitive : trace -> primitive -> value list -> value list;
  i_process_custom_jvp :
    trace ->
    primal:(value list -> value list) ->
    jvp:(value list -> value list) ->
    value list ->
    value list;
  i_process_custom_vjp :
    trace ->
    primal:(value list -> value list) ->
    fwd:(value list -> value list) ->
    bwd:(value list -> value list) ->
    value list ->
    value list;
}

let num_kinds = 5

let kind_index = function
  | KEval -> 0
  | KJVP -> 1
  | KBatch -> 2
  | KJaxpr -> 3
  | KPE -> 4

let interpreters : interpreter option array = Array.make num_kinds None

let register_interpreter kind interp =
  interpreters.(kind_index kind) <- Some interp

let interpreter_for trace =
  match interpreters.(kind_index trace.kind) with
  | Some interp -> interp
  | None -> failwith "core: no interpreter registered for this trace kind"

let eval_trace = { level = 0; kind = KEval; global_data = GNone }
let trace_stack : trace list ref = ref [ eval_trace ]
let dynamic_trace : trace option ref = ref None

let with_new_main kind global_data f =
  let level = List.length !trace_stack in
  let main = { level; kind; global_data } in
  trace_stack := main :: !trace_stack;
  Fun.protect
    ~finally:(fun () -> trace_stack := List.tl !trace_stack)
    (fun () -> f main)

let new_dynamic main f =
  let prev = !dynamic_trace in
  dynamic_trace := Some main;
  Fun.protect ~finally:(fun () -> dynamic_trace := prev) (fun () -> f ())

let find_top_trace args =
  let top =
    List.fold_left
      (fun acc v ->
        match v with
        | Tracer t -> if t.trace.level > acc.level then t.trace else acc
        | Concrete _ -> acc
        | Device _ -> acc)
      eval_trace args
  in
  match !dynamic_trace with Some dt when dt.level > top.level -> dt | _ -> top

let full_raise trace v =
  match v with
  | Concrete _ -> (interpreter_for trace).i_pure trace v
  | Device b ->
      (interpreter_for trace).i_pure trace (Concrete (Pjrt.Buffer.to_host b))
  | Tracer t ->
      if t.trace == trace then v
      else if t.trace.level < trace.level then
        (interpreter_for trace).i_lift trace v
      else if t.trace.level > trace.level then
        invalid_arg
          (Printf.sprintf "Can't lift level %d to %d." t.trace.level trace.level)
      else invalid_arg "Different traces at same level."

let full_lower v =
  match v with
  | Concrete _ -> v
  | Device _ -> v
  | Tracer t -> (interpreter_for t.trace).i_full_lower v

let process_primitive trace prim args =
  (interpreter_for trace).i_process_primitive trace prim args

let process_custom_jvp trace primal ~jvp args =
  (interpreter_for trace).i_process_custom_jvp trace ~primal ~jvp args

let process_custom_vjp trace primal ~fwd ~bwd args =
  (interpreter_for trace).i_process_custom_vjp trace ~primal ~fwd ~bwd args

let bind prim args =
  let top = find_top_trace args in
  let tracers = List.map (full_raise top) args in
  let outs = process_primitive top prim tracers in
  List.map full_lower outs

let bind1 prim args =
  match bind prim args with
  | [ out ] -> out
  | _ -> invalid_arg "bind1: expected a single output"

let get_aval = function
  | Concrete a ->
      { shape = Ndarray.shape a; dtype = Ndarray.dtype a; weak_type = false }
  | Tracer t -> t.aval
  | Device b ->
      {
        shape = Pjrt.Buffer.dimensions b;
        dtype = Pjrt.Buffer.element_type b;
        weak_type = false;
      }

let as_concrete = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> failwith "eval: expected a concrete value"

let eval_process_primitive _trace prim args =
  let inputs = List.map as_concrete args in
  let results = rules.impl prim inputs in
  List.map
    (fun r -> Concrete (Ndarray.canonicalize (Ndarray.dtype r) r))
    results

let eval_interpreter =
  {
    i_pure = (fun _ v -> v);
    i_lift = (fun _ v -> v);
    i_full_lower = (fun v -> v);
    i_process_primitive = eval_process_primitive;
    i_process_custom_jvp = (fun _ ~primal ~jvp:_ args -> primal args);
    i_process_custom_vjp = (fun _ ~primal ~fwd:_ ~bwd:_ args -> primal args);
  }

let () = register_interpreter KEval eval_interpreter
