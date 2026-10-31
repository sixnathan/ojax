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

type custom_rule =
  | Jvp_rule of {
      primal : value list -> value list;
      jvp : value list -> value list -> value list * value list;
    }
  | Vjp_rule of {
      primal : value list -> value list;
      fwd : value list -> value list * value list;
      bwd : value list -> value list -> value list;
    }
  | Linear_rule of {
      forward : value list -> value list -> value list;
      transpose : value list -> value list -> value list;
      res_differentiable : bool;
    }
  | Callback_rule of {
      host : value list -> value list;
      result_avals : aval list;
    }

let custom_rules : (int, custom_rule) Hashtbl.t = Hashtbl.create 16

let register_custom_rule rule =
  let id = fresh_id () in
  Hashtbl.replace custom_rules id rule;
  id

let custom_rule id =
  match Hashtbl.find_opt custom_rules id with
  | Some rule -> rule
  | None -> failwith "core: unknown custom derivative rule"

let with_eval_dynamic thunk = new_dynamic eval_trace thunk
let named_axis_trace : (string -> trace option) ref = ref (fun _ -> None)

let rejected_dtype name dt classes =
  raise
    (Errors.jax_type_error
       (Printf.sprintf "%s does not accept dtype %s. Accepted dtypes are %s."
          name (Dtype.short_name dt) classes))

let is_float_dtype = function
  | Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 -> true
  | _ -> false

let is_complex_dtype = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let check_operands prim args =
  let check name accepted classes =
    match args with
    | [] -> ()
    | first :: _ ->
        let dt =
          match first with
          | Concrete a -> Ndarray.dtype a
          | Tracer t -> t.aval.dtype
          | Device b -> Pjrt.Buffer.element_type b
        in
        if not (accepted dt) then rejected_dtype name dt classes
  in
  let floating = "subtypes of floating" in
  let inexact = "subtypes of floating, complexfloating" in
  let numeric = "subtypes of floating, complexfloating, integer" in
  let real = "subtypes of floating, integer, bool" in
  let real_numeric = "subtypes of floating, integer" in
  let is_inexact d = is_float_dtype d || is_complex_dtype d in
  let not_bool d = d <> Dtype.Bool in
  let not_complex d = not (is_complex_dtype d) in
  match prim with
  | Lgamma -> check "lgamma" is_float_dtype floating
  | Digamma -> check "digamma" is_float_dtype floating
  | Erf -> check "erf" is_float_dtype floating
  | Erfc -> check "erfc" is_float_dtype floating
  | Erf_inv -> check "erf_inv" is_float_dtype floating
  | Bessel_i0e -> check "bessel_i0e" is_float_dtype floating
  | Bessel_i1e -> check "bessel_i1e" is_float_dtype floating
  | Igamma -> check "igamma" is_float_dtype floating
  | Igammac -> check "igammac" is_float_dtype floating
  | Zeta -> check "zeta" is_float_dtype floating
  | Polygamma -> check "polygamma" is_float_dtype floating
  | Cbrt -> check "cbrt" is_float_dtype floating
  | Floor -> check "floor" is_float_dtype floating
  | Ceil -> check "ceil" is_float_dtype floating
  | Is_finite -> check "is_finite" is_float_dtype floating
  | Nextafter -> check "nextafter" is_float_dtype floating
  | Exp -> check "exp" is_inexact inexact
  | Log -> check "log" is_inexact inexact
  | Sin -> check "sin" is_inexact inexact
  | Cos -> check "cos" is_inexact inexact
  | Tan -> check "tan" is_inexact inexact
  | Acos -> check "acos" is_inexact inexact
  | Asin -> check "asin" is_inexact inexact
  | Atan -> check "atan" is_inexact inexact
  | Sinh -> check "sinh" is_inexact inexact
  | Cosh -> check "cosh" is_inexact inexact
  | Tanh -> check "tanh" is_inexact inexact
  | Asinh -> check "asinh" is_inexact inexact
  | Acosh -> check "acosh" is_inexact inexact
  | Atanh -> check "atanh" is_inexact inexact
  | Sqrt -> check "sqrt" is_inexact inexact
  | Rsqrt -> check "rsqrt" is_inexact inexact
  | Logistic -> check "logistic" is_inexact inexact
  | Expm1 -> check "expm1" is_inexact inexact
  | Log1p -> check "log1p" is_inexact inexact
  | Exp2 -> check "exp2" is_inexact inexact
  | Pow -> check "pow" is_inexact inexact
  | Neg -> check "neg" not_bool numeric
  | Sign -> check "sign" not_bool numeric
  | Abs ->
      check "abs"
        (function
          | Dtype.Bool | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint32
          | Dtype.Uint64 ->
              false
          | _ -> true)
        "subtypes of signedinteger, complexfloating, floating"
  | Square -> check "square" not_bool numeric
  | Sub -> check "sub" not_bool numeric
  | Add -> check "add" not_bool numeric
  | Mul -> check "mul" not_bool numeric
  | Div -> check "div" not_bool numeric
  | Atan2 -> check "atan2" is_inexact inexact
  | Integer_pow y ->
      check "integer_pow"
        (fun d -> not_bool d && (y >= 0 || is_inexact d))
        (if y >= 0 then numeric else inexact)
  | Real -> check "real" is_complex_dtype "subtypes of complexfloating"
  | Imag -> check "imag" is_complex_dtype "subtypes of complexfloating"
  | Conj ->
      check "conj"
        (function
          | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> true
          | _ -> false)
        "subtypes of complexfloating, float32, float64"
  | Round | Round_nearest_even -> check "round" is_float_dtype floating
  | Rem -> check "rem" (fun d -> not_bool d && not_complex d) real_numeric
  | Lt -> check "lt" not_complex real
  | Gt -> check "gt" not_complex real
  | Le -> check "le" not_complex real
  | Ge -> check "ge" not_complex real
  | _ -> ()

let bind prim args =
  check_operands prim args;
  let top = find_top_trace args in
  let top =
    match prim with
    | Collective { axis_name; _ } -> (
        match !named_axis_trace axis_name with
        | Some named when named.level > top.level -> named
        | _ -> top)
    | _ -> top
  in
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
  if Config.value Config.no_execution then
    failwith "JAX tried to execute a primitive, but no_execution is set";
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
