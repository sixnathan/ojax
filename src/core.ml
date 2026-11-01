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

let tracing_dynamically () =
  match !dynamic_trace with Some t -> t.level > 0 | None -> false

let current_level () = List.length !trace_stack - 1

let enter_main weak kind global_data f =
  let level = List.length !trace_stack in
  let main = { level; kind; global_data } in
  Weak.set weak 0 (Some main);
  trace_stack := main :: !trace_stack;
  Fun.protect
    ~finally:(fun () -> trace_stack := List.tl !trace_stack)
    (fun () -> f main)

let trace_kind_name = function
  | KEval -> "EvalTrace"
  | KJVP -> "JVPTrace"
  | KBatch -> "BatchTrace"
  | KJaxpr -> "DynamicJaxprTrace"
  | KPE -> "JaxprTrace"

let leak_checked = function KJaxpr | KBatch -> true | _ -> false

let with_new_main kind global_data f =
  let weak = Weak.create 1 in
  let result = enter_main weak kind global_data f in
  if Config.value Config.check_tracer_leaks && leak_checked kind then (
    Gc.full_major ();
    if Weak.check weak 0 then
      failwith
        (Printf.sprintf "Leaked trace %s. A tracer outlived its transformation."
           (trace_kind_name kind)));
  result

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
      fwd_perturbed :
        (bool list -> value list -> value list * value list) option;
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

let custom_names : (int, string * string) Hashtbl.t = Hashtbl.create 16

let register_custom_names id ~name ~jvp_name =
  Hashtbl.replace custom_names id (name, jvp_name)

let custom_names_of id = Hashtbl.find_opt custom_names id

let custom_rule id =
  match Hashtbl.find_opt custom_rules id with
  | Some rule -> rule
  | None -> failwith "core: unknown custom derivative rule"

let custom_call_jaxprs : (int * aval list * bool, closed_jaxpr) Hashtbl.t =
  Hashtbl.create 64

let note_custom_call_jaxpr id avals c =
  Hashtbl.replace custom_call_jaxprs (id, avals, Config.x64_enabled ()) c

let find_custom_call_jaxpr id avals =
  Hashtbl.find_opt custom_call_jaxprs (id, avals, Config.x64_enabled ())

let custom_call_jaxprs_of id =
  Hashtbl.fold
    (fun (rule, _, _) c acc -> if rule = id then c :: acc else acc)
    custom_call_jaxprs []

let with_eval_dynamic thunk = new_dynamic eval_trace thunk

let ffi_impl : (primitive -> Ndarray.t list -> Ndarray.t list) ref =
  ref (fun _ _ -> failwith "ffi_call needs the Ffi module to be linked")

let named_axis_trace : (string -> trace option) ref = ref (fun _ -> None)

let check_f16_reduction axes = function
  | [ v ] -> (
      let shape, dtype =
        match v with
        | Concrete a -> (Ndarray.shape a, Ndarray.dtype a)
        | Tracer t -> (t.aval.shape, t.aval.dtype)
        | Device b -> (Pjrt.Buffer.dimensions b, Pjrt.Buffer.element_type b)
      in
      match dtype with
      | (Dtype.F16 | Dtype.Bf16)
        when Array.exists (fun axis -> shape.(axis) <> 1) axes ->
          invalid_arg
            (Printf.sprintf
               "reduce_sum on operand %s[%s] is not allowed when \
                jax_allow_f16_reductions=False."
               (if dtype = Dtype.F16 then "f16" else "bf16")
               (String.concat ","
                  (List.map string_of_int (Array.to_list shape))))
      | _ -> ())
  | _ -> ()

type shard_rule = {
  sr_mesh : (string * int) list;
  sr_devices : int array;
  sr_manual : string list;
  sr_in : string list array list;
  sr_out : string list array list;
  sr_body : closed_jaxpr;
}

type constraint_rule = {
  cr_mesh : (string * int) list;
  cr_devices : int array;
  cr_spec : string list array;
}

let shard_rules : (int, shard_rule) Hashtbl.t = Hashtbl.create 8
let constraint_rules : (int, constraint_rule) Hashtbl.t = Hashtbl.create 8
let shard_rule id = Hashtbl.find_opt shard_rules id
let constraint_rule id = Hashtbl.find_opt constraint_rules id
let register_shard_rule id rule = Hashtbl.replace shard_rules id rule
let register_constraint_rule id rule = Hashtbl.replace constraint_rules id rule
let mesh_axes : (string * int) list ref = ref []

let with_mesh_axes axes thunk =
  let saved = !mesh_axes in
  mesh_axes := axes @ saved;
  Fun.protect ~finally:(fun () -> mesh_axes := saved) thunk

let placed_eval : (primitive -> value list -> value list option) ref =
  ref (fun _ _ -> None)

let is_placed = function Device (Pjrt.Buffer.Placed _) -> true | _ -> false

let eager_eval : (primitive -> value list -> value list option) ref =
  ref (fun _ _ -> None)

let placed_dispatch (top : trace) prim args =
  if top.kind = KEval && List.exists is_placed args then !placed_eval prim args
  else None

let rejected_dtype name dt classes =
  raise
    (Errors.jax_type_error
       (Printf.sprintf "%s does not accept dtype %s. Accepted dtypes are %s."
          name (Dtype.short_name dt) classes))

let is_float_dtype d = Dtypes.issubdtype d `Floating

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
          | Dtype.Bool | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint8 | Dtype.Uint16
          | Dtype.Uint32 | Dtype.Uint64 ->
              false
          | _ -> true)
        "subtypes of signedinteger, complexfloating, floating"
  | Square -> check "square" not_bool numeric
  | Sub -> check "sub" not_bool numeric
  | Add -> check "add" not_bool numeric
  | Add_any -> ()
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
  | Cholesky | Lu | Qr _ | Eig _ | Eigh _ | Svd _ | Schur _ | Hessenberg
  | Triangular_solve _ | Tridiagonal _ | Tridiagonal_solve _ ->
      check "linalg"
        (function
          | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> true
          | _ -> false)
        "float32, float64, complex64, complex128"
  | Rem -> check "rem" (fun d -> not_bool d && not_complex d) real_numeric
  | Lt -> check "lt" not_complex real
  | Gt -> check "gt" not_complex real
  | Le -> check "le" not_complex real
  | Ge -> check "ge" not_complex real
  | _ -> ()

let is_elementwise = function
  | Add | Add_any | Sub | Mul | Div | Max | Min | Pow | And | Atan2 | Mulhi
  | Nextafter | Or | Rem | Shift_left | Shift_right_arithmetic
  | Shift_right_logical | Xor | Eq | Lt | Gt | Ge | Le | Eq_to | Le_to | Lt_to
  | Ne | Complex | Igamma | Igammac | Igamma_grad_a | Polygamma | Zeta ->
      true
  | _ -> false

let value_shape = function
  | Concrete a -> Ndarray.shape a
  | Tracer t -> t.aval.shape
  | Device b -> Pjrt.Buffer.dimensions b

let elementwise_shape shapes =
  let n = List.fold_left (fun acc s -> max acc (Array.length s)) 0 shapes in
  Array.init n (fun i ->
      List.fold_left
        (fun acc s ->
          let k = i - (n - Array.length s) in
          if k < 0 || s.(k) = 1 then acc else s.(k))
        1 shapes)

let inline_calls : (int, unit) Hashtbl.t = Hashtbl.create 64
let mark_inline_call jid = Hashtbl.replace inline_calls jid ()
let is_inline_call jid = Hashtbl.mem inline_calls jid
let inherit_inline ~from jid = if is_inline_call from then mark_inline_call jid

let call_direct : (closed_jaxpr -> value list -> value list) ref =
  ref (fun _ _ -> invalid_arg "Core.call_direct: not installed")

let call_inline : (closed_jaxpr -> value list -> value list) ref =
  ref (fun _ _ -> invalid_arg "Core.call_inline: not installed")

let keeps_scalar_operands (top : trace) =
  match top.kind with KJaxpr | KJVP | KPE | KBatch -> true | KEval -> false

let rec dispatch prim args =
  let top = find_top_trace args in
  match (prim, args) with
  | Xla_call c, _ when top.kind = KEval -> !call_inline c args
  | Xla_call c, _ when top.kind = KJaxpr && is_inline_call c.jid ->
      !call_direct c args
  | Transpose perm, [ _ ]
    when Array.for_all Fun.id (Array.mapi (fun i p -> i = p) perm) ->
      args
  | Convert_element_type dt, [ Concrete a ]
    when top.kind = KJaxpr && Array.length (Ndarray.shape a) = 0 -> (
      match rules.impl prim [ a ] with
      | [ r ] -> [ Concrete (Ndarray.canonicalize dt r) ]
      | _ -> invalid_arg "dispatch: convert arity")
  | Select_n, which :: first :: rest
    when List.exists (fun v -> value_shape v <> value_shape first) rest
         || Array.length (value_shape which) > 0
            && value_shape which <> value_shape first ->
      let out = elementwise_shape (List.map value_shape args) in
      let widen v =
        let s = value_shape v in
        if s = out then v
        else
          let k = Array.length out - Array.length s in
          List.hd
            (dispatch
               (Broadcast_in_dim
                  {
                    shape = out;
                    dims = Array.init (Array.length s) (fun i -> i + k);
                  })
               [ v ])
      in
      let which =
        if Array.length (value_shape which) = 0 && keeps_scalar_operands top
        then which
        else widen which
      in
      dispatch Select_n (which :: List.map widen (first :: rest))
  | Select_n, which :: first :: _
    when (not (keeps_scalar_operands top))
         && Array.length (value_shape which) = 0
         && Array.length (value_shape first) > 0 ->
      let out = value_shape first in
      let widened =
        List.hd
          (dispatch (Broadcast_in_dim { shape = out; dims = [||] }) [ which ])
      in
      dispatch Select_n (widened :: List.tl args)
  | _ -> dispatch_widened top prim args

and dispatch_widened top prim args =
  if
    is_elementwise prim
    && (not (keeps_scalar_operands top))
    &&
    match args with
    | first :: rest ->
        let s = value_shape first in
        List.exists (fun v -> value_shape v <> s) rest
    | [] -> false
  then begin
    let out = elementwise_shape (List.map value_shape args) in
    let compiled =
      if
        (not (Config.value Config.no_execution))
        && (not (List.exists is_placed args))
        && List.for_all
             (fun v ->
               let s = value_shape v in
               s = out || Array.length s = 0)
             args
      then !eager_eval prim args
      else None
    in
    match compiled with
    | Some outs -> outs
    | None ->
        let widen v =
          let s = value_shape v in
          if s = out then v
          else
            let k = Array.length out - Array.length s in
            match
              dispatch
                (Broadcast_in_dim
                   {
                     shape = out;
                     dims = Array.init (Array.length s) (fun i -> i + k);
                   })
                [ v ]
            with
            | [ w ] -> w
            | _ -> invalid_arg "dispatch: broadcast arity"
        in
        dispatch prim (List.map widen args)
  end
  else dispatch_equal top prim args

and dispatch_equal top prim args =
  let top =
    match prim with
    | Collective { axis_name; _ } -> (
        match !named_axis_trace axis_name with
        | Some named when named.level > top.level -> named
        | _ -> top)
    | _ -> top
  in
  match placed_dispatch top prim args with
  | Some outs -> outs
  | None ->
      let tracers = List.map (full_raise top) args in
      let outs = process_primitive top prim tracers in
      List.map full_lower outs

let bind prim args =
  check_operands prim args;
  (match prim with
  | Reduce_sum axes when not (Config.value Config.allow_f16_reductions) ->
      check_f16_reduction axes args
  | _ -> ());
  if Key_reuse.idle () then dispatch prim args
  else Key_reuse.apply prim args (fun () -> dispatch prim args)

let bind1 prim args =
  match bind prim args with
  | [ out ] -> out
  | _ -> invalid_arg "bind1: expected a single output"

let weak_convert = ref false

let convert_like ~weak v dt =
  let old_dt, old_weak =
    match v with
    | Concrete a -> (Ndarray.dtype a, Ndarray.weak a)
    | Tracer t -> (t.aval.dtype, t.aval.weak_type)
    | Device b -> (Pjrt.Buffer.element_type b, false)
  in
  if old_dt = dt && old_weak = weak then v
  else
    match v with
    | Concrete a when Array.length (Ndarray.shape a) = 0 -> (
        match rules.impl (Convert_element_type dt) [ a ] with
        | [ r ] ->
            let r = Ndarray.canonicalize dt r in
            Concrete (if weak then Ndarray.as_weak r else r)
        | _ -> invalid_arg "convert_like: arity")
    | _ ->
        let saved = !weak_convert in
        weak_convert := weak;
        Fun.protect
          ~finally:(fun () -> weak_convert := saved)
          (fun () -> bind1 (Convert_element_type dt) [ v ])

let top_is_staging args = (find_top_trace args).kind <> KEval

let staging () =
  match !dynamic_trace with Some t -> t.kind = KJaxpr | None -> false

let splat_host dt sh n x =
  match dt with
  | Dtype.Complex64 | Dtype.Complex128 ->
      Ndarray.of_complex dt sh (Array.make n { Complex.re = x; im = 0.0 })
  | _ -> Ndarray.of_floats dt sh (Array.make n x)

let splat dt sh x =
  if Array.length sh = 0 || not (staging ()) then
    Concrete (splat_host dt sh (Array.fold_left ( * ) 1 sh) x)
  else
    bind1
      (Broadcast_in_dim { shape = sh; dims = [||] })
      [ Concrete (splat_host dt [||] 1 x) ]

let py_scalar dt x = Concrete (Ndarray.as_weak (splat_host dt [||] 1 x))

let get_aval = function
  | Concrete a ->
      {
        shape = Ndarray.shape a;
        dtype = Ndarray.dtype a;
        weak_type = Ndarray.weak a;
      }
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
  (match prim with
  | Dot_general _ -> Config.check_matmul_preset_on_cpu ()
  | _ -> ());
  if Config.value Config.no_execution then
    failwith "JAX tried to execute a primitive, but no_execution is set";
  match !eager_eval prim args with
  | Some outs -> outs
  | None ->
      let args =
        if is_elementwise prim then begin
          let shapes = List.map value_shape args in
          let out = elementwise_shape shapes in
          List.map
            (fun v ->
              let s = value_shape v in
              if s = out then v
              else
                let k = Array.length out - Array.length s in
                bind1
                  (Broadcast_in_dim
                     {
                       shape = out;
                       dims = Array.init (Array.length s) (fun i -> i + k);
                     })
                  [ v ])
            args
        end
        else args
      in
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

let divide_shape_sizes s1 s2 =
  let size = Array.fold_left ( * ) 1 in
  let sz1 = size s1 and sz2 = size s2 in
  if sz1 = sz2 then 1
  else
    let tuple s =
      match Array.to_list s with
      | [ d ] -> Printf.sprintf "(%d,)" d
      | ds -> "(" ^ String.concat ", " (List.map string_of_int ds) ^ ")"
    in
    let r = sz1 mod sz2 in
    if r <> 0 then
      raise
        (Errors.Inconclusive_dimension_operation
           (Printf.sprintf
              "Cannot divide evenly the sizes of shapes %s and %s. The \
               remainder %d should be 0."
              (tuple s1) (tuple s2) r));
    sz1 / sz2
