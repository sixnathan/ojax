open Types

let b1 = Core.bind1
let add a b = b1 Add [ a; b ]
let sub a b = b1 Sub [ a; b ]
let mul a b = b1 Mul [ a; b ]
let div a b = b1 Div [ a; b ]
let neg a = b1 Neg [ a ]
let zeros_like_value = Ad_util.zeros_like_value

let ones_like_aval a =
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n 1.0))

let ones_like_value v = ones_like_aval (Core.get_aval v)
let arity () = failwith "ad: rule arity mismatch"

let jvp_rule prim (primals : value list) (tangents : value list) : value * value
    =
  match prim with
  | Add -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (add x y, add tx ty)
      | _ -> arity ())
  | Sub -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (sub x y, sub tx ty)
      | _ -> arity ())
  | Mul -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (mul x y, add (mul tx y) (mul x ty))
      | _ -> arity ())
  | Div -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (div x y, add (div tx y) (div (neg (mul x ty)) (mul y y)))
      | _ -> arity ())
  | Neg -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (neg x, neg tx)
      | _ -> arity ())
  | Sin -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Sin [ x ], mul (b1 Cos [ x ]) tx)
      | _ -> arity ())
  | Cos -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Cos [ x ], mul (neg (b1 Sin [ x ])) tx)
      | _ -> arity ())
  | Exp -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Exp [ x ] in
          (po, mul po tx)
      | _ -> arity ())
  | Log -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Log [ x ], div tx x)
      | _ -> arity ())
  | Tanh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Tanh [ x ] in
          (po, mul (sub (ones_like_value po) (mul po po)) tx)
      | _ -> arity ())
  | Max -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (b1 Max [ x; y ], b1 Select_n [ b1 Gt [ x; y ]; ty; tx ])
      | _ -> arity ())
  | Min -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (b1 Min [ x; y ], b1 Select_n [ b1 Lt [ x; y ]; ty; tx ])
      | _ -> arity ())
  | Pow -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          let po = b1 Pow [ x; y ] in
          let one = ones_like_value y in
          let dx = mul (mul y (b1 Pow [ x; sub y one ])) tx in
          let dy = mul (mul (b1 Log [ x ]) po) ty in
          (po, add dx dy)
      | _ -> arity ())
  | Abs -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Abs [ x ], mul (b1 Sign [ x ]) tx)
      | _ -> arity ())
  | Sign -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Sign [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Eq | Lt | Gt -> (
      match (primals, tangents) with
      | [ x; y ], [ _; _ ] ->
          let po = b1 prim [ x; y ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Select_n -> (
      match (primals, tangents) with
      | which :: cases, _ :: tcases ->
          (b1 Select_n (which :: cases), b1 Select_n (which :: tcases))
      | _ -> arity ())
  | Convert_element_type dt -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 (Convert_element_type dt) [ x ] in
          let to_ =
            match dt with
            | Dtype.F32 | Dtype.F64 -> b1 (Convert_element_type dt) [ tx ]
            | _ -> zeros_like_value po
          in
          (po, to_)
      | _ -> arity ())
  | Broadcast_in_dim p -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 (Broadcast_in_dim p) [ x ], b1 (Broadcast_in_dim p) [ tx ])
      | _ -> arity ())
  | Reshape ns -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 (Reshape ns) [ x ], b1 (Reshape ns) [ tx ])
      | _ -> arity ())
  | Reduce_sum ax -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 (Reduce_sum ax) [ x ], b1 (Reduce_sum ax) [ tx ])
      | _ -> arity ())
  | Dot_general dd -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          ( b1 (Dot_general dd) [ x; y ],
            add (b1 (Dot_general dd) [ tx; y ]) (b1 (Dot_general dd) [ x; ty ])
          )
      | _ -> arity ())
  | Xla_call _ | Cond _ ->
      failwith "ad: jvp of control primitive not supported in M1"

let new_jvp_tracer trace primal tangent : tracer =
  {
    id = Core.fresh_id ();
    trace;
    aval = Core.get_aval primal;
    payload = JVP { primal; tangent };
  }

let jvp_pure trace v = Tracer (new_jvp_tracer trace v (zeros_like_value v))

let as_jvp trace v =
  match v with
  | Tracer t when t.trace.level = trace.level -> (
      match t.payload with
      | JVP { primal; tangent } -> (primal, tangent)
      | _ -> failwith "ad: expected a JVP tracer")
  | _ -> (v, zeros_like_value v)

let jvp_process_primitive trace prim args =
  let pairs = List.map (as_jvp trace) args in
  let primals = List.map fst pairs and tangents = List.map snd pairs in
  let po, to_ = jvp_rule prim primals tangents in
  [ Tracer (new_jvp_tracer trace po to_) ]

let interpreter : Core.interpreter =
  {
    i_pure = jvp_pure;
    i_lift = jvp_pure;
    i_full_lower = (fun v -> v);
    i_process_primitive = jvp_process_primitive;
    i_process_custom_jvp =
      (fun _ ~primal:_ ~jvp:_ _ ->
        failwith "ad: custom_jvp not supported in M1");
    i_process_custom_vjp =
      (fun _ ~primal:_ ~fwd:_ ~bwd:_ _ ->
        failwith "ad: custom_vjp not supported in M1");
  }

let install () = Core.register_interpreter KJVP interpreter
let () = install ()

let jvp (f : value list -> value list) (primals : value list)
    (tangents : value list) : value list * value list =
  Core.with_new_main KJVP GNone (fun main ->
      let ins =
        List.map2 (fun x t -> Tracer (new_jvp_tracer main x t)) primals tangents
      in
      let outs = f ins in
      let pairs =
        List.map
          (fun v ->
            match Core.full_raise main v with
            | Tracer { payload = JVP { primal; tangent }; _ } ->
                (primal, tangent)
            | _ -> failwith "ad.jvp: expected a JVP tracer")
          outs
      in
      List.split pairs)

let linearize (f : value list -> value list) (primals : value list) :
    value list * (value list -> value list) =
  let pvals_in =
    List.map Partial_eval.partial_val_known primals
    @ List.map
        (fun x -> Partial_eval.partial_val_unknown (Core.get_aval x))
        primals
  in
  let f_jvp inputs =
    let px, tx = Util.split_half inputs in
    let po, to_ = jvp f px tx in
    po @ to_
  in
  let jaxpr, consts, pvals_out =
    Partial_eval.partial_eval_flat f_jvp pvals_in
  in
  let primal_pvals, _ = Util.split_half pvals_out in
  let primals_out =
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> failwith "ad.linearize: primal output not known")
      primal_pvals
  in
  let f_lin tangents = Jaxpr.eval_jaxpr jaxpr (consts @ tangents) in
  (primals_out, f_lin)

type tval = Prim of value | Undef of aval

let is_undef = function Undef _ -> true | Prim _ -> false

let prim_val = function
  | Prim v -> v
  | Undef _ -> failwith "ad: undefined primal"

let in_aval = function Undef a -> a | Prim v -> Core.get_aval v

let broadcast_transpose (shape : int array) (dims : int array)
    (in_shape : int array) ct =
  let out_ndim = Array.length shape in
  let bdim_of = Array.make out_ndim (-1) in
  Array.iteri (fun i d -> bdim_of.(d) <- i) dims;
  let sum_axes = ref [] in
  for d = out_ndim - 1 downto 0 do
    let i = bdim_of.(d) in
    if i < 0 then sum_axes := d :: !sum_axes
    else if in_shape.(i) = 1 && shape.(d) <> 1 then sum_axes := d :: !sum_axes
  done;
  let reduced = b1 (Reduce_sum (Array.of_list !sum_axes)) [ ct ] in
  b1 (Reshape in_shape) [ reduced ]

let reduce_sum_transpose (axes : int array) (in_shape : int array) ct =
  let in_ndim = Array.length in_shape in
  let is_red = Array.make in_ndim false in
  Array.iter (fun a -> is_red.(a) <- true) axes;
  let kept = ref [] in
  for d = in_ndim - 1 downto 0 do
    if not is_red.(d) then kept := d :: !kept
  done;
  b1 (Broadcast_in_dim { shape = in_shape; dims = Array.of_list !kept }) [ ct ]

let transpose_rule prim (cts : value list) (primals : tval list) :
    value option list =
  let ct1 () =
    match cts with
    | [ c ] -> c
    | _ -> failwith "ad: transpose expects a single cotangent"
  in
  match prim with
  | Add ->
      let ct = ct1 () in
      [ Some ct; Some ct ]
  | Sub ->
      let ct = ct1 () in
      [ Some ct; Some (neg ct) ]
  | Neg -> [ Some (neg (ct1 ())) ]
  | Mul -> (
      let ct = ct1 () in
      match primals with
      | [ x; y ] ->
          if is_undef x then [ Some (mul ct (prim_val y)); None ]
          else [ None; Some (mul (prim_val x) ct) ]
      | _ -> arity ())
  | Div -> (
      let ct = ct1 () in
      match primals with
      | [ _; y ] -> [ Some (div ct (prim_val y)); None ]
      | _ -> arity ())
  | Select_n -> (
      let ct = ct1 () in
      match primals with
      | which :: cases ->
          let wv = prim_val which in
          None
          :: List.mapi
               (fun i p ->
                 if is_undef p then
                   let zs =
                     List.mapi
                       (fun j _ -> if j = i then ct else zeros_like_value ct)
                       cases
                   in
                   Some (b1 Select_n (wv :: zs))
                 else None)
               cases
      | _ -> arity ())
  | Convert_element_type _ -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (b1 (Convert_element_type (in_aval x).dtype) [ ct ]) ]
      | _ -> arity ())
  | Broadcast_in_dim { shape; dims } -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (broadcast_transpose shape dims (in_aval x).shape ct) ]
      | _ -> arity ())
  | Reshape _ -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (b1 (Reshape (in_aval x).shape) [ ct ]) ]
      | _ -> arity ())
  | Reduce_sum axes -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (reduce_sum_transpose axes (in_aval x).shape ct) ]
      | _ -> arity ())
  | Sin | Cos | Exp | Log | Tanh | Max | Min | Pow | Abs | Sign | Eq | Lt | Gt
  | Dot_general _ | Xla_call _ | Cond _ ->
      failwith "ad: primitive has no transpose rule in M1"

let eval_jaxpr_transposed (jx : jaxpr) (args : tval list) (cts : value list) :
    value list =
  let primal_env : (int, tval) Hashtbl.t = Hashtbl.create 32 in
  let read_primal = function
    | A_var v -> (
        match Hashtbl.find_opt primal_env v.vid with
        | Some tv -> tv
        | None -> Undef v.vaval)
    | A_lit nd -> Prim (Concrete nd)
    | DropVar a -> Undef a
  in
  let write_primal (v : var) tv =
    match tv with
    | Undef _ -> ()
    | Prim _ -> Hashtbl.replace primal_env v.vid tv
  in
  List.iter2 write_primal jx.in_binders args;
  let ct_env : (int, value) Hashtbl.t = Hashtbl.create 32 in
  let read_ct (v : var) =
    match Hashtbl.find_opt ct_env v.vid with
    | Some c ->
        Hashtbl.remove ct_env v.vid;
        c
    | None -> Ad_util.zeros_like_aval v.vaval
  in
  let write_ct atom ct =
    match atom with
    | A_var v -> (
        match Hashtbl.find_opt ct_env v.vid with
        | Some c -> Hashtbl.replace ct_env v.vid (Ad_util.add_jaxvals c ct)
        | None -> Hashtbl.replace ct_env v.vid ct)
    | _ -> ()
  in
  List.iter2 write_ct jx.outs cts;
  List.iter
    (fun (e : eqn) ->
      let cts_in = List.map read_ct e.outs in
      let primals_in = List.map read_primal e.inputs in
      let cts_out = transpose_rule e.prim cts_in primals_in in
      List.iter2
        (fun atom cto ->
          match cto with Some ct -> write_ct atom ct | None -> ())
        e.inputs cts_out)
    (List.rev jx.eqns);
  List.fold_right2
    (fun (v : var) tv acc ->
      match tv with Undef _ -> read_ct v :: acc | Prim _ -> acc)
    jx.in_binders args []

let vjp (f : value list -> value list) (primals : value list) :
    value list * (value list -> value list) =
  let pvals_in =
    List.map Partial_eval.partial_val_known primals
    @ List.map
        (fun x -> Partial_eval.partial_val_unknown (Core.get_aval x))
        primals
  in
  let f_jvp inputs =
    let px, tx = Util.split_half inputs in
    let po, to_ = jvp f px tx in
    po @ to_
  in
  let jaxpr, consts, pvals_out =
    Partial_eval.partial_eval_flat f_jvp pvals_in
  in
  let primal_pvals, _ = Util.split_half pvals_out in
  let primals_out =
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> failwith "ad.vjp: primal output not known")
      primal_pvals
  in
  let transpose_inputs =
    List.map (fun c -> Prim c) consts
    @ List.map (fun x -> Undef (Core.get_aval x)) primals
  in
  let f_vjp cts = eval_jaxpr_transposed jaxpr transpose_inputs cts in
  (primals_out, f_vjp)

let grad (f : value list -> value) (xs : value list) : value list =
  let primals_out, f_vjp = vjp (fun args -> [ f args ]) xs in
  let y =
    match primals_out with
    | [ y ] -> y
    | _ -> failwith "ad.grad: expected a single output"
  in
  f_vjp [ ones_like_value y ]
