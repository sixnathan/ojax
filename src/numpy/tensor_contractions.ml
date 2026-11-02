module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)

let pref preferred vs =
  match preferred with Some d -> d | None -> NL.result_type vs

let astype v dt = if dtype v = dt then v else NL.astype v dt

let dot_general ?precision ?preferred lc rc lb rb a b =
  let preferred_element_type =
    match preferred with
    | Some p -> Dtypes.canonicalize_dtype p
    | None -> Dtypes.promote_types (dtype a) (dtype b)
  in
  Lax_api.dot_general ?precision ~preferred_element_type a b ~lhs_contract:lc
    ~rhs_contract:rc ~lhs_batch:lb ~rhs_batch:rb

let contraction_inputs preferred a b =
  match preferred with
  | Some _ -> (a, b)
  | None ->
      let pdt = NL.result_type [ a; b ] in
      (astype a pdt, astype b pdt)

let mul_prim dt = if dt = D.Bool then T.And else T.Mul

let bcast_binop prim a b =
  let dt = NL.result_type [ a; b ] in
  let a, b = Jnp_util.promote_ranks (astype a dt) (astype b dt) in
  C.bind1 prim [ a; b ]

let dot ?precision ?preferred a b =
  let a, b = contraction_inputs preferred a b in
  let na = ndim a and nb = ndim b in
  let lc, rc =
    if na = 0 || nb = 0 then ([||], [||])
    else if nb = 1 then ([| na - 1 |], [| 0 |])
    else ([| na - 1 |], [| nb - 2 |])
  in
  dot_general ?precision ?preferred lc rc [||] [||] a b

let matmul ?precision ?preferred a b =
  Eager_jit.one "numpy.tensor_contractions.matmul" (precision, preferred) []
    [ a; b ] (fun _ vals ->
      let a, b = match vals with [ a; b ] -> (a, b) | _ -> assert false in
      let typeof v =
        let aval = C.get_aval v in
        Printf.sprintf "%s%s[%s]"
          (if aval.T.weak_type then "~" else "")
          (Dtype.numpy_name aval.T.dtype)
          (String.concat ","
             (List.map string_of_int (Array.to_list aval.T.shape)))
      in
      let a_type = typeof a and b_type = typeof b in
      let a, b = contraction_inputs preferred a b in
      let na = ndim a and nb = ndim b in
      List.iteri
        (fun i n ->
          if n < 1 then
            Py_exn.value_error
              (Printf.sprintf
                 "matmul input operand %d must have ndim at least 1, but it \
                  has ndim %d"
                 i n))
        [ na; nb ];
      let a_is_mat = na > 1 and b_is_mat = nb > 1 in
      let batch_of sh is_mat =
        if is_mat then Array.to_list (Array.sub sh 0 (Array.length sh - 2))
        else []
      in
      let a_dims = batch_of (shape a) a_is_mat
      and b_dims = batch_of (shape b) b_is_mat in
      let num_batch = max (List.length a_dims) (List.length b_dims) in
      let pad dims =
        List.init (num_batch - List.length dims) (fun _ -> None)
        @ List.map Option.some dims
      in
      let a_squeeze = ref []
      and b_squeeze = ref []
      and both_squeeze = ref []
      and a_batch = ref []
      and b_batch = ref []
      and idx_batch = ref []
      and idx_a = ref []
      and idx_b = ref [] in
      let len r = List.length !r in
      let push r v = r := !r @ [ v ] in
      List.iteri
        (fun i pair ->
          match pair with
          | None, _ -> push idx_b i
          | _, None -> push idx_a i
          | Some 1, Some 1 ->
              push a_squeeze (len idx_batch + len idx_a + len a_squeeze);
              push b_squeeze (len idx_batch + len idx_b + len b_squeeze);
              push both_squeeze i
          | Some 1, Some _ ->
              push idx_b i;
              push a_squeeze (len idx_batch + len idx_a + len a_squeeze)
          | Some _, Some 1 ->
              push idx_a i;
              push b_squeeze (len idx_batch + len idx_b + len b_squeeze)
          | Some ba, Some bb when ba = bb ->
              push a_batch (len idx_batch + len idx_a);
              push b_batch (len idx_batch + len idx_b);
              push idx_batch i
          | Some _, Some _ ->
              Py_exn.value_error
                (Printf.sprintf
                   "Incompatible types for matmul arguments: %s and %s" a_type
                   b_type))
        (List.combine (pad a_dims) (pad b_dims));
      if a_is_mat then push idx_a num_batch;
      if b_is_mat then push idx_b (num_batch + if a_is_mat then 1 else 0);
      let order = !idx_batch @ !idx_a @ !idx_b in
      let perm =
        Array.of_list
          (List.map snd
             (List.sort compare (List.mapi (fun pos v -> (v, pos)) order)))
      in
      let squeezed v axes =
        if axes = [] then v else NL.squeeze ~axis:(Array.of_list axes) v
      in
      let a = squeezed a !a_squeeze and b = squeezed b !b_squeeze in
      let out =
        dot_general ?precision ?preferred
          [| ndim a - 1 |]
          [| (ndim b - 1 - if b_is_mat then 1 else 0) |]
          (Array.of_list !a_batch) (Array.of_list !b_batch) a b
      in
      let result = C.bind1 (T.Transpose perm) [ out ] in
      if !both_squeeze = [] then result
      else NL.expand_dims result (Array.of_list !both_squeeze))

let matvec x1 x2 =
  let n2 = ndim x2 in
  let x2e = NL.expand_dims x2 [| n2 |] in
  let r = matmul x1 x2e in
  NL.squeeze ~axis:[| ndim r - 1 |] r

let matvec p0 p1 =
  Jnp_util.jit1 "matvec"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> matvec p0 p1 | _ -> invalid_arg "matvec: arity")
    [ p0; p1 ]

let vecmat x1 x2 =
  let n1 = ndim x1 in
  let x1e = NL.expand_dims x1 [| n1 - 1 |] in
  let r = matmul x1e x2 in
  NL.squeeze ~axis:[| ndim r - 2 |] r

let vecmat p0 p1 =
  Jnp_util.jit1 "vecmat"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> vecmat p0 p1 | _ -> invalid_arg "vecmat: arity")
    [ p0; p1 ]

let vdot ?precision ?preferred a b =
  let a =
    match (C.get_aval a).T.dtype with
    | D.Complex64 | D.Complex128 -> C.bind1 T.Conj [ a ]
    | _ -> a
  in
  dot ?precision ?preferred (NL.ravel a) (NL.ravel b)

let vecdot ?(axis = -1) ?precision ?preferred x1 x2 =
  if
    Validate.tuple_index (shape x1) axis <> Validate.tuple_index (shape x2) axis
  then
    Py_exn.value_error
      (Printf.sprintf "axes must match; got shapes %s and %s with axis=%d"
         (Validate.python_tuple (shape x1))
         (Validate.python_tuple (shape x2))
         axis);
  ignore
    (Validate.broadcast_shapes
       [
         Array.of_list
           (List.filteri
              (fun i _ -> i <> Validate.canonicalize_axis axis (ndim x1))
              (Array.to_list (shape x1)));
         Array.of_list
           (List.filteri
              (fun i _ -> i <> Validate.canonicalize_axis axis (ndim x2))
              (Array.to_list (shape x2)));
       ]);
  let x1 = NL.moveaxis [| axis |] [| ndim x1 - 1 |] x1 in
  let x2 = NL.moveaxis [| axis |] [| ndim x2 - 1 |] x2 in
  let x1 =
    match dtype x1 with
    | D.Complex64 | D.Complex128 -> C.bind1 T.Conj [ x1 ]
    | _ -> x1
  in
  let x1, x2 = contraction_inputs preferred x1 x2 in
  let sh = NL.broadcast_shapes_n [ shape x1; shape x2 ] in
  let bcast v = if shape v = sh then v else NL.broadcast_to v sh in
  let last = Array.length sh - 1 in
  let batch = Array.init last Fun.id in
  dot_general ?precision ?preferred [| last |] [| last |] batch batch (bcast x1)
    (bcast x2)

type td_axes = Ax_int of int | Ax_pair of int array * int array

let tensordot ?precision ?preferred ?(axes = Ax_int 2) a b =
  let a, b = contraction_inputs preferred a b in
  let na = ndim a and nb = ndim b in
  let lc, rc =
    match axes with
    | Ax_int k ->
        if k > min na nb then
          Py_exn.type_error
            (Printf.sprintf
               "Number of tensordot axes (axes %d) exceeds input ranks (%s and \
                %s)"
               k
               (Validate.python_tuple (shape a))
               (Validate.python_tuple (shape b)));
        (Array.init k (fun i -> na - k + i), Array.init k Fun.id)
    | Ax_pair (ax1, ax2) ->
        if Array.length ax1 <> Array.length ax2 then
          Py_exn.type_error
            (Printf.sprintf
               "tensordot requires axes lists to have equal length, got %s and \
                %s."
               (Validate.python_tuple ax1)
               (Validate.python_tuple ax2));
        ( Array.map (fun i -> Validate.canonicalize_axis i na) ax1,
          Array.map (fun i -> Validate.canonicalize_axis i nb) ax2 )
  in
  dot_general ?precision ?preferred lc rc [||] [||] a b

let inner ?precision ?preferred a b =
  if ndim a = 0 || ndim b = 0 then begin
    let pdt = pref preferred [ a; b ] in
    bcast_binop (mul_prim pdt) (astype a pdt) (astype b pdt)
  end
  else tensordot ?precision ?preferred ~axes:(Ax_pair ([| -1 |], [| -1 |])) a b

let outer a b =
  let pdt = pref None [ a; b ] in
  let a = NL.ravel (astype a pdt) and b = NL.ravel (astype b pdt) in
  let a2 = NL.expand_dims a [| 1 |] in
  let b2 = NL.expand_dims b [| 0 |] in
  bcast_binop (mul_prim pdt) a2 b2

let matvec a0 a1 =
  Eager_jit.one "numpy.tensor_contractions.matvec" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> matvec a0 a1
      | _ -> assert false)

let vecmat a0 a1 =
  Eager_jit.one "numpy.tensor_contractions.vecmat" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> vecmat a0 a1
      | _ -> assert false)

let vdot ?precision ?preferred a0 a1 =
  Eager_jit.one "numpy.tensor_contractions.vdot" (precision, preferred) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> vdot ?precision ?preferred a0 a1
      | _ -> assert false)

let inner ?precision ?preferred a0 a1 =
  Eager_jit.one "numpy.tensor_contractions.inner" (precision, preferred) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> inner ?precision ?preferred a0 a1
      | _ -> assert false)

let outer a0 a1 =
  Eager_jit.one "numpy.tensor_contractions.outer" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> outer a0 a1
      | _ -> assert false)
