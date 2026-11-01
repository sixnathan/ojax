module C = Core
module T = Types
module D = Dtype

let shape v = (C.get_aval v).T.shape
let dtype v = (C.get_aval v).T.dtype
let weak v = (C.get_aval v).T.weak_type
let py_float x = C.py_scalar (Dtypes.default_float_dtype ()) x

let py_float_arg x =
  match py_float x with
  | T.Concrete nd -> T.Concrete (Np_ints.mark_exact_float x nd)
  | v -> v

let py_complex z =
  let dt =
    if Dtypes.default_float_dtype () = Dtype.F64 then Dtype.Complex128
    else Dtype.Complex64
  in
  T.Concrete (Ndarray.as_weak (Ndarray.of_complex dt [||] [| z |]))

let py_int x = C.py_scalar (Dtypes.default_int_dtype ()) (float_of_int x)
let jit name ?key f args = Jaxpr.named_jit ~name ?key f args

let jit1 name ?key f args =
  match Jaxpr.named_jit ~name ?key (fun a -> [ f a ]) args with
  | [ v ] -> v
  | _ -> invalid_arg "Jnp_util.jit1: arity"

let broadcast_to v sh =
  let s = shape v in
  if s = sh then v
  else begin
    let nlead = Array.length sh - Array.length s in
    if nlead < 0 then invalid_arg "broadcast_to: incompatible shapes";
    Array.iteri
      (fun i d ->
        if not (d = sh.(i + nlead) || d = 1) then
          invalid_arg "broadcast_to: incompatible shapes")
      s;
    C.bind1
      (T.Broadcast_in_dim
         { shape = sh; dims = Array.init (Array.length s) (fun i -> i + nlead) })
      [ v ]
  end

let promote_ranks a b =
  let ra = Array.length (shape a) and rb = Array.length (shape b) in
  let lift v k =
    let s = shape v in
    C.bind1
      (T.Broadcast_in_dim
         {
           shape = Array.append (Array.make k 1) s;
           dims = Array.init (Array.length s) (fun i -> i + k);
         })
      [ v ]
  in
  if ra = rb || ra = 0 || rb = 0 then (a, b)
  else if ra < rb then (lift a (rb - ra), b)
  else (a, lift b (ra - rb))

let broadcast_arrays vs =
  match vs with
  | [] -> []
  | first :: rest ->
      if List.for_all (fun v -> shape v = shape first) rest then vs
      else
        let sh = C.elementwise_shape (List.map shape vs) in
        List.map (fun v -> broadcast_to v sh) vs

let where c x y =
  jit1 "_where"
    (function
      | [ c; x; y ] -> (
          let c =
            if dtype c = D.Bool then c
            else C.bind1 T.Ne [ c; C.splat (dtype c) [||] 0.0 ]
          in
          let dt, is_weak =
            Dtypes.result_type [ (dtype x, weak x); (dtype y, weak y) ]
          in
          let conv v = C.convert_like ~weak:is_weak v dt in
          let x = conv x in
          let y = conv y in
          if Array.length (shape c) = 0 then
            match broadcast_arrays [ x; y ] with
            | [ a; b ] -> C.bind1 T.Select_n [ c; b; a ]
            | _ -> assert false
          else
            match broadcast_arrays [ c; x; y ] with
            | [ c; a; b ] -> C.bind1 T.Select_n [ c; b; a ]
            | _ -> assert false)
      | _ -> invalid_arg "where: arity")
    [ c; x; y ]

let where_weak ?(x_weak = false) ?(y_weak = false) c x y =
  let key = (if x_weak then "w" else "s") ^ if y_weak then "w" else "s" in
  jit1 "_where" ~key
    (function
      | [ c; x; y ] -> (
          let c =
            if dtype c = D.Bool then c
            else C.bind1 T.Ne [ c; C.splat (dtype c) [||] 0.0 ]
          in
          let dt, is_weak =
            Dtypes.result_type
              [ (dtype x, weak x || x_weak); (dtype y, weak y || y_weak) ]
          in
          let conv v = C.convert_like ~weak:is_weak v dt in
          let x = conv x in
          let y = conv y in
          if Array.length (shape c) = 0 then
            match broadcast_arrays [ x; y ] with
            | [ a; b ] -> C.bind1 T.Select_n [ c; b; a ]
            | _ -> assert false
          else
            match broadcast_arrays [ c; x; y ] with
            | [ c; a; b ] -> C.bind1 T.Select_n [ c; b; a ]
            | _ -> assert false)
      | _ -> invalid_arg "where: arity")
    [ c; x; y ]
