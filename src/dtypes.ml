let default_int_dtype () =
  if Config.x64_enabled () then Dtype.I64 else Dtype.I32

let default_float_dtype () =
  if Config.x64_enabled () then Dtype.F64 else Dtype.F32

let default_complex_dtype () =
  if Config.x64_enabled () then Dtype.Complex128 else Dtype.Complex64

let canonicalize_dtype (d : Dtype.t) : Dtype.t =
  if Config.x64_enabled () then d
  else
    match d with
    | Dtype.F64 -> Dtype.F32
    | Dtype.I64 -> Dtype.I32
    | Dtype.Complex128 -> Dtype.Complex64
    | d -> d

type node =
  | Bool
  | Weak_int
  | U32
  | I32
  | I64
  | Weak_float
  | F32
  | F64
  | Weak_complex
  | C64
  | C128

let rank = function
  | Bool -> 0
  | Weak_int -> 1
  | U32 -> 2
  | I32 -> 3
  | I64 -> 4
  | Weak_float -> 5
  | F32 -> 6
  | F64 -> 7
  | Weak_complex -> 8
  | C64 -> 9
  | C128 -> 10

let jax_type (d : Dtype.t) (weak : bool) : node =
  match (d, weak) with
  | Dtype.Bool, _ -> Bool
  | (Dtype.I32 | Dtype.I64), true -> Weak_int
  | (Dtype.F32 | Dtype.F64), true -> Weak_float
  | (Dtype.Complex64 | Dtype.Complex128), true -> Weak_complex
  | Dtype.Uint32, _ -> U32
  | Dtype.I32, false -> I32
  | Dtype.I64, false -> I64
  | Dtype.F32, false -> F32
  | Dtype.F64, false -> F64
  | Dtype.Complex64, false -> C64
  | Dtype.Complex128, false -> C128

let dtype_of_node : node -> Dtype.t = function
  | Bool -> Dtype.Bool
  | U32 -> Dtype.Uint32
  | I32 -> Dtype.I32
  | I64 -> Dtype.I64
  | F32 -> Dtype.F32
  | F64 -> Dtype.F64
  | C64 -> Dtype.Complex64
  | C128 -> Dtype.Complex128
  | Weak_int -> default_int_dtype ()
  | Weak_float -> default_float_dtype ()
  | Weak_complex -> default_complex_dtype ()

let is_weak_node = function
  | Weak_int | Weak_float | Weak_complex -> true
  | _ -> false

let is_complex_node = function Weak_complex | C64 | C128 -> true | _ -> false

let real_projection = function
  | Weak_complex -> Weak_float
  | C64 -> F32
  | C128 -> F64
  | n -> n

let max_by_rank = function
  | [] -> invalid_arg "least_upper_bound: empty"
  | n :: rest ->
      List.fold_left (fun a b -> if rank b > rank a then b else a) n rest

let least_upper_bound nodes =
  let has = List.mem in
  (if has U32 nodes then
     let is_signed = function I32 | I64 | Weak_int -> true | _ -> false in
     if List.exists is_signed nodes then
       invalid_arg
         "dtypes: uint32 promoted against a signed integer joins to int64 in \
          jax (incomparable lattice edge); unsupported in this port");
  if List.exists is_complex_node nodes then
    let real_lub = max_by_rank (List.map real_projection nodes) in
    if real_lub = F64 then C128
    else if is_weak_node real_lub then Weak_complex
    else C64
  else max_by_rank nodes

let promote_types (a : Dtype.t) (b : Dtype.t) : Dtype.t =
  dtype_of_node (least_upper_bound [ jax_type a false; jax_type b false ])

let default_of_kind (d : Dtype.t) : Dtype.t =
  match d with
  | Dtype.F32 | Dtype.F64 -> default_float_dtype ()
  | Dtype.I32 | Dtype.I64 -> default_int_dtype ()
  | Dtype.Uint32 -> Dtype.Uint32
  | Dtype.Bool -> Dtype.Bool
  | Dtype.Complex64 | Dtype.Complex128 -> default_complex_dtype ()

let lattice_result_type (args : (Dtype.t * bool) list) : Dtype.t * bool =
  match args with
  | [] -> invalid_arg "lattice_result_type: at least one input required"
  | [ (d, w) ] -> (d, d <> Dtype.Bool && w)
  | _ ->
      let dtypes = List.map fst args in
      let weaks = List.map snd args in
      let all_weak = List.for_all Fun.id weaks in
      let all_same =
        match dtypes with
        | d0 :: rest -> List.for_all (fun d -> d = d0) rest
        | [] -> true
      in
      let out_dtype, out_weak =
        if all_same && not all_weak then (List.hd dtypes, false)
        else if all_weak then
          let r =
            least_upper_bound (List.map (fun d -> jax_type d false) dtypes)
          in
          (dtype_of_node r, true)
        else
          let r =
            least_upper_bound (List.map (fun (d, w) -> jax_type d w) args)
          in
          (dtype_of_node r, is_weak_node r)
      in
      (out_dtype, out_dtype <> Dtype.Bool && out_weak)

let result_type (args : (Dtype.t * bool) list) : Dtype.t * bool =
  let d, w = lattice_result_type args in
  ((if w then default_of_kind d else d), w)
