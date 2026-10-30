let default_int_dtype () =
  if Config.x64_enabled () then Dtype.I64 else Dtype.I32

let default_float_dtype () =
  if Config.x64_enabled () then Dtype.F64 else Dtype.F32

let to_inexact_dtype = function
  | Dtype.Bool | Dtype.I8 | Dtype.Uint8 | Dtype.I16 | Dtype.Uint16 | Dtype.I32
  | Dtype.Uint32 ->
      Dtype.F32
  | Dtype.I64 | Dtype.Uint64 -> Dtype.F64
  | ( Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 | Dtype.Complex64
    | Dtype.Complex128 ) as d ->
      d

let default_complex_dtype () =
  if Config.x64_enabled () then Dtype.Complex128 else Dtype.Complex64

let canonicalize_dtype (d : Dtype.t) : Dtype.t =
  if Config.x64_enabled () then d
  else
    match d with
    | Dtype.F64 -> Dtype.F32
    | Dtype.I64 -> Dtype.I32
    | Dtype.Complex128 -> Dtype.Complex64
    | Dtype.Uint64 -> Dtype.Uint32
    | d -> d

type node =
  | Bool
  | Weak_int
  | U8
  | U16
  | U32
  | U64
  | I8
  | I16
  | I32
  | I64
  | Weak_float
  | F16
  | Bf16
  | F32
  | F64
  | Weak_complex
  | C64
  | C128

let parents = function
  | Bool -> [ Weak_int ]
  | Weak_int -> [ U8; I8 ]
  | U8 -> [ I16; U16 ]
  | U16 -> [ I32; U32 ]
  | U32 -> [ I64; U64 ]
  | U64 -> [ Weak_float ]
  | I8 -> [ I16 ]
  | I16 -> [ I32 ]
  | I32 -> [ I64 ]
  | I64 -> [ Weak_float ]
  | Weak_float -> [ Weak_complex; F16; Bf16 ]
  | F16 -> [ F32 ]
  | Bf16 -> [ F32 ]
  | F32 -> [ F64; C64 ]
  | F64 -> [ C128 ]
  | Weak_complex -> [ C64 ]
  | C64 -> [ C128 ]
  | C128 -> []

let all_nodes =
  [
    Bool;
    Weak_int;
    U8;
    U16;
    U32;
    U64;
    I8;
    I16;
    I32;
    I64;
    Weak_float;
    F16;
    Bf16;
    F32;
    F64;
    Weak_complex;
    C64;
    C128;
  ]

let rec uppers node = node :: List.concat_map uppers (parents node)

let jax_type (d : Dtype.t) (weak : bool) : node =
  match (d, weak) with
  | Dtype.Bool, _ -> Bool
  | (Dtype.I32 | Dtype.I64 | Dtype.I8 | Dtype.I16), true -> Weak_int
  | (Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16), true -> Weak_float
  | (Dtype.Complex64 | Dtype.Complex128), true -> Weak_complex
  | Dtype.Uint8, _ -> U8
  | Dtype.Uint16, _ -> U16
  | Dtype.Uint32, _ -> U32
  | Dtype.Uint64, _ -> U64
  | Dtype.I8, false -> I8
  | Dtype.I16, false -> I16
  | Dtype.I32, false -> I32
  | Dtype.I64, false -> I64
  | Dtype.F16, false -> F16
  | Dtype.Bf16, false -> Bf16
  | Dtype.F32, false -> F32
  | Dtype.F64, false -> F64
  | Dtype.Complex64, false -> C64
  | Dtype.Complex128, false -> C128

let dtype_of_node : node -> Dtype.t = function
  | Bool -> Dtype.Bool
  | U8 -> Dtype.Uint8
  | U16 -> Dtype.Uint16
  | U32 -> Dtype.Uint32
  | U64 -> Dtype.Uint64
  | I8 -> Dtype.I8
  | I16 -> Dtype.I16
  | I32 -> Dtype.I32
  | I64 -> Dtype.I64
  | F16 -> Dtype.F16
  | Bf16 -> Dtype.Bf16
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

let least_upper_bound nodes =
  let common =
    List.filter
      (fun candidate ->
        List.for_all (fun n -> List.mem candidate (uppers n)) nodes)
      all_nodes
  in
  match
    List.filter
      (fun candidate ->
        List.for_all (fun other -> List.mem other (uppers candidate)) common)
      common
  with
  | best :: _ -> best
  | [] -> invalid_arg "dtypes: no unique least upper bound"

let promote_types (a : Dtype.t) (b : Dtype.t) : Dtype.t =
  dtype_of_node (least_upper_bound [ jax_type a false; jax_type b false ])

let default_of_kind (d : Dtype.t) : Dtype.t =
  match d with
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 -> default_float_dtype ()
  | Dtype.I32 | Dtype.I64 | Dtype.I8 | Dtype.I16 -> default_int_dtype ()
  | Dtype.Uint32 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint64 -> d
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

type finfo = {
  bits : int;
  eps : float;
  epsneg : float;
  fmax : float;
  fmin : float;
  tiny : float;
  smallest_subnormal : float;
  precision : int;
  resolution : float;
  nmant : int;
  nexp : int;
}

let finfo (d : Dtype.t) =
  let build bits nmant nexp precision resolution =
    let eps = Float.ldexp 1.0 (-nmant) in
    let max_exponent = (1 lsl (nexp - 1)) - 1 in
    let tiny = Float.ldexp 1.0 (1 - max_exponent) in
    let fmax = (2.0 -. eps) *. Float.ldexp 1.0 max_exponent in
    {
      bits;
      eps;
      epsneg = eps /. 2.0;
      fmax;
      fmin = -.fmax;
      tiny;
      smallest_subnormal = tiny *. eps;
      precision;
      resolution;
      nmant;
      nexp;
    }
  in
  match d with
  | Dtype.F16 -> build 16 10 5 3 1e-3
  | Dtype.Bf16 -> build 16 7 8 2 1e-2
  | Dtype.F32 | Dtype.Complex64 -> build 32 23 8 6 1e-6
  | Dtype.F64 | Dtype.Complex128 -> build 64 52 11 15 1e-15
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.Uint32 | Dtype.Uint64 | Dtype.Bool ->
      invalid_arg "finfo requires a floating point dtype"

type iinfo = { imin : float; imax : float; ibits : int }

let iinfo (d : Dtype.t) =
  let build signed ibits =
    let span = Float.ldexp 1.0 ibits in
    if signed then
      { imin = -.(span /. 2.0); imax = (span /. 2.0) -. 1.0; ibits }
    else { imin = 0.0; imax = span -. 1.0; ibits }
  in
  match d with
  | Dtype.I8 -> build true 8
  | Dtype.I16 -> build true 16
  | Dtype.I32 -> build true 32
  | Dtype.I64 -> build true 64
  | Dtype.Uint8 -> build false 8
  | Dtype.Uint16 -> build false 16
  | Dtype.Uint32 -> build false 32
  | Dtype.Uint64 -> build false 64
  | Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 | Dtype.Complex64
  | Dtype.Complex128 | Dtype.Bool ->
      invalid_arg "iinfo requires an integer dtype"

let issubdtype (d : Dtype.t) kind =
  match (kind, d) with
  | `Generic, _ -> true
  | `Bool, Dtype.Bool -> true
  | `Number, Dtype.Bool -> false
  | `Number, _ -> true
  | ( `Integer,
      ( Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.Uint8
      | Dtype.Uint16 | Dtype.Uint32 | Dtype.Uint64 ) ) ->
      true
  | `Signed, (Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64) -> true
  | `Unsigned, (Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint32 | Dtype.Uint64) ->
      true
  | ( `Inexact,
      ( Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 | Dtype.Complex64
      | Dtype.Complex128 ) ) ->
      true
  | `Floating, (Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64) -> true
  | `Complex, (Dtype.Complex64 | Dtype.Complex128) -> true
  | _ -> false

let isdtype dt kind =
  let f = Dtype.short_name dt in
  ignore f;
  let matches = function
    | "bool" -> dt = Dtype.Bool
    | "signed integer" -> List.mem dt Dtype.[ I8; I16; I32; I64 ]
    | "unsigned integer" -> List.mem dt Dtype.[ Uint8; Uint16; Uint32; Uint64 ]
    | "integral" ->
        List.mem dt Dtype.[ I8; I16; I32; I64; Uint8; Uint16; Uint32; Uint64 ]
    | "real floating" -> List.mem dt Dtype.[ F16; Bf16; F32; F64 ]
    | "complex floating" -> List.mem dt Dtype.[ Complex64; Complex128 ]
    | "numeric" -> dt <> Dtype.Bool
    | k -> invalid_arg ("Unrecognized data type kind: " ^ k)
  in
  matches kind
