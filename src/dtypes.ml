let default_int_dtype () =
  if Config.x64_enabled () then Dtype.I64 else Dtype.I32

let default_float_dtype () =
  if Config.x64_enabled () then Dtype.F64 else Dtype.F32

let to_inexact_dtype = function
  | Dtype.Bool | Dtype.I4 | Dtype.Uint4 | Dtype.I8 | Dtype.Uint8 | Dtype.I16
  | Dtype.Uint16 | Dtype.I32 | Dtype.Uint32 ->
      Dtype.F32
  | Dtype.I64 | Dtype.Uint64 -> Dtype.F64
  | d -> d

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
  | U2
  | U4
  | U8
  | U16
  | U32
  | U64
  | I2
  | I4
  | I8
  | I16
  | I32
  | I64
  | Weak_float
  | F4_e2m1fn
  | F6_e2m3fn
  | F6_e3m2fn
  | F8_e3m4
  | F8_e4m3
  | F8_e4m3b11fnuz
  | F8_e4m3fn
  | F8_e4m3fnuz
  | F8_e5m2
  | F8_e5m2fnuz
  | F8_e8m0fnu
  | F16
  | Bf16
  | F32
  | F64
  | Weak_complex
  | C64
  | C128

let parents = function
  | Bool -> [ Weak_int ]
  | Weak_int -> [ U8; U2; U4; I8; I2; I4 ]
  | U2 | U4 | I2 | I4 -> []
  | U8 -> [ I16; U16 ]
  | U16 -> [ I32; U32 ]
  | U32 -> if Config.x64_enabled () then [ I64; U64 ] else [ I32; U64 ]
  | U64 -> [ Weak_float ]
  | I8 -> [ I16 ]
  | I16 -> [ I32 ]
  | I32 -> [ I64 ]
  | I64 -> [ Weak_float ]
  | Weak_float ->
      [
        F4_e2m1fn;
        F6_e2m3fn;
        F6_e3m2fn;
        F8_e3m4;
        F8_e4m3;
        F8_e4m3b11fnuz;
        F8_e4m3fn;
        F8_e4m3fnuz;
        F8_e5m2;
        F8_e5m2fnuz;
        F8_e8m0fnu;
        Bf16;
        F16;
        Weak_complex;
      ]
  | F4_e2m1fn | F6_e2m3fn | F6_e3m2fn | F8_e3m4 | F8_e4m3 | F8_e4m3b11fnuz
  | F8_e4m3fn | F8_e4m3fnuz | F8_e5m2 | F8_e5m2fnuz | F8_e8m0fnu ->
      []
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
    U2;
    U4;
    U8;
    U16;
    U32;
    U64;
    I2;
    I4;
    I8;
    I16;
    I32;
    I64;
    Weak_float;
    F4_e2m1fn;
    F6_e2m3fn;
    F6_e3m2fn;
    F8_e3m4;
    F8_e4m3;
    F8_e4m3b11fnuz;
    F8_e4m3fn;
    F8_e4m3fnuz;
    F8_e5m2;
    F8_e5m2fnuz;
    F8_e8m0fnu;
    F16;
    Bf16;
    F32;
    F64;
    Weak_complex;
    C64;
    C128;
  ]

let rec uppers node = node :: List.concat_map uppers (parents node)

let strong_node : Dtype.t -> node = function
  | Dtype.Bool -> Bool
  | Dtype.Uint2 -> U2
  | Dtype.Uint4 -> U4
  | Dtype.Uint8 -> U8
  | Dtype.Uint16 -> U16
  | Dtype.Uint32 -> U32
  | Dtype.Uint64 -> U64
  | Dtype.I2 -> I2
  | Dtype.I4 -> I4
  | Dtype.I8 -> I8
  | Dtype.I16 -> I16
  | Dtype.I32 -> I32
  | Dtype.I64 -> I64
  | Dtype.F4_e2m1fn -> F4_e2m1fn
  | Dtype.F6_e2m3fn -> F6_e2m3fn
  | Dtype.F6_e3m2fn -> F6_e3m2fn
  | Dtype.F8_e3m4 -> F8_e3m4
  | Dtype.F8_e4m3 -> F8_e4m3
  | Dtype.F8_e4m3b11fnuz -> F8_e4m3b11fnuz
  | Dtype.F8_e4m3fn -> F8_e4m3fn
  | Dtype.F8_e4m3fnuz -> F8_e4m3fnuz
  | Dtype.F8_e5m2 -> F8_e5m2
  | Dtype.F8_e5m2fnuz -> F8_e5m2fnuz
  | Dtype.F8_e8m0fnu -> F8_e8m0fnu
  | Dtype.F16 -> F16
  | Dtype.Bf16 -> Bf16
  | Dtype.F32 -> F32
  | Dtype.F64 -> F64
  | Dtype.Complex64 -> C64
  | Dtype.Complex128 -> C128
  | Dtype.Float0 ->
      invalid_arg "dtype=float0 is not a valid dtype for JAX type promotion."

let jax_type (d : Dtype.t) (weak : bool) : node =
  match (d, weak) with
  | Dtype.Bool, _ -> Bool
  | (Dtype.I2 | Dtype.I4 | Dtype.Uint2 | Dtype.Uint4), true -> Weak_int
  | (Dtype.I32 | Dtype.I64 | Dtype.I8 | Dtype.I16), true -> Weak_int
  | ( ( Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
      | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
      | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
      | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ),
      true ) ->
      Weak_float
  | (Dtype.Complex64 | Dtype.Complex128), true -> Weak_complex
  | _ -> strong_node d

let dtype_of_node : node -> Dtype.t = function
  | Bool -> Dtype.Bool
  | U2 -> Dtype.Uint2
  | U4 -> Dtype.Uint4
  | U8 -> Dtype.Uint8
  | U16 -> Dtype.Uint16
  | U32 -> Dtype.Uint32
  | U64 -> Dtype.Uint64
  | I2 -> Dtype.I2
  | I4 -> Dtype.I4
  | I8 -> Dtype.I8
  | I16 -> Dtype.I16
  | I32 -> Dtype.I32
  | I64 -> Dtype.I64
  | F4_e2m1fn -> Dtype.F4_e2m1fn
  | F6_e2m3fn -> Dtype.F6_e2m3fn
  | F6_e3m2fn -> Dtype.F6_e3m2fn
  | F8_e3m4 -> Dtype.F8_e3m4
  | F8_e4m3 -> Dtype.F8_e4m3
  | F8_e4m3b11fnuz -> Dtype.F8_e4m3b11fnuz
  | F8_e4m3fn -> Dtype.F8_e4m3fn
  | F8_e4m3fnuz -> Dtype.F8_e4m3fnuz
  | F8_e5m2 -> Dtype.F8_e5m2
  | F8_e5m2fnuz -> Dtype.F8_e5m2fnuz
  | F8_e8m0fnu -> Dtype.F8_e8m0fnu
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

exception Type_promotion_error of string

let node_name = function
  | Weak_int -> "<class 'int'>"
  | Weak_float -> "<class 'float'>"
  | Weak_complex -> "<class 'complex'>"
  | n -> Dtype.numpy_name (dtype_of_node n)

let promotion_message nodes =
  let names =
    "("
    ^ String.concat ", "
        (List.map
           (fun n ->
             match n with
             | Weak_int | Weak_float | Weak_complex -> "\"" ^ node_name n ^ "\""
             | _ -> "'" ^ node_name n ^ "'")
           nodes)
    ^ ")"
  in
  let narrow_float width =
    Printf.sprintf
      "Input dtypes %s have no available implicit dtype promotion path. To \
       avoid unintended promotion, %d-bit floats do not support implicit \
       promotion. If you'd like your inputs to be promoted to another type, \
       you can do so explicitly using e.g. x.astype('float32')"
      names width
  in
  let has group = List.exists (fun n -> List.mem n group) nodes in
  if
    has
      [
        F8_e3m4;
        F8_e4m3;
        F8_e4m3b11fnuz;
        F8_e4m3fn;
        F8_e4m3fnuz;
        F8_e5m2;
        F8_e5m2fnuz;
        F8_e8m0fnu;
      ]
  then narrow_float 8
  else if has [ F6_e2m3fn; F6_e3m2fn ] then narrow_float 6
  else if has [ F4_e2m1fn ] then narrow_float 4
  else if has [ I2; I4; U2; U4 ] then
    Printf.sprintf
      "Input dtypes %s have no available implicit dtype promotion path. To \
       avoid unintended promotion, 1-bit, 2-bit and 4-bit integers do not \
       support implicit promotion. If you'd like your inputs to be promoted to \
       another type, you can do so explicitly using e.g. x.astype('int32')"
      names
  else
    Printf.sprintf
      "Input dtypes %s have no available implicit dtype promotion path. Try \
       explicitly casting inputs to the desired output type."
      names

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
  | [] -> raise (Type_promotion_error (promotion_message nodes))

let promote_types (a : Dtype.t) (b : Dtype.t) : Dtype.t =
  match least_upper_bound [ jax_type a false; jax_type b false ] with
  | Weak_float -> Dtype.F64
  | node -> dtype_of_node node

let default_of_kind (d : Dtype.t) : Dtype.t =
  match d with
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      default_float_dtype ()
  | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.I64 | Dtype.I8 | Dtype.I16 ->
      default_int_dtype ()
  | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.Uint64 | Dtype.Bool | Dtype.Float0 ->
      d
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
  let small bits nmant nexp precision resolution eps epsneg fmax fmin tiny
      smallest_subnormal =
    {
      bits;
      eps;
      epsneg;
      fmax;
      fmin;
      tiny;
      smallest_subnormal;
      precision;
      resolution;
      nmant;
      nexp;
    }
  in
  match d with
  | Dtype.F4_e2m1fn -> small 4 1 2 0 1.0 0.5 0.5 6.0 (-6.0) 1.0 0.5
  | Dtype.F6_e2m3fn -> small 6 3 2 0 1.0 0.125 0.125 7.5 (-7.5) 1.0 0.125
  | Dtype.F6_e3m2fn -> small 6 2 3 0 1.0 0.25 0.125 28.0 (-28.0) 0.25 0.0625
  | Dtype.F8_e3m4 ->
      small 8 4 3 1 0.09375 0.0625 0.03125 15.5 (-15.5) 0.25 0.015625
  | Dtype.F8_e4m3 ->
      small 8 3 4 1 0.1015625 0.125 0.0625 240.0 (-240.0) 0.015625 0.001953125
  | Dtype.F8_e4m3b11fnuz ->
      small 8 3 4 1 0.1015625 0.125 0.0625 30.0 (-30.0) 0.0009765625
        0.0001220703125
  | Dtype.F8_e4m3fn ->
      small 8 3 4 1 0.1015625 0.125 0.0625 448.0 (-448.0) 0.015625 0.001953125
  | Dtype.F8_e4m3fnuz ->
      small 8 3 4 1 0.1015625 0.125 0.0625 240.0 (-240.0) 0.0078125 0.0009765625
  | Dtype.F8_e5m2 ->
      small 8 2 5 1 0.09375 0.25 0.125 57344.0 (-57344.0) 6.103515625e-05
        1.52587890625e-05
  | Dtype.F8_e5m2fnuz ->
      small 8 2 5 1 0.09375 0.25 0.125 57344.0 (-57344.0) 3.0517578125e-05
        7.62939453125e-06
  | Dtype.F8_e8m0fnu ->
      let tiny = Float.ldexp 1.0 (-127) in
      small 8 0 8 1 0.125 1.0 0.5 (Float.ldexp 1.0 127) tiny tiny tiny
  | Dtype.F16 -> build 16 10 5 3 1e-3
  | Dtype.Bf16 -> build 16 7 8 2 1e-2
  | Dtype.F32 | Dtype.Complex64 -> build 32 23 8 6 1e-6
  | Dtype.F64 | Dtype.Complex128 -> build 64 52 11 15 1e-15
  | Dtype.I2 | Dtype.I4 | Dtype.Uint2 | Dtype.Uint4 | Dtype.I8 | Dtype.I16
  | Dtype.I32 | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint32
  | Dtype.Uint64 | Dtype.Bool | Dtype.Float0 ->
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
  | Dtype.I2 -> build true 2
  | Dtype.I4 -> build true 4
  | Dtype.Uint2 -> build false 2
  | Dtype.Uint4 -> build false 4
  | Dtype.I8 -> build true 8
  | Dtype.I16 -> build true 16
  | Dtype.I32 -> build true 32
  | Dtype.I64 -> build true 64
  | Dtype.Uint8 -> build false 8
  | Dtype.Uint16 -> build false 16
  | Dtype.Uint32 -> build false 32
  | Dtype.Uint64 -> build false 64
  | _ -> invalid_arg "iinfo requires an integer dtype"

let signed_dtypes = Dtype.[ I2; I4; I8; I16; I32; I64 ]
let unsigned_dtypes = Dtype.[ Uint2; Uint4; Uint8; Uint16; Uint32; Uint64 ]

let real_float_dtypes =
  Dtype.
    [
      F4_e2m1fn;
      F6_e2m3fn;
      F6_e3m2fn;
      F8_e3m4;
      F8_e4m3;
      F8_e4m3b11fnuz;
      F8_e4m3fn;
      F8_e4m3fnuz;
      F8_e5m2;
      F8_e5m2fnuz;
      F8_e8m0fnu;
      Bf16;
      F16;
      F32;
      F64;
    ]

let complex_dtypes = Dtype.[ Complex64; Complex128 ]

let issubdtype (d : Dtype.t) kind =
  let is group = List.mem d group in
  match kind with
  | `Generic -> true
  | `Bool -> d = Dtype.Bool
  | `Number ->
      is signed_dtypes || is unsigned_dtypes || is real_float_dtypes
      || is complex_dtypes
  | `Integer -> is signed_dtypes || is unsigned_dtypes
  | `Signed -> is signed_dtypes
  | `Unsigned -> is unsigned_dtypes
  | `Inexact -> is real_float_dtypes || is complex_dtypes
  | `Floating -> is real_float_dtypes
  | `Complex -> is complex_dtypes
  | `Extended | `Prng_key -> false

let supports_inf (d : Dtype.t) =
  match d with
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
  | Dtype.F8_e5m2fnuz ->
      false
  | d -> issubdtype d `Inexact

let max_identity d =
  if supports_inf d then Float.neg_infinity else (finfo d).fmin

let min_identity d = if supports_inf d then Float.infinity else (finfo d).fmax
let float0 = Dtype.Float0
let bfloat16 = Dtype.Bf16

let itemsize_bits (d : Dtype.t) =
  match d with
  | Dtype.Bool -> 8
  | Dtype.Float0 -> invalid_arg "unexpected input: dtype=float0"
  | Dtype.Complex64 -> 64
  | Dtype.Complex128 -> 128
  | d when List.mem d real_float_dtypes -> (finfo d).bits
  | d -> (iinfo d).ibits

type scalar_type = Scalar_bool | Scalar_int | Scalar_float | Scalar_complex

let scalar_type_of (d : Dtype.t) =
  match d with
  | Dtype.Bool -> Scalar_bool
  | Dtype.Float0 -> invalid_arg "Invalid scalar value of dtype float0"
  | d when List.mem d complex_dtypes -> Scalar_complex
  | d when List.mem d real_float_dtypes -> Scalar_float
  | _ -> Scalar_int

let isdtype dt kind =
  let matches = function
    | "bool" -> dt = Dtype.Bool
    | "signed integer" -> List.mem dt signed_dtypes
    | "unsigned integer" -> List.mem dt unsigned_dtypes
    | "integral" -> List.mem dt signed_dtypes || List.mem dt unsigned_dtypes
    | "real floating" -> List.mem dt real_float_dtypes
    | "complex floating" -> List.mem dt complex_dtypes
    | "numeric" -> issubdtype dt `Number
    | k -> invalid_arg ("Unrecognized data type kind: " ^ k)
  in
  matches kind
