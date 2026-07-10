let target_version = "1.17.0"

let element_type = function
  | Dtype.F32 -> "f32"
  | Dtype.F64 -> "f64"
  | Dtype.I32 -> "i32"
  | Dtype.I64 -> "i64"
  | Dtype.Bool -> "i1"
  | Dtype.Uint32 -> "ui32"
  | Dtype.Complex64 -> "complex<f32>"
  | Dtype.Complex128 -> "complex<f64>"

let tensor_type dtype shape =
  let elt = element_type dtype in
  if Array.length shape = 0 then Printf.sprintf "tensor<%s>" elt
  else
    let dims =
      Array.to_list shape |> List.map string_of_int |> String.concat "x"
    in
    Printf.sprintf "tensor<%sx%s>" dims elt

let tensor_type_of_aval (a : Types.aval) = tensor_type a.dtype a.shape
let is_f32 = function Dtype.F32 -> true | _ -> false
let f32_round v = Int32.float_of_bits (Int32.bits_of_float v)

let bits_equal dtype a b =
  if is_f32 dtype then
    Int32.equal (Int32.bits_of_float a) (Int32.bits_of_float b)
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let hex_bits dtype v =
  if is_f32 dtype then Printf.sprintf "0x%08lX" (Int32.bits_of_float v)
  else Printf.sprintf "0x%016LX" (Int64.bits_of_float v)

let canonical_nan dtype =
  if is_f32 dtype then "0x7FC00000" else "0x7FF8000000000000"

let stage1 dtype v =
  let s = Printf.sprintf "%.5e" v in
  let ei = String.index s 'e' in
  let spliced =
    String.sub s 0 ei ^ "0" ^ String.sub s ei (String.length s - ei)
  in
  if bits_equal dtype (float_of_string spliced) v then Some spliced else None

let strip_trailing_zeros digits =
  let n = ref (String.length digits) in
  while !n > 1 && digits.[!n - 1] = '0' do
    decr n
  done;
  String.sub digits 0 !n

let sig_figs dtype = if is_f32 dtype then 9 else 17

let stage2 dtype v =
  let p = sig_figs dtype in
  let s = Printf.sprintf "%.*e" (p - 1) v in
  let ei = String.index s 'e' in
  let mant = String.sub s 0 ei in
  let e = int_of_string (String.sub s (ei + 1) (String.length s - ei - 1)) in
  let neg = mant.[0] = '-' in
  let mant = if neg then String.sub mant 1 (String.length mant - 1) else mant in
  let digits =
    strip_trailing_zeros (String.concat "" (String.split_on_char '.' mant))
  in
  let sign = if neg then "-" else "" in
  let ndig = String.length digits in
  if e >= -3 && e <= p - 2 then
    let pp = e + 1 in
    if pp >= ndig then hex_bits dtype v
    else if pp <= 0 then sign ^ "0." ^ String.make (-pp) '0' ^ digits
    else sign ^ String.sub digits 0 pp ^ "." ^ String.sub digits pp (ndig - pp)
  else
    let m =
      if ndig = 1 then digits ^ ".0"
      else String.sub digits 0 1 ^ "." ^ String.sub digits 1 (ndig - 1)
    in
    let es =
      if e >= 0 then Printf.sprintf "+%d" e else Printf.sprintf "-%d" (-e)
    in
    sign ^ m ^ "E" ^ es

let float_literal dtype v0 =
  if not (is_f32 dtype || dtype = Dtype.F64) then
    invalid_arg "Ir.float_literal: not a floating-point dtype";
  let v = if is_f32 dtype then f32_round v0 else v0 in
  match Float.classify_float v with
  | FP_nan -> canonical_nan dtype
  | FP_infinite -> hex_bits dtype v
  | _ -> ( match stage1 dtype v with Some s -> s | None -> stage2 dtype v)

let int_literal v = Int64.to_string v
let bool_literal b = if b then "true" else "false"
let dense body = "dense<" ^ body ^ ">"

let int_array_attr xs =
  "[" ^ (Array.to_list xs |> List.map string_of_int |> String.concat ", ") ^ "]"

let enum_attr kind value = Printf.sprintf "#stablehlo<%s %s>" kind value
