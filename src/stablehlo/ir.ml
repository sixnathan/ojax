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
  | Dtype.I8 -> "i8"
  | Dtype.I16 -> "i16"
  | Dtype.Uint8 -> "ui8"
  | Dtype.Uint16 -> "ui16"
  | Dtype.Uint64 -> "ui64"
  | Dtype.F16 -> "f16"
  | Dtype.Bf16 -> "bf16"
  | Dtype.I2 -> "i2"
  | Dtype.I4 -> "i4"
  | Dtype.Uint2 -> "ui2"
  | Dtype.Uint4 -> "ui4"
  | Dtype.F4_e2m1fn -> "f4E2M1FN"
  | Dtype.F6_e2m3fn -> "f6E2M3FN"
  | Dtype.F6_e3m2fn -> "f6E3M2FN"
  | Dtype.F8_e3m4 -> "f8E3M4"
  | Dtype.F8_e4m3 -> "f8E4M3"
  | Dtype.F8_e4m3b11fnuz -> "f8E4M3B11FNUZ"
  | Dtype.F8_e4m3fn -> "f8E4M3FN"
  | Dtype.F8_e4m3fnuz -> "f8E4M3FNUZ"
  | Dtype.F8_e5m2 -> "f8E5M2"
  | Dtype.F8_e5m2fnuz -> "f8E5M2FNUZ"
  | Dtype.F8_e8m0fnu -> "f8E8M0FNU"
  | Dtype.Float0 -> invalid_arg "Ir.element_type: float0 has no stablehlo type"

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
let f32_round = Ndarray.(fun v -> float_of_f32_bits (f32_bits_of_float v))

let bits_equal dtype a b =
  if is_f32 dtype then
    Int32.equal (Ndarray.f32_bits_of_float a) (Ndarray.f32_bits_of_float b)
  else Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let hex_bits dtype v =
  if is_f32 dtype then Printf.sprintf "0x%08lX" (Ndarray.f32_bits_of_float v)
  else Printf.sprintf "0x%016LX" (Int64.bits_of_float v)

let big_of_int n =
  let rec limbs n = if n = 0 then [] else (n land 0xffff) :: limbs (n lsr 16) in
  limbs n

let big_mul_small big k =
  let rec go carry = function
    | [] -> if carry = 0 then [] else big_of_int carry
    | limb :: rest ->
        let p = (limb * k) + carry in
        (p land 0xffff) :: go (p lsr 16) rest
  in
  go 0 big

let big_div_small big k =
  let rem = ref 0 in
  let quotient =
    List.fold_left
      (fun acc limb ->
        let cur = (!rem lsl 16) lor limb in
        rem := cur mod k;
        (cur / k) :: acc)
      [] (List.rev big)
  in
  let rec trim = function 0 :: rest -> trim rest | l -> l in
  (List.rev (trim (List.rev quotient)), !rem)

let big_active_bits big =
  match List.rev big with
  | [] -> 0
  | top :: rest ->
      let rec width n = if n = 0 then 0 else 1 + width (n lsr 1) in
      width top + (16 * List.length rest)

let rec big_pow_mul big k n =
  if n = 0 then big else big_pow_mul (big_mul_small big k) k (n - 1)

let rec big_digits big exp acc =
  if big = [] then (acc, exp)
  else
    let q, d = big_div_small big 10 in
    if acc = [] && d = 0 then big_digits q (exp + 1) acc
    else big_digits q exp (d :: acc)

let shortest_binary v =
  let frac, e = Float.frexp v in
  let rec strip m e = if m land 1 = 0 then strip (m / 2) (e + 1) else (m, e) in
  strip (int_of_float (Float.ldexp frac 53)) (e - 53)

let round_digits digits exp precision =
  let n = List.length digits in
  if n <= precision then (digits, exp)
  else
    let kept = List.filteri (fun i _ -> i < precision) digits in
    let at_length out = (out, exp + n - List.length out) in
    if List.nth digits precision < 5 then
      let rec trim = function 0 :: rest -> trim rest | l -> l in
      at_length (List.rev (trim (List.rev kept)))
    else
      let rec carry = function
        | [] -> None
        | 9 :: rest -> carry rest
        | d :: rest -> Some ((d + 1) :: rest)
      in
      match carry (List.rev kept) with
      | Some rest -> at_length (List.rev rest)
      | None -> ([ 1 ], exp + n)

let scientific_digits ?(precision = 6) v =
  let m, e = shortest_binary v in
  let big, exp10 =
    if e >= 0 then (big_pow_mul (big_of_int m) 2 e, 0)
    else (big_pow_mul (big_of_int m) 5 (-e), e)
  in
  let required = ((precision * 196) + 58) / 59 in
  let bits = big_active_bits big in
  let tens = if bits > required then (bits - required) * 59 / 196 else 0 in
  let rec drop big k =
    if k = 0 then big else drop (fst (big_div_small big 10)) (k - 1)
  in
  let digits, exp = big_digits (drop big tens) (exp10 + tens) [] in
  round_digits digits exp precision

let small_finite_text v =
  let sign = if Float.sign_bit v then "-" else "" in
  if v = 0.0 then sign ^ "0.000000e+00"
  else
    let digits, exp = scientific_digits (Float.abs v) in
    let text = String.concat "" (List.map string_of_int digits) in
    let sci = exp + String.length text - 1 in
    let padded = text ^ String.make (7 - String.length text) '0' in
    Printf.sprintf "%s%c.%se%+03d" sign padded.[0] (String.sub padded 1 6) sci

let stage1 dtype v =
  let spliced = small_finite_text v in
  if bits_equal dtype (float_of_string spliced) v then Some spliced else None

let strip_trailing_zeros digits =
  let n = ref (String.length digits) in
  while !n > 1 && digits.[!n - 1] = '0' do
    decr n
  done;
  String.sub digits 0 !n

let apfloat_default_text ~precision v =
  let sign = if Float.sign_bit v then "-" else "" in
  let digits, exp = scientific_digits ~precision (Float.abs v) in
  let buffer = String.concat "" (List.map string_of_int digits) in
  let n = String.length buffer in
  let scientific =
    if exp >= 0 then exp > 3 || n + exp > precision
    else
      let msd = exp + n - 1 in
      if msd >= 0 then false else -msd > 3
  in
  if scientific then
    let e = exp + n - 1 in
    sign
    ^ String.make 1 buffer.[0]
    ^ "."
    ^ (if n = 1 then "0" else String.sub buffer 1 (n - 1))
    ^ "E"
    ^ (if e >= 0 then "+" else "-")
    ^ string_of_int (abs e)
  else if exp >= 0 then sign ^ buffer ^ String.make exp '0'
  else
    let whole = exp + n in
    if whole > 0 then
      sign ^ String.sub buffer 0 whole ^ "."
      ^ String.sub buffer whole (n - whole)
    else sign ^ "0." ^ String.make (-whole) '0' ^ buffer

let stage2 dtype v =
  let precision = if is_f32 dtype then 9 else 17 in
  let text = apfloat_default_text ~precision v in
  if String.contains text '.' then text else hex_bits dtype v

let double_attr_text v =
  let short = small_finite_text v in
  if float_of_string short = v then short
  else apfloat_default_text ~precision:9 v

let narrow_stage2 dtype v =
  let precision = match dtype with Dtype.F16 -> 5 | _ -> 4 in
  apfloat_default_text ~precision v

let round_half_even x =
  let r = Float.floor x in
  let d = x -. r in
  if d > 0.5 || (d = 0.5 && Float.rem r 2.0 <> 0.0) then r +. 1.0 else r

let narrow_round dtype v =
  if v = 0.0 || not (Float.is_finite v) then v
  else
    let precision, emin =
      match dtype with Dtype.F16 -> (11, -14) | _ -> (8, -126)
    in
    let m, e = Float.frexp v in
    let bits =
      if e - 1 < emin then precision - (emin - (e - 1)) else precision
    in
    if bits < 0 then Float.copy_sign 0.0 v
    else Float.ldexp (round_half_even (Float.ldexp m bits)) (e - bits)

let narrow_finite dtype v0 =
  let v = narrow_round dtype v0 in
  let spliced = small_finite_text v in
  if narrow_round dtype (float_of_string spliced) = v then spliced
  else narrow_stage2 dtype v

let narrow_literal dtype v =
  let nan_bits, inf_bits =
    match dtype with
    | Dtype.F16 -> (Ndarray.bits_of_half, 0x7C00)
    | _ -> (Ndarray.bits_of_bfloat, 0x7F80)
  in
  match Float.classify_float v with
  | FP_nan -> Printf.sprintf "0x%04X" (nan_bits v)
  | FP_infinite ->
      Printf.sprintf "0x%04X"
        (if v < 0.0 then inf_bits lor 0x8000 else inf_bits)
  | _ -> narrow_finite dtype v

let small_literal f v0 =
  let v = Float_format.round f v0 in
  if Float.is_finite v then small_finite_text v
  else Printf.sprintf "0x%02X" (Float_format.encode f v)

let rec float_literal dtype v0 =
  match dtype with
  | Dtype.F16 | Dtype.Bf16 -> narrow_literal dtype v0
  | Dtype.F32 | Dtype.F64 -> wide_literal dtype v0
  | d -> (
      match Float_format.of_dtype d with
      | Some f -> small_literal f v0
      | None -> invalid_arg "Ir.float_literal: not a floating-point dtype")

and wide_literal dtype v0 =
  let v = if is_f32 dtype then f32_round v0 else v0 in
  match Float.classify_float v with
  | FP_nan -> hex_bits dtype v
  | FP_infinite -> hex_bits dtype v
  | _ -> ( match stage1 dtype v with Some s -> s | None -> stage2 dtype v)

let int_literal v = Int64.to_string v
let bool_literal b = if b then "true" else "false"
let dense body = "dense<" ^ body ^ ">"

let int_array_attr xs =
  "[" ^ (Array.to_list xs |> List.map string_of_int |> String.concat ", ") ^ "]"

let enum_attr kind value = Printf.sprintf "#stablehlo<%s %s>" kind value

let complex_literal dtype (z : Complex.t) =
  let part =
    match dtype with Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32
  in
  Printf.sprintf "(%s,%s)" (float_literal part z.re) (float_literal part z.im)
