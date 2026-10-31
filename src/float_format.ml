type special = Ieee | Fn | Fnuz | Unsigned_fn | No_special

type t = {
  bits : int;
  nmant : int;
  nexp : int;
  min_exp : int;
  fmax : float;
  special : special;
}

let make bits nmant nexp min_exp fmax special =
  { bits; nmant; nexp; min_exp; fmax; special }

let of_dtype = function
  | Dtype.F4_e2m1fn -> Some (make 4 1 2 0 6.0 No_special)
  | Dtype.F6_e2m3fn -> Some (make 6 3 2 0 7.5 No_special)
  | Dtype.F6_e3m2fn -> Some (make 6 2 3 (-2) 28.0 No_special)
  | Dtype.F8_e3m4 -> Some (make 8 4 3 (-2) 15.5 Ieee)
  | Dtype.F8_e4m3 -> Some (make 8 3 4 (-6) 240.0 Ieee)
  | Dtype.F8_e4m3b11fnuz -> Some (make 8 3 4 (-10) 30.0 Fnuz)
  | Dtype.F8_e4m3fn -> Some (make 8 3 4 (-6) 448.0 Fn)
  | Dtype.F8_e4m3fnuz -> Some (make 8 3 4 (-7) 240.0 Fnuz)
  | Dtype.F8_e5m2 -> Some (make 8 2 5 (-14) 57344.0 Ieee)
  | Dtype.F8_e5m2fnuz -> Some (make 8 2 5 (-15) 57344.0 Fnuz)
  | Dtype.F8_e8m0fnu ->
      Some (make 8 0 8 (-127) (Float.ldexp 1.0 127) Unsigned_fn)
  | _ -> None

let overflow f v =
  match f.special with
  | Ieee -> Float.copy_sign Float.infinity v
  | Fn -> Float.copy_sign Float.nan v
  | Fnuz | Unsigned_fn -> Float.nan
  | No_special -> Float.copy_sign f.fmax v

let not_a_number f = match f.special with No_special -> -0.0 | _ -> Float.nan

let to_nearest_even q =
  let t = Float.trunc q in
  if Float.abs (q -. t) = 0.5 then
    if Float.rem t 2.0 = 0.0 then t else Float.round q
  else Float.round q

let round_finite f v =
  let _, e = Float.frexp v in
  let exp = e - 1 in
  let lowest = Stdlib.max (exp - f.nmant) (f.min_exp - f.nmant) in
  let out = Float.ldexp (to_nearest_even (Float.ldexp v (-lowest))) lowest in
  if Float.abs out > f.fmax then overflow f v else out

let round_unsigned f v =
  if v <= 0.0 || not (Float.is_finite v) then Float.nan
  else
    let least = Float.ldexp 1.0 f.min_exp in
    if v <= least then least
    else if v < 2.0 *. least then 2.0 *. least
    else round_finite f v

let narrow_to_f32 v = Int32.float_of_bits (Int32.bits_of_float v)

let round f v0 =
  let v = narrow_to_f32 v0 in
  if Float.is_nan v then
    if f.special = No_special then -0.0 else Float.copy_sign Float.nan v0
  else if f.special = Unsigned_fn then round_unsigned f v
  else if not (Float.is_finite v) then overflow f v
  else if v = 0.0 then if f.special = Fnuz then 0.0 else v
  else
    let out = round_finite f v in
    if out = 0.0 && f.special = Fnuz then 0.0 else out

let sign_bit f = 1 lsl (f.bits - 1)
let exponent_ones f = (1 lsl f.nexp) - 1
let mantissa_ones f = (1 lsl f.nmant) - 1

let nan_bits f =
  match f.special with
  | Ieee -> (exponent_ones f lsl f.nmant) lor (1 lsl (f.nmant - 1))
  | Fn -> (1 lsl (f.bits - 1)) - 1
  | Fnuz -> sign_bit f
  | Unsigned_fn -> 0xff
  | No_special -> invalid_arg "Float_format: dtype has no nan"

let encode_finite f v =
  let m = Float.abs v in
  if m = 0.0 then 0
  else
    let _, e = Float.frexp m in
    let exp = e - 1 in
    if exp < f.min_exp then int_of_float (Float.ldexp m (f.nmant - f.min_exp))
    else
      let biased = exp - f.min_exp + 1 in
      let fraction = int_of_float (Float.ldexp m (f.nmant - exp)) in
      (biased lsl f.nmant) lor (fraction land mantissa_ones f)

let encode f v0 =
  let v = round f v0 in
  if Float.is_nan v then
    nan_bits f
    lor if f.special <> Fnuz && Float.sign_bit v then sign_bit f else 0
  else if f.special = Unsigned_fn then
    let _, e = Float.frexp v in
    e - 1 - f.min_exp
  else
    let sign = if Float.sign_bit v then sign_bit f else 0 in
    if Float.is_finite v then sign lor encode_finite f v
    else sign lor (exponent_ones f lsl f.nmant)

let decode f bits =
  if f.special = Unsigned_fn then
    if bits = 0xff then Float.nan else Float.ldexp 1.0 (bits + f.min_exp)
  else
    let body = bits land (sign_bit f - 1) in
    let negative = bits land sign_bit f <> 0 in
    let biased = body lsr f.nmant and fraction = body land mantissa_ones f in
    let is_nan =
      match f.special with
      | Ieee -> biased = exponent_ones f && fraction <> 0
      | Fn -> body = nan_bits f
      | Fnuz -> bits = nan_bits f
      | Unsigned_fn | No_special -> false
    in
    if is_nan then if negative then -.Float.nan else Float.nan
    else if f.special = Ieee && biased = exponent_ones f then
      if negative then Float.neg_infinity else Float.infinity
    else
      let magnitude =
        if biased = 0 then
          Float.ldexp (float_of_int fraction) (f.min_exp - f.nmant)
        else
          Float.ldexp
            (float_of_int (fraction + (1 lsl f.nmant)))
            (biased - 1 + f.min_exp - f.nmant)
      in
      if negative then -.magnitude else magnitude

let next_after f x y =
  let xb = encode f x and yb = encode f y in
  let sign = sign_bit f in
  let is_nan v = Float.is_nan v && f.special <> No_special in
  if is_nan x || is_nan y then not_a_number f
  else if xb = yb then y
  else
    let xm = xb land (sign - 1) and ym = yb land (sign - 1) in
    let out =
      if xm = 0 then if ym = 0 then yb else yb land sign lor 1
      else if xb land sign <> yb land sign || xm > ym then xb - 1
      else xb + 1
    in
    decode f out
