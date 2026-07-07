let default_tol dtype =
  match dtype with
  | "bool" | "int2" | "int4" | "int8" | "int16" | "int32" | "int64" | "uint2"
  | "uint4" | "uint8" | "uint16" | "uint32" | "uint64" | "float0" ->
      0.0
  | "float4_e2m1fn" | "float8_e8m0fnu" -> 1e0
  | "float8_e3m4" | "float8_e4m3" | "float8_e4m3b11fnuz" | "float8_e4m3fn"
  | "float8_e4m3fnuz" | "float8_e5m2" | "float8_e5m2fnuz" ->
      1e-1
  | "bfloat16" -> 1e-2
  | "float16" -> 1e-3
  | "float32" -> 1e-6
  | "float64" -> 1e-15
  | "complex64" -> 1e-6
  | "complex128" -> 1e-15
  | _ -> failwith ("compare: unknown dtype " ^ dtype)

let canonical_dtype_x64_off dtype =
  match dtype with
  | "float64" -> "float32"
  | "int64" -> "int32"
  | "uint64" -> "uint32"
  | "complex128" -> "complex64"
  | d -> d

let assert_tol dtype atol rtol =
  let expected = default_tol dtype in
  if atol <> expected || rtol <> expected then
    failwith
      (Printf.sprintf "compare: manifest tol (%g,%g) != table %g for %s" atol
         rtol expected dtype)

let shapes_equal a b =
  Array.length a = Array.length b && Array.for_all2 (fun x y -> x = y) a b

let both_nan a b = Float.is_nan a && Float.is_nan b

let allclose_float name atol rtol xa xb =
  Array.iteri
    (fun i a ->
      let b = xb.(i) in
      let inf x = x = Float.infinity || x = Float.neg_infinity in
      let ok =
        if Float.is_nan a || Float.is_nan b then both_nan a b
        else if inf a || inf b then a = b
        else Float.abs (a -. b) <= atol +. (rtol *. Float.abs b)
      in
      if not ok then
        failwith
          (Printf.sprintf "%s: element %d actual=%.17g expected=%.17g" name i a
             b))
    xa

let allclose_complex name atol rtol xa xb =
  let cnan (z : Complex.t) = Float.is_nan z.re || Float.is_nan z.im in
  Array.iteri
    (fun i (a : Complex.t) ->
      let b = xb.(i) in
      let ok =
        if cnan a || cnan b then cnan a && cnan b
        else Complex.norm (Complex.sub a b) <= atol +. (rtol *. Complex.norm b)
      in
      if not ok then
        failwith
          (Printf.sprintf
             "%s: element %d actual=(%.17g,%.17g) expected=(%.17g,%.17g)" name i
             a.re a.im b.re b.im))
    xa

let exact_int name xa xb =
  Array.iteri
    (fun i a ->
      let b = xb.(i) in
      if Int64.compare a b <> 0 then
        failwith
          (Printf.sprintf "%s: element %d actual=%Ld expected=%Ld" name i a b))
    xa

let check ~name ~compare ~atol ~rtol ~(expected : Npz.t) ~(actual : Npz.t) =
  if not (shapes_equal expected.shape actual.shape) then
    failwith (name ^ ": shape mismatch");
  match (compare, expected.data, actual.data) with
  | "exact", Npz.I ea, Npz.I aa -> exact_int name aa ea
  | "allclose", Npz.F ea, Npz.F aa -> allclose_float name atol rtol aa ea
  | "allclose", Npz.C ea, Npz.C aa -> allclose_complex name atol rtol aa ea
  | "allclose", Npz.I ea, Npz.I aa -> exact_int name aa ea
  | _ -> failwith (name ^ ": data-kind/compare mismatch")
