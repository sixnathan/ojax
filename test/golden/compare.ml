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

let valid_tol_reason = function
  | "reduction_reassociation" | "argmax_tie" | "transcendental_ulp"
  | "matmul_reassociation" ->
      true
  | _ -> false

let assert_tol_widened dtype atol rtol reason =
  match reason with
  | None -> assert_tol dtype atol rtol
  | Some r ->
      let base = default_tol dtype in
      if not (valid_tol_reason r) then
        failwith ("compare: invalid tol widening reason " ^ r)
      else if atol <> rtol then
        failwith "compare: widened tol requires atol = rtol"
      else if base = 0.0 then
        failwith "compare: cannot widen an exact tolerance"
      else
        let ok = ref false in
        List.iter
          (fun k -> if atol = base *. (10.0 ** float_of_int k) then ok := true)
          [ 0; 1; 2 ];
        if not !ok then
          failwith
            (Printf.sprintf
               "compare: widened tol %g not a 1..100x multiple of base %g for \
                %s"
               atol base dtype)

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

type gauge = Columns | Rows | Whole

let gauge_group gauge shape i =
  let r = Array.length shape in
  match gauge with
  | Whole -> 0
  | Rows -> i / shape.(r - 1)
  | Columns ->
      let n = shape.(r - 1) in
      let m = shape.(r - 2) in
      (i / (m * n) * n) + (i mod n)

let align_signs gauge ~(expected : Npz.t) ~(actual : Npz.t) =
  match (expected.data, actual.data) with
  | Npz.F ea, Npz.F aa when Array.length ea = Array.length aa ->
      let groups = Hashtbl.create 16 in
      Array.iteri
        (fun i a ->
          let g = gauge_group gauge actual.shape i in
          let d = Option.value (Hashtbl.find_opt groups g) ~default:0.0 in
          Hashtbl.replace groups g (d +. (a *. ea.(i))))
        aa;
      let flip i a =
        if Hashtbl.find groups (gauge_group gauge actual.shape i) < 0.0 then -.a
        else a
      in
      { actual with data = Npz.F (Array.mapi flip aa) }
  | _ -> actual

let suite = ref "unknown"
let set_suite name = suite := name
let tallies : (string, int ref * int ref) Hashtbl.t = Hashtbl.create 64

let exact_log =
  lazy
    (Option.bind (Sys.getenv_opt "OJAX_EXACT_LOG") (fun path ->
         try Some (open_out_gen [ Open_append; Open_creat ] 0o644 path)
         with Sys_error _ -> None))

let same_bits a b =
  Int64.bits_of_float a = Int64.bits_of_float b
  || (Float.is_nan a && Float.is_nan b)

let measure_exact (expected : Npz.t) (actual : Npz.t) =
  let largest = ref 0.0 and differing = ref 0 and count = ref 0 in
  let compare_float a b =
    incr count;
    if not (same_bits a b) then begin
      incr differing;
      let relative = Float.abs (a -. b) /. Float.max (Float.abs b) 1e-300 in
      if relative > !largest then largest := relative
    end
  in
  let compare_int a b =
    incr count;
    if a <> b then incr differing
  in
  if not (shapes_equal expected.shape actual.shape) then differing := -1
  else
    begin match (expected.data, actual.data) with
    | Npz.F e, Npz.F a when Array.length e = Array.length a ->
        Array.iteri (fun i x -> compare_float x e.(i)) a
    | Npz.C e, Npz.C a when Array.length e = Array.length a ->
        Array.iteri
          (fun i (x : Complex.t) ->
            compare_float x.re e.(i).re;
            compare_float x.im e.(i).im)
          a
    | Npz.I e, Npz.I a when Array.length e = Array.length a ->
        Array.iteri (fun i x -> compare_int x e.(i)) a
    | _ -> differing := -1
    end;
  (!count, !differing, !largest)

let record_exact name expected actual =
  let count, differing, largest = measure_exact expected actual in
  let identical, total =
    match Hashtbl.find_opt tallies !suite with
    | Some tally -> tally
    | None ->
        let tally = (ref 0, ref 0) in
        Hashtbl.replace tallies !suite tally;
        tally
  in
  incr total;
  if differing = 0 then incr identical;
  match Lazy.force exact_log with
  | None -> ()
  | Some oc -> (
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\n%!" name count differing largest
          !suite
      with Sys_error _ -> ())

let parse_floor text =
  List.filter_map
    (fun line ->
      match String.split_on_char '\t' line with
      | suite :: identical :: _ -> Some (suite, int_of_string identical)
      | _ -> None)
    (String.split_on_char '\n' text)

let ratchet_failures ~floor ~require_all =
  List.filter_map
    (fun (suite, wanted) ->
      match Hashtbl.find_opt tallies suite with
      | Some (identical, _) when !identical < wanted ->
          Some
            (Printf.sprintf "%s: %d bit-identical, floor %d" suite !identical
               wanted)
      | Some _ -> None
      | None when require_all ->
          Some (Printf.sprintf "%s: did not run, floor %d" suite wanted)
      | None -> None)
    floor

let matched_setup () =
  Ojax.Pjrt.Discover.is_exact ()
  && String.starts_with ~prefix:"libscipy_openblas"
       (Filename.basename Ojax.Linalg.Discover.library_path)

let check_ratchet ~floor_text () =
  if not (matched_setup ()) then
    print_endline
      "exact ratchet: skipped, the floors hold only for the plugin built from \
       jax's XLA commit with scipy's OpenBLAS"
  else begin
    let require_all = Array.length Sys.argv <= 2 in
    match ratchet_failures ~floor:(parse_floor floor_text) ~require_all with
    | [] -> ()
    | failures -> failwith ("exact ratchet fell: " ^ String.concat "; " failures)
  end

let check ~name ~compare ~atol ~rtol ~(expected : Npz.t) ~(actual : Npz.t) =
  record_exact name expected actual;
  if not (shapes_equal expected.shape actual.shape) then
    failwith (name ^ ": shape mismatch");
  match (compare, expected.data, actual.data) with
  | "exact", Npz.I ea, Npz.I aa -> exact_int name aa ea
  | "allclose", Npz.F ea, Npz.F aa -> allclose_float name atol rtol aa ea
  | "allclose", Npz.C ea, Npz.C aa -> allclose_complex name atol rtol aa ea
  | "allclose", Npz.I ea, Npz.I aa -> exact_int name aa ea
  | _ -> failwith (name ^ ": data-kind/compare mismatch")
