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

let raw_format = function
  | "float16" -> Some (2, 0x7C00L, 0x3FFL)
  | "bfloat16" -> Some (2, 0x7F80L, 0x7FL)
  | "float32" | "complex64" -> Some (4, 0x7F800000L, 0x7FFFFFL)
  | "float64" | "complex128" -> Some (8, 0x7FF0000000000000L, 0xFFFFFFFFFFFFFL)
  | _ -> None

let raw_word s off = function
  | 2 -> Int64.of_int (String.get_uint16_le s off)
  | 4 -> Int64.logand (Int64.of_int32 (String.get_int32_le s off)) 0xFFFFFFFFL
  | _ -> String.get_int64_le s off

let raw_kinds dtype a b =
  let nan_sign = ref 0 and nan_payload = ref 0 in
  let neg_zero = ref 0 and value = ref 0 and first = ref "" in
  begin match raw_format dtype with
  | None -> String.iteri (fun i x -> if x <> b.[i] then incr value) a
  | Some (width, exponent, mantissa) ->
      let sign = Int64.shift_left 1L ((8 * width) - 1) in
      let is_nan w =
        Int64.logand w exponent = exponent && Int64.logand w mantissa <> 0L
      in
      for k = 0 to (String.length a / width) - 1 do
        let x = raw_word a (k * width) width in
        let y = raw_word b (k * width) width in
        if x <> y && !first = "" then
          first := Printf.sprintf "%d:%Lx!=%Lx" k x y;
        if x <> y then
          if is_nan x && is_nan y then
            incr (if Int64.logxor x y = sign then nan_sign else nan_payload)
          else if Int64.logor x y = sign then incr neg_zero
          else incr value
      done
  end;
  (!nan_sign, !nan_payload, !neg_zero, !value, !first)

let raw_identical = "same"

let raw_status (expected : Npz.t) (actual : Npz.t) =
  let e = expected.raw and a = actual.raw in
  if e.raw_dtype <> a.raw_dtype then
    Printf.sprintf "dtype:%s!=%s" a.raw_dtype e.raw_dtype
  else if not (shapes_equal expected.shape actual.shape) then
    let text shape =
      String.concat "x" (Array.to_list (Array.map string_of_int shape))
    in
    Printf.sprintf "shape:%s!=%s" (text actual.shape) (text expected.shape)
  else if String.length e.bytes <> String.length a.bytes then "length"
  else if String.equal e.bytes a.bytes then raw_identical
  else
    let nan_sign, nan_payload, neg_zero, value, first =
      raw_kinds e.raw_dtype a.bytes e.bytes
    in
    Printf.sprintf "nan_sign=%d,nan_payload=%d,neg_zero=%d,value=%d,first=%s"
      nan_sign nan_payload neg_zero value first

let raw_prefix = "raw:"
let ratcheted_prefixes = [ "fuzz:"; "fuzz_s2:"; "coverage:" ]

let is_ratcheted suite =
  List.exists
    (fun prefix -> String.starts_with ~prefix suite)
    ratcheted_prefixes

let row_results : (string * string, bool * bool) Hashtbl.t = Hashtbl.create 4096

let note_row suite name ~bit ~raw =
  if is_ratcheted suite then Hashtbl.replace row_results (suite, name) (bit, raw)

let tally_of suite =
  match Hashtbl.find_opt tallies suite with
  | Some tally -> tally
  | None ->
      let tally = (ref 0, ref 0) in
      Hashtbl.replace tallies suite tally;
      tally

let bump suite same =
  let identical, total = tally_of suite in
  incr total;
  if same then incr identical

let record_exact ?suite:named name expected actual =
  let suite = Option.value named ~default:!suite in
  let count, differing, largest = measure_exact expected actual in
  let status = raw_status expected actual in
  bump suite (differing = 0);
  bump (raw_prefix ^ suite) (status = raw_identical);
  note_row suite name ~bit:(differing = 0) ~raw:(status = raw_identical);
  match Lazy.force exact_log with
  | None -> ()
  | Some oc -> (
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\t%s\n%!" name count differing
          largest suite status
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

let baseline_path suite =
  Filename.concat "identical_rows"
    (String.map (fun c -> if c = ':' then '_' else c) suite ^ ".tsv")

let read_lines path =
  In_channel.with_open_text path In_channel.input_all
  |> String.split_on_char '\n'
  |> List.filter (fun line -> line <> "")

let row_failures ~require_all ~baseline =
  Hashtbl.fold
    (fun suite _ acc ->
      if not (is_ratcheted suite) then acc
      else
        let listed = baseline suite in
        List.filter_map
          (fun (row, raw_listed) ->
            match Hashtbl.find_opt row_results (suite, row) with
            | Some (false, _) ->
                Some (Printf.sprintf "%s %s: no longer identical" suite row)
            | Some (true, false) when raw_listed ->
                Some
                  (Printf.sprintf "%s %s: no longer identical in raw bytes"
                     suite row)
            | Some _ -> None
            | None when require_all ->
                Some (Printf.sprintf "%s %s: did not run" suite row)
            | None -> None)
          listed
        @ acc)
    tallies []

let read_baseline suite =
  let path = baseline_path suite in
  if not (Sys.file_exists path) then
    failwith ("row ratchet: no baseline " ^ path);
  List.map
    (fun line ->
      match String.split_on_char '\t' line with
      | [ row; raw ] -> (row, raw = "1")
      | _ -> failwith ("row ratchet: bad line in " ^ path ^ ": " ^ line))
    (read_lines path)

let row_ratchet_text ~require_all =
  match row_failures ~require_all ~baseline:read_baseline with
  | [] -> []
  | failures ->
      let sorted = List.sort compare failures in
      let shown = List.filteri (fun i _ -> i < 40) sorted in
      [
        Printf.sprintf "row ratchet: %d rows fell: %s" (List.length sorted)
          (String.concat "; " shown);
      ]

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
    let floors =
      match ratchet_failures ~floor:(parse_floor floor_text) ~require_all with
      | [] -> []
      | failures -> [ "exact ratchet fell: " ^ String.concat "; " failures ]
    in
    match floors @ row_ratchet_text ~require_all with
    | [] -> ()
    | failures -> failwith (String.concat "; " failures)
  end

let check_tolerance ~name ~compare ~atol ~rtol ~(expected : Npz.t)
    ~(actual : Npz.t) =
  if not (shapes_equal expected.shape actual.shape) then
    failwith (name ^ ": shape mismatch");
  match (compare, expected.data, actual.data) with
  | "exact", Npz.I ea, Npz.I aa -> exact_int name aa ea
  | "allclose", Npz.F ea, Npz.F aa -> allclose_float name atol rtol aa ea
  | "allclose", Npz.C ea, Npz.C aa -> allclose_complex name atol rtol aa ea
  | "allclose", Npz.I ea, Npz.I aa -> exact_int name aa ea
  | _ -> failwith (name ^ ": data-kind/compare mismatch")

let check ~name ~compare ~atol ~rtol ~(expected : Npz.t) ~(actual : Npz.t) =
  record_exact name expected actual;
  check_tolerance ~name ~compare ~atol ~rtol ~expected ~actual

let check_with_reference ~name ~compare ~atol ~rtol ~eager_suite
    ~(expected : Npz.t) ~(actual : Npz.t) ~(reference : Npz.t)
    ~(reference_actual : Npz.t) =
  record_exact name reference reference_actual;
  record_exact ~suite:eager_suite name expected actual;
  check_tolerance ~name ~compare ~atol ~rtol ~expected:reference
    ~actual:reference_actual
