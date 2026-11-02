module U = Yojson.Safe.Util
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

type member = { descr : string; shape : int array; start : int; length : int }

let members b =
  let eocd = Npz.find_eocd b in
  let entries = Npz.u16 b (eocd + 10) in
  let cd_off = Npz.u32 b (eocd + 16) in
  let table = Hashtbl.create 64 in
  let pos = ref cd_off in
  for _ = 1 to entries do
    let p = !pos in
    let name_len = Npz.u16 b (p + 28) in
    let extra_len = Npz.u16 b (p + 30) in
    let comment_len = Npz.u16 b (p + 32) in
    let local = Npz.u32 b (p + 42) in
    let name = Bytes.sub_string b (p + 46) name_len in
    let l_name = Npz.u16 b (local + 26) in
    let l_extra = Npz.u16 b (local + 28) in
    let off = local + 30 + l_name + l_extra in
    let major = Bytes.get_uint8 b (off + 6) in
    let hlen, htext =
      if major = 1 then (Npz.u16 b (off + 8), off + 10)
      else (Npz.u32 b (off + 8), off + 12)
    in
    let header = Bytes.sub_string b htext hlen in
    let descr = Npz.parse_descr header in
    let shape = Npz.parse_shape header in
    let width =
      match fst (Npz.elem_kind descr) with
      | "bool" | "int8" | "uint8" -> 1
      | "int16" | "uint16" | "float16" | "bfloat16" -> 2
      | "int32" | "uint32" | "float32" -> 4
      | "complex128" -> 16
      | _ -> 8
    in
    let n = Array.fold_left ( * ) 1 shape in
    Hashtbl.replace table
      (Filename.remove_extension name)
      { descr; shape; start = htext + hlen; length = n * width };
    pos := p + 46 + name_len + extra_len + comment_len
  done;
  table

let cached = ref None

let bundle path =
  match !cached with
  | Some (p, b, t) when p = path -> (b, t)
  | _ ->
      let b = Npz.read_file path in
      let t = members b in
      cached := Some (path, b, t);
      (b, t)

let raw_of_nd nd =
  let shape = Nd.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  let rank = Array.length shape in
  let index flat =
    let idx = Array.make rank 0 in
    let f = ref flat in
    for d = rank - 1 downto 0 do
      idx.(d) <- !f mod shape.(d);
      f := !f / shape.(d)
    done;
    idx
  in
  let dtype = Nd.dtype nd in
  let width =
    match dtype with
    | D.Bool | D.I8 | D.Uint8 -> 1
    | D.I16 | D.Uint16 | D.F16 | D.Bf16 -> 2
    | D.I32 | D.Uint32 | D.F32 -> 4
    | D.Complex128 -> 16
    | _ -> 8
  in
  let out = Bytes.create (n * width) in
  for i = 0 to n - 1 do
    let o = i * width in
    match dtype with
    | D.F64 ->
        Bytes.set_int64_le out o (Int64.bits_of_float (Nd.get_f nd (index i)))
    | D.F32 ->
        Bytes.set_int32_le out o (Int32.bits_of_float (Nd.get_f nd (index i)))
    | D.F16 ->
        Bytes.set_uint16_le out o (Nd.bits_of_half (Nd.get_f nd (index i)))
    | D.Bf16 ->
        Bytes.set_uint16_le out o (Nd.bits_of_bfloat (Nd.get_f nd (index i)))
    | D.Complex64 ->
        let z = Nd.get_c nd (index i) in
        Bytes.set_int32_le out o (Int32.bits_of_float z.re);
        Bytes.set_int32_le out (o + 4) (Int32.bits_of_float z.im)
    | D.Complex128 ->
        let z = Nd.get_c nd (index i) in
        Bytes.set_int64_le out o (Int64.bits_of_float z.re);
        Bytes.set_int64_le out (o + 8) (Int64.bits_of_float z.im)
    | _ -> (
        let v = Nd.get_i64 nd (index i) in
        match width with
        | 1 -> Bytes.set_int8 out o (Int64.to_int v land 0xFF)
        | 2 -> Bytes.set_int16_le out o (Int64.to_int v land 0xFFFF)
        | 4 -> Bytes.set_int32_le out o (Int64.to_int32 v)
        | _ -> Bytes.set_int64_le out o v)
  done;
  out

let classify ~dtype ~width expected i actual j =
  let value b o =
    match (dtype, width) with
    | ("float32" | "complex64"), _ ->
        Some (Int32.float_of_bits (Bytes.get_int32_le b o))
    | ("float64" | "complex128"), _ ->
        Some (Int64.float_of_bits (Bytes.get_int64_le b o))
    | "float16", _ -> Some (Nd.half_of_bits (Bytes.get_uint16_le b o))
    | "bfloat16", _ -> Some (Nd.bfloat_of_bits (Bytes.get_uint16_le b o))
    | _ -> None
  in
  match (value expected i, value actual j) with
  | Some e, Some a ->
      if Float.is_nan e && Float.is_nan a then ("nan_bits", 0.0)
      else if Float.is_nan e || Float.is_nan a then
        ("nan_vs_num", Float.infinity)
      else if e = a then ("signed_zero", 0.0)
      else if Float.abs e = Float.infinity || Float.abs a = Float.infinity then
        ("inf_vs_num", Float.infinity)
      else if e = 0.0 || a = 0.0 then ("zero_vs_num", Float.infinity)
      else
        let relative = Float.abs (a -. e) /. Float.abs e in
        ((if relative < 1e-3 then "ulps" else "value"), relative)
  | _ -> ("int", 0.0)

let log_row ?detail ~suite name count differing largest classes =
  Compare.bump suite (differing = 0);
  Compare.bump (Compare.raw_prefix ^ suite) (classes = Compare.raw_identical);
  Compare.note_row suite name ~bit:(differing = 0)
    ~raw:(classes = Compare.raw_identical);
  match Lazy.force Compare.exact_log with
  | None -> ()
  | Some oc -> (
      let tail = match detail with None -> "" | Some d -> "\t" ^ d in
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\t%s%s\n%!" name count differing
          largest suite classes tail
      with Sys_error _ -> ())

let dump_channel =
  lazy
    (Option.map
       (open_out_gen [ Open_append; Open_creat ] 0o644)
       (Sys.getenv_opt "OJAX_FUZZ_DUMP"))

let dump name i b e actual a step =
  match Lazy.force dump_channel with
  | None -> ()
  | Some oc ->
      let hex bytes off =
        String.concat ""
          (List.init step (fun k ->
               Printf.sprintf "%02x"
                 (Bytes.get_uint8 bytes (off + step - 1 - k))))
      in
      Printf.fprintf oc "%s %d jax=%s ojax=%s\n%!" name i (hex b e)
        (hex actual a)

let compare_output ~suite ~name ~string_of_dtype b (m : member) nd =
  let expected_dtype = fst (Npz.elem_kind m.descr) in
  let got_dtype = string_of_dtype (Nd.dtype nd) in
  if got_dtype <> expected_dtype then
    log_row ~suite name 0 (-3) 0.0
      (Printf.sprintf "dtype:%s_vs_%s" got_dtype expected_dtype)
  else if not (Compare.shapes_equal (Nd.shape nd) m.shape) then
    log_row ~suite name 0 (-4) 0.0 "shape"
  else begin
    let actual = raw_of_nd nd in
    let elements = Array.fold_left ( * ) 1 m.shape in
    let width = if elements = 0 then 1 else m.length / elements in
    let scalars =
      if expected_dtype = "complex64" || expected_dtype = "complex128" then 2
      else 1
    in
    let step = width / scalars in
    let differing = ref 0 and largest = ref 0.0 in
    let classes = Hashtbl.create 4 in
    if Bytes.length actual <> m.length then differing := -6
    else
      for i = 0 to (elements * scalars) - 1 do
        let e = m.start + (i * step) and a = i * step in
        if Bytes.sub b e step <> Bytes.sub actual a step then begin
          incr differing;
          let cls, rel =
            classify ~dtype:expected_dtype ~width:step b e actual a
          in
          Hashtbl.replace classes cls ();
          if rel > !largest then largest := rel;
          dump name i b e actual a step
        end
      done;
    let names =
      List.sort compare (Hashtbl.fold (fun k () acc -> k :: acc) classes [])
    in
    log_row ~suite name (elements * scalars) !differing !largest
      (if !differing = 0 then Compare.raw_identical else String.concat "," names)
  end

let run_guarded f = try Ok (f ()) with e -> Error e

let one_line message =
  let text = Error_types.one_line message in
  if String.length text > 120 then String.sub text 0 120 else text

let check_mode ~suite ~case_id ~mode ~string_of_dtype ~concrete ~jax_error b
    table outputs run =
  let prefix = case_id ^ ":" ^ mode in
  let result = run_guarded run in
  match (outputs, result) with
  | None, Error e ->
      let row = prefix ^ ":both_raise" in
      let detail = Error_types.detail ~jax:jax_error e in
      if Error_types.tier ~jax:jax_error e <> None then
        log_row ~detail ~suite row 0 0 0.0 Error_types.both_raise
      else
        log_row ~detail ~suite row 0 Error_types.mismatch_code 0.0
          (Error_types.mismatch ~jax:jax_error e)
  | None, Ok _ ->
      log_row ~suite (prefix ^ ":jax_raises") 0 (-2) 0.0 "jax_raises"
  | Some _, Error e ->
      log_row ~suite (prefix ^ ":ojax_raises") 0 (-1) 0.0
        ("ojax_raises:" ^ one_line (Printexc.to_string e))
  | Some names, Ok values ->
      if List.length names <> List.length values then
        log_row ~suite (prefix ^ ":arity") 0 (-5) 0.0 "arity"
      else
        List.iter2
          (fun name value ->
            let m = Hashtbl.find table (case_id ^ "/" ^ mode ^ "_" ^ name) in
            let row = prefix ^ ":" ^ name in
            match run_guarded (fun () -> concrete value) with
            | Ok nd -> compare_output ~suite ~name:row ~string_of_dtype b m nd
            | Error _ -> log_row ~suite row 0 (-1) 0.0 "ojax_raises:concrete")
          names values

let output_names case key =
  match U.member key case with
  | `Null -> None
  | l ->
      Some (List.map (fun o -> U.to_string (U.member "name" o)) (U.to_list l))

let check_case ~suite ~set_dir ~x64 ~nd_of_npz ~string_of_dtype ~concrete case
    () =
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let case_id = U.to_string (U.member "case_id" case) in
  let op = U.to_string (U.member "op" case) in
  let params = U.member "params" case in
  let names =
    List.map
      (fun a -> U.to_string (U.member "name" a))
      (U.to_list (U.member "args" case))
  in
  let b, table =
    bundle (Filename.concat (Filename.concat set_dir "ops") (op ^ ".npz"))
  in
  let input name =
    let m = Hashtbl.find table (case_id ^ "/in_" ^ name) in
    nd_of_npz (Npz.decode b m.start m.descr m.shape)
  in
  let values = List.map (fun n -> Ojax.Types.Concrete (input n)) names in
  let lookup args name =
    let rec go ns vs =
      match (ns, vs) with
      | n :: _, v :: _ when n = name -> v
      | _ :: nt, _ :: vt -> go nt vt
      | _ -> failwith ("fuzz: missing input " ^ name)
    in
    go names args
  in
  let direct () = Parity_cases.run op params (lookup values) in
  let jax_error key =
    match U.member key case with `String name -> name | _ -> "unknown"
  in
  check_mode ~suite ~case_id ~mode:"eager" ~string_of_dtype ~concrete
    ~jax_error:(jax_error "eager_error") b table
    (output_names case "eager")
    direct;
  let jit () =
    if Parity_cases.needs_concrete op then direct ()
    else
      Ojax.Api.jit_flat
        (fun args -> Parity_cases.run op params (lookup args))
        values
  in
  if Parity_cases.needs_concrete op && output_names case "jit" = None then ()
  else
    check_mode ~suite ~case_id ~mode:"jit" ~string_of_dtype ~concrete
      ~jax_error:(jax_error "jit_error") b table (output_names case "jit") jit

let primary = "fuzz"
let second_seed = "fuzz_s2"

let available ~goldens_root ~name =
  Sys.file_exists
    (Filename.concat
       (Filename.concat (Filename.concat goldens_root name) "x64_off")
       "manifest.json")

let suites ~goldens_root ~nd_of_npz ~string_of_dtype ~concrete =
  List.concat_map
    (fun name ->
      List.filter_map
        (fun (set_name, x64) ->
          let set_dir =
            Filename.concat (Filename.concat goldens_root name) set_name
          in
          let path = Filename.concat set_dir "manifest.json" in
          if not (Sys.file_exists path) then None
          else
            let suite = name ^ ":" ^ set_name in
            let cases =
              Yojson.Safe.from_file path |> U.member "cases" |> U.to_list
            in
            Some
              ( suite,
                List.map
                  (fun case ->
                    Alcotest.test_case
                      (U.to_string (U.member "case_id" case))
                      `Quick
                      (check_case ~suite ~set_dir ~x64 ~nd_of_npz
                         ~string_of_dtype ~concrete case))
                  cases ))
        [ ("x64_off", false); ("x64_on", true) ])
    [ primary; second_seed ]
