module U = Yojson.Safe.Util
module E = Ojax.Numpy.Extras
module S = Ojax.Numpy.Setops
module J = Ojax.Numpy.Jnp_util
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let name = "size_ops"
let one f = function [ x ] -> f x | _ -> failwith "size_ops: arity"
let two f = function [ x; y ] -> f x y | _ -> failwith "size_ops: arity"

let three f = function
  | [ x; y; z ] -> f x y z
  | _ -> failwith "size_ops: arity"

let cases =
  [
    ("argwhere_size", one (E.argwhere ~size:4));
    ( "argwhere_size_fill",
      one (fun x -> E.argwhere ~size:5 ~fill_value:(J.py_int (-1)) x) );
    ("argwhere_scalar", one (E.argwhere ~size:2));
    ("argwhere_scalar_zero", one (E.argwhere ~size:2));
    ("unique_size", one (fun x -> List.hd (E.unique ~size:4 x)));
    ("unique_size_nan", one (fun x -> List.hd (E.unique ~size:6 x)));
    ( "unique_size_fill",
      one (fun x -> List.hd (E.unique ~size:8 ~fill_value:(-9.0) x)) );
    ( "unique_size_nan_noequal",
      one (fun x -> List.hd (E.unique ~size:7 ~equal_nan:false x)) );
    ("intersect1d_size", two (E.intersect1d ~size:3));
    ( "intersect1d_size_fill",
      two (fun x y -> E.intersect1d ~size:6 ~fill_value:(J.py_float (-1.0)) x y)
    );
    ("intersect1d_size_unique", two (E.intersect1d ~assume_unique:true ~size:3));
    ("intersect1d_size_nan", two (E.intersect1d ~size:4));
    ( "intersect1d_size_int",
      two (fun x y -> E.intersect1d ~size:5 ~fill_value:(J.py_int (-1)) x y) );
    ("setdiff1d_size", two (E.setdiff1d ~size:3));
    ("setdiff1d_size_fill", two (E.setdiff1d ~size:6 ~fill_value:(-1.0)));
    ( "setdiff1d_size_unique",
      two (E.setdiff1d ~assume_unique:true ~size:4 ~fill_value:9.0) );
    ("setdiff1d_size_int", two (E.setdiff1d ~size:5));
    ("setxor1d_size", two (E.setxor1d ~size:4));
    ( "setxor1d_size_fill",
      two (fun x y -> E.setxor1d ~size:9 ~fill_value:(J.py_float (-1.0)) x y) );
    ("setxor1d_size_unique", two (E.setxor1d ~assume_unique:true ~size:6));
    ( "setxor1d_size_int",
      two (fun x y -> E.setxor1d ~size:7 ~fill_value:(J.py_int (-1)) x y) );
    ("isin", two (S.isin ?assume_unique:None ?invert:None ?method_:None));
    ("isin_invert", two (S.isin ~invert:true ?assume_unique:None ?method_:None));
    ("place_tile", three (E.place ~inplace:false));
    ("place_slice", three (E.place ~inplace:false));
    ("fill_diagonal_tile", two (E.fill_diagonal ~inplace:false));
    ("fill_diagonal_slice", two (E.fill_diagonal ~inplace:false));
  ]

let unhex s =
  String.init
    (String.length s / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub s (2 * i) 2)))

let input arg =
  let dt =
    match D.of_numpy_name (U.to_string (U.member "dtype" arg)) with
    | Some d -> d
    | None -> failwith "size_ops: dtype"
  in
  let shape =
    Array.of_list (List.map U.to_int (U.to_list (U.member "shape" arg)))
  in
  let b = Bytes.of_string (unhex (U.to_string (U.member "hex" arg))) in
  let n = Array.fold_left ( * ) 1 shape in
  let xs =
    Array.init n (fun i ->
        match dt with
        | D.F32 -> Nd.float_of_f32_bits (Bytes.get_int32_le b (4 * i))
        | D.I32 -> Int32.to_float (Bytes.get_int32_le b (4 * i))
        | D.Bool -> float_of_int (Bytes.get_uint8 b i)
        | _ -> failwith "size_ops: input dtype")
  in
  Ojax.Types.Concrete (Nd.of_floats dt shape xs)

let host = function
  | Ojax.Types.Concrete nd -> nd
  | Ojax.Types.Device b -> Ojax.Pjrt.Buffer.to_host b
  | Ojax.Types.Tracer _ -> failwith "size_ops: tracer result"

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let out_text v =
  let nd = host v in
  let raw = Npz.raw_of_nd nd in
  raw.Npz.raw_dtype ^ "["
  ^ String.concat "," (Array.to_list (Array.map string_of_int (Nd.shape nd)))
  ^ "]" ^ hex raw.Npz.bytes

let jax_out o =
  U.to_string (U.member "dtype" o)
  ^ "["
  ^ String.concat ","
      (List.map
         (fun s -> string_of_int (U.to_int s))
         (U.to_list (U.member "shape" o)))
  ^ "]"
  ^ U.to_string (U.member "hex" o)

let error_detail e =
  match Ojax.Py_exn.class_name e with
  | Some c -> c ^ ": " ^ Error_types.message_of e
  | None -> Error_types.ojax_class e ^ ": " ^ Error_types.message_of e

let str_opt row field =
  match U.member field row with `String s -> Some s | _ -> None

let expected row mode =
  match (U.member mode row, str_opt row (mode ^ "_error")) with
  | `Assoc _, _ -> jax_out (U.member mode row)
  | _, Some e -> e
  | _ -> "missing"

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let log field jax ojax =
    let same = String.equal jax ojax in
    Fuzz.log_row
      ~detail:(String.escaped jax ^ "\t" ^ String.escaped ojax)
      ~suite
      (id ^ ":" ^ field)
      1
      (if same then 0 else 1)
      0.0
      (if same then Compare.raw_identical else "differ")
  in
  let f = List.assoc id cases in
  let args = List.map input (U.to_list (U.member "args" row)) in
  let guard g = try g () with e -> error_detail e in
  log "eager" (expected row "eager") (guard (fun () -> out_text (f args)));
  log "jit" (expected row "jit")
    (guard (fun () ->
         out_text (List.hd (Ojax.Api.jit_flat (fun a -> [ f a ]) args))));
  let avals = List.map Ojax.Core.get_aval args in
  let closed () = Ojax.Jaxpr.make_jaxpr avals (fun a -> [ f a ]) in
  let text_error = str_opt row "text_error" in
  let want field =
    match (str_opt row field, text_error) with
    | Some t, _ -> t
    | None, Some e -> e
    | None, None -> "missing"
  in
  log "jaxpr" (want "jaxpr")
    (guard (fun () ->
         Ojax.Pretty_printer.closed_jaxpr_to_jax_string (closed ())));
  log "hlo" (want "hlo")
    (guard (fun () -> Ojax.Stablehlo.Emit.emit_jax_text ~name:"f" (closed ())))

let read_rows path =
  In_channel.with_open_text path (fun ic ->
      let rec loop acc =
        match In_channel.input_line ic with
        | Some line -> loop (Yojson.Safe.from_string line :: acc)
        | None -> List.rev acc
      in
      loop [])

let suites ~goldens_root =
  let path =
    Filename.concat (Filename.concat goldens_root name) "cases.jsonl"
  in
  if not (Sys.file_exists path) then []
  else
    let rows = read_rows path in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              if mode = "x64_on" = U.to_bool (U.member "x64" row) then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
