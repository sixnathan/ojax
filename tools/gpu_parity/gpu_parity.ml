module U = Yojson.Safe.Util
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module B = Ojax.Pjrt.Buffer

let dtype_of_name = function
  | "float32" -> D.F32
  | "float64" -> D.F64
  | "int32" -> D.I32
  | "int64" -> D.I64
  | "bool" -> D.Bool
  | "uint32" -> D.Uint32
  | "complex64" -> D.Complex64
  | "complex128" -> D.Complex128
  | "int8" -> D.I8
  | "int16" -> D.I16
  | "uint8" -> D.Uint8
  | "uint16" -> D.Uint16
  | "uint64" -> D.Uint64
  | "float16" -> D.F16
  | "bfloat16" -> D.Bf16
  | other -> failwith ("gpu_parity: dtype " ^ other)

let name_of_dtype = function
  | D.F32 -> "float32"
  | D.F64 -> "float64"
  | D.I32 -> "int32"
  | D.I64 -> "int64"
  | D.Bool -> "bool"
  | D.Uint32 -> "uint32"
  | D.Complex64 -> "complex64"
  | D.Complex128 -> "complex128"
  | D.I8 -> "int8"
  | D.I16 -> "int16"
  | D.Uint8 -> "uint8"
  | D.Uint16 -> "uint16"
  | D.Uint64 -> "uint64"
  | D.F16 -> "float16"
  | D.Bf16 -> "bfloat16"
  | _ -> "other"

let bytes_of_hex hex =
  let n = String.length hex / 2 in
  let buf = Bigarray.(Array1.create int8_unsigned c_layout n) in
  for i = 0 to n - 1 do
    buf.{i} <- int_of_string ("0x" ^ String.sub hex (2 * i) 2)
  done;
  buf

let hex_of_bytes buf =
  let text = Buffer.create (2 * Bigarray.Array1.dim buf) in
  for i = 0 to Bigarray.Array1.dim buf - 1 do
    Buffer.add_string text (Printf.sprintf "%02x" buf.{i})
  done;
  Buffer.contents text

let input_of j =
  let dtype = dtype_of_name (U.to_string (U.member "dtype" j)) in
  let shape =
    Array.of_list (List.map U.to_int (U.to_list (U.member "shape" j)))
  in
  ( U.to_string (U.member "name" j),
    T.Concrete
      (B.host_of_bytes dtype shape
         (bytes_of_hex (U.to_string (U.member "hex" j)))) )

let host = function
  | T.Concrete nd -> nd
  | T.Device buffer -> B.to_host buffer
  | T.Tracer _ -> failwith "gpu_parity: tracer result"

let out_row i v =
  let nd = host v in
  `Assoc
    [
      ("name", `String (Printf.sprintf "out%d" i));
      ("dtype", `String (name_of_dtype (Nd.dtype nd)));
      ("shape", `List (List.map (fun d -> `Int d) (Array.to_list (Nd.shape nd))));
      ("hex", `String (hex_of_bytes (B.host_bytes nd)));
    ]

let guarded f =
  try `List (List.mapi out_row (f ())) with
  | Stack_overflow -> `String "Stack_overflow"
  | Out_of_memory -> `String "Out_of_memory"
  | e -> (
      match Ojax.Errors.message_of e with
      | Some m -> `String m
      | None -> `String (Printexc.to_string e))

let run_case oc case =
  let set = U.to_string (U.member "set" case) in
  let id = U.to_string (U.member "case" case) in
  let op = U.to_string (U.member "op" case) in
  let params = U.member "params" case in
  let inputs = List.map input_of (U.to_list (U.member "inputs" case)) in
  let names = List.map fst inputs and values = List.map snd inputs in
  let lookup args name =
    let rec go ns vs =
      match (ns, vs) with
      | n :: _, v :: _ when n = name -> v
      | _ :: nt, _ :: vt -> go nt vt
      | _ -> failwith ("gpu_parity: missing input " ^ name)
    in
    go names args
  in
  let direct () = Parity_cases.run op params (lookup values) in
  let jitted () =
    if Parity_cases.needs_concrete op then direct ()
    else
      Ojax.Api.jit_flat
        (fun args -> Parity_cases.run op params (lookup args))
        values
  in
  Ojax.Config.with_value Ojax.Config.enable_x64 (set = "x64_on") (fun () ->
      List.iter
        (fun mode ->
          let result =
            match mode with "jit" -> guarded jitted | _ -> guarded direct
          in
          let row =
            match result with
            | `List _ as outs -> [ ("outs", outs) ]
            | error -> [ ("error", error) ]
          in
          Yojson.Safe.to_channel oc
            (`Assoc
               ([
                  ("set", `String set);
                  ("case", `String id);
                  ("mode", `String mode);
                ]
               @ row));
          output_char oc '\n';
          flush oc)
        (U.to_list (U.member "modes" case) |> List.map U.to_string))

let () =
  Ojax.Lax.install ();
  Ojax.Random.Prng.install ();
  Ojax.Random.Threefry.install ();
  let cases = Sys.argv.(1) and out = Sys.argv.(2) in
  let shard = int_of_string Sys.argv.(3) in
  let shards = int_of_string Sys.argv.(4) in
  let oc = open_out_gen [ Open_append; Open_creat ] 0o644 out in
  let ic = open_in cases in
  let index = ref 0 in
  (try
     while true do
       let line = input_line ic in
       if !index mod shards = shard then
         run_case oc (Yojson.Safe.from_string line);
       incr index
     done
   with End_of_file -> ());
  close_in ic;
  close_out oc
