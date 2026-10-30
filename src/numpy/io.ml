module T = Types
module D = Dtype
module Nd = Ndarray

let host = function
  | T.Concrete nd -> nd
  | T.Device b -> Pjrt.Buffer.to_host b
  | T.Tracer _ -> invalid_arg "numpy io: a tracer cannot be written to a file"

let descr_of = function
  | D.Bool -> "|b1"
  | D.I8 -> "|i1"
  | D.I16 -> "<i2"
  | D.I32 -> "<i4"
  | D.I64 -> "<i8"
  | D.Uint8 -> "|u1"
  | D.Uint16 -> "<u2"
  | D.Uint32 -> "<u4"
  | D.Uint64 -> "<u8"
  | D.F32 -> "<f4"
  | D.F64 -> "<f8"
  | D.Complex64 -> "<c8"
  | D.Complex128 -> "<c16"
  | d -> invalid_arg ("numpy io: no file format for dtype " ^ D.short_name d)

let dtype_of_descr = function
  | "|b1" -> D.Bool
  | "|i1" -> D.I8
  | "<i2" -> D.I16
  | "<i4" -> D.I32
  | "<i8" -> D.I64
  | "|u1" -> D.Uint8
  | "<u2" -> D.Uint16
  | "<u4" -> D.Uint32
  | "<u8" -> D.Uint64
  | "<f4" -> D.F32
  | "<f8" -> D.F64
  | "<c8" -> D.Complex64
  | "<c16" -> D.Complex128
  | d -> invalid_arg ("numpy io: unsupported dtype descriptor " ^ d)

let width = function
  | D.Bool | D.I8 | D.Uint8 -> 1
  | D.I16 | D.Uint16 -> 2
  | D.I32 | D.Uint32 | D.F32 -> 4
  | D.I64 | D.Uint64 | D.F64 | D.Complex64 -> 8
  | D.Complex128 -> 16
  | d -> invalid_arg ("numpy io: no element width for " ^ D.short_name d)

let index_of_flat sh f =
  let idx = Array.make (Array.length sh) 0 in
  let r = ref f in
  for d = Array.length sh - 1 downto 0 do
    idx.(d) <- !r mod sh.(d);
    r := !r / sh.(d)
  done;
  idx

let encode nd =
  let dt = Nd.dtype nd and sh = Nd.shape nd in
  let count = Array.fold_left ( * ) 1 sh in
  let w = width dt in
  let out = Bytes.create (count * w) in
  for f = 0 to count - 1 do
    let idx = index_of_flat sh f in
    let at = f * w in
    match dt with
    | D.Bool | D.I8 | D.Uint8 ->
        Bytes.set_uint8 out at (int_of_float (Nd.get_f nd idx) land 255)
    | D.I16 | D.Uint16 ->
        Bytes.set_uint16_le out at (int_of_float (Nd.get_f nd idx) land 65535)
    | D.I32 | D.Uint32 ->
        Bytes.set_int32_le out at
          (Int64.to_int32 (Int64.of_float (Nd.get_f nd idx)))
    | D.I64 | D.Uint64 -> Bytes.set_int64_le out at (Nd.get_i64 nd idx)
    | D.F32 -> Bytes.set_int32_le out at (Int32.bits_of_float (Nd.get_f nd idx))
    | D.F64 -> Bytes.set_int64_le out at (Int64.bits_of_float (Nd.get_f nd idx))
    | D.Complex64 ->
        let z = Nd.get_c nd idx in
        Bytes.set_int32_le out at (Int32.bits_of_float z.Complex.re);
        Bytes.set_int32_le out (at + 4) (Int32.bits_of_float z.Complex.im)
    | D.Complex128 ->
        let z = Nd.get_c nd idx in
        Bytes.set_int64_le out at (Int64.bits_of_float z.Complex.re);
        Bytes.set_int64_le out (at + 8) (Int64.bits_of_float z.Complex.im)
    | _ -> assert false
  done;
  Bytes.unsafe_to_string out

let decode dt sh data offset =
  let count = Array.fold_left ( * ) 1 sh in
  let w = width dt in
  if String.length data - offset < count * w then
    invalid_arg "numpy io: not enough data for the requested shape";
  let at f = offset + (f * w) in
  match dt with
  | D.Complex64 ->
      Nd.of_complex dt sh
        (Array.init count (fun f ->
             {
               Complex.re =
                 Int32.float_of_bits (String.get_int32_le data (at f));
               im = Int32.float_of_bits (String.get_int32_le data (at f + 4));
             }))
  | D.Complex128 ->
      Nd.of_complex dt sh
        (Array.init count (fun f ->
             {
               Complex.re =
                 Int64.float_of_bits (String.get_int64_le data (at f));
               im = Int64.float_of_bits (String.get_int64_le data (at f + 8));
             }))
  | D.F32 ->
      Nd.of_floats dt sh
        (Array.init count (fun f ->
             Int32.float_of_bits (String.get_int32_le data (at f))))
  | D.F64 ->
      Nd.of_floats dt sh
        (Array.init count (fun f ->
             Int64.float_of_bits (String.get_int64_le data (at f))))
  | D.I64 | D.Uint64 ->
      Nd.of_int64s dt sh
        (Array.init count (fun f -> String.get_int64_le data (at f)))
  | _ ->
      let read f =
        match dt with
        | D.Bool | D.Uint8 -> String.get_uint8 data (at f)
        | D.I8 -> String.get_int8 data (at f)
        | D.Uint16 -> String.get_uint16_le data (at f)
        | D.I16 -> String.get_int16_le data (at f)
        | D.I32 -> Int32.to_int (String.get_int32_le data (at f))
        | D.Uint32 ->
            Int32.to_int (String.get_int32_le data (at f)) land 0xFFFFFFFF
        | _ -> assert false
      in
      Nd.of_floats dt sh (Array.init count (fun f -> float_of_int (read f)))

let npy_bytes v =
  let nd = host v in
  let sh = Nd.shape nd in
  let shape_text =
    match sh with
    | [||] -> "()"
    | [| n |] -> Printf.sprintf "(%d,)" n
    | _ ->
        "("
        ^ String.concat ", " (Array.to_list (Array.map string_of_int sh))
        ^ ")"
  in
  let dict =
    Printf.sprintf "{'descr': '%s', 'fortran_order': False, 'shape': %s, }"
      (descr_of (Nd.dtype nd))
      shape_text
  in
  let prefix = 10 in
  let unpadded = prefix + String.length dict + 1 in
  let pad = (64 - (unpadded mod 64)) mod 64 in
  let header = dict ^ String.make pad ' ' ^ "\n" in
  let b = Buffer.create (prefix + String.length header) in
  Buffer.add_string b "\147NUMPY\001\000";
  Buffer.add_uint16_le b (String.length header);
  Buffer.add_string b header;
  Buffer.add_string b (encode nd);
  Buffer.contents b

let find_after text key =
  let klen = String.length key in
  let rec go i =
    if i + klen > String.length text then
      invalid_arg ("numpy io: malformed header, missing " ^ key)
    else if String.sub text i klen = key then i + klen
    else go (i + 1)
  in
  go 0

let parse_npy data =
  if String.length data < 10 || String.sub data 0 6 <> "\147NUMPY" then
    invalid_arg "numpy io: not an npy file";
  let major = Char.code data.[6] in
  let hlen, start =
    if major = 1 then (String.get_uint16_le data 8, 10)
    else (Int32.to_int (String.get_int32_le data 8), 12)
  in
  let header = String.sub data start hlen in
  let descr =
    let i = find_after header "'descr': '" in
    String.sub header i (String.index_from header i '\'' - i)
  in
  let fortran =
    let i = find_after header "'fortran_order': " in
    String.sub header i 4 = "True"
  in
  let sh =
    let i = find_after header "'shape': (" in
    let j = String.index_from header i ')' in
    String.sub header i (j - i)
    |> String.split_on_char ','
    |> List.filter_map (fun p ->
        match String.trim p with "" -> None | t -> Some (int_of_string t))
    |> Array.of_list
  in
  let dt = dtype_of_descr descr in
  if not fortran then decode dt sh data (start + hlen)
  else begin
    let reversed = Array.of_list (List.rev (Array.to_list sh)) in
    let raw = decode dt reversed data (start + hlen) in
    let axes =
      Array.init (Array.length sh) (fun i -> Array.length sh - 1 - i)
    in
    match Lax_numpy.transpose ~axes (T.Concrete raw) with
    | T.Concrete nd -> nd
    | v -> host v
  end

let write_file path contents =
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc contents)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let with_npy path =
  if Filename.check_suffix path ".npy" then path else path ^ ".npy"

let with_npz path =
  if Filename.check_suffix path ".npz" then path else path ^ ".npz"

let save path v = write_file (with_npy path) (npy_bytes v)
let load path = T.Concrete (parse_npy (read_file path))

let crc_table =
  lazy
    (Array.init 256 (fun i ->
         let c = ref (Int32.of_int i) in
         for _ = 0 to 7 do
           c :=
             if Int32.logand !c 1l <> 0l then
               Int32.logxor 0xEDB88320l (Int32.shift_right_logical !c 1)
             else Int32.shift_right_logical !c 1
         done;
         !c))

let crc32 s =
  let table = Lazy.force crc_table in
  let c = ref 0xFFFFFFFFl in
  String.iter
    (fun ch ->
      let i =
        Int32.to_int
          (Int32.logand (Int32.logxor !c (Int32.of_int (Char.code ch))) 0xFFl)
      in
      c := Int32.logxor table.(i) (Int32.shift_right_logical !c 8))
    s;
  Int32.logxor !c 0xFFFFFFFFl

let savez path arrays =
  let out = Buffer.create 1024 and central = Buffer.create 256 in
  let entry (name, v) =
    let member = name ^ ".npy" in
    let body = npy_bytes v in
    let crc = crc32 body in
    let size = Int32.of_int (String.length body) in
    let offset = Int32.of_int (Buffer.length out) in
    let common b =
      Buffer.add_uint16_le b 20;
      Buffer.add_uint16_le b 0;
      Buffer.add_uint16_le b 0;
      Buffer.add_uint16_le b 0;
      Buffer.add_uint16_le b 33;
      Buffer.add_int32_le b crc;
      Buffer.add_int32_le b size;
      Buffer.add_int32_le b size;
      Buffer.add_uint16_le b (String.length member);
      Buffer.add_uint16_le b 0
    in
    Buffer.add_int32_le out 0x04034b50l;
    common out;
    Buffer.add_string out member;
    Buffer.add_string out body;
    Buffer.add_int32_le central 0x02014b50l;
    Buffer.add_uint16_le central 20;
    common central;
    Buffer.add_uint16_le central 0;
    Buffer.add_uint16_le central 0;
    Buffer.add_uint16_le central 0;
    Buffer.add_int32_le central 0l;
    Buffer.add_int32_le central offset;
    Buffer.add_string central member
  in
  List.iter entry arrays;
  let start = Buffer.length out in
  Buffer.add_buffer out central;
  Buffer.add_int32_le out 0x06054b50l;
  Buffer.add_uint16_le out 0;
  Buffer.add_uint16_le out 0;
  Buffer.add_uint16_le out (List.length arrays);
  Buffer.add_uint16_le out (List.length arrays);
  Buffer.add_int32_le out (Int32.of_int (Buffer.length central));
  Buffer.add_int32_le out (Int32.of_int start);
  Buffer.add_uint16_le out 0;
  write_file (with_npz path) (Buffer.contents out)

let load_npz path =
  let data = read_file path in
  let n = String.length data in
  let rec find_end i =
    if i < 0 then invalid_arg "numpy io: not a zip archive"
    else if String.get_int32_le data i = 0x06054b50l then i
    else find_end (i - 1)
  in
  let eocd = find_end (n - 22) in
  let count = String.get_uint16_le data (eocd + 10) in
  let rec read k pos acc =
    if k = count then List.rev acc
    else begin
      if String.get_int32_le data pos <> 0x02014b50l then
        invalid_arg "numpy io: malformed zip central directory";
      let methd = String.get_uint16_le data (pos + 10) in
      if methd <> 0 then
        invalid_arg
          "numpy io: compressed npz members are not supported, use savez \
           rather than savez_compressed";
      let size = Int32.to_int (String.get_int32_le data (pos + 24)) in
      let name_len = String.get_uint16_le data (pos + 28) in
      let extra_len = String.get_uint16_le data (pos + 30) in
      let comment_len = String.get_uint16_le data (pos + 32) in
      let local = Int32.to_int (String.get_int32_le data (pos + 42)) in
      let member = String.sub data (pos + 46) name_len in
      let local_name = String.get_uint16_le data (local + 26) in
      let local_extra = String.get_uint16_le data (local + 28) in
      let body = String.sub data (local + 30 + local_name + local_extra) size in
      let name =
        if Filename.check_suffix member ".npy" then
          Filename.chop_suffix member ".npy"
        else member
      in
      read (k + 1)
        (pos + 46 + name_len + extra_len + comment_len)
        ((name, T.Concrete (parse_npy body)) :: acc)
    end
  in
  read 0 (Int32.to_int (String.get_int32_le data (eocd + 16))) []

let frombuffer ?(dtype = D.F64) ?(count = -1) ?(offset = 0) data =
  let w = width dtype in
  let available = String.length data - offset in
  let n =
    if count >= 0 then count
    else if available mod w <> 0 then
      invalid_arg "buffer size must be a multiple of element size"
    else available / w
  in
  T.Concrete (decode dtype [| n |] data offset)

let fromstring ?(dtype = D.F64) ?(sep = " ") text =
  let parts =
    if String.trim sep = "" then
      String.split_on_char ' '
        (String.map (function '\n' | '\t' | '\r' -> ' ' | c -> c) text)
    else String.split_on_char sep.[0] text
  in
  let values =
    List.filter_map
      (fun p ->
        match String.trim p with "" -> None | t -> Some (float_of_string t))
      parts
  in
  let arr = Array.of_list values in
  T.Concrete (Nd.of_floats dtype [| Array.length arr |] arr)

let fromfile ?(dtype = D.F64) ?(count = -1) ?(sep = "") ?(offset = 0) path =
  let data = read_file path in
  if sep = "" then frombuffer ~dtype ~count ~offset data
  else
    let parsed = fromstring ~dtype ~sep data in
    if count < 0 then parsed
    else
      T.Concrete
        (Nd.of_floats dtype [| count |]
           (Array.init count (fun i -> Nd.get_f (host parsed) [| i |])))

let fromiter ?(dtype = D.F64) ?(count = -1) seq =
  let values =
    if count >= 0 then Seq.take count seq |> List.of_seq else List.of_seq seq
  in
  if count >= 0 && List.length values < count then
    invalid_arg "iterator too short";
  let arr = Array.of_list values in
  T.Concrete (Nd.of_floats dtype [| Array.length arr |] arr)
