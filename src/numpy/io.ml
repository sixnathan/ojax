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
  | D.F16 -> "<f2"
  | D.Bf16 -> "<V2"
  | D.F32 -> "<f4"
  | D.F64 -> "<f8"
  | D.Complex64 -> "<c8"
  | D.Complex128 -> "<c16"

let dtype_of_descr descr =
  let order, kind =
    if descr = "" then invalid_arg "numpy io: empty dtype descriptor"
    else
      match descr.[0] with
      | ('<' | '>' | '|' | '=') as c ->
          (c, String.sub descr 1 (String.length descr - 1))
      | _ -> ('=', descr)
  in
  let dt =
    match kind with
    | "b1" -> D.Bool
    | "i1" -> D.I8
    | "i2" -> D.I16
    | "i4" -> D.I32
    | "i8" -> D.I64
    | "u1" -> D.Uint8
    | "u2" -> D.Uint16
    | "u4" -> D.Uint32
    | "u8" -> D.Uint64
    | "f2" -> D.F16
    | "V2" -> D.Bf16
    | "f4" -> D.F32
    | "f8" -> D.F64
    | "c8" -> D.Complex64
    | "c16" -> D.Complex128
    | _ -> invalid_arg ("numpy io: unsupported dtype descriptor " ^ descr)
  in
  (dt, order = '>')

let width = function
  | D.Bool | D.I8 | D.Uint8 -> 1
  | D.I16 | D.Uint16 | D.F16 | D.Bf16 -> 2
  | D.I32 | D.Uint32 | D.F32 -> 4
  | D.I64 | D.Uint64 | D.F64 | D.Complex64 -> 8
  | D.Complex128 -> 16

let component_width = function
  | D.Complex64 -> 4
  | D.Complex128 -> 8
  | dt -> width dt

let to_little_endian dt data offset count =
  let w = component_width dt and total = count * width dt in
  let out = Bytes.of_string data in
  let elements = total / w in
  for e = 0 to elements - 1 do
    for b = 0 to w - 1 do
      Bytes.set out (offset + (e * w) + b) data.[offset + (e * w) + (w - 1 - b)]
    done
  done;
  Bytes.unsafe_to_string out

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
    | D.F16 -> Bytes.set_uint16_le out at (Nd.bits_of_half (Nd.get_f nd idx))
    | D.Bf16 -> Bytes.set_uint16_le out at (Nd.bits_of_bfloat (Nd.get_f nd idx))
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
  | D.F16 ->
      Nd.of_floats dt sh
        (Array.init count (fun f ->
             Nd.half_of_bits (String.get_uint16_le data (at f))))
  | D.Bf16 ->
      Nd.of_floats dt sh
        (Array.init count (fun f ->
             Nd.bfloat_of_bits (String.get_uint16_le data (at f))))
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
  let dt, swapped = dtype_of_descr descr in
  let count = Array.fold_left ( * ) 1 sh in
  let data =
    if swapped && String.length data >= start + hlen + (count * width dt) then
      to_little_endian dt data (start + hlen) count
    else data
  in
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

let load path =
  let nd = parse_npy (read_file path) in
  let target = Dtypes.canonicalize_dtype (Nd.dtype nd) in
  if target = Nd.dtype nd then T.Concrete nd
  else Lax_numpy.astype (T.Concrete nd) target

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

type inflate_state = {
  source : string;
  mutable position : int;
  mutable bit_buffer : int;
  mutable bit_count : int;
}

let next_bits st count =
  while st.bit_count < count do
    if st.position >= String.length st.source then
      invalid_arg "numpy io: truncated deflate stream";
    st.bit_buffer <-
      st.bit_buffer lor (Char.code st.source.[st.position] lsl st.bit_count);
    st.position <- st.position + 1;
    st.bit_count <- st.bit_count + 8
  done;
  let v = st.bit_buffer land ((1 lsl count) - 1) in
  st.bit_buffer <- st.bit_buffer lsr count;
  st.bit_count <- st.bit_count - count;
  v

let huffman_table lengths =
  let counts = Array.make 16 0 in
  Array.iter (fun l -> counts.(l) <- counts.(l) + 1) lengths;
  counts.(0) <- 0;
  let offsets = Array.make 16 0 in
  for l = 1 to 14 do
    offsets.(l + 1) <- offsets.(l) + counts.(l)
  done;
  let symbols = Array.make (Array.length lengths) 0 in
  Array.iteri
    (fun sym l ->
      if l <> 0 then begin
        symbols.(offsets.(l)) <- sym;
        offsets.(l) <- offsets.(l) + 1
      end)
    lengths;
  (counts, symbols)

let next_symbol st (counts, symbols) =
  let rec go len code first index =
    if len > 15 then invalid_arg "numpy io: invalid deflate code"
    else
      let code = code lor next_bits st 1 in
      let count = counts.(len) in
      if code - count < first then symbols.(index + (code - first))
      else go (len + 1) (code lsl 1) ((first + count) lsl 1) (index + count)
  in
  go 1 0 0 0

let length_base =
  [|
    3;
    4;
    5;
    6;
    7;
    8;
    9;
    10;
    11;
    13;
    15;
    17;
    19;
    23;
    27;
    31;
    35;
    43;
    51;
    59;
    67;
    83;
    99;
    115;
    131;
    163;
    195;
    227;
    258;
  |]

let length_extra =
  [|
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    0;
    1;
    1;
    1;
    1;
    2;
    2;
    2;
    2;
    3;
    3;
    3;
    3;
    4;
    4;
    4;
    4;
    5;
    5;
    5;
    5;
    0;
  |]

let distance_base =
  [|
    1;
    2;
    3;
    4;
    5;
    7;
    9;
    13;
    17;
    25;
    33;
    49;
    65;
    97;
    129;
    193;
    257;
    385;
    513;
    769;
    1025;
    1537;
    2049;
    3073;
    4097;
    6145;
    8193;
    12289;
    16385;
    24577;
  |]

let distance_extra =
  [|
    0;
    0;
    0;
    0;
    1;
    1;
    2;
    2;
    3;
    3;
    4;
    4;
    5;
    5;
    6;
    6;
    7;
    7;
    8;
    8;
    9;
    9;
    10;
    10;
    11;
    11;
    12;
    12;
    13;
    13;
  |]

let code_length_order =
  [| 16; 17; 18; 0; 8; 7; 9; 6; 10; 5; 11; 4; 12; 3; 13; 2; 14; 1; 15 |]

let fixed_tables =
  lazy
    (let literal =
       Array.init 288 (fun i ->
           if i < 144 then 8
           else if i < 256 then 9
           else if i < 280 then 7
           else 8)
     in
     (huffman_table literal, huffman_table (Array.make 30 5)))

let dynamic_tables st =
  let hlit = next_bits st 5 + 257 in
  let hdist = next_bits st 5 + 1 in
  let hclen = next_bits st 4 + 4 in
  let code_lengths = Array.make 19 0 in
  for i = 0 to hclen - 1 do
    code_lengths.(code_length_order.(i)) <- next_bits st 3
  done;
  let table = huffman_table code_lengths in
  let lengths = Array.make (hlit + hdist) 0 in
  let i = ref 0 in
  let fill value times =
    if !i + times > hlit + hdist then
      invalid_arg "numpy io: invalid deflate code lengths";
    Array.fill lengths !i times value;
    i := !i + times
  in
  while !i < hlit + hdist do
    match next_symbol st table with
    | 16 ->
        if !i = 0 then invalid_arg "numpy io: invalid deflate code lengths";
        fill lengths.(!i - 1) (3 + next_bits st 2)
    | 17 -> fill 0 (3 + next_bits st 3)
    | 18 -> fill 0 (11 + next_bits st 7)
    | sym -> fill sym 1
  done;
  ( huffman_table (Array.sub lengths 0 hlit),
    huffman_table (Array.sub lengths hlit hdist) )

let inflate source expected =
  let st = { source; position = 0; bit_buffer = 0; bit_count = 0 } in
  let out = Buffer.create expected in
  let compressed (literal, distance) =
    let rec go () =
      let sym = next_symbol st literal in
      if sym < 256 then begin
        Buffer.add_char out (Char.chr sym);
        go ()
      end
      else if sym > 256 then begin
        let l = sym - 257 in
        if l >= 29 then invalid_arg "numpy io: invalid deflate length";
        let len = length_base.(l) + next_bits st length_extra.(l) in
        let d = next_symbol st distance in
        if d >= 30 then invalid_arg "numpy io: invalid deflate distance";
        let dist = distance_base.(d) + next_bits st distance_extra.(d) in
        if dist > Buffer.length out then
          invalid_arg "numpy io: invalid deflate distance";
        for _ = 1 to len do
          Buffer.add_char out (Buffer.nth out (Buffer.length out - dist))
        done;
        go ()
      end
    in
    go ()
  in
  let rec blocks () =
    let final = next_bits st 1 in
    (match next_bits st 2 with
    | 0 ->
        st.bit_buffer <- 0;
        st.bit_count <- 0;
        if st.position + 4 > String.length source then
          invalid_arg "numpy io: truncated deflate stream";
        let len = String.get_uint16_le source st.position in
        if String.get_uint16_le source (st.position + 2) <> len lxor 0xFFFF then
          invalid_arg "numpy io: corrupt stored deflate block";
        if st.position + 4 + len > String.length source then
          invalid_arg "numpy io: truncated deflate stream";
        Buffer.add_substring out source (st.position + 4) len;
        st.position <- st.position + 4 + len
    | 1 -> compressed (Lazy.force fixed_tables)
    | 2 -> compressed (dynamic_tables st)
    | _ -> invalid_arg "numpy io: invalid deflate block type");
    if final = 0 then blocks ()
  in
  blocks ();
  Buffer.contents out

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
      let crc = String.get_int32_le data (pos + 16) in
      let packed =
        Int32.to_int (String.get_int32_le data (pos + 20)) land 0xFFFFFFFF
      in
      let size =
        Int32.to_int (String.get_int32_le data (pos + 24)) land 0xFFFFFFFF
      in
      let name_len = String.get_uint16_le data (pos + 28) in
      let extra_len = String.get_uint16_le data (pos + 30) in
      let comment_len = String.get_uint16_le data (pos + 32) in
      let local = Int32.to_int (String.get_int32_le data (pos + 42)) in
      let member = String.sub data (pos + 46) name_len in
      let local_name = String.get_uint16_le data (local + 26) in
      let local_extra = String.get_uint16_le data (local + 28) in
      let start = local + 30 + local_name + local_extra in
      let body =
        match methd with
        | 0 -> String.sub data start size
        | 8 -> inflate (String.sub data start packed) size
        | m ->
            invalid_arg
              (Printf.sprintf "numpy io: unsupported npz compression method %d"
                 m)
      in
      if String.length body <> size || crc32 body <> crc then
        invalid_arg ("numpy io: bad checksum for npz member " ^ member);
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
