type data = I of int64 array | F of float array | C of Complex.t array
type t = { dtype : string; shape : int array; data : data }

let u16 b off = Bytes.get_uint16_le b off
let u32 b off = Int32.to_int (Bytes.get_int32_le b off) land 0xFFFFFFFF

let read_file path =
  let ic = open_in_bin path in
  let len = in_channel_length ic in
  let b = Bytes.create len in
  really_input ic b 0 len;
  close_in ic;
  b

let find_eocd b =
  let sig_bytes = "PK\005\006" in
  let n = Bytes.length b in
  let rec loop i =
    if i < 0 then failwith "npz: end-of-central-directory not found"
    else if Bytes.sub_string b i 4 = sig_bytes then i
    else loop (i - 1)
  in
  loop (n - 4)

let contains hay needle from =
  let hn = String.length hay and nn = String.length needle in
  let rec loop i =
    if i + nn > hn then -1
    else if String.sub hay i nn = needle then i
    else loop (i + 1)
  in
  loop from

let parse_shape header =
  let lp = contains header "'shape':" 0 in
  let op = String.index_from header lp '(' in
  let cp = String.index_from header op ')' in
  let inner = String.sub header (op + 1) (cp - op - 1) in
  let parts = String.split_on_char ',' inner in
  let dims =
    List.filter_map
      (fun p ->
        let s = String.trim p in
        if s = "" then None else Some (int_of_string s))
      parts
  in
  Array.of_list dims

let parse_descr header =
  let key = "'descr':" in
  let vp = contains header key 0 + String.length key in
  let q1 = String.index_from header vp '\'' in
  let q2 = String.index_from header (q1 + 1) '\'' in
  String.sub header (q1 + 1) (q2 - q1 - 1)

let parse_fortran header =
  let key = "'fortran_order':" in
  let lp = contains header key 0 in
  let vp = lp + String.length key in
  let comma = String.index_from header vp ',' in
  let v = String.trim (String.sub header vp (comma - vp)) in
  v = "True"

let elem_kind descr =
  let core =
    if
      String.length descr > 0
      && (descr.[0] = '<'
         || descr.[0] = '>'
         || descr.[0] = '|'
         || descr.[0] = '=')
    then String.sub descr 1 (String.length descr - 1)
    else descr
  in
  if String.length descr > 0 && descr.[0] = '>' then
    failwith ("npz: big-endian descr unsupported " ^ descr);
  match core with
  | "b1" -> ("bool", `Int (1, false))
  | "i1" -> ("int8", `Int (1, false))
  | "i2" -> ("int16", `Int (2, false))
  | "i4" -> ("int32", `Int (4, false))
  | "i8" -> ("int64", `Int (8, false))
  | "u1" -> ("uint8", `Int (1, true))
  | "u2" -> ("uint16", `Int (2, true))
  | "u4" -> ("uint32", `Int (4, true))
  | "u8" -> ("uint64", `Int (8, true))
  | "f4" -> ("float32", `Float 4)
  | "f8" -> ("float64", `Float 8)
  | "c8" -> ("complex64", `Complex 4)
  | "c16" -> ("complex128", `Complex 8)
  | _ -> failwith ("npz: unsupported descr " ^ descr)

let read_int b off width unsigned =
  match (width, unsigned) with
  | 1, false -> Int64.of_int (Bytes.get_int8 b off)
  | 1, true -> Int64.of_int (Bytes.get_uint8 b off)
  | 2, false -> Int64.of_int (Bytes.get_int16_le b off)
  | 2, true -> Int64.of_int (Bytes.get_uint16_le b off)
  | 4, false -> Int64.of_int32 (Bytes.get_int32_le b off)
  | 4, true ->
      Int64.logand (Int64.of_int32 (Bytes.get_int32_le b off)) 0xFFFFFFFFL
  | 8, _ -> Bytes.get_int64_le b off
  | _ -> failwith "npz: bad int width"

let read_f32 b off = Int32.float_of_bits (Bytes.get_int32_le b off)
let read_f64 b off = Int64.float_of_bits (Bytes.get_int64_le b off)

let decode b start descr shape =
  let dtype, kind = elem_kind descr in
  let n = Array.fold_left ( * ) 1 shape in
  let data =
    match kind with
    | `Int (w, u) ->
        I (Array.init n (fun i -> read_int b (start + (i * w)) w u))
    | `Float 4 -> F (Array.init n (fun i -> read_f32 b (start + (i * 4))))
    | `Float 8 -> F (Array.init n (fun i -> read_f64 b (start + (i * 8))))
    | `Float _ -> failwith "npz: bad float width"
    | `Complex 4 ->
        C
          (Array.init n (fun i ->
               let o = start + (i * 8) in
               { Complex.re = read_f32 b o; im = read_f32 b (o + 4) }))
    | `Complex 8 ->
        C
          (Array.init n (fun i ->
               let o = start + (i * 16) in
               { Complex.re = read_f64 b o; im = read_f64 b (o + 8) }))
    | `Complex _ -> failwith "npz: bad complex width"
  in
  { dtype; shape; data }

let parse_npy b off =
  if Bytes.sub_string b off 6 <> "\147NUMPY" then failwith "npz: bad npy magic";
  let major = Bytes.get_uint8 b (off + 6) in
  let hlen, htext_off =
    if major = 1 then (u16 b (off + 8), off + 10)
    else (u32 b (off + 8), off + 12)
  in
  let header = Bytes.sub_string b htext_off hlen in
  if parse_fortran header then failwith "npz: fortran_order arrays unsupported";
  let descr = parse_descr header in
  let shape = parse_shape header in
  let data_off = htext_off + hlen in
  decode b data_off descr shape

let read path =
  let b = read_file path in
  let eocd = find_eocd b in
  let entries = u16 b (eocd + 10) in
  let cd_off = u32 b (eocd + 16) in
  let rec loop pos remaining acc =
    if remaining = 0 then List.rev acc
    else begin
      if Bytes.sub_string b pos 4 <> "PK\001\002" then
        failwith "npz: bad central directory signature";
      let name_len = u16 b (pos + 28) in
      let extra_len = u16 b (pos + 30) in
      let comment_len = u16 b (pos + 32) in
      let local_off = u32 b (pos + 42) in
      let name = Bytes.sub_string b (pos + 46) name_len in
      let method_ = u16 b (pos + 10) in
      if method_ <> 0 then failwith "npz: compressed members unsupported";
      let l_name_len = u16 b (local_off + 26) in
      let l_extra_len = u16 b (local_off + 28) in
      let data_start = local_off + 30 + l_name_len + l_extra_len in
      let member = Filename.remove_extension name in
      let arr = parse_npy b data_start in
      let next = pos + 46 + name_len + extra_len + comment_len in
      loop next (remaining - 1) ((member, arr) :: acc)
    end
  in
  loop cd_off entries []
