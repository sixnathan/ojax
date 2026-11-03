type builder = {
  mutable bytes : Bytes.t;
  mutable head : int;
  mutable minalign : int;
  mutable vtable : int array;
  mutable object_end : int;
  vtables : (int list, int) Hashtbl.t;
  mutable nested : bool;
  mutable vector_num_elems : int;
  mutable finished : bool;
}

let create initial_size =
  {
    bytes = Bytes.make initial_size '\000';
    head = initial_size;
    minalign = 1;
    vtable = [||];
    object_end = 0;
    vtables = Hashtbl.create 16;
    nested = false;
    vector_num_elems = 0;
    finished = false;
  }

let offset b = Bytes.length b.bytes - b.head

let grow b =
  let old = Bytes.length b.bytes in
  let size = max 1 (old * 2) in
  let fresh = Bytes.make size '\000' in
  Bytes.blit b.bytes 0 fresh (size - old) old;
  b.bytes <- fresh

let pad b n =
  if n > 0 then begin
    Bytes.fill b.bytes (b.head - n) n '\000';
    b.head <- b.head - n
  end

let prep b size additional =
  if size > b.minalign then b.minalign <- size;
  let align_size =
    (lnot (Bytes.length b.bytes - b.head + additional) + 1) land (size - 1)
  in
  let needed = align_size + size + additional in
  while b.head < needed do
    let old = Bytes.length b.bytes in
    grow b;
    b.head <- b.head + (Bytes.length b.bytes - old)
  done;
  pad b align_size

let write_le bytes pos width v =
  for i = 0 to width - 1 do
    Bytes.set bytes (pos + i) (Char.chr ((v lsr (8 * i)) land 0xff))
  done

let place b width v =
  b.head <- b.head - width;
  write_le b.bytes b.head width v

let prepend b width v =
  prep b width 0;
  place b width v

let prepend_u8 b v = prepend b 1 v
let prepend_u16 b v = prepend b 2 v
let prepend_u32 b v = prepend b 4 v

let prepend_uoffset_relative b off =
  prep b 4 0;
  if off > offset b then failwith "flatbuffers: Offset arithmetic error.";
  place b 4 (offset b - off + 4)

let prepend_soffset_relative b off =
  prep b 4 0;
  if off > offset b then failwith "flatbuffers: Offset arithmetic error.";
  place b 4 (offset b - off + 4)

let start_object b numfields =
  if b.nested then failwith "flatbuffers: IsNestedError";
  b.vtable <- Array.make numfields 0;
  b.object_end <- offset b;
  b.nested <- true

let slot b n = b.vtable.(n) <- offset b

let end_object b =
  b.nested <- false;
  prepend_soffset_relative b 0;
  let object_offset = offset b in
  let n = Array.length b.vtable in
  let key = ref [] in
  let trim = ref true in
  for i = n - 1 downto 0 do
    let elem = b.vtable.(i) in
    if elem = 0 then (if not !trim then key := 0 :: !key)
    else begin
      key := (object_offset - elem) :: !key;
      trim := false
    end
  done;
  let object_size = object_offset - b.object_end in
  let key = List.rev (object_size :: !key) in
  (match Hashtbl.find_opt b.vtables key with
  | None ->
      let trailing = ref 0 in
      let trim = ref true in
      for i = n - 1 downto 0 do
        let elem = b.vtable.(i) in
        if elem = 0 && !trim then incr trailing
        else begin
          let off =
            if elem = 0 then 0
            else begin
              trim := false;
              object_offset - elem
            end
          in
          prepend b 2 off
        end
      done;
      prepend b 2 object_size;
      prepend b 2 ((n - !trailing + 2) * 2);
      let object_start = Bytes.length b.bytes - object_offset in
      write_le b.bytes object_start 4
        ((offset b - object_offset) land 0xffffffff);
      Hashtbl.replace b.vtables key (offset b)
  | Some vt2 ->
      let object_start = Bytes.length b.bytes - object_offset in
      b.head <- object_start;
      write_le b.bytes b.head 4 ((vt2 - object_offset) land 0xffffffff));
  b.vtable <- [||];
  object_offset

let start_vector b elem_size num align =
  if b.nested then failwith "flatbuffers: IsNestedError";
  b.nested <- true;
  b.vector_num_elems <- num;
  prep b 4 (elem_size * num);
  prep b align (elem_size * num);
  offset b

let end_vector b =
  b.nested <- false;
  place b 4 b.vector_num_elems;
  offset b

let raw_vector b s ~terminator =
  if b.nested then failwith "flatbuffers: IsNestedError";
  b.nested <- true;
  let n = String.length s in
  prep b 4 (if terminator then n + 1 else n);
  if terminator then place b 1 0;
  b.head <- b.head - n;
  Bytes.blit_string s 0 b.bytes b.head n;
  b.vector_num_elems <- n;
  end_vector b

let create_string b s = raw_vector b s ~terminator:true
let create_byte_vector b s = raw_vector b s ~terminator:false

let create_uint_vector b width values =
  ignore (start_vector b width (List.length values) width);
  List.iter (fun v -> place b width v) (List.rev values);
  b.vector_num_elems <- List.length values;
  end_vector b

let offsets_vector b offs =
  ignore (start_vector b 4 (List.length offs) 4);
  List.iter (prepend_uoffset_relative b) (List.rev offs);
  end_vector b

let add_int_slot b width n v =
  if v <> 0 then begin
    prepend b width v;
    slot b n
  end

let add_offset_slot b n off =
  if off <> 0 then begin
    prepend_uoffset_relative b off;
    slot b n
  end

let finish b root =
  prep b b.minalign 4;
  prepend_uoffset_relative b root;
  b.finished <- true

let output b = Bytes.sub_string b.bytes b.head (Bytes.length b.bytes - b.head)

type table = { buf : string; pos : int }

let u8 s p = Char.code s.[p]
let u16 s p = u8 s p lor (u8 s (p + 1) lsl 8)
let u32 s p = u16 s p lor (u16 s (p + 2) lsl 16)

let i32 s p =
  let v = u32 s p in
  if v land 0x80000000 <> 0 then v - 0x100000000 else v

let check s p width =
  if p < 0 || p + width > String.length s then
    invalid_arg "flatbuffers: offset out of range"

let root s =
  check s 0 4;
  let p = u32 s 0 in
  check s p 4;
  { buf = s; pos = p }

let field t n =
  let vt = t.pos - i32 t.buf t.pos in
  check t.buf vt 4;
  let size = u16 t.buf vt in
  let o = 4 + (2 * n) in
  if o < size then u16 t.buf (vt + o) else 0

let has t n = field t n <> 0

let get_int t n width ~default =
  match field t n with
  | 0 -> default
  | o ->
      let p = t.pos + o in
      check t.buf p width;
      let v = ref 0 in
      for i = width - 1 downto 0 do
        v := (!v lsl 8) lor u8 t.buf (p + i)
      done;
      !v

let get_int8 t n =
  let v = get_int t n 1 ~default:0 in
  if v >= 128 then v - 256 else v

let indirect s p =
  check s p 4;
  p + u32 s p

let vector t n =
  match field t n with
  | 0 -> None
  | o ->
      let p = indirect t.buf (t.pos + o) in
      check t.buf p 4;
      Some (p + 4, u32 t.buf p)

let vector_length t n = match vector t n with None -> 0 | Some (_, l) -> l

let bytes_at s p =
  check s p 4;
  let n = u32 s p in
  check s (p + 4) n;
  String.sub s (p + 4) n

let get_string t n =
  match field t n with
  | 0 -> None
  | o -> Some (bytes_at t.buf (indirect t.buf (t.pos + o)))

let get_bytes t n =
  match field t n with
  | 0 -> ""
  | o -> bytes_at t.buf (indirect t.buf (t.pos + o))

let get_table t n =
  match field t n with
  | 0 -> None
  | o -> Some { buf = t.buf; pos = indirect t.buf (t.pos + o) }

let tables t n =
  match vector t n with
  | None -> []
  | Some (start, len) ->
      List.init len (fun i ->
          { buf = t.buf; pos = indirect t.buf (start + (4 * i)) })

let strings t n =
  match vector t n with
  | None -> []
  | Some (start, len) ->
      List.init len (fun i -> bytes_at t.buf (indirect t.buf (start + (4 * i))))

let uints t n width =
  match vector t n with
  | None -> []
  | Some (start, len) ->
      check t.buf start (len * width);
      List.init len (fun i ->
          let p = start + (i * width) in
          if width = 2 then u16 t.buf p
          else if width = 4 then u32 t.buf p
          else u8 t.buf p)
