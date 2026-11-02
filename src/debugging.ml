open Types

let render v =
  match v with
  | Concrete _ | Device _ -> Numpy.Extras.array_str v
  | Tracer _ -> "<traced>"

let format_message fmt args =
  let buffer = Buffer.create (String.length fmt + 16) in
  let remaining = ref args in
  let n = String.length fmt in
  let i = ref 0 in
  while !i < n do
    (match fmt.[!i] with
    | '{' when !i + 1 < n && fmt.[!i + 1] = '{' ->
        Buffer.add_char buffer '{';
        incr i
    | '}' when !i + 1 < n && fmt.[!i + 1] = '}' ->
        Buffer.add_char buffer '}';
        incr i
    | '{' -> (
        (match String.index_from_opt fmt !i '}' with
        | Some close -> i := close
        | None -> invalid_arg "debug.print: unmatched brace");
        match !remaining with
        | v :: rest ->
            remaining := rest;
            Buffer.add_string buffer (render v)
        | [] -> invalid_arg "debug.print: not enough arguments")
    | c -> Buffer.add_char buffer c);
    incr i
  done;
  Buffer.contents buffer

let callback f args =
  let host values =
    f values;
    []
  in
  ignore (Custom_derivatives.pure_callback host ~result_avals:[] args)

type field = { fname : string; conv : char option; spec : string }

let parse_format fmt =
  let n = String.length fmt in
  let chunks = ref [] in
  let lit = Buffer.create 16 in
  let flush field =
    chunks := (Buffer.contents lit, field) :: !chunks;
    Buffer.clear lit
  in
  let i = ref 0 in
  while !i < n do
    let c = fmt.[!i] in
    if c = '{' && !i + 1 < n && fmt.[!i + 1] = '{' then begin
      Buffer.add_char lit '{';
      flush None;
      i := !i + 2
    end
    else if c = '}' && !i + 1 < n && fmt.[!i + 1] = '}' then begin
      Buffer.add_char lit '}';
      flush None;
      i := !i + 2
    end
    else if c = '}' then
      Py_exn.value_error "Single '}' encountered in format string"
    else if c = '{' then begin
      if !i + 1 >= n then
        Py_exn.value_error "Single '{' encountered in format string";
      let depth = ref 1 and j = ref (!i + 1) in
      while !depth > 0 && !j < n do
        (match fmt.[!j] with '{' -> incr depth | '}' -> decr depth | _ -> ());
        if !depth > 0 then incr j
      done;
      if !depth > 0 then Py_exn.value_error "expected '}' before end of string";
      let body = String.sub fmt (!i + 1) (!j - !i - 1) in
      let m = String.length body in
      let k = ref 0 in
      while !k < m && body.[!k] <> '!' && body.[!k] <> ':' do
        incr k
      done;
      let fname = String.sub body 0 !k in
      let conv, rest =
        if !k < m && body.[!k] = '!' then begin
          if !k + 1 >= m then
            Py_exn.value_error
              "end of string while looking for conversion specifier";
          let cv = body.[!k + 1] in
          if !k + 2 < m && body.[!k + 2] <> ':' then
            Py_exn.value_error "expected ':' after conversion specifier";
          (Some cv, !k + 2)
        end
        else (None, !k)
      in
      let spec =
        if rest < m && body.[rest] = ':' then
          String.sub body (rest + 1) (m - rest - 1)
        else ""
      in
      flush (Some { fname; conv; spec });
      i := !j + 1
    end
    else begin
      Buffer.add_char lit c;
      incr i
    end
  done;
  if Buffer.length lit > 0 then flush None;
  List.rev !chunks

let is_digits s = s <> "" && String.for_all (fun c -> c >= '0' && c <= '9') s

let resolve chunks ~nargs ~kwargs =
  let auto = ref (Some 0) in
  let used_pos = Hashtbl.create 8 and used_kw = Hashtbl.create 8 in
  let fields =
    List.filter_map
      (fun (lit, field) ->
        match field with
        | None -> Some (lit, None)
        | Some f ->
            let key =
              if f.fname = "" then
                begin match !auto with
                | None ->
                    Py_exn.value_error
                      "cannot switch from manual field specification to \
                       automatic field numbering"
                | Some k ->
                    auto := Some (k + 1);
                    `Pos k
                end
              else if is_digits f.fname then begin
                (match !auto with
                | Some k when k > 0 ->
                    Py_exn.value_error
                      "cannot switch from manual field specification to \
                       automatic field numbering"
                | _ -> ());
                auto := None;
                `Pos (int_of_string f.fname)
              end
              else `Kw f.fname
            in
            (match key with
            | `Pos k ->
                if k >= nargs then Py_exn.index_error "tuple index out of range";
                Hashtbl.replace used_pos k ()
            | `Kw name ->
                if not (List.mem name kwargs) then
                  Py_exn.key_error ("'" ^ name ^ "'");
                Hashtbl.replace used_kw name ());
            Some (lit, Some (key, f)))
      chunks
  in
  (fields, used_pos, used_kw)

let tracer_repr v =
  let a = Core.get_aval v in
  Printf.sprintf "JitTracer(%s%s[%s])"
    (if a.weak_type then "~" else "")
    (Dtype.numpy_name a.dtype)
    (String.concat "," (List.map string_of_int (Array.to_list a.shape)))

let value_repr v =
  match v with Tracer _ -> tracer_repr v | _ -> Numpy.Extras.repr v

let py_list items = "[" ^ String.concat ", " items ^ "]"

type spec = {
  fill : char;
  align : char option;
  sign : char;
  alt : bool;
  zero : bool;
  width : int;
  precision : int option;
  ftype : char option;
}

let parse_spec s =
  let n = String.length s in
  let i = ref 0 in
  let is_align c = c = '<' || c = '>' || c = '^' || c = '=' in
  let fill, align =
    if n >= 2 && is_align s.[1] then begin
      i := 2;
      (s.[0], Some s.[1])
    end
    else if n >= 1 && is_align s.[0] then begin
      i := 1;
      (' ', Some s.[0])
    end
    else (' ', None)
  in
  let sign =
    if !i < n && (s.[!i] = '+' || s.[!i] = '-' || s.[!i] = ' ') then begin
      incr i;
      s.[!i - 1]
    end
    else '-'
  in
  let alt =
    if !i < n && s.[!i] = '#' then begin
      incr i;
      true
    end
    else false
  in
  let zero =
    if !i < n && s.[!i] = '0' then begin
      incr i;
      true
    end
    else false
  in
  let digits () =
    let st = !i in
    while !i < n && s.[!i] >= '0' && s.[!i] <= '9' do
      incr i
    done;
    if !i > st then Some (int_of_string (String.sub s st (!i - st))) else None
  in
  let width = Option.value (digits ()) ~default:0 in
  let precision =
    if !i < n && s.[!i] = '.' then begin
      incr i;
      match digits () with
      | Some p -> Some p
      | None -> Py_exn.value_error "Format specifier missing precision"
    end
    else None
  in
  let ftype =
    if !i < n then begin
      incr i;
      Some s.[!i - 1]
    end
    else None
  in
  if !i < n then Py_exn.value_error "Invalid format specifier";
  let fill, align =
    if zero && align = None then ('0', Some '=') else (fill, align)
  in
  { fill; align; sign; alt; zero; width; precision; ftype }

let pad sp ~default_align body =
  let neg = String.length body > 0 && body.[0] = '-' in
  let has_sign =
    neg || (String.length body > 0 && (body.[0] = '+' || body.[0] = ' '))
  in
  let len = String.length body in
  if len >= sp.width then body
  else
    let k = sp.width - len in
    let fill n = String.make n sp.fill in
    match Option.value sp.align ~default:default_align with
    | '<' -> body ^ fill k
    | '^' -> fill (k / 2) ^ body ^ fill (k - (k / 2))
    | '=' when has_sign ->
        String.sub body 0 1 ^ fill k ^ String.sub body 1 (len - 1)
    | '=' -> fill k ^ body
    | _ -> fill k ^ body

let with_sign sp neg body =
  if neg then "-" ^ body
  else match sp.sign with '+' -> "+" ^ body | ' ' -> " " ^ body | _ -> body

let strip_g alt s =
  if alt then s
  else
    let mant, exp =
      match String.index_opt s 'e' with
      | Some k -> (String.sub s 0 k, String.sub s k (String.length s - k))
      | None -> (s, "")
    in
    let mant =
      if String.contains mant '.' then begin
        let m = ref (String.length mant) in
        while !m > 0 && mant.[!m - 1] = '0' do
          decr m
        done;
        if !m > 0 && mant.[!m - 1] = '.' then decr m;
        String.sub mant 0 !m
      end
      else mant
    in
    mant ^ exp

let format_float sp x =
  let neg = Float.sign_bit x && not (Float.is_nan x) in
  let ax = Float.abs x in
  let upper =
    match sp.ftype with Some ('E' | 'F' | 'G') -> true | _ -> false
  in
  let special =
    if Float.is_nan x then Some "nan"
    else if Float.is_integer ax || Float.is_finite ax then None
    else Some "inf"
  in
  let body =
    match (special, sp.ftype) with
    | Some s, Some '%' -> s ^ "%"
    | Some s, _ -> if upper then String.uppercase_ascii s else s
    | None, (Some ('f' | 'F') | None) when sp.ftype <> None ->
        Printf.sprintf "%.*f" (Option.value sp.precision ~default:6) ax
    | None, Some ('e' | 'E') ->
        let s =
          Printf.sprintf "%.*e" (Option.value sp.precision ~default:6) ax
        in
        if upper then String.uppercase_ascii s else s
    | None, Some ('g' | 'G' | 'n') ->
        let p =
          match sp.precision with Some 0 -> 1 | Some p -> p | None -> 6
        in
        let s =
          strip_g sp.alt
            (Printf.sprintf (if sp.alt then "%#.*g" else "%.*g") p ax)
        in
        if upper then String.uppercase_ascii s else s
    | None, Some '%' ->
        Printf.sprintf "%.*f"
          (Option.value sp.precision ~default:6)
          (ax *. 100.0)
        ^ "%"
    | None, None -> (
        match sp.precision with
        | None ->
            let r = Pretty_printer.python_repr_float ax in
            r
        | Some p ->
            let p = if p = 0 then 1 else p in
            let s = strip_g false (Printf.sprintf "%.*g" p ax) in
            if String.contains s '.' || String.contains s 'e' then s
            else s ^ ".0")
    | None, Some c ->
        Py_exn.value_error
          (Printf.sprintf "Unknown format code '%c' for object of type 'float'"
             c)
  in
  pad sp ~default_align:'>' (with_sign sp neg body)

let format_int sp (x : int64) =
  match sp.ftype with
  | Some ('e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%') ->
      format_float sp (Int64.to_float x)
  | _ ->
      if sp.precision <> None then
        Py_exn.value_error "Precision not allowed in integer format specifier";
      let neg = Int64.compare x 0L < 0 in
      let ax = Int64.abs x in
      let body =
        match sp.ftype with
        | None | Some ('d' | 'n') -> Printf.sprintf "%Lu" ax
        | Some 'x' -> (if sp.alt then "0x" else "") ^ Printf.sprintf "%Lx" ax
        | Some 'X' -> (if sp.alt then "0X" else "") ^ Printf.sprintf "%LX" ax
        | Some 'o' -> (if sp.alt then "0o" else "") ^ Printf.sprintf "%Lo" ax
        | Some 'b' ->
            let rec bits v acc =
              if Int64.equal v 0L then if acc = "" then "0" else acc
              else
                bits
                  (Int64.shift_right_logical v 1)
                  ((if Int64.logand v 1L = 1L then "1" else "0") ^ acc)
            in
            (if sp.alt then "0b" else "") ^ bits ax ""
        | Some c ->
            Py_exn.value_error
              (Printf.sprintf
                 "Unknown format code '%c' for object of type 'int'" c)
      in
      pad sp ~default_align:'>' (with_sign sp neg body)

let format_value ?(options = Numpy.Array_print.default_options) v spec =
  let nd =
    match v with
    | Concrete nd -> nd
    | Device b -> Pjrt.Buffer.to_host b
    | Tracer _ -> failwith "debug.print: expected a concrete value"
  in
  if Array.length (Ndarray.shape nd) > 0 then
    if spec = "" then Numpy.Array_print.array_str ~options nd
    else
      Py_exn.type_error
        "unsupported format string passed to numpy.ndarray.__format__"
  else
    let dt = Ndarray.dtype nd in
    if Dtypes.isdtype dt "complex floating" then
      Numpy.Array_print.array_str ~options nd
    else if dt = Dtype.Bool then
      let b = Ndarray.get_f nd [||] <> 0.0 in
      if spec = "" then if b then "True" else "False"
      else format_int (parse_spec spec) (if b then 1L else 0L)
    else if Dtypes.isdtype dt "real floating" then
      let x = Ndarray.get_f nd [||] in
      if spec = "" then Pretty_printer.python_repr_float x
      else format_float (parse_spec spec) x
    else
      let x = Ndarray.get_i64 nd [||] in
      if spec = "" then
        if dt = Dtype.Uint64 then Printf.sprintf "%Lu" x else Int64.to_string x
      else format_int (parse_spec spec) x

let host_nd = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> failwith "debug.print: expected a concrete value"

let print_sink : (string -> unit) ref = ref print_endline

let np_printoptions_text (o : Numpy.Extras.printoptions) =
  Printf.sprintf
    "(('edgeitems', %d), ('threshold', %d), ('floatmode', 'maxprec'), \
     ('precision', %d), ('suppress', %s), ('linewidth', 75), ('nanstr', \
     'nan'), ('infstr', 'inf'), ('sign', '-'), ('formatter', None), ('legacy', \
     False), ('override_repr', None))"
    o.edgeitems o.threshold o.precision
    (if o.suppress then "True" else "False")

let array_options (o : Numpy.Extras.printoptions) =
  {
    Numpy.Array_print.default_options with
    precision = o.precision;
    threshold = o.threshold;
    edgeitems = o.edgeitems;
    suppress = o.suppress;
  }

let in_tree_text nargs kw_names =
  let pos =
    match nargs with
    | 0 -> "()"
    | 1 -> "(*,)"
    | n -> "(" ^ String.concat ", " (List.init n (fun _ -> "*")) ^ ")"
  in
  let kw =
    "{"
    ^ String.concat ", " (List.map (fun k -> "'" ^ k ^ "': *") kw_names)
    ^ "}"
  in
  Printf.sprintf "PyTreeDef((%s, %s))" pos kw

let print ?(ordered = false) ?(kwargs = []) fmt args =
  let kwargs = List.sort (fun (a, _) (b, _) -> compare a b) kwargs in
  let kw_names = List.map fst kwargs in
  let nargs = List.length args in
  let chunks = parse_format fmt in
  let fields, used_pos, used_kw = resolve chunks ~nargs ~kwargs:kw_names in
  let unused_pos =
    List.filteri (fun i _ -> not (Hashtbl.mem used_pos i)) args
  in
  if unused_pos <> [] then
    Py_exn.value_error
      ("Unused positional arguments to `jax.debug.print`: "
      ^ py_list (List.map value_repr unused_pos));
  let unused_kw = List.filter (fun k -> not (Hashtbl.mem used_kw k)) kw_names in
  if unused_kw <> [] then
    Py_exn.value_error
      ("Unused keyword arguments to `jax.debug.print`: "
      ^ py_list (List.map (fun k -> "'" ^ k ^ "'") unused_kw)
      ^ ". You may be passing an f-string (i.e, `f\"{x}\"`) into \
         `jax.debug.print` and instead should pass in a regular string.");
  let has_placeholders =
    match chunks with (_, Some _) :: _ -> true | _ -> false
  in
  let popts = Numpy.Extras.get_printoptions () in
  let options = array_options popts in
  let render values =
    let pos = List.filteri (fun i _ -> i < nargs) values in
    let kw = List.filteri (fun i _ -> i >= nargs) values in
    let kw = List.combine kw_names kw in
    if has_placeholders then
      String.concat ""
        (List.map
           (fun (lit, f) ->
             match f with
             | None -> lit
             | Some (key, f) ->
                 let v =
                   match key with
                   | `Pos k -> List.nth pos k
                   | `Kw k -> List.assoc k kw
                 in
                 let shown =
                   match f.conv with
                   | Some 'r' -> Numpy.Extras.repr v
                   | Some 's' ->
                       Numpy.Array_print.array_str ~options (host_nd v)
                   | _ -> format_value ~options v f.spec
                 in
                 lit ^ shown)
           fields)
    else
      String.concat " "
        (fmt
        :: List.map
             (fun v -> Numpy.Array_print.array_str ~options (host_nd v))
             pos)
  in
  let params =
    [
      ("fmt", fmt);
      ("has_placeholders", if has_placeholders then "True" else "False");
      ("in_tree", in_tree_text nargs kw_names);
      ("logging_record", "None");
      ("np_printoptions", np_printoptions_text popts);
      ("ordered", if ordered then "True" else "False");
      ("partitioned", "False");
      ("static_args", "()");
    ]
  in
  let host values =
    !print_sink (render values);
    []
  in
  ignore
    (Custom_derivatives.callback_with_view ~ordered "debug_print" params host
       ~result_avals:[]
       (args @ List.map snd kwargs))

let log ?(level = "INFO") fmt args =
  callback
    (fun values ->
      prerr_endline (Printf.sprintf "%s: %s" level (format_message fmt values)))
    args

let default_breakpoint values =
  List.iteri
    (fun i v -> prerr_endline (Printf.sprintf "%d: %s" i (render v)))
    values

let breakpoint_handler = ref default_breakpoint
let set_breakpoint_handler f = breakpoint_handler := f
let breakpoint args = callback (fun values -> !breakpoint_handler values) args

let inspect_array_sharding v ~callback =
  match v with
  | Tracer _ -> ()
  | Concrete _ | Device _ ->
      callback (Spmd.sharding_to_string (Spmd.array_sharding v))

let visualize_sharding ?scale ?min_width ?max_width shape sharding =
  print_string
    (Spmd.visualize_sharding_string ?scale ?min_width ?max_width shape sharding)

let visualize_array_sharding ?scale ?min_width ?max_width v =
  visualize_sharding ?scale ?min_width ?max_width (Core.get_aval v).shape
    (Spmd.array_sharding v)
