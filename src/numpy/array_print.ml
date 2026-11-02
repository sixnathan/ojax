type options = {
  precision : int;
  threshold : int;
  edgeitems : int;
  suppress : bool;
  widen_elements : bool;
  linewidth : int;
}

let default_options =
  {
    precision = 8;
    threshold = 1000;
    edgeitems = 3;
    suppress = false;
    widen_elements = false;
    linewidth = 75;
  }

let f32_of v = Int32.float_of_bits (Int32.bits_of_float v)

let rounder dt =
  match dt with
  | Dtype.F32 | Dtype.Complex64 -> f32_of
  | Dtype.F64 | Dtype.Complex128 -> Fun.id
  | other -> fun v -> Ndarray.get_f (Ndarray.of_floats other [||] [| v |]) [||]

let mantissa_bits dt =
  match dt with
  | Dtype.F32 | Dtype.Complex64 -> 23
  | Dtype.F64 | Dtype.Complex128 -> 52
  | Dtype.F16 -> 10
  | Dtype.Bf16 -> 7
  | other -> (
      match Float_format.of_dtype other with Some f -> f.nmant | None -> 52)

let finfo_precision dt =
  int_of_float (Float.of_int (mantissa_bits dt) *. Float.log10 2.0)

let shortest_sci round v =
  let rec go p =
    let s = Printf.sprintf "%.*e" p v in
    if p >= 17 || round (float_of_string s) = v then s else go (p + 1)
  in
  go 0

let split_exp s =
  let at = String.index s 'e' in
  ( String.sub s 0 at,
    int_of_string
      (let t = String.sub s (at + 1) (String.length s - at - 1) in
       if t.[0] = '+' then String.sub t 1 (String.length t - 1) else t) )

let digits_of_mantissa m =
  let b = Buffer.create 20 in
  String.iter (fun c -> if c >= '0' && c <= '9' then Buffer.add_char b c) m;
  Buffer.contents b

let trim_zeros frac =
  let n = ref (String.length frac) in
  while !n > 0 && frac.[!n - 1] = '0' do
    decr n
  done;
  String.sub frac 0 !n

let shortest_frac_len round v =
  let mantissa, e = split_exp (shortest_sci round v) in
  let digits = trim_zeros (digits_of_mantissa mantissa) in
  max 0 (String.length digits - 1 - e)

let shortest_positional round v =
  let mantissa, e = split_exp (shortest_sci round (Float.abs v)) in
  let digits = trim_zeros (digits_of_mantissa mantissa) in
  let digits = if digits = "" then "0" else digits in
  let n = String.length digits in
  let int_part, frac =
    if e >= n - 1 then (digits ^ String.make (e - n + 1) '0', "")
    else if e >= 0 then
      (String.sub digits 0 (e + 1), String.sub digits (e + 1) (n - e - 1))
    else ("0", String.make (-e - 1) '0' ^ digits)
  in
  (int_part, frac)

let positional_parts round precision v =
  let av = Float.abs v in
  let int_part, frac =
    if av = 0.0 then ("0", "")
    else if shortest_frac_len round av <= precision then
      shortest_positional round av
    else
      let s = Printf.sprintf "%.*f" precision av in
      let dot = String.index s '.' in
      ( String.sub s 0 dot,
        trim_zeros (String.sub s (dot + 1) (String.length s - dot - 1)) )
  in
  (int_part, frac)

let signed plus v body =
  if Float.sign_bit v then "-" ^ body else if plus then "+" ^ body else body

let sci_parts round precision v =
  let av = Float.abs v in
  if av = 0.0 then ("0", "", 0)
  else
    let mantissa, e = split_exp (shortest_sci round av) in
    let digits = trim_zeros (digits_of_mantissa mantissa) in
    if String.length digits - 1 <= precision then
      (String.sub digits 0 1, String.sub digits 1 (String.length digits - 1), e)
    else
      let m, e = split_exp (Printf.sprintf "%.*e" precision av) in
      let d = digits_of_mantissa m in
      (String.sub d 0 1, trim_zeros (String.sub d 1 (String.length d - 1)), e)

let exp_string e digits =
  let sign = if e < 0 then "-" else "+" in
  Printf.sprintf "e%s%0*d" sign digits (abs e)

let pad_to n s =
  let k = String.length s in
  if k >= n then s else String.make (n - k) ' ' ^ s

let pad_right_to n s =
  let k = String.length s in
  if k >= n then s else s ^ String.make (n - k) ' '

let float_formatter ~options ~plus ~dtype values =
  let round = rounder dtype in
  let precision = options.precision in
  let finite = List.filter Float.is_finite values in
  let nonzero =
    List.filter_map
      (fun v -> if v <> 0.0 then Some (Float.abs v) else None)
      finite
  in
  let exp_format =
    match nonzero with
    | [] -> false
    | first :: rest ->
        let hi = List.fold_left Float.max first rest in
        let lo = List.fold_left Float.min first rest in
        let cutoff = 10.0 ** Float.of_int (min 8 (finfo_precision dtype)) in
        hi >= cutoff
        || ((not options.suppress) && (lo < 0.0001 || round (hi /. lo) > 1000.0))
  in
  let has_nonfinite = List.compare_lengths finite values <> 0 in
  let has_neg_inf = List.exists (fun v -> v = Float.neg_infinity) values in
  let neginf = plus || has_neg_inf in
  let nan_pads pad_left pad_right =
    if has_nonfinite then
      let offset = pad_right + 1 in
      let neg = if neginf then 1 else 0 in
      max pad_left (max (3 - offset) (3 + neg - offset))
    else pad_left
  in
  let nonfinite pad_left pad_right v =
    let body =
      if Float.is_nan v then if plus then "+nan" else "nan"
      else if v < 0.0 then "-inf"
      else if plus then "+inf"
      else "inf"
    in
    pad_to (pad_left + pad_right + 1) body
  in
  if finite = [] then
    let pad_left = nan_pads 0 0 in
    fun v -> nonfinite pad_left 0 v
  else if exp_format then
    let parts = List.map (fun v -> (v, sci_parts round precision v)) finite in
    let frac_len =
      List.fold_left
        (fun acc (_, (_, f, _)) -> max acc (String.length f))
        0 parts
    in
    let exp_digits =
      List.fold_left
        (fun acc (_, (_, _, e)) ->
          max acc (String.length (string_of_int (abs e))))
        2 parts
    in
    let int_len =
      List.fold_left
        (fun acc (v, (i, _, _)) -> max acc (String.length (signed plus v i)))
        0 parts
    in
    let pad_right = exp_digits + 2 + frac_len in
    let pad_left = nan_pads int_len pad_right in
    fun v ->
      if not (Float.is_finite v) then nonfinite pad_left pad_right v
      else
        let av = Float.abs v in
        let m, e =
          if av = 0.0 then (Printf.sprintf "0.%s" (String.make frac_len '0'), 0)
          else
            let m, e = split_exp (Printf.sprintf "%.*e" frac_len av) in
            let m = if frac_len = 0 then m ^ "." else m in
            (m, e)
        in
        let dot = String.index m '.' in
        let int_part = String.sub m 0 dot in
        let rest = String.sub m dot (String.length m - dot) in
        pad_to pad_left (signed plus v int_part)
        ^ rest ^ exp_string e exp_digits
  else
    let parts =
      List.map (fun v -> (v, positional_parts round precision v)) finite
    in
    let int_len =
      List.fold_left
        (fun acc (v, (i, _)) -> max acc (String.length (signed plus v i)))
        0 parts
    in
    let pad_right =
      List.fold_left (fun acc (_, (_, f)) -> max acc (String.length f)) 0 parts
    in
    let pad_left = nan_pads int_len pad_right in
    fun v ->
      if not (Float.is_finite v) then nonfinite pad_left pad_right v
      else
        let element_round = if options.widen_elements then Fun.id else round in
        let i, f = positional_parts element_round precision v in
        pad_to pad_left (signed plus v i) ^ "." ^ pad_right_to pad_right f

let rec leading_trailing edge shape axis prefix acc =
  if axis = Array.length shape then List.rev prefix :: acc
  else
    let n = shape.(axis) in
    let picks =
      if n > 2 * edge then
        List.init edge Fun.id @ List.init edge (fun i -> n - edge + i)
      else List.init n Fun.id
    in
    List.fold_left
      (fun acc i -> leading_trailing edge shape (axis + 1) (i :: prefix) acc)
      acc picks

let selected_indices ~summarize ~edge shape =
  let all = if summarize then edge else max_int / 4 in
  List.rev_map Array.of_list (leading_trailing all shape 0 [] [])

let is_complex = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let is_float dt =
  match dt with
  | Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      true
  | _ -> false

let unsigned_string (v : int64) dt =
  match dt with
  | Dtype.Uint64 -> Printf.sprintf "%Lu" v
  | _ -> Int64.to_string v

let make_formatter ~options nd indices =
  let dt = Ndarray.dtype nd in
  if dt = Dtype.Bool then
    let true_str =
      if Array.length (Ndarray.shape nd) = 0 then "True" else " True"
    in
    fun idx -> if Ndarray.get_f nd idx <> 0.0 then true_str else "False"
  else if is_complex dt then begin
    let values = List.map (fun i -> Ndarray.get_c nd i) indices in
    let re =
      float_formatter ~options ~plus:false ~dtype:dt
        (List.map (fun (z : Complex.t) -> z.re) values)
    in
    let im =
      float_formatter ~options ~plus:true ~dtype:dt
        (List.map (fun (z : Complex.t) -> z.im) values)
    in
    fun idx ->
      let z = Ndarray.get_c nd idx in
      let r = re z.re and i = im z.im in
      let trimmed =
        let n = ref (String.length i) in
        while !n > 0 && i.[!n - 1] = ' ' do
          decr n
        done;
        !n
      in
      r ^ String.sub i 0 trimmed ^ "j"
      ^ String.sub i trimmed (String.length i - trimmed)
  end
  else if is_float dt then
    let f =
      float_formatter ~options ~plus:false ~dtype:dt
        (List.map (fun i -> Ndarray.get_f nd i) indices)
    in
    fun idx -> f (Ndarray.get_f nd idx)
  else
    let width =
      List.fold_left
        (fun acc i ->
          max acc (String.length (unsigned_string (Ndarray.get_i64 nd i) dt)))
        0 indices
    in
    fun idx -> pad_to width (unsigned_string (Ndarray.get_i64 nd idx) dt)

let rstrip s =
  let n = ref (String.length s) in
  while !n > 0 && (s.[!n - 1] = ' ' || s.[!n - 1] = '\n') do
    decr n
  done;
  String.sub s 0 !n

let extend_line s line word width next_prefix =
  let needs_wrap = String.length line + String.length word > width in
  let needs_wrap =
    needs_wrap && String.length line > String.length next_prefix
  in
  if needs_wrap then (s ^ rstrip line ^ "\n", next_prefix ^ word)
  else (s, line ^ word)

let extend_line_pretty s line word width next_prefix =
  match String.split_on_char '\n' word with
  | [ _ ] -> extend_line s line word width next_prefix
  | words ->
      let longest =
        List.fold_left (fun acc w -> max acc (String.length w)) 0 words
      in
      let first = List.hd words in
      let s, line, indent =
        if
          String.length line + longest > width
          && String.length line > String.length next_prefix
        then (s ^ rstrip line ^ "\n", next_prefix ^ first, next_prefix)
        else (s, line ^ first, String.make (String.length line) ' ')
      in
      let s, line =
        List.fold_left
          (fun (s, line) w -> (s ^ rstrip line ^ "\n", indent ^ w))
          (s, line) (List.tl words)
      in
      let last = List.nth words (List.length words - 1) in
      (s, line ^ String.make (longest - String.length last) ' ')

let format_array nd format ~linewidth ~next_prefix ~separator ~edge ~summary =
  let shape = Ndarray.shape nd in
  let rank = Array.length shape in
  let rec go index hanging curr_width =
    let axis = List.length index in
    let axes_left = rank - axis in
    if axes_left = 0 then format (Array.of_list (List.rev index))
    else
      let next_hanging = hanging ^ " " in
      let next_width = curr_width - 1 in
      let a_len = shape.(axis) in
      let show_summary = summary <> "" && 2 * edge < a_len in
      let leading, trailing =
        if show_summary then (edge, edge) else (0, a_len)
      in
      let sep_trim = rstrip separator in
      let body =
        if axes_left = 1 then begin
          let elem_width = curr_width - max (String.length sep_trim) 1 in
          let s = ref "" and line = ref hanging in
          let add word =
            let s', l' = extend_line_pretty !s !line word elem_width hanging in
            s := s';
            line := l'
          in
          for i = 0 to leading - 1 do
            add (go (i :: index) next_hanging next_width);
            line := !line ^ separator
          done;
          if show_summary then begin
            let s', l' = extend_line !s !line summary elem_width hanging in
            s := s';
            line := l' ^ separator
          end;
          for i = trailing downto 2 do
            add (go ((a_len - i) :: index) next_hanging next_width);
            line := !line ^ separator
          done;
          add (go ((a_len - 1) :: index) next_hanging next_width);
          !s ^ !line
        end
        else begin
          let line_sep = sep_trim ^ String.make (axes_left - 1) '\n' in
          let b = Buffer.create 128 in
          for i = 0 to leading - 1 do
            Buffer.add_string b hanging;
            Buffer.add_string b (go (i :: index) next_hanging next_width);
            Buffer.add_string b line_sep
          done;
          if show_summary then begin
            Buffer.add_string b hanging;
            Buffer.add_string b summary;
            Buffer.add_string b line_sep
          end;
          for i = trailing downto 2 do
            Buffer.add_string b hanging;
            Buffer.add_string b
              (go ((a_len - i) :: index) next_hanging next_width);
            Buffer.add_string b line_sep
          done;
          Buffer.add_string b hanging;
          Buffer.add_string b
            (go ((a_len - 1) :: index) next_hanging next_width);
          Buffer.contents b
        end
      in
      let h = String.length hanging in
      "[" ^ String.sub body h (String.length body - h) ^ "]"
  in
  go [] next_prefix linewidth

let size nd = Array.fold_left ( * ) 1 (Ndarray.shape nd)

let array2string ?(options = default_options) ?(separator = " ") ?(prefix = "")
    ?(suffix = "") nd =
  if size nd = 0 then "[]"
  else
    let summarize = size nd > options.threshold in
    let indices =
      selected_indices ~summarize ~edge:options.edgeitems (Ndarray.shape nd)
    in
    let format = make_formatter ~options nd indices in
    format_array nd format
      ~linewidth:(options.linewidth - String.length suffix)
      ~next_prefix:(" " ^ String.make (String.length prefix) ' ')
      ~separator ~edge:options.edgeitems
      ~summary:(if summarize then "..." else "")

let python_shape shape =
  match Array.to_list shape with
  | [ d ] -> Printf.sprintf "(%d,)" d
  | dims -> "(" ^ String.concat ", " (List.map string_of_int dims) ^ ")"

let array_repr ?(options = default_options) nd =
  let prefix = "Array(" in
  let dtype_str = "dtype=" ^ Dtype.numpy_name (Ndarray.dtype nd) in
  let s =
    if size nd = 0 then "[], shape=" ^ python_shape (Ndarray.shape nd)
    else array2string ~options ~prefix ~suffix:"," ~separator:", " nd
  in
  let last_newline =
    match String.rindex_opt s '\n' with Some i -> i | None -> -1
  in
  let last_line_len = String.length s - last_newline + 1 in
  let sep =
    if last_line_len + String.length dtype_str + 2 > options.linewidth then
      String.make (String.length prefix) ' '
    else " "
  in
  prefix ^ s ^ "," ^ sep ^ dtype_str ^ ")"

let numpy_array_repr ?(options = default_options) nd =
  let prefix = "ArrayImpl(" in
  let dtype = Ndarray.dtype nd in
  let implied =
    Ndarray.shape nd <> [| 0 |]
    &&
    match dtype with
    | Dtype.F64 | Dtype.I64 | Dtype.Bool | Dtype.Complex128 -> true
    | _ -> false
  in
  let s =
    if size nd = 0 then "[], shape=" ^ python_shape (Ndarray.shape nd)
    else array2string ~options ~prefix ~suffix:"," ~separator:", " nd
  in
  if implied then prefix ^ s ^ ")"
  else
    let dtype_str = "dtype=" ^ Dtype.numpy_name dtype in
    let last_newline =
      match String.rindex_opt s '\n' with Some i -> i | None -> -1
    in
    let last_line_len = String.length s - last_newline + 1 in
    let sep =
      if last_line_len + String.length dtype_str + 2 > options.linewidth then
        "\n" ^ String.make (String.length prefix) ' '
      else " "
    in
    prefix ^ s ^ "," ^ sep ^ dtype_str ^ ")"

let scalar_str nd =
  let dt = Ndarray.dtype nd in
  let one v =
    let round = rounder dt in
    if not (Float.is_finite v) then
      if Float.is_nan v then "nan" else if v < 0.0 then "-inf" else "inf"
    else
      let av = Float.abs v in
      if av = 0.0 || (av >= 1e-4 && av < 1e16) then
        let i, f = shortest_positional round v in
        signed false v (i ^ "." ^ if f = "" then "0" else f)
      else
        let m, e = split_exp (shortest_sci round av) in
        let d = trim_zeros (digits_of_mantissa m) in
        let head = String.sub d 0 1 in
        let tail = String.sub d 1 (String.length d - 1) in
        signed false v
          (head ^ (if tail = "" then "" else "." ^ tail) ^ exp_string e 2)
  in
  if dt = Dtype.Bool then
    if Ndarray.get_f nd [||] <> 0.0 then "True" else "False"
  else if is_complex dt then
    let z = Ndarray.get_c nd [||] in
    let part v =
      let s = one v in
      if String.ends_with ~suffix:".0" s then
        String.sub s 0 (String.length s - 2)
      else s
    in
    let im = part z.im in
    let sign =
      if String.length im > 0 && (im.[0] = '-' || im.[0] = '+') then "" else "+"
    in
    if z.re = 0.0 && not (Float.sign_bit z.re) then im ^ "j"
    else "(" ^ part z.re ^ sign ^ im ^ "j)"
  else if is_float dt then one (Ndarray.get_f nd [||])
  else unsigned_string (Ndarray.get_i64 nd [||]) dt

let array_str ?(options = default_options) nd =
  if Array.length (Ndarray.shape nd) = 0 then scalar_str nd
  else array2string ~options nd
