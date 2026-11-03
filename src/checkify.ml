module T = Types
module C = Core
module D = Dtype
module U = Numpy.Ufuncs

type error_category = Failed_check | Nan | Oob | Division_by_zero

let user_checks = [ Failed_check ]
let nan_checks = [ Nan ]
let index_checks = [ Oob ]
let div_checks = [ Division_by_zero ]
let float_checks = nan_checks @ div_checks
let automatic_checks = float_checks @ index_checks
let all_checks = automatic_checks @ user_checks

type eff = {
  category : error_category;
  shape_dtypes : (string * int array) list;
}

type exn_meta =
  | Failed_check_meta of {
      fmt_string : string;
      nargs : int;
      kwargs : string list;
    }

type entry = {
  eff : eff;
  pred : T.value;
  code : T.value;
  payload : T.value list;
}

type error = { entries : entry list; metadata : (int * exn_meta) list }

let init_error = { entries = []; metadata = [] }
let counter = ref 0

let next_code () =
  incr counter;
  !counter

let reset_codes () = counter := 0

let class_string = function
  | Failed_check -> "<class 'jax._src.checkify.FailedCheckError'>"
  | Nan -> "<class 'jax._src.checkify.NaNError'>"
  | Oob -> "<class 'jax._src.checkify.OOBError'>"
  | Division_by_zero -> "<class 'jax._src.checkify.DivisionByZeroError'>"

let compare_effect a b =
  compare
    (class_string a.category, a.shape_dtypes)
    (class_string b.category, b.shape_dtypes)

let aval v = C.get_aval v
let py_false () = C.py_scalar D.Bool 0.0
let py_int i = Numpy.Jnp_util.py_int i

let update_error error pred code metadata payload eff =
  let cur =
    List.find_opt (fun e -> compare_effect e.eff eff = 0) error.entries
  in
  let err_of_type = match cur with Some e -> e.pred | None -> py_false () in
  let out_err = U.bitwise_or err_of_type pred in
  let old_code = match cur with Some e -> e.code | None -> py_int (-1) in
  let out_code = C.bind1 T.Select_n [ err_of_type; code; old_code ] in
  let out_payload =
    match cur with
    | Some e ->
        List.map2
          (fun c p -> C.bind1 T.Select_n [ err_of_type; p; c ])
          e.payload payload
    | None -> payload
  in
  let entry = { eff; pred = out_err; code = out_code; payload = out_payload } in
  let others =
    List.filter (fun e -> compare_effect e.eff eff <> 0) error.entries
  in
  {
    entries = List.sort (fun a b -> compare_effect a.eff b.eff) (entry :: others);
    metadata = error.metadata @ metadata;
  }

let effect_of_payload category payload =
  {
    category;
    shape_dtypes =
      List.map
        (fun v ->
          let a = aval v in
          (D.numpy_name a.T.dtype, a.T.shape))
        payload;
  }

let assert_func error pred meta payload =
  let code = next_code () in
  let eff = effect_of_payload Failed_check payload in
  update_error error pred (py_int code) [ (code, meta) ] payload eff

type scope = { enabled : error_category list; mutable current : error }

let scopes : scope list ref = ref []

let error_leaves error =
  List.map (fun e -> e.pred) error.entries
  @ List.map (fun e -> e.code) error.entries
  @ List.concat_map (fun e -> e.payload) error.entries

let host v =
  match v with
  | T.Concrete nd -> nd
  | T.Device b -> Pjrt.Buffer.to_host b
  | T.Tracer _ -> failwith "checkify: traced error value"

let truthy v = Ndarray.get_f (host v) [||] <> 0.0
let int_of v = int_of_float (Ndarray.get_f (host v) [||])

let format fmt args kwargs =
  let b = Buffer.create 64 in
  let n = String.length fmt in
  let auto = ref 0 in
  let i = ref 0 in
  while !i < n do
    let c = fmt.[!i] in
    if c = '{' && !i + 1 < n && fmt.[!i + 1] = '{' then begin
      Buffer.add_char b '{';
      i := !i + 2
    end
    else if c = '}' && !i + 1 < n && fmt.[!i + 1] = '}' then begin
      Buffer.add_char b '}';
      i := !i + 2
    end
    else if c = '{' then begin
      let j = String.index_from fmt !i '}' in
      let field = String.sub fmt (!i + 1) (j - !i - 1) in
      let v =
        if field = "" then begin
          let v = List.nth args !auto in
          incr auto;
          v
        end
        else
          match int_of_string_opt field with
          | Some k -> List.nth args k
          | None -> List.assoc field kwargs
      in
      Buffer.add_string b (Numpy.Array_print.array_str (host v));
      i := j + 1
    end
    else begin
      Buffer.add_char b c;
      incr i
    end
  done;
  Buffer.contents b

let message_of meta payload =
  match meta with
  | Failed_check_meta { fmt_string; nargs; kwargs } ->
      let args = List.filteri (fun i _ -> i < nargs) payload in
      let kw = List.filteri (fun i _ -> i >= nargs) payload in
      format fmt_string args (List.combine kwargs kw) ^ " (`check` failed)"

let get error =
  let best =
    List.fold_left
      (fun acc e ->
        if truthy e.pred then
          let code = int_of e.code in
          match acc with
          | Some (c, _) when c <= code -> acc
          | _ -> Some (code, e)
        else acc)
      None error.entries
  in
  match best with
  | None -> None
  | Some (code, e) ->
      Some (message_of (List.assoc code error.metadata) e.payload)

let throw error =
  match get error with
  | Some m -> raise (Py_exn.Jax_runtime_error m)
  | None -> ()

let check_ids : (int, unit) Hashtbl.t = Hashtbl.create 4

let check_error error =
  let leaves = error_leaves error in
  let host_fun args =
    let n = List.length error.entries in
    let rec take k l =
      if k = 0 then [] else List.hd l :: take (k - 1) (List.tl l)
    in
    let rec drop k l = if k = 0 then l else drop (k - 1) (List.tl l) in
    let preds = take n args and codes = take n (drop n args) in
    let rest = ref (drop (2 * n) args) in
    let entries =
      List.map2
        (fun e (p, c) ->
          let k = List.length e.payload in
          let pl = take k !rest in
          rest := drop k !rest;
          { e with pred = p; code = c; payload = pl })
        error.entries (List.combine preds codes)
    in
    (match get { error with entries } with
    | Some m -> raise (Py_exn.Jax_runtime_error m)
    | None -> ());
    []
  in
  ignore
    (Custom_derivatives.callback_with_view ~ordered:false "check"
       [ ("debug", "False") ]
       host_fun ~result_avals:[] leaves)

let is_scalar_pred v =
  let a = aval v in
  a.T.shape = [||] && a.T.dtype = D.Bool

let check ?(kwargs = []) pred msg fmt_args =
  if not (is_scalar_pred pred) then
    Py_exn.type_error
      (Printf.sprintf "check takes a scalar pred as argument, got %s"
         (Numpy.Extras.array_str pred));
  let kwargs = List.stable_sort (fun (a, _) (b, _) -> compare a b) kwargs in
  let payload = fmt_args @ List.map snd kwargs in
  let meta =
    Failed_check_meta
      {
        fmt_string = msg;
        nargs = List.length fmt_args;
        kwargs = List.map fst kwargs;
      }
  in
  let error = assert_func init_error (U.logical_not pred) meta payload in
  match !scopes with
  | scope :: _ ->
      let discharged =
        List.fold_left
          (fun acc e ->
            if List.mem e.eff.category scope.enabled then
              update_error acc e.pred e.code [] e.payload e.eff
            else acc)
          scope.current error.entries
      in
      scope.current <-
        { discharged with metadata = error.metadata @ discharged.metadata }
  | [] -> check_error error

let with_leaves error leaves =
  let n = List.length error.entries in
  let rec take k l =
    if k = 0 then [] else List.hd l :: take (k - 1) (List.tl l)
  in
  let rec drop k l = if k = 0 then l else drop (k - 1) (List.tl l) in
  let preds = take n leaves and codes = take n (drop n leaves) in
  let rest = ref (drop (2 * n) leaves) in
  let entries =
    List.map2
      (fun e (p, c) ->
        let k = List.length e.payload in
        let pl = take k !rest in
        rest := drop k !rest;
        { e with pred = p; code = c; payload = pl })
      error.entries (List.combine preds codes)
  in
  { error with entries }

let checkify ?(errors = user_checks) f args =
  let scope = { enabled = errors; current = init_error } in
  scopes := scope :: !scopes;
  let outs =
    Fun.protect
      ~finally:(fun () -> scopes := List.tl !scopes)
      (fun () -> f args)
  in
  (scope.current, outs)
