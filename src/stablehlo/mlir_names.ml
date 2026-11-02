module Names = Set.Make (String)
module Tbl = Map.Make (String)

type state = {
  next_value : int;
  next_arg : int;
  next_conflict : int;
  used : Names.t;
}

let fresh_state =
  { next_value = 0; next_arg = 0; next_conflict = 0; used = Names.empty }

type region = {
  first : int;
  last : int;
  children : (int * region) list;
  mutable names : string Tbl.t;
}

let is_token_char c =
  (c >= 'a' && c <= 'z')
  || (c >= 'A' && c <= 'Z')
  || (c >= '0' && c <= '9')
  || c = '_'

let tokens_of line =
  let n = String.length line in
  let rec go i acc =
    if i >= n then List.rev acc
    else if line.[i] = '%' then begin
      let j = ref (i + 1) in
      while !j < n && is_token_char line.[!j] do
        incr j
      done;
      go !j (String.sub line i (!j - i) :: acc)
    end
    else go (i + 1) acc
  in
  go 0 []

let starts_with prefix s =
  String.length s >= String.length prefix
  && String.sub s 0 (String.length prefix) = prefix

let ends_with suffix s =
  let n = String.length s and m = String.length suffix in
  n >= m && String.sub s (n - m) m = suffix

let contains needle s =
  let n = String.length s and m = String.length needle in
  let rec go i = i + m <= n && (String.sub s i m = needle || go (i + 1)) in
  go 0

let closes line = starts_with "}" (String.trim line)
let opens line = ends_with "{" (String.trim line)

let build_regions lines =
  let rec walk i =
    let rec loop i children =
      if i >= Array.length lines then (i, List.rev children)
      else if closes lines.(i) then (i, List.rev children)
      else if opens lines.(i) then
        let close, child = region_from i (i + 1) in
        continue_after close children child i
      else loop (i + 1) children
    and continue_after close children child owner =
      let children = (owner, child) :: children in
      if close < Array.length lines && opens lines.(close) then
        let close', child' = region_from close (close + 1) in
        continue_after close' children child' close
      else loop (close + 1) children
    in
    loop i []
  and region_from _owner first =
    let last, children = walk first in
    (last, { first; last; children; names = Tbl.empty })
  in
  region_from 0 0

let unique_name st base =
  if not (Names.mem base st.used) then
    (base, { st with used = Names.add base st.used })
  else
    let rec probe st =
      let candidate = base ^ "_" ^ string_of_int st.next_conflict in
      let st = { st with next_conflict = st.next_conflict + 1 } in
      if Names.mem candidate st.used then probe st
      else (candidate, { st with used = Names.add candidate st.used })
    in
    probe st

let assign region tok text = region.names <- Tbl.add tok text region.names

let bind_named region st tok base =
  let name, st = unique_name st base in
  assign region tok ("%" ^ name);
  st

let bind_arg region st tok =
  let st = bind_named region st tok ("arg" ^ string_of_int st.next_arg) in
  { st with next_arg = st.next_arg + 1 }

let bind_numbered region st tok =
  assign region tok ("%" ^ string_of_int st.next_value);
  { st with next_value = st.next_value + 1 }

let definition_of line =
  let t = String.trim line in
  if not (starts_with "%" t) then None
  else
    match String.index_opt t '=' with
    | Some i when i >= 2 && t.[i - 1] = ' ' -> (
        match tokens_of (String.sub t 0 i) with
        | [ tok ] -> Some (tok, String.sub t (i + 1) (String.length t - i - 1))
        | _ -> None)
    | _ -> None

let constant_base rhs =
  if not (contains "stablehlo.constant " rhs) then None
  else
    let open_at = String.rindex rhs '<' in
    let elt = String.sub rhs (open_at + 1) (String.length rhs - open_at - 1) in
    let elt =
      match String.rindex_opt elt 'x' with
      | Some i
        when (not (starts_with "complex" elt))
             && i + 1 < String.length elt
             && String.for_all
                  (fun c -> (c >= '0' && c <= '9') || c = 'x')
                  (String.sub elt 0 i) ->
          String.sub elt (i + 1) (String.length elt - i - 1)
      | _ -> elt
    in
    let integer =
      (starts_with "i" elt || starts_with "ui" elt)
      && not (starts_with "index" elt)
    in
    if integer then Some "c" else Some "cst"

let block_args line =
  let t = String.trim line in
  if starts_with "^bb0(" t then tokens_of t else []

let while_args line =
  if contains "stablehlo.while(" line then (
    let at = ref 0 in
    let n = String.length line in
    let key = "stablehlo.while(" in
    let m = String.length key in
    while !at + m <= n && String.sub line !at m <> key do
      incr at
    done;
    let open_at = !at + m in
    let close_at = String.index_from line open_at ')' in
    String.split_on_char ',' (String.sub line open_at (close_at - open_at))
    |> List.filter_map (fun part ->
        match tokens_of part with first :: _ -> Some first | [] -> None))
  else []

let func_args line =
  let t = String.trim line in
  if starts_with "func.func " t then
    let close_at =
      let depth = ref 0 and result = ref (-1) in
      String.iteri
        (fun i c ->
          if !result < 0 then
            if c = '(' then incr depth
            else if c = ')' then begin
              decr depth;
              if !depth = 0 then result := i
            end)
        t;
      !result
    in
    tokens_of (String.sub t 0 (max close_at 0))
  else []

let reducer_args line =
  if starts_with "reducer(" (String.trim line) then
    let tokens = tokens_of line in
    let at parity = List.filteri (fun i _ -> i mod 2 = parity) tokens in
    at 0 @ at 1
  else []

let top_k_results line =
  if contains "chlo.top_k(" line then
    match tokens_of line with
    | values :: indices :: _ -> Some (values, indices)
    | _ -> None
  else None

let manual_computation_args line =
  if contains "sdy.manual_computation(" line then begin
    let key = "manual_axes=" in
    let n = String.length line and m = String.length key in
    let rec find i =
      if i + m > n then None
      else if String.sub line i m = key then Some i
      else find (i + 1)
    in
    match find 0 with
    | Some at -> tokens_of (String.sub line at (n - at))
    | None -> []
  end
  else []

let rng_bit_generator_results line =
  if contains "stablehlo.rng_bit_generator " line then
    match tokens_of line with
    | state :: output :: _ -> Some (state, output)
    | _ -> None
  else None

let rec nearest_while lines i =
  if i < 0 then None
  else if contains "stablehlo.while(" lines.(i) then Some i
  else nearest_while lines (i - 1)

let is_while_body line =
  let t = String.trim line in
  t = "cond {" || t = "} do {"

let rec number lines region st =
  let st = ref st in
  if region.first < Array.length lines then
    List.iter
      (fun tok -> st := bind_arg region !st tok)
      (block_args lines.(region.first));
  let spans = List.map (fun (_, r) -> (r.first, r.last)) region.children in
  let in_child i = List.exists (fun (a, b) -> i >= a && i < b) spans in
  for i = region.first to min region.last (Array.length lines) - 1 do
    if not (in_child i) then
      match top_k_results lines.(i) with
      | Some (values, indices) ->
          st := bind_named region !st values "values";
          st := bind_named region !st indices "indices"
      | None -> (
          match rng_bit_generator_results lines.(i) with
          | Some (state, output) ->
              st := bind_named region !st state "output_state";
              st := bind_named region !st output "output"
          | None -> (
              match definition_of lines.(i) with
              | Some (tok, rhs) -> (
                  match constant_base rhs with
                  | Some base -> st := bind_named region !st tok base
                  | None -> st := bind_numbered region !st tok)
              | None -> ()))
  done;
  let finished = !st in
  List.iter
    (fun (owner, child) ->
      let line = lines.(owner) in
      let start =
        if starts_with "func.func " (String.trim line) then
          List.fold_left (bind_arg child) fresh_state (func_args line)
        else if is_while_body line then
          match nearest_while lines owner with
          | Some header ->
              List.fold_left
                (fun acc tok -> bind_named child acc tok "iterArg")
                finished
                (while_args lines.(header))
          | None -> finished
        else if contains "sdy.manual_computation(" line then
          List.fold_left (bind_arg child) finished
            (manual_computation_args line)
        else List.fold_left (bind_arg child) finished (reducer_args line)
      in
      number lines child start)
    region.children

let rewrite env line =
  let b = Buffer.create (String.length line) in
  let n = String.length line in
  let i = ref 0 in
  while !i < n do
    if line.[!i] = '%' then begin
      let j = ref (!i + 1) in
      while !j < n && is_token_char line.[!j] do
        incr j
      done;
      let tok = String.sub line !i (!j - !i) in
      Buffer.add_string b
        (match Tbl.find_opt tok env with Some s -> s | None -> tok);
      i := !j
    end
    else begin
      Buffer.add_char b line.[!i];
      incr i
    end
  done;
  Buffer.contents b

let overlay base extra = Tbl.union (fun _ _ x -> Some x) base extra

let rec apply lines out region env =
  let env = overlay env region.names in
  let spans = List.map (fun (_, r) -> (r.first, r.last)) region.children in
  let in_child i = List.exists (fun (a, b) -> i >= a && i < b) spans in
  let first_while_child =
    List.find_map (fun (owner, child) ->
        if String.trim lines.(owner) = "cond {" then Some (owner, child)
        else None)
  in
  let cond_of_header =
    let table = Hashtbl.create 2 in
    let rec go = function
      | [] -> ()
      | children -> (
          match first_while_child children with
          | Some (owner, child) ->
              (match nearest_while lines owner with
              | Some header -> Hashtbl.replace table header child.names
              | None -> ());
              go (List.filter (fun (o, _) -> o <> owner) children)
          | None -> ())
    in
    go region.children;
    table
  in
  for i = region.first to min region.last (Array.length lines) - 1 do
    if not (in_child i) then
      out.(i) <-
        (match Hashtbl.find_opt cond_of_header i with
        | Some names -> rewrite (overlay env names) lines.(i)
        | None -> rewrite env lines.(i))
  done;
  List.iter
    (fun (owner, child) ->
      apply lines out child env;
      let t = String.trim lines.(owner) in
      if
        starts_with "func.func " t || starts_with "reducer(" t
        || contains "sdy.manual_computation(" t
      then out.(owner) <- rewrite (overlay env child.names) lines.(owner))
    region.children

let split_results body =
  let parts = ref [] and depth = ref 0 and start = ref 0 in
  String.iteri
    (fun i c ->
      match c with
      | '<' | '{' | '[' | '(' -> incr depth
      | '>' | '}' | ']' | ')' -> decr depth
      | ',' when !depth = 0 ->
          parts := String.sub body !start (i - !start) :: !parts;
          start := i + 1
      | _ -> ())
    body;
  parts := String.sub body !start (String.length body - !start) :: !parts;
  List.rev_map String.trim !parts

let add_result_info ?names line =
  let t = String.trim line in
  if not (starts_with "func.func public @main(" t) then line
  else
    let key = ") -> (" in
    let n = String.length line and m = String.length key in
    let rec find i =
      if i + m > n then None
      else if String.sub line i m = key then Some i
      else find (i + 1)
    in
    match find 0 with
    | None -> line
    | Some at ->
        let start = at + m in
        let stop = String.rindex line ')' in
        let body = String.sub line start (stop - start) in
        if body = "" then
          String.sub line 0 at ^ ")" ^ String.sub line (stop + 1) (n - stop - 1)
        else
          let types = split_results body in
          let is_token ty = starts_with "!stablehlo.token" ty in
          let many =
            List.length (List.filter (fun ty -> not (is_token ty)) types) > 1
          in
          let tokens = List.length (List.filter is_token types) in
          let annotated =
            List.mapi
              (fun i ty ->
                if is_token ty then ty
                else
                  let i = i - tokens in
                  let label =
                    match names with
                    | Some names when i < List.length names -> List.nth names i
                    | _ ->
                        if many then Printf.sprintf "result[%d]" i else "result"
                  in
                  let key = " {" in
                  let k = String.length ty and m = String.length key in
                  let rec brace j =
                    if j + m > k then None
                    else if String.sub ty j m = key then Some j
                    else brace (j + 1)
                  in
                  match brace 0 with
                  | Some j ->
                      Printf.sprintf "%s {jax.result_info = \"%s\", %s"
                        (String.sub ty 0 j) label
                        (String.sub ty (j + m) (k - j - m))
                  | None ->
                      Printf.sprintf "%s {jax.result_info = \"%s\"}" ty label)
              types
          in
          String.sub line 0 start
          ^ String.concat ", " annotated
          ^ String.sub line stop (n - stop)

let find_from line key from =
  let n = String.length line and m = String.length key in
  let rec go i =
    if i + m > n then None
    else if String.sub line i m = key then Some i
    else go (i + 1)
  in
  go from

let merge_attribute arg attr =
  match find_from arg " {" 0 with
  | None -> arg ^ " {" ^ attr ^ "}"
  | Some j ->
      let inner = String.sub arg (j + 2) (String.length arg - j - 3) in
      let attrs = List.sort compare (attr :: split_results inner) in
      String.sub arg 0 j ^ " {" ^ String.concat ", " attrs ^ "}"

let add_arg_attributes attribute_of line =
  let t = String.trim line in
  if not (starts_with "func.func public @main(" t) then line
  else
    match (find_from line "@main(" 0, find_from line ") -> (" 0) with
    | Some open_at, Some close_at ->
        let start = open_at + String.length "@main(" in
        let body = String.sub line start (close_at - start) in
        if String.trim body = "" then line
        else
          let parts = split_results body in
          let is_token arg = contains ": !stablehlo.token" arg in
          let tokens = List.length (List.filter is_token parts) in
          let args =
            List.mapi
              (fun i arg ->
                if is_token arg then arg
                else
                  match attribute_of (i - tokens) with
                  | Some attr -> merge_attribute arg attr
                  | None -> arg)
              parts
          in
          String.sub line 0 start ^ String.concat ", " args
          ^ String.sub line close_at (String.length line - close_at)
    | _ -> line

let symbol_name name =
  let bare = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '_' | '$' | '.' -> true
    | _ -> false
  in
  if String.for_all bare name then name else "\"" ^ name ^ "\""

let splat_index_layouts line =
  let open_key = "dense<[" and close_key = "]> : tensor<1xindex>" in
  let b = Buffer.create (String.length line) in
  let n = String.length line in
  let m = String.length open_key and c = String.length close_key in
  let rec go i =
    if i >= n then ()
    else if i + m <= n && String.sub line i m = open_key then begin
      let j = ref (i + m) in
      while !j < n && line.[!j] >= '0' && line.[!j] <= '9' do
        incr j
      done;
      if !j + c <= n && String.sub line !j c = close_key then begin
        Buffer.add_string b "dense<";
        Buffer.add_string b (String.sub line (i + m) (!j - i - m));
        Buffer.add_string b "> : tensor<1xindex>";
        go (!j + c)
      end
      else begin
        Buffer.add_char b line.[i];
        go (i + 1)
      end
    end
    else begin
      Buffer.add_char b line.[i];
      go (i + 1)
    end
  in
  go 0;
  Buffer.contents b

let renumber_callbacks text =
  let key = "mhlo.backend_config = {index = " in
  let k = String.length key in
  let seen = Hashtbl.create 4 in
  let line_of line =
    if not (contains "@xla_ffi_python_cpu_callback(" line) then line
    else
      let n = String.length line in
      let rec find i =
        if i + k > n then None
        else if String.sub line i k = key then Some (i + k)
        else find (i + 1)
      in
      match find 0 with
      | None -> line
      | Some start ->
          let stop = ref start in
          while !stop < n && line.[!stop] >= '0' && line.[!stop] <= '9' do
            incr stop
          done;
          let index = Hashtbl.length seen in
          Hashtbl.replace seen index ();
          String.sub line 0 start ^ string_of_int index
          ^ String.sub line !stop (n - !stop)
  in
  String.concat "\n" (List.map line_of (String.split_on_char '\n' text))

let constant_of line =
  let t = String.trim line in
  if contains "stablehlo.constant " t then
    match definition_of line with
    | Some (tok, rhs) -> Some (tok, String.trim rhs)
    | None -> None
  else None

let opens_isolated line =
  let t = String.trim line in
  starts_with "reducer(" t
  || ends_with "({" t
     && (not (contains "stablehlo.case" t))
     && not (contains "stablehlo.if" t)

let replace_first ~sub ~by line =
  let n = String.length line and m = String.length sub in
  let rec find i =
    if i + m > n then None
    else if String.sub line i m = sub then Some i
    else find (i + 1)
  in
  match find 0 with
  | None -> None
  | Some i ->
      Some (String.sub line 0 i ^ by ^ String.sub line (i + m) (n - i - m))

let print_empty_dynamic_slices text =
  String.split_on_char '\n' text
  |> List.map (fun line ->
      if contains "stablehlo.dynamic_slice" line then
        Option.value ~default:line
          (replace_first ~sub:", sizes = [] :" ~by:", , sizes = [] :" line)
      else line)
  |> String.concat "\n"

let add_platform_index_arg ~arg text =
  let param = Printf.sprintf "%%%d" arg in
  let scalar = "tensor<i32>" in
  let attr =
    Printf.sprintf "%s: %s {jax.global_constant = \"_platform_index\"}" param
      scalar
  in
  let first_paren line =
    let i = String.index line '(' in
    ( String.sub line 0 (i + 1),
      String.sub line (i + 1) (String.length line - i - 1) )
  in
  let with_first x rest =
    if String.length rest > 0 && rest.[0] = ')' then x ^ rest
    else x ^ ", " ^ rest
  in
  let rewrite line =
    let t = String.trim line in
    if starts_with "func.func " t then
      let head, rest = first_paren line in
      head ^ with_first attr rest
    else
      let is_call =
        starts_with "call @" t
        || starts_with "func.call @" t
        || contains " = call @" t
        || contains " = func.call @" t
      in
      if not is_call then line
      else
        let head, rest = first_paren line in
        let line = head ^ with_first param rest in
        match replace_first ~sub:" : (" ~by:" : (\000" line with
        | None -> line
        | Some marked ->
            let i = String.index marked '\000' in
            let before = String.sub marked 0 i in
            let after =
              String.sub marked (i + 1) (String.length marked - i - 1)
            in
            before ^ with_first scalar after
  in
  String.concat "\n" (List.map rewrite (String.split_on_char '\n' text))

let mentions_host_callback text = contains "@xla_ffi_python_cpu_callback" text

let drop_empty_backend_config text =
  let sub = "{backend_config = \"\", " in
  String.concat "\n"
    (List.map
       (fun line ->
         if contains "stablehlo.custom_call" line then
           Option.value (replace_first ~sub ~by:"{" line) ~default:line
         else line)
       (String.split_on_char '\n' text))

let hoist_constants text =
  let lines = Array.of_list (String.split_on_char '\n' text) in
  let n = Array.length lines in
  let rec scope i =
    let kept = ref [] and constants = ref [] and renames = ref Tbl.empty in
    let depth = ref 0 in
    let at_head = ref true in
    let rec walk i =
      if i >= n then i
      else
        let line = lines.(i) in
        match constant_of line with
        | Some (tok, rhs) ->
            (match List.assoc_opt rhs !constants with
            | Some first -> renames := Tbl.add tok first !renames
            | None ->
                if !at_head && !depth = 0 then
                  constants := !constants @ [ (rhs, tok) ]
                else constants := (rhs, tok) :: !constants);
            walk (i + 1)
        | None ->
            at_head := false;
            if closes line && not (opens line) then
              if !depth = 0 then i
              else begin
                decr depth;
                kept := line :: !kept;
                walk (i + 1)
              end
            else if opens_isolated line then begin
              kept := line :: !kept;
              let header, i' =
                if i + 1 < n && starts_with "^bb0(" (String.trim lines.(i + 1))
                then ([ lines.(i + 1) ], i + 2)
                else ([], i + 1)
              in
              kept := List.rev_append header !kept;
              let inner, next = scope i' in
              kept := List.rev_append inner !kept;
              walk next
            end
            else if closes line then begin
              kept := line :: !kept;
              walk (i + 1)
            end
            else begin
              if opens line then incr depth;
              kept := line :: !kept;
              walk (i + 1)
            end
    in
    let next = walk i in
    let body = List.rev !kept in
    let indent =
      match body with
      | first :: _ ->
          let k = ref 0 in
          while !k < String.length first && first.[!k] = ' ' do
            incr k
          done;
          String.make !k ' '
      | [] -> "    "
    in
    let hoisted =
      List.map
        (fun (rhs, tok) -> Printf.sprintf "%s%s = %s" indent tok rhs)
        !constants
    in
    (List.map (rewrite !renames) (hoisted @ body), next)
  in
  let out = ref [] in
  let i = ref 0 in
  while !i < n do
    let line = lines.(!i) in
    out := line :: !out;
    incr i;
    if starts_with "func.func " (String.trim line) then begin
      let inner, next = scope !i in
      out := List.rev_append inner !out;
      i := next
    end
  done;
  String.concat "\n" (List.rev !out)

let render ?(num_partitions = 1) ?(module_attributes = []) ?result_names
    ?(arg_attribute = fun _ -> None) ~module_name text =
  let lines = Array.of_list (String.split_on_char '\n' text) in
  let root = snd (build_regions lines) in
  number lines root fresh_state;
  let out = Array.copy lines in
  apply lines out root Tbl.empty;
  let out =
    Array.map
      (fun l ->
        splat_index_layouts
          (add_arg_attributes arg_attribute
             (add_result_info ?names:result_names l)))
      out
  in
  if Array.length out > 0 && String.trim out.(0) = "module {" then
    out.(0) <-
      Printf.sprintf
        "module @%s attributes {%smhlo.num_partitions = %d : i32, \
         mhlo.num_replicas = 1 : i32} {"
        (symbol_name ("jit_" ^ module_name))
        (String.concat "" (List.map (fun a -> a ^ ", ") module_attributes))
        num_partitions;
  String.concat "\n" (Array.to_list out)
