type ns_elem = Scope of string | Transform of string
type name_stack = { stack : ns_elem list }

let empty_name_stack : name_stack = { stack = [] }

let new_name_stack (name : string) : name_stack =
  if name = "" then empty_name_stack else { stack = [ Scope name ] }

let extend (ns : name_stack) (name : string) : name_stack =
  { stack = ns.stack @ [ Scope name ] }

let transform (ns : name_stack) (name : string) : name_stack =
  { stack = ns.stack @ [ Transform name ] }

let name_stack_str (ns : name_stack) : string =
  let scope = ref [] in
  let append s = scope := !scope @ [ s ] in
  let modify_last f =
    match List.rev !scope with
    | last :: rest -> scope := List.rev (f last :: rest)
    | [] -> ()
  in
  List.iter
    (fun elem ->
      match elem with
      | Scope name -> append name
      | Transform name -> (
          match !scope with
          | [] -> append (Printf.sprintf "%s()" name)
          | _ -> modify_last (fun last -> Printf.sprintf "%s(%s)" name last)))
    (List.rev ns.stack);
  String.concat "/" (List.rev !scope)

type t = { name_stack : name_stack }

let new_source_info () : t = { name_stack = empty_name_stack }
let context : t ref = ref (new_source_info ())
let current () : t = !context
let name_stack (si : t) : name_stack = si.name_stack
let current_name_stack () : name_stack = (current ()).name_stack
let summarize (_ : t) : string = ""

let concat (a : name_stack) (b : name_stack) : name_stack =
  { stack = a.stack @ b.stack }

let with_name_stack (ns : name_stack) f =
  let saved = !context in
  context := { name_stack = ns };
  Fun.protect ~finally:(fun () -> context := saved) f

let transform_name_stack name f =
  with_name_stack (transform (current_name_stack ()) name) f

let reset_name_stack f = with_name_stack empty_name_stack f
let eqn_stacks : (int, name_stack) Hashtbl.t = Hashtbl.create 1024

let note_eqn_outs (vids : int list) =
  let ns = current_name_stack () in
  if ns.stack <> [] then
    List.iter (fun v -> Hashtbl.replace eqn_stacks v ns) vids

let eqn_name_stack (vids : int list) =
  match List.find_map (Hashtbl.find_opt eqn_stacks) vids with
  | Some ns -> ns
  | None -> empty_name_stack

let register_exclusion (_ : string) : unit = ()
let api_boundary (f : 'a -> 'b) : 'a -> 'b = f
