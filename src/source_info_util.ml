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
let register_exclusion (_ : string) : unit = ()
let api_boundary (f : 'a -> 'b) : 'a -> 'b = f
