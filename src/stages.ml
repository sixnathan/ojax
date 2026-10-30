open Types

type lowered = {
  closed : closed_jaxpr;
  in_avals : aval list;
  out_avals : aval list;
}

type compiled = { run : value list -> value list; source : lowered }

let lower f avals =
  let closed = Jaxpr.make_jaxpr avals f in
  let out_avals =
    List.map
      (function
        | A_var v -> v.vaval
        | A_lit nd ->
            {
              shape = Ndarray.shape nd;
              dtype = Ndarray.dtype nd;
              weak_type = false;
            }
        | DropVar a -> a)
      closed.jaxpr.outs
  in
  { closed; in_avals = avals; out_avals }

let as_text lowered = Stablehlo.Emit.emit_closed_jaxpr lowered.closed
let jaxpr_text lowered = Pretty_printer.closed_jaxpr_to_string lowered.closed

let compile lowered =
  { run = Backend.executor lowered.closed; source = lowered }

let call compiled args =
  let actual = List.map Core.get_aval args in
  List.iter2
    (fun (a : aval) (e : aval) ->
      if a.shape <> e.shape || a.dtype <> e.dtype then
        invalid_arg
          "Argument types differ from the types the function was compiled with")
    actual compiled.source.in_avals;
  compiled.run args

let cost_analysis _ = []
