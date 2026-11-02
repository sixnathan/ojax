module U = Yojson.Safe.Util
module T = Ojax.Types
module TR = Ojax.Tree_util
module F = Ojax.Numpy.Ufuncs
module NL = Ojax.Numpy.Lax_numpy
module A = Ojax.Api
module P = Ojax.Parallel

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type result = {
  kind : string;
  cls : string;
  message : string;
  text : string;
  stdout : string;
  warnings : (string * string) list;
}

let float_dt () =
  if Ojax.Config.x64_enabled () then Ojax.Dtype.F64 else Ojax.Dtype.F32

let int_dt () =
  if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32

let py_float x = Ojax.Core.py_scalar (float_dt ()) x

let run_case thunk =
  let out = Buffer.create 64 in
  let saved = !Ojax.Debugging.print_sink in
  (Ojax.Debugging.print_sink :=
     fun s ->
       Buffer.add_string out s;
       Buffer.add_char out '\n');
  let outcome, warnings =
    Fun.protect
      ~finally:(fun () -> Ojax.Debugging.print_sink := saved)
      (fun () ->
        Ojax.Py_warnings.catch (fun () ->
            match thunk () with t -> Ok t | exception e -> Error e))
  in
  let base =
    {
      kind = "ok";
      cls = "";
      message = "";
      text = "";
      stdout = Buffer.contents out;
      warnings;
    }
  in
  match outcome with
  | Ok text -> { base with text }
  | Error e -> (
      match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
      | Some cls, Some message -> { base with kind = "raise"; cls; message }
      | _ ->
          {
            base with
            kind = "raise";
            cls = "OCaml";
            message = Printexc.to_string e;
          })

let text_of v =
  let nd =
    match v with
    | T.Concrete nd -> nd
    | T.Device b -> Ojax.Pjrt.Buffer.to_host b
    | T.Tracer _ -> failwith "vmap_coll: tracer result"
  in
  Printf.sprintf "%s [%s] %s"
    (Ojax.Dtype.numpy_name (Ojax.Ndarray.dtype nd))
    (String.concat ", "
       (List.map string_of_int (Array.to_list (Ojax.Ndarray.shape nd))))
    (Parity_cases.hex_of_nd nd)

let leaves t = TR.tree_leaves t
let tuple l = TR.Tuple (List.map (fun v -> TR.Leaf v) l)

let one = function
  | [ TR.Leaf v ] -> v
  | _ -> failwith "vmap_coll: expected one leaf"

let two = function
  | [ TR.Leaf a; TR.Leaf b ] -> (a, b)
  | _ -> failwith "vmap_coll: expected two leaves"

let lf = function TR.Leaf v -> v | _ -> failwith "vmap_coll: expected a leaf"
let sub v = F.subtract v (P.psum v ~axis_name:"i")
let pmax v = P.pmax v ~axis_name:"i"
let pmin v = P.pmin v ~axis_name:"i"

let mapped ?(in_axes = [ Some 0 ]) ?out_axes body args =
  lf
    (A.vmap ~axis_name:"i" ?out_axes
       (fun xs -> TR.Leaf (body (one xs)))
       in_axes args)

let double vs =
  List.map
    (fun v ->
      F.multiply v (Ojax.Core.py_scalar (Ojax.Core.get_aval v).T.dtype 2.0))
    vs

let io ordered v =
  let aval = Ojax.Core.get_aval v in
  match
    Ojax.Jax_top.io_callback ~ordered double
      ~result_avals:[ { aval with T.weak_type = false } ]
      [ v ]
  with
  | [ r ] -> r
  | _ -> failwith "vmap_coll: io_callback arity"

let x () = NL.arange ~dtype:(float_dt ()) 3.0
let m () = NL.reshape (NL.arange ~dtype:(float_dt ()) 6.0) [| 2; 3 |]
let row1 () = Ojax.Numpy.Getitem.take (m ()) [ Ojax.Numpy.Getitem.Int 1 ]
let sum v = Ojax.Numpy.Reductions.sum v
let add1 v = F.add v (py_float 1.0)

let nested name f v =
  match
    A.jit_flat ~name (function [ b ] -> [ f b ] | _ -> assert false) [ v ]
  with
  | [ r ] -> r
  | _ -> failwith "vmap_coll: nested arity"

let case_fn name =
  let lam f args = ("_lambda", f, args) in
  match name with
  | "psum_sub" -> lam (fun a -> tuple [ mapped sub a ]) [ x () ]
  | "psum_in1" ->
      lam (fun a -> tuple [ mapped ~in_axes:[ Some 1 ] sub a ]) [ m () ]
  | "psum_out1" -> lam (fun a -> tuple [ mapped ~out_axes:1 sub a ]) [ m () ]
  | "pmax" -> lam (fun a -> tuple [ mapped pmax a ]) [ x () ]
  | "pmin_in1" ->
      lam (fun a -> tuple [ mapped ~in_axes:[ Some 1 ] pmin a ]) [ m () ]
  | "psum_int" ->
      lam
        (fun a -> tuple [ mapped (fun v -> P.psum v ~axis_name:"i") a ])
        [ NL.arange ~dtype:(int_dt ()) 4.0 ]
  | "psum_grad" ->
      lam
        (fun a ->
          tuple [ lf (A.grad (fun b -> TR.Leaf (sum (mapped sub b))) a) ])
        [ x () ]
  | "psum_jvp" ->
      lam
        (fun a ->
          let p, t =
            A.jvp
              (fun b -> TR.Leaf (mapped sub b))
              a
              [ TR.Leaf (F.multiply (one a) (py_float 2.0)) ]
          in
          tuple [ lf p; lf t ])
        [ x () ]
  | "psum_nested" ->
      lam
        (fun a ->
          tuple [ lf (A.vmap (fun r -> TR.Leaf (mapped sub r)) [ Some 0 ] a) ])
        [ m () ]
  | "pmax_nested" ->
      lam
        (fun a ->
          tuple [ lf (A.vmap (fun r -> TR.Leaf (mapped pmax r)) [ Some 0 ] a) ])
        [ m () ]
  | "psum_unmapped" ->
      lam
        (fun a ->
          tuple
            [
              lf
                (A.vmap ~axis_name:"i"
                   (fun vs ->
                     let v, w = two vs in
                     TR.Leaf (F.multiply v (P.psum w ~axis_name:"i")))
                   [ Some 0; None ] a);
            ])
        [ x (); row1 () ]
  | "pmax_unmapped" ->
      lam
        (fun a ->
          tuple
            [
              lf
                (A.vmap ~axis_name:"i"
                   (fun vs ->
                     let v, w = two vs in
                     TR.Leaf (F.add v (pmax w)))
                   [ Some 0; None ] a);
            ])
        [ x (); row1 () ]
  | "psum_vjp" ->
      lam
        (fun a ->
          let b, c = two a in
          let _, pull =
            A.vjp (fun xs -> TR.Leaf (mapped sub xs)) [ TR.Leaf b ]
          in
          tuple [ lf (List.hd (pull (TR.Leaf c))) ])
        [ x (); add1 (x ()) ]
  | "psum_grad_in1" ->
      lam
        (fun a ->
          tuple
            [
              lf
                (A.grad
                   (fun b ->
                     let y = mapped ~in_axes:[ Some 1 ] sub b in
                     TR.Leaf
                       (sum (F.power y (Ojax.Core.py_scalar (int_dt ()) 2.0))))
                   a);
            ])
        [ m () ]
  | "pmax_grad" ->
      lam
        (fun a ->
          tuple [ lf (A.grad (fun b -> TR.Leaf (sum (mapped pmax b))) a) ])
        [ x () ]
  | "pmin_jvp" ->
      lam
        (fun a ->
          let p, t = A.jvp (fun b -> TR.Leaf (mapped pmin b)) a a in
          tuple [ lf p; lf t ])
        [ x () ]
  | "io_ordered" -> lam (fun a -> tuple [ io true (one a) ]) [ m () ]
  | "io_ordered_two" ->
      lam (fun a -> tuple [ io true (add1 (io true (one a))) ]) [ x () ]
  | "io_ordered_multi" ->
      lam
        (fun a ->
          let b, c = two a in
          let r1 = io true b in
          tuple [ r1; io true c ])
        [ x (); m () ]
  | "io_unordered" ->
      lam
        (fun a -> tuple [ F.multiply (io false (one a)) (py_float 3.0) ])
        [ m () ]
  | "dprint_ordered_two" ->
      ( "f",
        (fun a ->
          let v = one a in
          Ojax.Debugging.print ~ordered:true "a {}" [ v ];
          Ojax.Debugging.print ~ordered:true "b {}" [ add1 v ];
          tuple [ F.multiply v (py_float 2.0) ]),
        [ x () ] )
  | "io_ordered_noout" ->
      ( "g",
        (fun a ->
          let v = one a in
          ignore
            (Ojax.Jax_top.io_callback ~ordered:true
               (fun _ -> [])
               ~result_avals:[] [ v ]);
          tuple [ add1 v ]),
        [ x () ] )
  | "dprint_then_io" ->
      ( "h",
        (fun a ->
          let v = one a in
          Ojax.Debugging.print ~ordered:true "a {}" [ v ];
          tuple [ io true v ]),
        [ x () ] )
  | "io_then_dprint" ->
      ( "k",
        (fun a ->
          let r = io true (one a) in
          Ojax.Debugging.print ~ordered:true "b {}" [ r ];
          tuple [ r ]),
        [ x () ] )
  | "io_nested_jit" ->
      lam
        (fun a -> tuple [ add1 (nested "inner" (fun v -> io true v) (one a)) ])
        [ x () ]
  | "io_nested_twice" ->
      lam
        (fun a ->
          let inner = function
            | [ b ] -> [ io true b ]
            | _ -> failwith "vmap_coll: inner arity"
          in
          let jitted = A.jit_flat ~name:"inner" inner in
          let f v =
            match jitted [ v ] with
            | [ r ] -> r
            | _ -> failwith "vmap_coll: inner arity"
          in
          tuple [ f (f (one a)) ])
        [ x () ]
  | "dprint_nested_jit" ->
      lam
        (fun a ->
          let show v =
            Ojax.Debugging.print ~ordered:true "in {}" [ v ];
            F.multiply v (py_float 2.0)
          in
          tuple [ add1 (nested "show" show (one a)) ])
        [ x () ]
  | "mixed_io_nested" ->
      lam
        (fun a ->
          let both v =
            Ojax.Debugging.print ~ordered:true "both {}" [ v ];
            io true v
          in
          let v = one a in
          let r = nested "both" both v in
          tuple [ r; io true v ])
        [ x () ]
  | _ -> failwith ("vmap_coll: unknown case " ^ name)

let thunk name mode () =
  let module_name, fn, args = case_fn name in
  let args = List.map (fun v -> TR.Leaf v) args in
  let values t = String.concat "\n" (List.map text_of (leaves t)) in
  match mode with
  | "eager" -> values (fn args)
  | "jit" -> values (A.jit fn args)
  | "jaxpr" ->
      Ojax.Pretty_printer.closed_jaxpr_to_jax_string (A.make_jaxpr fn args)
  | "lowered" ->
      let wrapped = Ojax.Stages.jit ~name:module_name fn in
      Ojax.Stages.as_text (wrapped.Ojax.Stages.wrapped_lower args)
  | other -> failwith ("vmap_coll: unknown mode " ^ other)

let expected_of row =
  let str k = match U.member k row with `String s -> s | _ -> "" in
  {
    kind = str "kind";
    cls = str "class";
    message = str "message";
    text = str "text";
    stdout = str "stdout";
    warnings =
      (match U.member "warnings" row with
      | `List l ->
          List.map
            (fun w ->
              match U.to_list w with
              | [ c; m ] -> (U.to_string c, U.to_string m)
              | _ -> failwith "vmap_coll: warning entry")
            l
      | _ -> []);
  }

let same e a =
  e.kind = a.kind && e.cls = a.cls && e.message = a.message && e.text = a.text
  && e.stdout = a.stdout && e.warnings = a.warnings

let record suite name same =
  Compare.bump suite same;
  Compare.bump (Compare.raw_prefix ^ suite) same;
  Compare.note_row suite name ~bit:same ~raw:same;
  match Lazy.force Compare.exact_log with
  | None -> ()
  | Some oc -> (
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\t%s\n%!" name 1
          (if same then 0 else 1)
          0.0 suite
          (if same then Compare.raw_identical else "differ")
      with Sys_error _ -> ())

let show r =
  Printf.sprintf "%s %s %S text=%S stdout=%S" r.kind r.cls r.message r.text
    r.stdout

let check_row ~x64 row () =
  let id = U.to_string (U.member "id" row) in
  let expected = expected_of row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
        run_case
          (thunk
             (U.to_string (U.member "case" row))
             (U.to_string (U.member "mode" row))))
  in
  let suite = "vmap_coll:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = same expected actual in
  record suite id ok;
  if not ok then
    Printf.eprintf "VMAP_COLL DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (show expected) (show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "vmap_coll") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "vmap_coll_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "id" row))
          `Quick (check_row ~x64 row))
      rows )
