module U = Yojson.Safe.Util
module T = Ojax.Types
module F = Ojax.Numpy.Ufuncs
module G = Ojax.Numpy.Getitem

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

let leaf v = Ojax.Tree_util.Leaf v

let float_dt () =
  if Ojax.Config.x64_enabled () then Ojax.Dtype.F64 else Ojax.Dtype.F32

let py_float x = Ojax.Core.py_scalar (float_dt ()) x

let py_int x =
  Ojax.Core.py_scalar
    (if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32)
    x

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

let idx k = [ G.Int k ]
let at v k = G.take v (idx k)
let astype = Ojax.Numpy.Lax_numpy.astype
let div a k = F.true_divide a (py_int k)

module L = Ojax.Lax_api

let dprint_body name x =
  let p = Ojax.Debugging.print in
  let i = Ojax.Numpy.Lax_numpy.arange ~dtype:Ojax.Dtype.I32 3.0 in
  (match name with
  | "kw_pos" ->
      p ~kwargs:[ ("x", x) ] "x={x} y={}" [ F.multiply x (py_float 2.0) ]
  | "spec_ordered" -> p ~ordered:true "v {:.2f} {}" [ at x 0; x ]
  | "no_placeholders" -> p "hello" [ x ]
  | "indexed" -> p "{1} then {0}" [ x; F.add x (py_float 1.0) ]
  | "int_spec" -> p "n={:d} m={:5d}" [ at i 1; at i 2 ]
  | "escape" -> p "{{literal}} {}" [ x ]
  | "bool_arr" -> p "b={}" [ F.greater x (py_int 1.0) ]
  | "int_arr" -> p "i={}" [ F.multiply i (py_int 7.0) ]
  | "mat" ->
      p
        ~kwargs:
          [
            ( "m",
              F.true_divide
                (Ojax.Numpy.Tensor_contractions.outer x x)
                (py_int 3.0) );
          ]
        "m=\n{m}" []
  | "scalar_default" ->
      p "s={}" [ F.true_divide (Ojax.Numpy.Reductions.sum x) (py_int 7.0) ]
  | "exp_spec" -> p "e={:.3e}" [ F.multiply (at x 2) (py_float 1234.5) ]
  | "align_spec" ->
      p "[{:>8.3f}] [{:<6}]" [ F.true_divide (at x 1) (py_int 3.0); at i 2 ]
  | "plain_text" -> p "just text" []
  | "two_kw" -> p ~kwargs:[ ("a", x); ("b", i) ] "{b} {a} {b}" []
  | "pct_spec" -> p "{:.1%}" [ F.true_divide (at x 1) (py_int 4.0) ]
  | "g_spec" ->
      p "{:g} {:.3g}"
        [
          F.true_divide (at x 2) (py_int 3.0);
          F.multiply (at x 2) (py_float 1e7);
        ]
  | "neg_scalar" -> p "{} {:+.2f}" [ F.negative (at x 1); F.negative (at x 2) ]
  | "weak_scalar" -> p "{}" [ F.add (at x 0) (py_float 0.5) ]
  | "f64_vals" -> p "{}" [ F.true_divide x (py_int 3.0) ]
  | "nan_inf" -> p "{}" [ F.log (F.subtract x (py_int 1.0)) ]
  | "f16_scalar" -> p "{}" [ astype (div (at x 1) 3.0) Ojax.Dtype.F16 ]
  | "bf16_arr" -> p "{}" [ astype (div x 3.0) Ojax.Dtype.Bf16 ]
  | "big_arr" ->
      p "{}"
        [
          F.add
            (Ojax.Numpy.Lax_numpy.arange ~dtype:Ojax.Dtype.F32 1500.0)
            (at x 0);
        ]
  | "complex_scalar" -> p "{}" [ L.complex (at x 1) (at x 2) ]
  | "u8_arr" -> p "{}" [ astype (F.multiply i (py_int 100.0)) Ojax.Dtype.Uint8 ]
  | "bool_scalar_spec" ->
      p "[{:>6}] [{}]"
        [ F.greater (at x 1) (py_int 0.0); F.greater (at x 0) (py_int 0.0) ]
  | "repr_conv" -> p "{!r}" [ x ]
  | "str_conv" -> p "{!s} {!r}" [ at x 1; at x 2 ]
  | "i64_arr" ->
      p "{}"
        [
          F.multiply
            (Ojax.Numpy.Lax_numpy.arange
               ~dtype:(Ojax.Dtypes.canonicalize_dtype Ojax.Dtype.I64)
               4.0)
            (py_int 1000000.0);
        ]
  | "f16_spec" -> p "{:.4f}" [ astype (div (at x 1) 3.0) Ojax.Dtype.F16 ]
  | "hex_spec" ->
      p "{:x} {:#X} {:b} {:o}"
        [
          F.multiply (at i 2) (py_int 100.0);
          F.multiply (at i 2) (py_int 100.0);
          at i 2;
          F.multiply (at i 2) (py_int 9.0);
        ]
  | "center_spec" -> p "[{:^9.2f}] [{:*^7}]" [ div (at x 1) 3.0; at i 2 ]
  | "zero_pad" -> p "{:08.3f} {:05d}" [ div (F.negative (at x 1)) 3.0; at i 2 ]
  | "int_as_float_spec" -> p "{:.2f} {:e}" [ at i 2; at i 1 ]
  | "neg_zero" -> p "{} {:.1f}" [ F.negative (at x 0); F.negative (at x 0) ]
  | "empty_arr" ->
      p "{}"
        [ G.take x [ G.Slice { start = Some 3; stop = None; step = None } ] ]
  | "two_d_int" ->
      p "{}"
        [ F.subtract (Ojax.Numpy.Tensor_contractions.outer i i) (py_int 3.0) ]
  | "inf_scalar" ->
      p "{} {:.2f} {:e}"
        [
          F.true_divide (py_int 1.0) (at x 0);
          F.true_divide (py_int (-1.0)) (at x 0);
          F.log (F.subtract (at x 0) (py_int 1.0));
        ]
  | "i8_arr" -> p "{}" [ astype (F.subtract i (py_int 1.0)) Ojax.Dtype.I8 ]
  | "f16_arr" -> p "{}" [ astype (div x 3.0) Ojax.Dtype.F16 ]
  | "u32_hex" ->
      p "{:08x}"
        [ astype (F.multiply (at i 2) (py_int 1000.0)) Ojax.Dtype.Uint32 ]
  | "big_scalar" ->
      p "{} {}"
        [
          F.multiply (at x 1) (py_float 1e16);
          F.multiply (at x 2) (py_float 1e20);
        ]
  | "tiny_scalar" ->
      p "{} {}"
        [
          F.multiply (at x 1) (py_float 1e-5);
          F.multiply (at x 2) (py_float 1e-30);
        ]
  | "tenth_scalar" -> p "{} {}" [ div (at x 1) 10.0; div x 10.0 ]
  | "printopts" ->
      Ojax.Numpy.Extras.with_printoptions ~precision:3 (fun () ->
          p "{} {}" [ div x 3.0; div (at x 1) 3.0 ])
  | "suppress_opts" ->
      Ojax.Numpy.Extras.with_printoptions ~suppress:true ~threshold:5
        ~edgeitems:2 (fun () ->
          p "{}"
            [
              F.add
                (F.multiply
                   (Ojax.Numpy.Lax_numpy.arange ~dtype:Ojax.Dtype.F32 8.0)
                   (py_float 1e-9))
                (at x 1);
            ])
  | "missing_kw" -> p "{y}" [ x ]
  | "missing_pos" -> p "{} {}" [ x ]
  | "spec_on_array" -> p "{:.2f}" [ x ]
  | "mixed_numbering" -> p "{} {0}" [ x; x ]
  | other -> failwith ("docs_p2: unknown dprint case " ^ other));
  F.add x (py_int 1.0)

let one f = function
  | [ Ojax.Tree_util.Leaf a ] -> f a
  | _ -> failwith "docs_p2: arity"

let dprint_thunk name mode () =
  let x = Ojax.Numpy.Lax_numpy.arange ~dtype:(float_dt ()) 3.0 in
  let body = one (fun a -> leaf (dprint_body name a)) in
  match mode with
  | "eager" ->
      ignore (dprint_body name x);
      ""
  | "jit" ->
      ignore (Ojax.Api.jit body [ leaf x ]);
      ""
  | "jaxpr" ->
      Ojax.Pretty_printer.closed_jaxpr_to_jax_string
        (Ojax.Api.make_jaxpr body [ leaf x ])
  | "lowered" ->
      let wrapped = Ojax.Stages.jit ~name:"body" body in
      Ojax.Stages.as_text (wrapped.Ojax.Stages.wrapped_lower [ leaf x ])
  | other -> failwith ("docs_p2: unknown mode " ^ other)

let xform_thunk name mode () =
  let p = Ojax.Debugging.print in
  let x = Ojax.Numpy.Lax_numpy.arange ~dtype:(float_dt ()) 3.0 in
  let m =
    Ojax.Numpy.Lax_numpy.reshape
      (Ojax.Numpy.Lax_numpy.arange ~dtype:(float_dt ()) 6.0)
      [| 2; 3 |]
  in
  let fn, arg =
    match name with
    | "vmap" ->
        ( Ojax.Api.vmap
            (one (fun a ->
                 p "row {}" [ a ];
                 leaf (F.multiply a (py_int 2.0))))
            [ Some 0 ],
          m )
    | "vmap_ordered" ->
        ( Ojax.Api.vmap
            (one (fun a ->
                 p ~ordered:true "row {} {}" [ at a 0; a ];
                 leaf (F.multiply a (py_int 2.0))))
            [ Some 0 ],
          m )
    | "grad" ->
        ( Ojax.Api.grad
            (one (fun a ->
                 p "a={}" [ a ];
                 leaf (Ojax.Numpy.Reductions.sum (F.sin a)))),
          x )
    | "jvp" ->
        ( one (fun a ->
              let f =
                one (fun b ->
                    p "a={}" [ b ];
                    leaf (F.sin b))
              in
              let pr, tg =
                Ojax.Api.jvp f
                  [ leaf a ]
                  [ leaf (Ojax.Numpy.Array_creation.ones_like a) ]
              in
              Ojax.Tree_util.Tuple [ pr; tg ]),
          x )
    | "cond" ->
        ( one (fun a ->
              match
                L.cond
                  (F.less (at a 0) (py_int 1.0))
                  (fun bs ->
                    p "yes {}" bs;
                    bs)
                  (fun bs ->
                    p "no {}" bs;
                    List.map F.negative bs)
                  [ a ]
              with
              | [ r ] -> leaf r
              | _ -> failwith "docs_p2: cond"),
          x )
    | "fori" ->
        ( one (fun a ->
              match
                L.fori_loop (py_int 0.0) (py_int 3.0)
                  (fun k cs ->
                    p "k={} c={}" (k :: cs);
                    List.map (fun c -> F.add c (py_int 1.0)) cs)
                  [ a ]
              with
              | [ r ] -> leaf r
              | _ -> failwith "docs_p2: fori"),
          x )
    | "scan" ->
        ( one (fun a ->
              match
                L.scan
                  (function
                    | [ c; e ] ->
                        p "c={} e={}" [ c; e ];
                        [ F.add c e; F.multiply c e ]
                    | _ -> failwith "docs_p2: scan body")
                  [ at a 0 ]
                  [ a ]
              with
              | [ c; ys ] -> Ojax.Tree_util.Tuple [ leaf c; leaf ys ]
              | _ -> failwith "docs_p2: scan"),
          x )
    | "while" ->
        ( one (fun a ->
              match
                L.while_loop
                  (function
                    | [ c ] -> F.less (at c 0) (py_int 2.0)
                    | _ -> failwith "docs_p2: while")
                  (fun cs ->
                    p "w {}" cs;
                    List.map (fun c -> F.add c (py_int 1.0)) cs)
                  [ a ]
              with
              | [ r ] -> leaf r
              | _ -> failwith "docs_p2: while"),
          x )
    | "remat" ->
        ( Ojax.Api.grad
            (one (fun a ->
                 let g = function
                   | [ b ] ->
                       p "r {}" [ b ];
                       [ F.sin b ]
                   | _ -> failwith "docs_p2: remat"
                 in
                 match Ojax.Api.checkpoint g [ a ] with
                 | [ r ] -> leaf (Ojax.Numpy.Reductions.sum r)
                 | _ -> failwith "docs_p2: remat")),
          x )
    | "grad_scan" ->
        ( Ojax.Api.grad
            (one (fun a ->
                 match
                   L.scan
                     (function
                       | [ c; e ] ->
                           p "c={}" [ c ];
                           [ F.multiply c e; e ]
                       | _ -> failwith "docs_p2: scan body")
                     [ py_float 1.0 ]
                     [ a ]
                 with
                 | [ c; _ ] -> leaf c
                 | _ -> failwith "docs_p2: grad_scan")),
          F.add x (py_int 1.0) )
    | "jit_grad" ->
        let g =
          Ojax.Api.grad
            (one (fun a ->
                 p "a={}" [ a ];
                 leaf (Ojax.Numpy.Reductions.sum (F.multiply a a))))
        in
        ( (fun args ->
            match
              Ojax.Api.jit_flat ~name:"f"
                (fun vs -> Ojax.Tree_util.tree_leaves (g (List.map leaf vs)))
                (Ojax.Tree_util.tree_leaves (Ojax.Tree_util.Tuple args))
            with
            | [ r ] -> leaf r
            | _ -> failwith "docs_p2: jit_grad"),
          x )
    | "nested_vmap" ->
        ( Ojax.Api.vmap
            (Ojax.Api.vmap
               (one (fun a ->
                    p "e {}" [ a ];
                    leaf (F.multiply a (py_int 2.0))))
               [ Some 0 ])
            [ Some 0 ],
          m )
    | "cond_ordered" ->
        ( one (fun a ->
              match
                L.cond
                  (F.less (at a 0) (py_int 1.0))
                  (fun bs ->
                    p ~ordered:true "yes {}" bs;
                    bs)
                  (fun bs ->
                    p ~ordered:true "no {}" bs;
                    List.map F.negative bs)
                  [ a ]
              with
              | [ r ] -> leaf r
              | _ -> failwith "docs_p2: cond"),
          x )
    | "nested_jit" ->
        ( one (fun a ->
              let g = function
                | [ b ] ->
                    p "inner {}" [ b ];
                    [ F.add b (py_int 1.0) ]
                | _ -> failwith "docs_p2: nested"
              in
              p "outer {}" [ a ];
              match Ojax.Api.jit_flat ~name:"g" g [ a ] with
              | [ r ] -> leaf (F.multiply r (py_int 2.0))
              | _ -> failwith "docs_p2: nested"),
          x )
    | "vmap_unbatched" -> (one (fun a -> leaf a), x)
    | other -> failwith ("docs_p2: unknown xform " ^ other)
  in
  let fn, args =
    if name = "vmap_unbatched" then
      ( Ojax.Api.vmap
          (function
            | [ Ojax.Tree_util.Leaf a; Ojax.Tree_util.Leaf b ] ->
                p "{} {}" [ a; b ];
                leaf (F.add a b)
            | _ -> failwith "docs_p2: vmap_unbatched")
          [ Some 0; None ],
        [ leaf m; leaf x ] )
    else (fn, [ leaf arg ])
  in
  match mode with
  | "eager" ->
      ignore (fn args);
      ""
  | "jit" when name = "jit_grad" ->
      ignore (fn args);
      ""
  | "jit" ->
      ignore (Ojax.Api.jit fn args);
      ""
  | _ ->
      Ojax.Pretty_printer.closed_jaxpr_to_jax_string
        (Ojax.Api.make_jaxpr fn args)

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
              | _ -> failwith "docs_p2: warning entry")
            l
      | _ -> []);
  }

let pairs_text j =
  String.concat "\n"
    (List.map
       (fun p ->
         match U.to_list p with
         | [ k; v ] -> U.to_string k ^ "=" ^ Yojson.Safe.to_string v
         | _ -> failwith "docs_p2: pair")
       (U.to_list j))

let f32_range shape div =
  let n = Array.fold_left ( * ) 1 shape in
  let f v = Int32.float_of_bits (Int32.bits_of_float v) in
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 shape
       (Array.init n (fun k -> f (float_of_int k /. div))))

let aot_fn name =
  let x34 = f32_range [| 3; 4 |] 7.0
  and y42 = f32_range [| 4; 2 |] 5.0
  and v = f32_range [| 16 |] 3.0
  and i =
    T.Concrete
      (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 6 |]
         (Array.init 6 float_of_int))
  in
  let mm a b = Ojax.Numpy.Tensor_contractions.matmul a b in
  let two f = function
    | [ Ojax.Tree_util.Leaf a; Ojax.Tree_util.Leaf b ] -> f a b
    | _ -> failwith "docs_p2: arity"
  in
  match name with
  | "sin_dot" ->
      ( two (fun a b -> leaf (F.add (mm (F.sin a) b) (py_float 1.0))),
        [ x34; y42 ] )
  | "add" -> (two (fun a b -> leaf (F.add a b)), [ v; v ])
  | "reduce" ->
      (one (fun a -> leaf (Ojax.Numpy.Reductions.sum (F.multiply a a))), [ v ])
  | "exp_log" -> (one (fun a -> leaf (F.log1p (F.exp a))), [ x34 ])
  | "matmul" -> (two (fun a b -> leaf (mm a b)), [ x34; y42 ])
  | "cumsum" -> (one (fun a -> leaf (Ojax.Numpy.Reductions.cumsum a)), [ v ])
  | "where" ->
      ( one (fun a ->
            leaf
              (Ojax.Numpy.Lax_numpy.where_
                 (F.greater a (py_float 2.0))
                 a (F.negative a))),
        [ v ] )
  | "int_mul" ->
      (one (fun a -> leaf (F.add (F.multiply a a) (py_int 3.0))), [ i ])
  | "two_out" ->
      ( two (fun a b ->
            Ojax.Tree_util.Tuple
              [
                leaf (F.add a (py_float 1.0));
                leaf (F.multiply b (py_float 2.0));
              ]),
        [ x34; y42 ] )
  | "tanh_big" -> (one (fun a -> leaf (F.multiply (F.tanh a) (F.cos a))), [ v ])
  | other -> failwith ("docs_p2: unknown aot case " ^ other)

let fmt_float x = Printf.sprintf "%s" (Yojson.Safe.to_string (`Float x))

let aot_text row =
  let name = U.to_string (U.member "case" row) in
  let f, args = aot_fn name in
  let wrapped = Ojax.Stages.jit f in
  let compiled =
    Ojax.Stages.compile (wrapped.Ojax.Stages.wrapped_lower (List.map leaf args))
  in
  match U.to_string (U.member "group" row) with
  | "aot_cost" ->
      String.concat "\n"
        (List.map
           (fun (k, v) -> k ^ "=" ^ fmt_float v)
           (List.sort compare (Ojax.Stages.cost_analysis compiled)))
  | _ ->
      let m = Ojax.Stages.memory_analysis compiled in
      let module E = Ojax.Pjrt.Executable in
      String.concat "\n"
        (List.map
           (fun (k, v) -> k ^ "=" ^ string_of_int v)
           (List.sort compare
              [
                ( "generated_code_size_in_bytes",
                  m.E.generated_code_size_in_bytes );
                ("argument_size_in_bytes", m.E.argument_size_in_bytes);
                ("output_size_in_bytes", m.E.output_size_in_bytes);
                ("alias_size_in_bytes", m.E.alias_size_in_bytes);
                ("temp_size_in_bytes", m.E.temp_size_in_bytes);
                ( "host_generated_code_size_in_bytes",
                  m.E.host_generated_code_size_in_bytes );
                ("host_argument_size_in_bytes", m.E.host_argument_size_in_bytes);
                ("host_output_size_in_bytes", m.E.host_output_size_in_bytes);
                ("host_alias_size_in_bytes", m.E.host_alias_size_in_bytes);
                ("host_temp_size_in_bytes", m.E.host_temp_size_in_bytes);
                ("peak_memory_in_bytes", m.E.peak_memory_in_bytes);
              ]))

let text_of v =
  let nd =
    match v with
    | T.Concrete nd -> nd
    | T.Device b -> Ojax.Pjrt.Buffer.to_host b
    | T.Tracer _ -> failwith "docs_p2: tracer result"
  in
  Printf.sprintf "%s [%s] %s"
    (Ojax.Dtype.numpy_name (Ojax.Ndarray.dtype nd))
    (String.concat ", "
       (List.map string_of_int (Array.to_list (Ojax.Ndarray.shape nd))))
    (Parity_cases.hex_of_nd nd)

let static_index_thunk name mode flag () =
  let x =
    T.Concrete
      (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 2; 3 |]
         (Array.init 6 (fun k -> float_of_int (k + 1))))
  and v =
    T.Concrete
      (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 3 |]
         (Array.init 3 (fun k -> float_of_int (k + 1))))
  in
  let f _ =
    match name with
    | "pos_oob" -> G.take v [ G.Int 5 ]
    | "neg_oob" -> G.take v [ G.Int (-4) ]
    | "second_axis" -> G.take x [ G.Int 1; G.Int 7 ]
    | "in_bounds" -> G.take x [ G.Int 1; G.Int 2 ]
    | "neg_in" -> G.take v [ G.Int (-3) ]
    | "at_clip" -> G.take ~mode:"clip" v [ G.Int 5 ]
    | "at_promise" -> G.take ~mode:"promise_in_bounds" v [ G.Int 5 ]
    | "at_fill" -> G.take ~mode:"fill" v [ G.Int 5 ]
    | "at_nonorm" -> G.take ~normalize_indices:false v [ G.Int (-1) ]
    | "slice_mix" ->
        G.take x
          [ G.Int 3; G.Slice { start = Some 1; stop = None; step = None } ]
    | "at_set" -> G.at_set v [ G.Int 5 ] (py_float 9.0)
    | "int_arr_idx" ->
        G.take v
          [
            G.Arr
              (T.Concrete
                 (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 2 |] [| 0.0; 4.0 |]));
          ]
    | other -> failwith ("docs_p2: unknown static_index case " ^ other)
  in
  Ojax.Config.with_value Ojax.Config.check_static_indices flag (fun () ->
      if mode = "jit" then
        match Ojax.Api.jit (one (fun a -> leaf (f a))) [ leaf v ] with
        | Ojax.Tree_util.Leaf r -> text_of r
        | _ -> failwith "docs_p2: jit result"
      else text_of (f v))

let ufunc_of = function
  | "maximum" -> (
      fun l -> match l with [ a; b ] -> F.maximum a b | _ -> assert false)
  | "minimum" -> (
      fun l -> match l with [ a; b ] -> F.minimum a b | _ -> assert false)
  | "logaddexp" -> (
      fun l -> match l with [ a; b ] -> F.logaddexp a b | _ -> assert false)
  | "logaddexp2" -> (
      fun l -> match l with [ a; b ] -> F.logaddexp2 a b | _ -> assert false)
  | "bitwise_and" -> (
      fun l -> match l with [ a; b ] -> F.bitwise_and a b | _ -> assert false)
  | "bitwise_or" -> (
      fun l -> match l with [ a; b ] -> F.bitwise_or a b | _ -> assert false)
  | "bitwise_xor" -> (
      fun l -> match l with [ a; b ] -> F.bitwise_xor a b | _ -> assert false)
  | "logical_and" -> (
      fun l -> match l with [ a; b ] -> F.logical_and a b | _ -> assert false)
  | "logical_or" -> (
      fun l -> match l with [ a; b ] -> F.logical_or a b | _ -> assert false)
  | "logical_xor" -> (
      fun l -> match l with [ a; b ] -> F.logical_xor a b | _ -> assert false)
  | "negative" -> (
      fun l -> match l with [ a ] -> F.negative a | _ -> assert false)
  | other -> failwith ("docs_p2: unknown ufunc " ^ other)

let ufunc_at_thunk fn idx_kind mode () =
  let integer = List.mem fn [ "bitwise_and"; "bitwise_or"; "bitwise_xor" ] in
  let dt = if integer then Ojax.Dtype.I32 else Ojax.Dtype.F32 in
  let arr shape vals = T.Concrete (Ojax.Ndarray.of_floats dt shape vals) in
  let v = arr [| 5 |] (Array.init 5 (fun k -> float_of_int (k + 1))) in
  let m = arr [| 2; 3 |] (Array.init 6 (fun k -> float_of_int (k + 1))) in
  let i l =
    G.Arr
      (T.Concrete
         (Ojax.Ndarray.of_floats Ojax.Dtype.I32
            [| List.length l |]
            (Array.of_list (List.map float_of_int l))))
  in
  let unary = fn = "negative" in
  let operands () =
    match idx_kind with
    | "scalar" -> (v, [ G.Int 2 ], arr [||] [| 3.0 |])
    | "dup" -> (v, [ i [ 0; 2; 2 ] ], arr [| 3 |] [| 2.0; 3.0; 4.0 |])
    | "two_d" -> (m, [ i [ 0; 1; 1 ]; i [ 2; 0; 0 ] ], arr [||] [| 2.0 |])
    | _ -> (m, [ i [ 1; 1 ] ], arr [| 3 |] [| 2.0; 3.0; 1.0 |])
  in
  let run a idx b =
    G.ufunc_at ~inplace:false ~name:fn (ufunc_of fn) a idx
      (if unary then [] else [ b ])
  in
  let a, idx, b = operands () in
  if mode = "jaxpr" then
    let idx_vals =
      List.map
        (function
          | G.Int k -> py_int (float_of_int k)
          | G.Arr v -> v
          | _ -> assert false)
        idx
    in
    let lead = if unary then [ a ] else [ a; b ] in
    Ojax.Pretty_printer.closed_jaxpr_to_jax_string
      (Ojax.Api.make_jaxpr
         (fun leaves ->
           let vals =
             List.map
               (function Ojax.Tree_util.Leaf v -> v | _ -> assert false)
               leaves
           in
           let nl = List.length lead in
           let a = List.hd vals in
           let b = if unary then a else List.nth vals 1 in
           let ix = List.filteri (fun k _ -> k >= nl) vals in
           leaf (run a (List.map (fun v -> G.Arr v) ix) b))
         (List.map leaf (lead @ idx_vals)))
  else text_of (run a idx b)

let api_thunk name () =
  let x =
    Ojax.Numpy.Lax_numpy.reshape
      (Ojax.Numpy.Lax_numpy.arange ~dtype:Ojax.Dtype.F32 6.0)
      [| 2; 3 |]
  in
  let like v =
    Ojax.Shape_dtype_struct.to_string (Ojax.Shape_dtype_struct.like v)
  in
  match name with
  | "flat" -> text_of (Ojax.Numpy.Array_methods.flat x)
  | "like_f32" -> like x
  | "like_weak" ->
      like (F.multiply (G.take x [ G.Int 0; G.Int 0 ]) (py_float 2.0))
  | "like_int" -> like (Ojax.Numpy.Lax_numpy.arange ~dtype:Ojax.Dtype.I32 4.0)
  | "like_bool" -> like (F.greater x (py_int 1.0))
  | "like_scalar_int" -> like (py_int 3.0)
  | other -> failwith ("docs_p2: unknown api case " ^ other)

let actual_of row =
  match U.to_string (U.member "group" row) with
  | "dprint" ->
      run_case
        (dprint_thunk
           (U.to_string (U.member "case" row))
           (U.to_string (U.member "mode" row)))
  | "dprint_x" ->
      run_case
        (xform_thunk
           (U.to_string (U.member "case" row))
           (U.to_string (U.member "mode" row)))
  | "ufunc_at" ->
      run_case
        (ufunc_at_thunk
           (U.to_string (U.member "case" row))
           (U.to_string (U.member "idx" row))
           (U.to_string (U.member "mode" row)))
  | "api" -> run_case (api_thunk (U.to_string (U.member "case" row)))
  | "static_index" ->
      run_case
        (static_index_thunk
           (U.to_string (U.member "case" row))
           (U.to_string (U.member "mode" row))
           (U.to_bool (U.member "flag" row)))
  | "aot_cost" | "aot_mem" -> run_case (fun () -> aot_text row)
  | g -> failwith ("docs_p2: unknown group " ^ g)

let expected_text row =
  match U.to_string (U.member "group" row) with
  | "aot_cost" ->
      { (expected_of row) with text = pairs_text (U.member "cost" row) }
  | "aot_mem" ->
      { (expected_of row) with text = pairs_text (U.member "mem" row) }
  | _ -> expected_of row

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
  let expected = expected_text row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () -> actual_of row)
  in
  let suite = "docs_p2:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = same expected actual in
  record suite id ok;
  if not ok then
    Printf.eprintf "DOCS_P2 DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (show expected) (show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "docs_p2") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "docs_p2_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "id" row))
          `Quick (check_row ~x64 row))
      rows )
