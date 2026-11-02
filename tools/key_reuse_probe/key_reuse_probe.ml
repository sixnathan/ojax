module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module R = Ojax.Random.Core
module L = Ojax.Lax_api
module A = Ojax.Api
module U = Ojax.Numpy.Ufuncs

let i32 v = T.Concrete (Nd.of_floats D.I32 [||] [| v |])
let tt = T.Concrete (Nd.of_floats D.Bool [||] [| 1. |])
let f32 v = T.Concrete (Nd.of_floats D.F32 [||] [| v |])
let k ?(seed = 0.) () = R.key (i32 seed)
let normal a = R.normal a ~shape:[||]
let uniform a = R.uniform a ~shape:[||] ~minval:0. ~maxval:1.
let first_key a = Ojax.Numpy.Indexing.take_index a (i32 0.)

let split2 a =
  let s = R.split a 2 in
  let piece i =
    Ojax.Numpy.Lax_numpy.squeeze ~axis:[| 0 |]
      (L.slice_in_dim ~axis:0 s i (i + 1))
  in
  (piece 0, piece 1)

let jit f args = ignore (A.jit_flat f args)
let one f = function [ a ] -> [ f a ] | _ -> failwith "arity"

let cases =
  [
    ( "direct",
      fun () ->
        let a = k () in
        ignore (normal a);
        ignore (normal a) );
    ( "split",
      fun () ->
        let x, y = split2 (k ()) in
        ignore (normal x);
        ignore (normal y) );
    ( "jit_reuse",
      fun () -> jit (one (fun a -> U.add (normal a) (normal a))) [ k () ] );
    ( "jit_ok",
      fun () ->
        jit
          (one (fun a ->
               let x, y = split2 a in
               U.add (normal x) (normal y)))
          [ k () ] );
    ( "cond_each_branch",
      fun () ->
        jit
          (function
            | [ a; p ] -> L.cond p (one normal) (one uniform) [ a ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_then_reuse",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                let x = L.cond p (one normal) (one (fun _ -> f32 0.)) [ a ] in
                [ U.add (List.hd x) (normal a) ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_one_side",
      fun () ->
        jit
          (function
            | [ a; p ] -> L.cond p (one normal) (one (fun _ -> f32 0.)) [ a ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_closure",
      fun () ->
        let a = k () in
        jit
          (function
            | [ p ] ->
                L.cond p (fun _ -> [ normal a ]) (fun _ -> [ uniform a ]) []
            | _ -> failwith "arity")
          [ tt ] );
    ( "eager_cond_after",
      fun () ->
        let a = k () in
        ignore (L.cond tt (one normal) (one (fun _ -> f32 0.)) [ a ]);
        ignore (normal a) );
    ( "scan_split",
      fun () ->
        ignore
          (L.scan ~length:3
             (function
               | [ c ] ->
                   let c, s = split2 c in
                   [ c; normal s ]
               | _ -> failwith "arity")
             [ k () ]
             []) );
    ( "scan_reuse",
      fun () ->
        ignore
          (L.scan ~length:3
             (function [ c ] -> [ c; normal c ] | _ -> failwith "arity")
             [ k () ]
             []) );
    ( "scan_closure",
      fun () ->
        let a = k () in
        ignore
          (L.scan ~length:3
             (function [ c ] -> [ c; normal a ] | _ -> failwith "arity")
             [ i32 0. ]
             []) );
    ( "scan_xs",
      fun () ->
        let ks = R.split (k ()) 3 in
        ignore
          (L.scan
             (function [ c; x ] -> [ c; normal x ] | _ -> failwith "arity")
             [ i32 0. ]
             [ ks ]) );
    ( "while_split",
      fun () ->
        ignore
          (L.while_loop
             (function [ _; i ] -> L.lt i (i32 3.) | _ -> failwith "arity")
             (function
               | [ a; i ] ->
                   let a, _ = split2 a in
                   [ a; L.add i (i32 1.) ]
               | _ -> failwith "arity")
             [ k (); i32 0. ]) );
    ( "while_reuse",
      fun () ->
        ignore
          (L.while_loop
             (function [ _; i ] -> L.lt i (i32 3.) | _ -> failwith "arity")
             (function
               | [ a; i ] ->
                   ignore (normal a);
                   [ a; L.add i (i32 1.) ]
               | _ -> failwith "arity")
             [ k (); i32 0. ]) );
    ( "while_closure",
      fun () ->
        let a = k () in
        ignore
          (L.while_loop
             (function [ s ] -> L.lt s (i32 3.) | _ -> failwith "arity")
             (function
               | [ s ] ->
                   ignore (normal a);
                   [ L.add s (i32 1.) ]
               | _ -> failwith "arity")
             [ i32 0. ]) );
    ( "fori_passthrough",
      fun () -> ignore (L.fori_loop (i32 0.) (i32 3.) (fun _ s -> s) [ k () ])
    );
    ( "switch_each",
      fun () ->
        jit
          (function
            | [ a; i ] ->
                L.switch i
                  [ one normal; one uniform; one (fun _ -> f32 0.) ]
                  [ a ]
            | _ -> failwith "arity")
          [ k (); i32 1. ] );
    ( "vmap_split",
      fun () ->
        ignore
          (Ojax.Interpreters.Batching.vmap_flat (one normal) [ Some 0 ]
             [ R.split (k ()) 3 ]) );
    ( "vmap_same",
      fun () ->
        let a = k () in
        ignore
          (Ojax.Interpreters.Batching.vmap_flat
             (one (fun _ -> normal a))
             [ Some 0 ]
             [ Ojax.Numpy.Lax_numpy.arange ~dtype:D.I32 3. ]) );
    ( "clone_ok",
      fun () ->
        let a = k () in
        ignore (normal (R.clone a));
        ignore (normal a) );
    ( "fold_in_twice",
      fun () ->
        let a = k () in
        ignore (R.fold_in a (i32 1.));
        ignore (R.fold_in a (i32 2.)) );
    ( "key_data_then",
      fun () ->
        let a = k () in
        ignore (R.key_data a);
        ignore (normal a) );
    ( "nested_jit_reuse",
      fun () ->
        let g = A.jit_flat (one normal) in
        jit (one (fun a -> U.add (List.hd (g [ a ])) (normal a))) [ k () ] );
    ( "nested_jit_ok",
      fun () ->
        let g = A.jit_flat (one normal) in
        jit
          (one (fun a ->
               let x, y = split2 a in
               U.add (List.hd (g [ x ])) (List.hd (g [ y ]))))
          [ k () ] );
    ( "cond_return_key",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                L.cond p (one Fun.id) (one (fun a -> fst (split2 a))) [ a ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_identity_alias_reuse",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                let b = List.hd (L.cond p (one Fun.id) (one Fun.id) [ a ]) in
                [ U.add (normal a) (normal b) ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_unused_then_use",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                ignore
                  (L.cond p
                     (one (fun _ -> f32 0.))
                     (one (fun _ -> f32 1.))
                     [ a ]);
                [ normal a ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_forward_then_use",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                [ normal (List.hd (L.cond p (one Fun.id) (one Fun.id) [ a ])) ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_mixed_out",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                [
                  normal
                    (List.hd
                       (L.cond p (one Fun.id)
                          (one (fun a -> fst (split2 a)))
                          [ a ]));
                ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "scan_pass_then_use",
      fun () ->
        let out =
          L.scan ~length:2
            (function [ c ] -> [ c; f32 0. ] | _ -> failwith "arity")
            [ k () ]
            []
        in
        ignore (normal (List.hd out)) );
    ( "scan_pass_then_both",
      fun () ->
        let a = k () in
        let out =
          L.scan ~length:2
            (function [ c ] -> [ c; f32 0. ] | _ -> failwith "arity")
            [ a ] []
        in
        ignore (normal (List.hd out));
        ignore (normal a) );
    ( "scan_consumes_init_then_use",
      fun () ->
        let a = k () in
        ignore
          (L.scan ~length:2
             (function
               | [ c ] ->
                   let c, s = split2 c in
                   [ c; normal s ]
               | _ -> failwith "arity")
             [ a ] []);
        ignore (normal a) );
    ( "scan_out_usable",
      fun () ->
        let out =
          L.scan ~length:2
            (function
              | [ c ] ->
                  let c, s = split2 c in
                  [ c; normal s ]
              | _ -> failwith "arity")
            [ k () ]
            []
        in
        ignore (normal (List.hd out)) );
    ( "while_pass_then_use",
      fun () ->
        let out =
          L.while_loop
            (function [ _; i ] -> L.lt i (i32 3.) | _ -> failwith "arity")
            (function
              | [ a; i ] -> [ a; L.add i (i32 1.) ] | _ -> failwith "arity")
            [ k (); i32 0. ]
        in
        ignore (normal (List.hd out)) );
    ( "while_cond_and_body",
      fun () ->
        ignore
          (L.while_loop
             (function
               | [ a; _ ] -> L.lt (normal a) (f32 3.) | _ -> failwith "arity")
             (function
               | [ a; i ] -> [ fst (split2 a); i ] | _ -> failwith "arity")
             [ k (); i32 0. ]) );
    ( "nested_identity_jit",
      fun () ->
        let g = A.jit_flat (one Fun.id) in
        jit (one (fun a -> normal (List.hd (g [ a ])))) [ k () ] );
    ( "nested_unused_jit",
      fun () ->
        let g = A.jit_flat (one (fun _ -> f32 0.)) in
        jit
          (one (fun a ->
               ignore (g [ a ]);
               normal a))
          [ k () ] );
    ( "cond_in_scan",
      fun () ->
        let flags = T.Concrete (Nd.of_floats D.Bool [| 2 |] [| 1.; 0. |]) in
        ignore
          (L.scan
             (function
               | [ c; p ] ->
                   let c, s = split2 c in
                   [
                     c;
                     List.hd
                       (L.cond p (one normal) (one (fun _ -> f32 0.)) [ s ]);
                   ]
               | _ -> failwith "arity")
             [ k () ]
             [ flags ]) );
    ( "scan_in_cond",
      fun () ->
        let inner a =
          let out =
            L.scan ~length:2
              (function
                | [ c ] ->
                    let c, s = split2 c in
                    [ c; normal s ]
                | _ -> failwith "arity")
              [ a ] []
          in
          Ojax.Numpy.Reductions.sum (List.nth out 1)
        in
        jit
          (function
            | [ a; p ] -> L.cond p (one inner) (one (fun _ -> f32 0.)) [ a ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "cond_branch_reuse",
      fun () ->
        jit
          (function
            | [ a; p ] ->
                L.cond p
                  (one (fun a -> U.add (normal a) (normal a)))
                  (one (fun _ -> f32 0.))
                  [ a ]
            | _ -> failwith "arity")
          [ k (); tt ] );
    ( "consumed_before_cond",
      fun () ->
        let a = k () in
        ignore (normal a);
        ignore (L.cond tt (one normal) (one (fun _ -> f32 0.)) [ a ]) );
    ( "consumed_before_unused_cond",
      fun () ->
        let a = k () in
        ignore (normal a);
        ignore (L.cond tt (one (fun _ -> f32 0.)) (one (fun _ -> f32 0.)) [ a ])
    );
    ( "consumed_closure_in_branch",
      fun () ->
        let a = k () in
        ignore (normal a);
        ignore (L.cond tt (fun _ -> [ normal a ]) (fun _ -> [ f32 0. ]) []) );
    ( "closure_after_cond",
      fun () ->
        let a = k () in
        ignore (L.cond tt (fun _ -> [ normal a ]) (fun _ -> [ f32 0. ]) []);
        ignore (normal a) );
    ("map_split", fun () -> ignore (L.map (one normal) [ R.split (k ()) 2 ]));
    ( "scan_carry_swap",
      fun () ->
        ignore
          (L.scan ~length:2
             (function [ a; b ] -> [ b; a; normal a ] | _ -> failwith "arity")
             [ k (); k ~seed:1. () ]
             []) );
  ]

let () =
  ignore first_key;
  A.debug_key_reuse true (fun () ->
      List.iter
        (fun (name, f) ->
          match f () with
          | () -> Printf.printf "%s OK\n%!" name
          | exception e ->
              let m =
                match Ojax.Errors.message_of e with
                | Some m -> m
                | None -> Printexc.to_string e
              in
              let m = String.concat " " (String.split_on_char '\n' m) in
              Printf.printf "%s ERR %s\n%!" name m)
        cases)
