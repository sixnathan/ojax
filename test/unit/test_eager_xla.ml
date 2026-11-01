open Ojax
open Ojax.Types

let () =
  Lax.install ();
  Random.Prng.install ();
  Random.Threefry.install ()

let need_plugin () =
  if Sys.getenv_opt Pjrt.Discover.env_var = None then Alcotest.skip ()

let f32 shape xs = Concrete (Ndarray.of_floats Dtype.F32 shape xs)

let floats = function
  | Concrete nd ->
      let n = Array.fold_left ( * ) 1 (Ndarray.shape nd) in
      Array.init n (fun i ->
          Ndarray.get_f nd
            (if Array.length (Ndarray.shape nd) = 0 then [||] else [| i |]))
  | Device _ | Tracer _ -> Alcotest.fail "expected a concrete result"

let bits xs = Array.map Int64.bits_of_float xs

let test_cache_hit () =
  need_plugin ();
  let x = f32 [| 3 |] [| 1.5; -2.25; 0.125 |] in
  let y = f32 [| 3 |] [| 4.0; 0.5; 8.0 |] in
  let first = Backend.eager_stats () in
  let a = Core.bind1 Mul [ x; y ] in
  let second = Backend.eager_stats () in
  let b = Core.bind1 Mul [ x; y ] in
  let third = Backend.eager_stats () in
  Alcotest.(check int)
    "the first call compiles" (first.eager_compiled + 1) second.eager_compiled;
  Alcotest.(check int)
    "the second call reuses the program" second.eager_compiled
    third.eager_compiled;
  Alcotest.(check int)
    "the second call hits the cache"
    (second.eager_cache_hits + 1)
    third.eager_cache_hits;
  Alcotest.(check (array int64))
    "same bytes"
    (bits (floats a))
    (bits (floats b))

let test_key_has_shape_and_dtype () =
  need_plugin ();
  let compiled () = (Backend.eager_stats ()).eager_compiled in
  let start = compiled () in
  ignore (Core.bind1 Neg [ f32 [| 2 |] [| 1.0; 2.0 |] ]);
  ignore (Core.bind1 Neg [ f32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] ]);
  ignore
    (Core.bind1 Neg
       [ Concrete (Ndarray.of_floats Dtype.F64 [| 2 |] [| 1.0; 2.0 |]) ]);
  Alcotest.(check int) "three programs" (start + 3) (compiled ())

let test_matches_exact_values () =
  need_plugin ();
  let x = f32 [| 2 |] [| 0.5; 2.0 |] in
  let y = f32 [| 2 |] [| 0.25; 4.0 |] in
  let out = floats (Core.bind1 Add [ x; y ]) in
  Alcotest.(check (array int64)) "sum" (bits [| 0.75; 6.0 |]) (bits out)

let test_fallback_recorded () =
  need_plugin ();
  let matrix = f32 [| 2; 2 |] [| 2.0; 1.0; 1.0; 3.0 |] in
  let outs = Core.bind Hessenberg [ matrix ] in
  Alcotest.(check bool) "host produced results" true (outs <> []);
  let names =
    List.map (fun (name, _, _) -> name) (Backend.eager_stats ()).eager_fallbacks
  in
  Alcotest.(check bool)
    "hessenberg fell back" true
    (List.mem "hessenberg" names)

let test_region_compiles_whole_function () =
  need_plugin ();
  let x = f32 [| 2 |] [| 1.0; 4.0 |] in
  let calls = ref 0 in
  let f v =
    incr calls;
    Core.bind1 Sqrt [ Core.bind1 Add [ v; v ] ]
  in
  let run () =
    Eager_jit.one "test.region" () [] [ x ] (fun _ vals ->
        match vals with [ v ] -> f v | _ -> Alcotest.fail "arguments")
  in
  let a = floats (run ()) in
  let traced = !calls in
  let b = floats (run ()) in
  Alcotest.(check int) "traced once" traced !calls;
  Alcotest.(check (array int64)) "same bytes" (bits a) (bits b)

let test_region_key_tells_options_apart () =
  need_plugin ();
  let x = f32 [| 2 |] [| 1.0; 4.0 |] in
  let y = f32 [| 2 |] [| 2.0; 8.0 |] in
  let run ~left ~right =
    Eager_jit.one "test.options" () [ left; right ] [ x ] (fun opts vals ->
        match (opts, vals) with
        | [ Some l; None ], [ v ] -> Core.bind1 Add [ v; l ]
        | [ None; Some r ], [ v ] -> Core.bind1 Mul [ v; r ]
        | _ -> Alcotest.fail "arguments")
  in
  let added = floats (run ~left:(Some y) ~right:None) in
  let multiplied = floats (run ~left:None ~right:(Some y)) in
  Alcotest.(check (array int64)) "add" (bits [| 3.0; 12.0 |]) (bits added);
  Alcotest.(check (array int64)) "mul" (bits [| 2.0; 32.0 |]) (bits multiplied)

let () =
  Alcotest.run "eager_xla"
    [
      ( "eager",
        [
          Alcotest.test_case "cache hit" `Quick test_cache_hit;
          Alcotest.test_case "cache key" `Quick test_key_has_shape_and_dtype;
          Alcotest.test_case "values" `Quick test_matches_exact_values;
          Alcotest.test_case "fallback" `Quick test_fallback_recorded;
          Alcotest.test_case "region" `Quick test_region_compiles_whole_function;
          Alcotest.test_case "region options" `Quick
            test_region_key_tells_options_apart;
        ] );
    ]
