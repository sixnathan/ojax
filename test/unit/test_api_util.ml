module Au = Ojax.Api_util
module Tu = Ojax.Tree_util
module Lu = Ojax.Linear_util

let int_tree =
  Alcotest.testable (Alcotest.pp (Alcotest.list Alcotest.int)) ( = )

let leaves t = Tu.tree_leaves t

let test_flatten_fun () =
  let f args =
    match args with
    | [ Tu.Leaf x; Tu.Tuple [ Tu.Leaf a; Tu.Leaf b ] ] ->
        Tu.Tuple [ Tu.Leaf (x + a); Tu.Leaf (x + b) ]
    | _ -> assert false
  in
  let arg_tuple = Tu.Tuple [ Tu.Leaf 10; Tu.Tuple [ Tu.Leaf 1; Tu.Leaf 2 ] ] in
  let flat, in_tree = Tu.tree_flatten arg_tuple in
  Alcotest.(check (list int)) "in flat" [ 10; 1; 2 ] flat;
  let wf, out_tree = Au.flatten_fun f in_tree in
  let out_flat = Lu.call_wrapped wf flat in
  Alcotest.(check (list int)) "out flat" [ 11; 12 ] out_flat;
  let out = Tu.tree_unflatten (out_tree ()) out_flat in
  Alcotest.check int_tree "roundtrip" [ 11; 12 ] (leaves out)

let test_argnums_partial () =
  let g args =
    match args with
    | [ Tu.Leaf a; Tu.Leaf b; Tu.Leaf c ] -> Tu.Leaf ((a * 100) + (b * 10) + c)
    | _ -> assert false
  in
  let args = [ Tu.Leaf 1; Tu.Leaf 2; Tu.Leaf 3 ] in
  let f_wrapped, dyn = Au.argnums_partial g [ 1 ] args in
  Alcotest.check int_tree "dyn" [ 2 ] (List.concat_map leaves dyn);
  let out = f_wrapped [ Tu.Leaf 5 ] in
  Alcotest.check int_tree "reassembled" [ 153 ] (leaves out)

let test_argnums_negative () =
  let g args =
    match args with [ _; _; Tu.Leaf c ] -> Tu.Leaf c | _ -> assert false
  in
  let args = [ Tu.Leaf 1; Tu.Leaf 2; Tu.Leaf 3 ] in
  let _, dyn = Au.argnums_partial g [ -1 ] args in
  Alcotest.check int_tree "neg resolves to last" [ 3 ]
    (List.concat_map leaves dyn)

let test_ensure_inbounds_error () =
  match Au.ensure_inbounds 2 [ 3 ] with
  | _ -> Alcotest.fail "expected out-of-bounds failure"
  | exception Invalid_argument _ -> ()

let () =
  Alcotest.run "api_util"
    [
      ( "api_util",
        [
          Alcotest.test_case "flatten_fun" `Quick test_flatten_fun;
          Alcotest.test_case "argnums_partial" `Quick test_argnums_partial;
          Alcotest.test_case "argnums_negative" `Quick test_argnums_negative;
          Alcotest.test_case "ensure_inbounds_error" `Quick
            test_ensure_inbounds_error;
        ] );
    ]
