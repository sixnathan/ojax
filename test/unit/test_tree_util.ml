module T = Ojax.Tree_util

let ints = Alcotest.(list int)

let sample =
  T.Tuple
    [
      T.Leaf 1;
      T.List [ T.Leaf 2; T.Null; T.Leaf 3 ];
      T.Dict [ ("b", T.Leaf 4); ("a", T.Leaf 5) ];
    ]

let leaves_sorted () =
  Alcotest.check ints "leaves emitted with dict keys sorted" [ 1; 2; 3; 5; 4 ]
    (T.tree_leaves sample)

let roundtrip () =
  let leaves, def = T.tree_flatten sample in
  let rebuilt = T.tree_unflatten def leaves in
  Alcotest.check ints "unflatten of flatten preserves leaves"
    (T.tree_leaves sample) (T.tree_leaves rebuilt);
  Alcotest.(check bool)
    "treedef stable after roundtrip" true
    (T.tree_structure rebuilt = def)

let unflatten_new_leaves () =
  let _, def = T.tree_flatten sample in
  let rebuilt = T.tree_unflatten def [ 10; 20; 30; 40; 50 ] in
  Alcotest.check ints "unflatten places new leaves positionally"
    [ 10; 20; 30; 40; 50 ] (T.tree_leaves rebuilt)

let map_preserves_structure () =
  let mapped = T.tree_map (fun x -> x * 2) sample in
  Alcotest.check ints "map over leaves" [ 2; 4; 6; 10; 8 ]
    (T.tree_leaves mapped);
  Alcotest.(check bool)
    "structure unchanged by map" true
    (T.tree_structure mapped = T.tree_structure sample)

let arity_errors () =
  let _, def = T.tree_flatten sample in
  Alcotest.check_raises "too few leaves"
    (Invalid_argument "tree_unflatten: too few leaves for treedef") (fun () ->
      ignore (T.tree_unflatten def [ 1; 2 ]));
  Alcotest.check_raises "too many leaves"
    (Invalid_argument "tree_unflatten: too many leaves for treedef") (fun () ->
      ignore (T.tree_unflatten def [ 1; 2; 3; 4; 5; 6 ]))

let () =
  Alcotest.run "tree_util"
    [
      ( "flatten",
        [
          Alcotest.test_case "leaves sorted" `Quick leaves_sorted;
          Alcotest.test_case "roundtrip" `Quick roundtrip;
          Alcotest.test_case "unflatten new leaves" `Quick unflatten_new_leaves;
          Alcotest.test_case "arity errors" `Quick arity_errors;
        ] );
      ( "map",
        [
          Alcotest.test_case "preserves structure" `Quick
            map_preserves_structure;
        ] );
    ]
