module L = Ojax.Linear_util

let ints = Alcotest.(list int)

let store_write_once () =
  let s = L.new_store () in
  L.store s 7;
  Alcotest.(check int) "stored value read back" 7 (L.store_val s);
  Alcotest.check_raises "double write raises"
    (L.Store_exception "Store occupied") (fun () -> L.store s 8)

let store_empty_read () =
  let s = L.new_store () in
  Alcotest.check_raises "empty read raises" (L.Store_exception "Store empty")
    (fun () -> ignore (L.store_val s))

let store_reset () =
  let s = L.new_store () in
  L.store s 1;
  L.reset s;
  L.store s 2;
  Alcotest.(check int) "reset allows rewrite" 2 (L.store_val s)

let wrap_and_call () =
  let w = L.wrap_init (fun xs -> List.map (fun x -> x + 1) xs) in
  Alcotest.check ints "call_wrapped runs the function" [ 2; 3; 4 ]
    (L.call_wrapped w [ 1; 2; 3 ])

let transformation_stack () =
  let base = L.wrap_init (fun xs -> List.map (fun x -> x * 10) xs) in
  let double_inputs down xs = down (List.map (fun x -> x * 2) xs) in
  let w = L.transformation2 double_inputs base in
  Alcotest.check ints "transform preprocesses args before downstream" [ 20; 40 ]
    (L.call_wrapped w [ 1; 2 ])

let transformation_with_aux () =
  let base = L.wrap_init (fun xs -> List.map (fun x -> x + 1) xs) in
  let gen down st xs =
    let ys = down xs in
    L.store st (List.length ys);
    List.map (fun y -> y * 2) ys
  in
  let w, aux = L.transformation_with_aux2 gen base in
  let out = L.call_wrapped w [ 5; 6; 7 ] in
  Alcotest.check ints "aux transform postprocesses results" [ 12; 14; 16 ] out;
  Alcotest.(check int) "aux thunk holds stored value" 3 (aux ())

let merge_aux () =
  let full v () = v in
  let empty () = raise (L.Store_exception "Store empty") in
  Alcotest.(check (pair bool int))
    "first occupied" (true, 9)
    (L.merge_linear_aux (full 9) empty);
  Alcotest.(check (pair bool int))
    "second occupied" (false, 4)
    (L.merge_linear_aux empty (full 4));
  Alcotest.check_raises "neither occupied"
    (L.Store_exception "neither store occupied") (fun () ->
      ignore (L.merge_linear_aux empty empty));
  Alcotest.check_raises "both occupied"
    (L.Store_exception "both stores occupied") (fun () ->
      ignore (L.merge_linear_aux (full 1) (full 2)))

let () =
  Alcotest.run "linear_util"
    [
      ( "store",
        [
          Alcotest.test_case "write once" `Quick store_write_once;
          Alcotest.test_case "empty read" `Quick store_empty_read;
          Alcotest.test_case "reset" `Quick store_reset;
        ] );
      ( "wrapped_fun",
        [
          Alcotest.test_case "wrap and call" `Quick wrap_and_call;
          Alcotest.test_case "transformation2" `Quick transformation_stack;
          Alcotest.test_case "transformation_with_aux2" `Quick
            transformation_with_aux;
          Alcotest.test_case "merge_linear_aux" `Quick merge_aux;
        ] );
    ]
