module S = Ojax.Source_info_util

let str = Alcotest.string

let name_stack_format () =
  let ns = S.new_name_stack "top" in
  Alcotest.check str "single scope" "top" (S.name_stack_str ns);
  let ns2 = S.extend ns "inner" in
  Alcotest.check str "nested scope" "top/inner" (S.name_stack_str ns2);
  let ns3 = S.transform ns2 "jvp" in
  Alcotest.check str "trailing transform" "top/inner/jvp()"
    (S.name_stack_str ns3);
  let ns4 = S.transform S.empty_name_stack "vmap" in
  Alcotest.check str "transform on empty" "vmap()" (S.name_stack_str ns4);
  let ns5 = S.transform (S.extend ns3 "more") "grad" in
  Alcotest.check str "transform wraps prior scope" "top/inner/jvp(more)/grad()"
    (S.name_stack_str ns5);
  Alcotest.check str "empty name stack" ""
    (S.name_stack_str (S.new_name_stack ""))

let source_info_stub () =
  let si = S.new_source_info () in
  Alcotest.check str "fresh source info has empty name stack" ""
    (S.name_stack_str (S.name_stack si));
  Alcotest.check str "summarize is empty" "" (S.summarize (S.current ()));
  S.register_exclusion "some/path";
  let f = S.api_boundary (fun x -> x + 1) in
  Alcotest.(check int) "api_boundary is identity" 6 (f 5)

let () =
  Alcotest.run "source_info_util"
    [
      ( "name_stack",
        [ Alcotest.test_case "formatting" `Quick name_stack_format ] );
      ("stub", [ Alcotest.test_case "opaque stubs" `Quick source_info_stub ]);
    ]
