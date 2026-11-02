open Ojax
open Ojax.Types
module T = Tree_util
module P = Shape_poly
module D = Dtype
module X = Interpreters

let () = Lax.install ()
let () = Config.set Config.num_cpu_devices 2
let () = Config.set Config.platforms "cpu"
let aval ?(weak = false) dtype shape = { shape; dtype; weak_type = weak }
let f32a = aval D.F32
let i32a = aval D.I32
let f32 shape xs = Concrete (Ndarray.of_floats D.F32 shape xs)
let i32 shape xs = Concrete (Ndarray.of_floats D.I32 shape xs)
let i32_arr shape xs = Ndarray.of_floats D.I32 shape xs

let host = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected concrete"

let flat v =
  let nd = host v in
  let shape = Ndarray.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  Array.init n (fun i ->
      let idx = Array.make (Array.length shape) 0 in
      let r = ref i in
      for d = Array.length shape - 1 downto 0 do
        idx.(d) <- !r mod shape.(d);
        r := !r / shape.(d)
      done;
      Ndarray.get_f nd idx)

let check_floats msg expected v =
  Alcotest.(check (array (float 0.0))) msg expected (flat v)

let check_shape msg expected v =
  Alcotest.(check (array int)) msg expected (Ndarray.shape (host v))

let check_dtype msg expected v =
  Alcotest.(check string)
    msg (D.numpy_name expected)
    (D.numpy_name (Ndarray.dtype (host v)))

let raises_msg msg expected f =
  match f () with
  | _ -> Alcotest.failf "%s: expected an error" msg
  | exception (Invalid_argument m | Failure m) ->
      Alcotest.(check string) msg expected m

let raises_prefix msg prefix f =
  match f () with
  | _ -> Alcotest.failf "%s: expected an error" msg
  | exception
      ( Invalid_argument m
      | Failure m
      | P.Inconclusive_dimension_operation m
      | Pjrt.Abi.Error m
      | Errors.Jax_runtime_error m ) ->
      if not (String.starts_with ~prefix m) then
        Alcotest.failf "%s: got %S, wanted prefix %S" msg m prefix

let dims_str dims = List.map P.dim_to_string (Array.to_list dims)

let ss ?constraints ?scope ?like s =
  P.symbolic_shape ?constraints ?scope ?like (Some s)

let check_strs = Alcotest.(check (list string))

let expr_strings () =
  check_strs "parse and normal form"
    [ "a"; "b + 1"; "2*c"; "a*b"; "a^2"; "floordiv(a, 2)"; "mod(a, 3)";
      "max(a, 2)"; "min(a, 2)"; "3"; "0"; "a"; "- a"; "a^2*b"; "b + a + 1";
      "b + a"; "a*b"; "5*a*b - 3*a + 2" ]
    (dims_str
       (ss
          "a, b+1, 2*c, a*b, a^2, floordiv(a,2), mod(a,3), max(a,2), min(a,2), \
           3, a-a, 2*a-a, -a, a*a*b, a+b+1, b+a, b*a, 5*a*b-3*a+2"))
[@@ocamlformat "disable"]

let two_vars () =
  match ss "a,b" with [| a; b |] -> (a, b) | _ -> Alcotest.fail "arity"

let arithmetic () =
  let a, b = two_vars () in
  let c = P.of_int in
  let s = P.dim_to_string in
  check_strs "ring ops"
    [ "a + 1"; "a + 1"; "a - 1"; "- a + 1"; "3*a"; "0"; "- a"; "a*b"; "a^2";
      "a^2"; "a^2 - 1"; "a^2 + 2*a + 1" ]
    [ s (P.add a (c 1)); s (P.add (c 1) a); s (P.sub a (c 1)); s (P.sub (c 1) a);
      s (P.mul a (c 3)); s (P.mul a (c 0)); s (P.neg a); s (P.mul a b);
      s (P.pow a 2); s (P.mul a a);
      s (P.mul (P.add a (c 1)) (P.sub a (c 1)));
      s (P.pow (P.add a (c 1)) 2) ];
  let two_a = P.mul (c 2) a in
  let two_a1 = P.add two_a (c 1) in
  let ab = P.mul a b in
  check_strs "division"
    [ "floordiv(a, 2)"; "mod(a, 2)"; "a"; "0"; "a"; "1"; "floordiv(4, a)";
      "mod(4, a)"; "b"; "0"; "b + 1"; "floordiv(b + a, a)"; "floordiv(a, b)";
      "mod(a, b)"; "floordiv(6*a, 4)"; "mod(6*a, 4)"; "a"; "3" ]
    [ s (P.floordiv_dim a (c 2)); s (P.mod_dim a (c 2));
      s (P.floordiv_dim two_a (c 2)); s (P.mod_dim two_a (c 2));
      s (P.floordiv_dim two_a1 (c 2)); s (P.mod_dim two_a1 (c 2));
      s (P.floordiv_dim (c 4) a); s (P.mod_dim (c 4) a);
      s (P.floordiv_dim ab a); s (P.mod_dim ab a);
      s (P.floordiv_dim (P.add ab a) a);
      s (P.floordiv_dim (P.add a b) a);
      s (P.floordiv_dim a b); s (P.mod_dim a b);
      s (fst (P.divmod (P.mul (c 6) a) (c 4)));
      s (snd (P.divmod (P.mul (c 6) a) (c 4)));
      s (fst (P.divmod (P.add (P.mul (c 7) a) (c 3)) (c 7)));
      s (snd (P.divmod (P.add (P.mul (c 7) a) (c 3)) (c 7))) ];
  Alcotest.(check (list bool))
    "predicates"
    [ true; false; false; true; false; true; false; true ]
    [ P.is_symbolic_dim a; P.is_symbolic_dim (c 3);
      P.is_symbolic_dim (P.sub a a); P.equal a a; P.equal a b;
      P.equal (P.add a (c 1)) (P.add (c 1) a); P.equal a (c 1);
      P.equal (P.sub a a) (c 0) ];
  check_strs "max and min"
    [ "a"; "a"; "a"; "max(a, 3)"; "max(3, a)"; "max(a, b)"; "1"; "0"; "0";
      "min(a, 3)"; "min(3, a)"; "min(a, b)" ]
    [ s (P.max_dim a (c 1)); s (P.max_dim a (c 0)); s (P.max_dim (c 1) a);
      s (P.max_dim a (c 3)); s (P.max_dim (c 3) a); s (P.max_dim a b);
      s (P.min_dim a (c 1)); s (P.min_dim a (c 0)); s (P.min_dim (c 0) a);
      s (P.min_dim a (c 3)); s (P.min_dim (c 3) a); s (P.min_dim a b) ];
  raises_msg "negative power"
    "Symbolic dimension cannot be raised to negative powers: 'a' ** '-1'"
    (fun () -> P.pow a (-1));
  check_strs "power zero" [ "1"; "a^3" ] [ s (P.pow a 0); s (P.pow a 3) ]
[@@ocamlformat "disable"]

let comparisons () =
  let a, b = two_vars () in
  let c = P.of_int in
  let inconclusive msg text f =
    raises_prefix msg
      (Printf.sprintf "Symbolic dimension comparison %s is inconclusive." text)
      f
  in
  Alcotest.(check (list bool)) "conclusive"
    [ true; true; false; true; false; true; true ]
    [ P.ge a (c 1); P.ge a (c 0); P.lt a (c 1); P.gt a (c 0); P.le a (c 0);
      P.gt (P.add a (c 1)) a; P.ge (P.mul (c 2) a) a ];
  inconclusive "a>=2" "'a' >= '2'" (fun () -> P.ge a (c 2));
  inconclusive "a<=1" "'a' <= '1'" (fun () -> P.le a (c 1));
  inconclusive "a>=b" "'a' >= 'b'" (fun () -> P.ge a b);
  inconclusive "ab>=a" "'a*b' >= 'a'" (fun () -> P.ge (P.mul a b) a);
  let m = P.mod_dim a (c 3) and f = P.floordiv_dim a (c 2) in
  Alcotest.(check (list bool)) "implicit constraints"
    [ true; true; true; true ]
    [ P.ge m (c 0); P.lt m (c 3); P.ge f (c 0);
      P.ge (P.sub a (P.mod_dim a (c 2))) (c 0) ];
  inconclusive "mod>=1" "'mod(a, 3)' >= '1'" (fun () -> P.ge m (c 1));
  inconclusive "floordiv>=1" "'floordiv(a, 2)' >= '1'" (fun () -> P.ge f (c 1))
[@@ocamlformat "disable"]

let bounds_table () =
  List.iter
    (fun (spec, lb, ub) ->
      match ss spec with
      | [| d |] ->
          let l, u = P.bounds_decision d P.Best in
          Alcotest.(check (pair (float 0.0) (float 0.0))) spec (lb, ub) (l, u)
      | _ -> Alcotest.fail "arity")
    [
      ("a", 1.0, infinity);
      ("a-1", 0.0, infinity);
      ("2*a+1", 3.0, infinity);
      ("a-b", neg_infinity, infinity);
      ("mod(a,3)", 0.0, 2.0);
      ("floordiv(a,2)", 0.0, infinity);
      ("a*b", 1.0, infinity);
      ("max(a,2)", 2.0, infinity);
      ("min(a,2)", 1.0, 2.0);
      ("floordiv(a,2)*2", 0.0, infinity);
      ("mod(a,3)+a", 1.0, infinity);
      ("1-a", neg_infinity, 0.0);
      ("a^2", 1.0, infinity);
      ("a*b-a", neg_infinity, infinity);
      ("max(a,b)-a", 0.0, infinity);
      ("min(a,b)-a", neg_infinity, 0.0);
      ("floordiv(a,b)", 0.0, infinity);
      ("mod(a,b)", 0.0, infinity);
      ("a-mod(a,2)", 0.0, infinity);
      ("floordiv(a,-2)", neg_infinity, -1.0);
      ("mod(a,-3)", -2.0, 0.0);
    ]

let constraints () =
  let scope =
    P.symbolic_scope
      ~constraints:
        [ "a >= 3"; "b <= 5"; "c == a+1"; "mod(d,4) == 0"; "e*2 == f" ]
      ()
  in
  let q = ss ~scope "a,b,c,d,e,f,a+c,e*2,4*floordiv(d,4),2*e+1,a*c" in
  check_strs "normalized by the scope"
    [ "a"; "b"; "a + 1"; "d"; "e"; "f"; "2*a + 1"; "f"; "d"; "f + 1"; "a^2 + a" ]
    (dims_str q);
  let bound i = P.bounds_decision q.(i) P.Best in
  Alcotest.(check (pair (float 0.0) (float 0.0)))
    "a" (3.0, infinity) (bound 0);
  Alcotest.(check (pair (float 0.0) (float 0.0))) "b" (1.0, 5.0) (bound 1);
  Alcotest.(check (pair (float 0.0) (float 0.0)))
    "c" (4.0, infinity) (bound 2);
  let c = P.of_int in
  Alcotest.(check (list bool)) "decisions"
    [ true; true; true; true ]
    [ P.ge q.(0) (c 3); P.le q.(1) (c 5); P.lt q.(1) (c 6); P.ge q.(2) (c 4) ];
  raises_prefix "a>=4" "Symbolic dimension comparison 'a' >= '4' is inconclusive."
    (fun () -> P.ge q.(0) (c 4));
  check_strs "mod normalizes to 0" [ "0" ] [ P.dim_to_string (P.mod_dim q.(3) (c 4)) ];
  let parse_err cs expected =
    raises_msg (String.concat ";" cs) expected (fun () ->
        P.symbolic_shape ~constraints:cs (Some "a"))
  in
  parse_err [ "a" ] "Constraint parsing error: must contain one of '==' or '>=' or '<='";
  parse_err [ "a >= b >= c" ]
    "syntax error in symbolic shape ' b >= c' in dimension 0: expecting one of \
     {STAR} but found GREATEREQUAL. Parsed ' b ', remaining '>= c'.";
  parse_err [ "1 >= 2" ] "Unsatisfiable explicit constraint: 1 >= 2";
  parse_err [ "2 == 3" ] "Unsatisfiable explicit constraint: 2 == 3";
  let eq_err =
    "Invalid equality constraint: {e1} == {e2}. The left-hand-side must be of \
     the form `term * coefficient`."
  in
  parse_err [ "a+b == 3" ] eq_err;
  parse_err [ "3 == a" ] eq_err;
  parse_err [ "a == 1"; "a == 2" ] "Unsatisfiable explicit constraint: a == 2";
  parse_err [ "a, b >= 2" ] "too many values to unpack (expected 1)";
  check_strs "unsat is not checked eagerly" [ "a" ]
    (dims_str (ss ~constraints:[ "a >= 3"; "a <= 2" ] "a"));
  raises_msg "scope and constraints"
    "Cannot specify both a `scope` and `constraints`." (fun () ->
      P.symbolic_shape ~constraints:[ "a>=1" ] ~scope:(P.symbolic_scope ())
        (Some "a"))
[@@ocamlformat "disable"]

let parser_table () =
  let some = Array.map (fun x -> Some x) in
  let ok ?like spec expected =
    check_strs spec expected (dims_str (ss ?like spec))
  in
  ok "" [];
  ok "()" [];
  ok "(a,b)" [ "a"; "b" ];
  ok "a," [ "a" ];
  ok "((a))" [ "a" ];
  ok "2,3" [ "2"; "3" ];
  ok ~like:(some [| 2; 3 |]) "..." [ "2"; "3" ];
  ok ~like:[| Some 2; None |] "_,a" [ "2"; "a" ];
  ok ~like:[| None; Some 3 |] "a,_" [ "a"; "3" ];
  ok ~like:(some [| 1; 2; 3 |]) "a,..." [ "a"; "2"; "3" ];
  ok ~like:(some [| 2 |]) "2" [ "2" ];
  ok "-3" [ "-3" ];
  ok "-a+2" [ "- a + 2" ];
  ok "+a" [ "a" ];
  ok "a-b-c" [ "- c - b + a" ];
  ok "a*-2" [ "- 2*a" ];
  ok "max(a,2)+min(a,2)" [ "min(a, 2) + max(a, 2)" ];
  let err ?like spec tail =
    raises_msg spec
      (Printf.sprintf "syntax error in symbolic shape '%s' in dimension %s" spec
         tail) (fun () -> ss ?like spec)
  in
  err ~like:[| None |] "_"
    "0: unexpected placeholder for unknown dimension; like=(None,). Parsed '', \
     remaining ''.";
  err ~like:(some [| 3 |]) "2"
    "0: different size 2 for known dimension; like=(3,). Parsed '', remaining \
     ''.";
  err ~like:(some [| 1 |]) "a,_,_"
    "1: cannot resolve placeholder '_' because we parsed 1 already and 'like' \
     shape has only 1 dimensions. Parsed 'a,', remaining '_,_'.";
  err "..."
    "0: spec contains ... but no 'like' shape was given. Parsed '', remaining \
     '...'.";
  err "_"
    "0: spec contains ... but no 'like' shape was given. Parsed '', remaining \
     '_'.";
  err "a+"
    "0: expecting one of {NAME, MINUS, NUMBER} but found ENDMARKER. Parsed '', \
     remaining ''.";
  err "a b" "0: expecting one of {STAR} but found NAME. Parsed 'a ', remaining 'b'.";
  err "1.5" "0: expecting integer, found 1.5. Parsed '', remaining '1.5'.";
  err "a/2" "0: expecting one of {STAR} but found SLASH. Parsed 'a', remaining '/2'.";
  err "a^b" "0: expecting one of {NUMBER} but found NAME. Parsed 'a^', remaining 'b'.";
  err "a^-1"
    "0: expecting one of {NUMBER} but found MINUS. Parsed 'a^', remaining '-1'.";
  err "a**2"
    "0: expecting one of {STAR} but found DOUBLESTAR. Parsed 'a', remaining \
     '**2'.";
  err "a,,b"
    "1: expecting one of {NAME, MINUS, NUMBER} but found COMMA. Parsed 'a,', \
     remaining ',b'.";
  err "(a" "0: expecting one of {STAR} but found NL. Parsed '(a', remaining ''.";
  err "a)" "1: expecting one of {ENDMARKER} but found RPAR. Parsed 'a', remaining ')'.";
  err "floordiv(a)"
    "0: expecting one of {COMMA} but found RPAR. Parsed 'floordiv(a', remaining \
     ')'.";
  err "mod(a,b,c)"
    "0: expecting one of {RPAR} but found COMMA. Parsed 'mod(a,b', remaining \
     ',c)'.";
  err "2a" "0: expecting one of {STAR} but found NAME. Parsed '2', remaining 'a'.";
  err "%"
    "0: expecting one of {NAME, MINUS, NUMBER} but found PERCENT. Parsed '', \
     remaining '%'.";
  err ~like:(some [| 2; 4 |]) "a,3"
    "1: different size 3 for known dimension; like=(2, 4). Parsed '', remaining \
     ''.";
  check_strs "none takes like" [ "2"; "3" ]
    (dims_str (P.symbolic_shape ~like:(some [| 2; 3 |]) None));
  raises_msg "none without like"
    "syntax error in symbolic shape None in dimension 0: spec contains ... but \
     no 'like' shape was given. Parsed '', remaining '...'." (fun () ->
      P.symbolic_shape None)
[@@ocamlformat "disable"]

let specs_of tree =
  List.map
    (fun (s : P.spec) -> (dims_str s.shape, D.numpy_name s.dtype))
    (T.tree_leaves tree)

let args_specs () =
  let x = f32a [| 2; 3 |] and y = i32a [| 4 |] in
  let args = T.Tuple [ T.Leaf x; T.Leaf y ] in
  let leaf s = T.Leaf (Some s) in
  let check msg expected spec =
    Alcotest.(check (list (pair (list string) string)))
      msg expected
      (specs_of (P.symbolic_args_specs args spec))
  in
  check "per arg"
    [ ([ "a"; "3" ], "float32"); ([ "c" ], "int32") ]
    (T.Tuple [ leaf "a,_"; leaf "c" ]);
  check "one string for all"
    [ ([ "a"; "3" ], "float32"); ([ "a" ], "int32") ]
    (leaf "a,...");
  check "none is static"
    [ ([ "2"; "3" ], "float32"); ([ "4" ], "int32") ]
    (T.Leaf None);
  check "none for one arg"
    [ ([ "2"; "3" ], "float32"); ([ "c" ], "int32") ]
    (T.Tuple [ T.Leaf None; leaf "c" ]);
  check "list prefix for a tuple"
    [ ([ "a"; "3" ], "float32"); ([ "b" ], "int32") ]
    (T.List [ leaf "a,3"; leaf "b" ]);
  let with_constraints =
    P.symbolic_args_specs ~constraints:[ "a>=2" ] args
      (T.Tuple [ leaf "a,b"; leaf "a" ])
  in
  Alcotest.(check (list (pair (list string) string)))
    "constraints"
    [ ([ "a"; "b" ], "float32"); ([ "a" ], "int32") ]
    (specs_of with_constraints);
  raises_prefix "prefix mismatch" "pytree structure error" (fun () ->
      P.symbolic_args_specs args (T.Tuple [ leaf "a"; leaf "b"; leaf "c" ]));
  raises_msg "scope and constraints" "Cannot use both `scope` and `constraints`"
    (fun () ->
      P.symbolic_args_specs ~constraints:[ "a>=1" ] ~scope:(P.symbolic_scope ())
        args (leaf "a,b"));
  let s1 = (ss "a").(0) and s2 = (ss "a").(0) in
  raises_prefix "scopes do not mix"
    "Invalid mixing of symbolic scopes for linear combination." (fun () ->
      P.add s1 s2);
  Alcotest.(check bool) "different scopes are unequal" false (P.equal s1 s2);
  raises_prefix "scopes do not compare"
    "Invalid mixing of symbolic scopes when comparing 'a' >= 'a'." (fun () ->
      P.ge s1 s2)

let spec shape dtype : P.spec = { shape; dtype }

let symbolic_export () =
  let sum_axis0 = function
    | [ x ] ->
        [ Lax_api.reduce_sum (Lax_api.add x (Lax_api.full_like x 1.0)) [| 0 |] ]
    | _ -> Alcotest.fail "arity"
  in
  let a = ss "a" in
  let e =
    Export.export_symbolic sum_axis0
      [ spec [| a.(0); P.of_int 3 |] D.F32 ]
  in
  let ones shape =
    let n = Array.fold_left ( * ) 1 shape in
    f32 shape (Array.make n 1.0)
  in
  (match Export.call_symbolic e [ ones [| 2; 3 |] ] with
  | [ r ] -> check_floats "a=2" [| 4.; 4.; 4. |] r
  | _ -> Alcotest.fail "arity");
  (match Export.call_symbolic e [ ones [| 5; 3 |] ] with
  | [ r ] -> check_floats "a=5" [| 10.; 10.; 10. |] r
  | _ -> Alcotest.fail "arity");
  raises_msg "constant mismatch"
    "Shape mismatch for args[0].shape[1] (expected same constant): expected (a, \
     3) and called with (2, 4)" (fun () ->
      Export.call_symbolic e [ ones [| 2; 4 |] ]);
  raises_msg "rank mismatch"
    "Rank mismatch for args[0]: expected (a, 3) and called with (2,)" (fun () ->
      Export.call_symbolic e [ ones [| 2 |] ]);
  raises_msg "dtype mismatch"
    "Dtype mismatch for args[0]: expected float32 and called with int32"
    (fun () -> Export.call_symbolic e [ i32 [| 2; 3 |] (Array.make 6 1.0) ]);
  raises_prefix "zero dim"
    "Input shapes do not match the polymorphic shapes specification. Expected \
     value >= 1 for dimension variable 'a'. Using the following polymorphic \
     shapes specifications: args[0].shape = (a, 3). Obtained dimension \
     variables: 'a' = 0 from specification 'a' for dimension args[0].shape[0] \
     (= 0), . Please see"
    (fun () -> Export.call_symbolic e [ ones [| 0; 3 |] ]);
  let sum1 = function [ x ] -> [ Lax_api.reduce_sum x [| 0 |] ] | _ -> [] in
  Alcotest.(check (list bool)) "export aliases" [ true; false ]
    [ Export.is_symbolic_dim (Export.symbolic_shape (Some "a")).(0);
      Export.is_symbolic_dim
        (Export.symbolic_shape ~scope:(Export.symbolic_scope ()) (Some "3")).(0) ];
  let run ?constraints text arg =
    let d = P.symbolic_shape ?constraints (Some text) in
    let f = function
      | [ x ] ->
          [
            Lax_api.reduce_sum x
              (Array.init (Array.length d) (fun i -> i));
          ]
      | _ -> []
    in
    let ex = Export.export_symbolic f [ spec d D.F32 ] in
    match Export.call_symbolic ex [ ones arg ] with
    | [ r ] -> (flat r).(0)
    | _ -> Alcotest.fail "arity"
  in
  Alcotest.(check (float 0.0)) "2*a" 4.0 (run "2*a" [| 4 |]);
  Alcotest.(check (float 0.0)) "a,a+1" 6.0 (run "a,a+1" [| 2; 3 |]);
  raises_prefix "a,a*b"
    "Cannot solve for values of dimension variables {'b'}. We can only solve \
     linear uni-variate constraints. Using the following polymorphic shapes \
     specifications: args[0].shape = (a, a*b). Unprocessed specifications: \
     'a*b' for dimension size args[0].shape[1]." (fun () ->
      run "a,a*b" [| 2; 6 |]);
  raises_prefix "a,2*a+1"
    "Input shapes do not match the polymorphic shapes specification. Found \
     inconsistency between dimension size args[0].shape[1] (= 6) and the \
     specification '2*a + 1' (= 5). Using the following polymorphic shapes \
     specifications: args[0].shape = (a, 2*a + 1). Obtained dimension \
     variables: 'a' = 2 from spe" (fun () -> run "a,2*a+1" [| 2; 6 |]);
  Alcotest.(check (float 0.0)) "a*2,a" 8.0 (run "a*2,a" [| 4; 2 |]);
  Alcotest.(check (float 0.0)) "mod(a,2),a" 3.0 (run "mod(a,2),a" [| 1; 3 |]);
  Alcotest.(check (float 0.0)) "floordiv(a,2),a" 3.0
    (run "floordiv(a,2),a" [| 1; 3 |]);
  Alcotest.(check (float 0.0)) "a+b,a" 10.0 (run "a+b,a" [| 5; 2 |]);
  raises_prefix "remainder"
    "Input shapes do not match the polymorphic shapes specification. Division \
     had remainder 1 when computing the value of 'a'. Using the following \
     polymorphic shapes specifications: args[0].shape = (2*a,)." (fun () ->
      run "2*a" [| 3 |]);
  raises_prefix "inconsistent"
    "Input shapes do not match the polymorphic shapes specification. Found \
     inconsistency between dimension size args[0].shape[1] (= 3) and the \
     specification 'a' (= 2). Using the following polymorphic shapes \
     specifications: args[0].shape = (a, a)." (fun () ->
      run "a,a" [| 2; 3 |]);
  raises_msg "unsolvable"
    "Cannot solve for values of dimension variables {'a', 'b'}. We can only \
     solve linear uni-variate constraints. Using the following polymorphic \
     shapes specifications: args[0].shape = (a*b,). Unprocessed specifications: \
     'a*b' for dimension size args[0].shape[0]. Please see \
     https://docs.jax.dev/en/latest/export/shape_poly.html#dimension-variables-must-be-solvable-from-the-input-shapes \
     for more details." (fun () -> run "a*b" [| 6 |]);
  raises_prefix "explicit constraint"
    "Input shapes do not match the symbolic shape constraint a>=3. Expected 'a \
     - 3' to be greater or equal to 0, but found -1." (fun () ->
      run ~constraints:[ "a>=3" ] "a,b" [| 2; 3 |]);
  Alcotest.(check (float 0.0)) "constraint holds" 6.0
    (run ~constraints:[ "a>=2" ] "a,b" [| 2; 3 |]);
  let d1 = ss "a" and d2 = ss "b" in
  raises_prefix "mixed scopes at export"
    "Invalid mixing of symbolic scopes when exporting f." (fun () ->
      Export.export_symbolic ~fun_name:"f" sum1
        [ spec d1 D.F32; spec d2 D.F32 ]);
  Alcotest.(check (list string)) "all_dim_vars" [ "a"; "b" ]
    (P.all_dim_vars [ spec (ss "b,a+b,3") D.F32 ]);
  Alcotest.(check int) "evaluate" 11
    (P.evaluate [ ("a", 2); ("b", 3) ] (ss "a*b + floordiv(b+a, 2) + mod(b, 2) + 2").(0))
[@@ocamlformat "disable"]

let square = function
  | [ x ] -> [ Lax_api.mul x x ]
  | _ -> Alcotest.fail "arity"

let export_metadata () =
  Alcotest.(check (list int)) "calling convention range" [ 9; 10 ]
    [ Export.minimum_supported_calling_convention_version;
      Export.maximum_supported_calling_convention_version ];
  Alcotest.(check string) "default platform" "cpu" (Export.default_export_platform ());
  let p = Export.platform_check and cc = Export.custom_call_check "a" in
  Alcotest.(check (list string)) "check names" [ "platform"; "custom_call:a" ]
    (List.map Export.disabled_check_to_string [ p; cc ]);
  Alcotest.(check (option string)) "custom target" (Some "a") (Export.is_custom_call cc);
  Alcotest.(check (option string)) "platform target" None (Export.is_custom_call p);
  let e = Export.export ~fun_name:"f" square [ f32a [||] ] in
  Alcotest.(check int) "version" 10 e.calling_convention_version;
  Alcotest.(check (list int)) "kept" [ 0 ] e.module_kept_var_idx;
  Alcotest.(check int) "devices" 1 e.nr_devices;
  Alcotest.(check bool) "globals" false e.uses_global_constants;
  Alcotest.(check (list string)) "platforms" [ "cpu" ] e.platforms;
  raises_msg "version range"
    "The requested export calling convention version 8 is outside the range of \
     supported versions [9..10]" (fun () ->
      Config.with_value Config.export_calling_convention_version 8 (fun () ->
          Export.export square [ f32a [||] ]));
  let first_only = function x :: _ -> [ Lax_api.add x x ] | [] -> [] in
  let pruned = Export.export first_only [ f32a [||]; f32a [||] ] in
  Alcotest.(check (list int)) "unused argument dropped" [ 0 ] pruned.module_kept_var_idx;
  (match Export.call pruned [ f32 [||] [| 3.0 |]; f32 [||] [| 9.0 |] ] with
  | [ r ] -> check_floats "pruned call" [| 6.0 |] r
  | _ -> Alcotest.fail "arity");
  raises_msg "structure"
    "The invocation args and kwargs must have the same pytree structure as when \
     the function 'f' was exported, but they have the following structural \
     differences:\n" (fun () ->
      Export.call e [ f32 [||] [| 3.0 |]; f32 [||] [| 3.0 |] ])
[@@ocamlformat "disable"]

let export_platforms () =
  let on_cuda = Export.export ~fun_name:"<lambda>" ~platforms:[ "cuda" ] square [ f32a [||] ] in
  raises_msg "platform mismatch"
    "Function '<lambda>' was exported for platforms '('cuda',)' but it is used \
     on '('cpu',)'." (fun () -> Export.call on_cuda [ f32 [||] [| 1.0 |] ]);
  let allowed =
    Export.export ~platforms:[ "cuda" ] ~disabled_checks:[ Export.platform_check ]
      square [ f32a [||] ]
  in
  (match Export.call allowed [ f32 [||] [| 3.0 |] ] with
  | [ r ] -> check_floats "platform check disabled" [| 9.0 |] r
  | _ -> Alcotest.fail "arity");
  let restored = Export.deserialize (Export.serialize allowed) in
  Alcotest.(check (list string)) "checks survive" [ "platform" ]
    (List.map Export.disabled_check_to_string restored.disabled_safety_checks);
  match Export.call restored [ f32 [||] [| 3.0 |] ] with
  | [ r ] -> check_floats "restored call" [| 9.0 |] r
  | _ -> Alcotest.fail "arity"
[@@ocamlformat "disable"]

let export_vjp () =
  let three = f32 [||] [| 3.0 |] in
  let e = Export.export square [ f32a [||] ] in
  Alcotest.(check bool) "has vjp" true (Export.has_vjp e);
  (match Export.call (Export.vjp e) [ three; f32 [||] [| 1.0 |] ] with
  | [ r ] -> check_floats "vjp" [| 6.0 |] r
  | _ -> Alcotest.fail "arity");
  let call_first args = [ List.hd (Export.call e args) ] in
  let grad_of f x =
    let _, pull = X.Ad.vjp f [ x ] in
    List.hd (pull [ f32 [||] [| 1.0 |] ])
  in
  check_floats "grad through call" [| 6.0 |] (grad_of call_first three);
  check_floats "second order" [| 2.0 |]
    (grad_of (fun args -> [ grad_of call_first (List.hd args) ]) three);
  check_floats "jit through call" [| 9.0 |] (List.hd (Api.jit_flat call_first [ three ]));
  raises_msg "vmap of call" "Batching rule for 'call_exported' not implemented"
    (fun () ->
      X.Batching.vmap_flat call_first [ Some 0 ] [ f32 [| 3 |] [| 0.; 1.; 2. |] ]);
  let plain = Export.deserialize (Export.serialize e) in
  Alcotest.(check bool) "no vjp without vjp_order" false (Export.has_vjp plain);
  raises_msg "no vjp" "No VJP is available" (fun () -> Export.vjp plain);
  let once = Export.deserialize (Export.serialize ~vjp_order:1 e) in
  Alcotest.(check bool) "vjp_order 1" true (Export.has_vjp once);
  Alcotest.(check bool) "vjp_order 1 stops" false (Export.has_vjp (Export.vjp once));
  let call_once args = [ List.hd (Export.call once args) ] in
  check_floats "grad after deserialize" [| 6.0 |] (grad_of call_once three)
[@@ocamlformat "disable"]

let point x y = T.Custom ("test.Point", "aux", [ T.Leaf x; T.Leaf y ])

let export_trees () =
  let f = function
    | T.Tuple [ T.Custom (n, aux, [ T.Leaf x; T.Leaf y ]) ] ->
        T.Dict
          [ ("sum", T.Leaf (Lax_api.add x y));
            ("p", T.Custom (n, aux, [ T.Leaf y; T.Leaf x ])) ]
    | _ -> Alcotest.fail "structure"
  in
  let args = T.Tuple [ point (f32a [||]) (f32a [||]) ] in
  let e = Export.export_tree f args in
  raises_prefix "unregistered"
    "Cannot serialize PyTreeDef containing an unregistered type `test.Point`."
    (fun () -> Export.serialize e);
  let aux_seen = ref [] in
  Export.register_pytree_node_serialization ~serialized_name:"my.Point"
    ~serialize_auxdata:(fun a -> aux_seen := a :: !aux_seen; String.uppercase_ascii a)
    ~deserialize_auxdata:String.lowercase_ascii "test.Point";
  raises_msg "duplicate type"
    "Duplicate serialization registration for type `test.Point`. Previous \
     registration was with serialized_name `my.Point`." (fun () ->
      Export.register_pytree_node_serialization ~serialized_name:"other"
        ~serialize_auxdata:Fun.id ~deserialize_auxdata:Fun.id "test.Point");
  raises_msg "duplicate name"
    "Duplicate serialization registration for serialized_name `my.Point`. \
     Previous registration was for type `test.Point`." (fun () ->
      Export.register_pytree_node_serialization ~serialized_name:"my.Point"
        ~serialize_auxdata:Fun.id ~deserialize_auxdata:Fun.id "test.Other");
  let blob = Export.serialize e in
  Alcotest.(check bool) "auxdata passed through" true (List.mem "aux" !aux_seen);
  let r = Export.deserialize blob in
  Alcotest.(check bool) "in_tree survives" true (r.in_tree = e.in_tree);
  Alcotest.(check bool) "out_tree survives" true (r.out_tree = e.out_tree);
  let one = f32 [||] [| 1.0 |] and two = f32 [||] [| 2.0 |] in
  (match Export.call_tree r (T.Tuple [ point one two ]) with
  | T.Dict [ ("p", T.Custom ("test.Point", "aux", [ T.Leaf a; T.Leaf b ])); ("sum", T.Leaf s) ] ->
      check_floats "swapped first" [| 2.0 |] a;
      check_floats "swapped second" [| 1.0 |] b;
      check_floats "sum" [| 3.0 |] s
  | _ -> Alcotest.fail "unexpected result structure");
  raises_msg "wrong structure"
    "The invocation args and kwargs must have the same pytree structure as when \
     the function 'exported' was exported, but they have the following \
     structural differences:\n" (fun () ->
      Export.call_tree r (T.Tuple [ T.Leaf one; T.Leaf two ]));
  Export.register_namedtuple_serialization ~serialized_name:"my.NT"
    ~fields:[ "x"; "y" ] "test.NT";
  let nt a b = T.Custom ("test.NT", "", [ T.Leaf a; T.Leaf b ]) in
  let swap = function
    | T.Tuple [ T.Custom (n, a, [ x; y ]) ] -> T.Custom (n, a, [ y; x ])
    | _ -> Alcotest.fail "structure"
  in
  let e2 = Export.export_tree swap (T.Tuple [ nt (f32a [||]) (f32a [||]) ]) in
  let r2 = Export.deserialize (Export.serialize e2) in
  Alcotest.(check bool) "namedtuple in_tree" true (r2.in_tree = e2.in_tree);
  (match Export.call_tree r2 (T.Tuple [ nt one two ]) with
  | T.Custom ("test.NT", _, [ T.Leaf a; T.Leaf b ]) ->
      check_floats "x" [| 2.0 |] a;
      check_floats "y" [| 1.0 |] b
  | _ -> Alcotest.fail "unexpected namedtuple result");
  raises_msg "namedtuple duplicate"
    "Duplicate serialization registration for type `test.NT`. Previous \
     registration was with serialized_name `my.NT`." (fun () ->
      Export.register_namedtuple_serialization ~serialized_name:"my.NT2"
        ~fields:[ "x" ] "test.NT")
[@@ocamlformat "disable"]

let lib = lazy (Ffi.dlopen (Filename.concat (Sys.getcwd ()) "ffi_target.so"))
let registered = ref false

let targets =
  [ "scale"; "minmax"; "fail"; "bump"; "layout_probe"; "fill_raw"; "alias_check";
    "attr_probe"; "type_id_read"; "add_pair" ]
[@@ocamlformat "disable"]

let register_all () =
  if not !registered then (
    registered := true;
    List.iter
      (fun n ->
        let name = "ojax_" ^ n in
        Ffi.register_ffi_target name (Ffi.dlsym (Lazy.force lib) name))
      targets)

let scale_attrs = [ ("scale", Ffi.Ffi_float 2.5); ("offset", Ffi.Ffi_int 1L) ]
let xs = f32 [| 3 |] [| 1.; 2.; 3. |]

let ffi_scale () =
  register_all ();
  let call =
    Ffi.ffi_call1 ~vmap_method:"sequential" "ojax_scale" (f32a [| 3 |])
  in
  check_floats "eager" [| 3.5; 6.; 8.5 |] (call ~attributes:scale_attrs [ xs ]);
  check_floats "device argument" [| 3.5; 6.; 8.5 |]
    (call ~attributes:scale_attrs [ Backend.of_host_value xs ]);
  check_floats "jit" [| 3.5; 6.; 8.5 |]
    (List.hd
       (Api.jit_flat (fun args -> [ call ~attributes:scale_attrs args ]) [ xs ]));
  check_floats "scalar" [| 4. |]
    (Ffi.ffi_call1 "ojax_scale" (f32a [||])
       ~attributes:[ ("scale", Ffi.Ffi_float 2.0) ]
       [ f32 [||] [| 2.0 |] ]);
  let scan_body carry i =
    Lax_api.add carry
      (Ffi.ffi_call1 "ojax_scale" (f32a [||])
         ~attributes:[ ("scale", Ffi.Ffi_float 2.0) ]
         [ i ])
  in
  let summed =
    List.hd
      (Api.jit_flat
         (fun args ->
           let inputs = List.hd args in
           let parts = Core.bind (Unstack 0) [ inputs ] in
           [ List.fold_left scan_body (f32 [||] [| 0.0 |]) parts ])
         [ xs ])
  in
  check_floats "inside jit with more ops" [| 12.0 |] summed

let ffi_multi () =
  register_all ();
  let arg = f32 [| 4 |] [| 3.; -1.; 7.; 2. |] in
  let call = Ffi.ffi_call "ojax_minmax" [ f32a [| 2 |]; i32a [||] ] in
  let check tag = function
    | [ range; count ] ->
        check_floats (tag ^ " range") [| -1.; 7. |] range;
        check_floats (tag ^ " count") [| 4. |] count;
        check_dtype (tag ^ " dtype") D.I32 count
    | _ -> Alcotest.fail "arity"
  in
  check "eager" (call [ arg ]);
  check "jit" (Api.jit_flat (fun a -> call a) [ arg ])

let ffi_errors () =
  register_all ();
  raises_prefix "handler error" "ojax_fail was asked to fail" (fun () ->
      Ffi.ffi_call1 "ojax_fail" (f32a [||]) [ f32 [||] [| 1. |] ]);
  raises_prefix "unregistered"
    "No FFI handler registered for ojax_nope on a platform Host (canonical \
     host)" (fun () ->
      Ffi.ffi_call1 "ojax_nope" (f32a [||]) [ f32 [||] [| 1. |] ]);
  Ffi.register_ffi_target ~platform:"cuda" "ojax_only_gpu"
    (Ffi.dlsym (Lazy.force lib) "ojax_scale");
  raises_prefix "other platform"
    "No FFI handler registered for ojax_only_gpu on a platform Host (canonical \
     host)" (fun () ->
      Ffi.ffi_call1 "ojax_only_gpu" (f32a [||]) [ f32 [||] [| 1. |] ]);
  raises_msg "bad vmap method"
    "vmap_method must be on of the allowed methods ['sequential', \
     'sequential_unrolled', 'expand_dims', 'broadcast_all', \
     'legacy_vectorized', None], but got: bad" (fun () ->
      Ffi.ffi_call ~vmap_method:"bad" "t" [ i32a [||] ]);
  raises_msg "legacy config"
    "The use of the legacy_backend_config parameter requires \
     custom_call_api_version < 4; got 4." (fun () ->
      Ffi.ffi_call ~legacy_backend_config:"a" "t" [ i32a [||] ]);
  let one = Concrete (Ndarray.of_floats D.F32 [||] [| 1.0 |]) in
  raises_msg "input layouts"
    "The number of input arguments (1) must equal the number of input layouts \
     (2)." (fun () ->
      Ffi.ffi_call
        ~input_layouts:[ Ffi.Default; Ffi.Default ]
        "t"
        [ i32a [||] ]
        [ one ]);
  raises_msg "output layouts"
    "The number of outputs (1) must equal the number of output layouts (2)."
    (fun () ->
      Ffi.ffi_call
        ~output_layouts:[ Ffi.Default; Ffi.Default ]
        "t"
        [ i32a [||] ]
        [ one ]);
  let alias pairs () =
    Ffi.ffi_call ~input_output_aliases:pairs "t" [ i32a [||] ] [ one ]
  in
  raises_msg "alias input"
    "input_output_aliases contains the mapping '1:0' with input index 1 \
     outside the range [0, 1)."
    (alias [ (1, 0) ]);
  raises_msg "alias output"
    "input_output_aliases contains the mapping '0:1' with output index 1 \
     outside the range [0, 1)."
    (alias [ (0, 1) ]);
  raises_msg "alias aval"
    "input_output_aliases contains the mapping '0:0' referring to an input \
     with abstract value float32[] and an output with a different abstract \
     value int32[]."
    (alias [ (0, 0) ]);
  raises_msg "type id"
    "register_ffi_type_id is not supported after jaxlib version 381." (fun () ->
      Ffi.register_ffi_type_id "n" 0n);
  raises_prefix "attributes need v4"
    "The use of ffi_call attributes requires a custom call API version of at \
     least 4" (fun () ->
      Ffi.ffi_call1 ~custom_call_api_version:2 "t" (f32a [||])
        ~attributes:[ ("a", Ffi.Ffi_int 1L) ]
        [ one ]);
  raises_prefix "untyped registration"
    "register_ffi_target: api_version=0 is not available" (fun () ->
      Ffi.register_ffi_target ~api_version:0 "ojax_v0"
        (Ffi.dlsym (Lazy.force lib) "ojax_scale"));
  raises_prefix "dlopen" "dlopen failed" (fun () ->
      Ffi.dlopen "/nonexistent/lib.so");
  raises_prefix "dlsym" "dlsym ojax_missing failed" (fun () ->
      Ffi.dlsym (Lazy.force lib) "ojax_missing")

let ffi_side_effects () =
  register_all ();
  let bump = Ffi.ffi_call1 ~has_side_effect:true "ojax_bump" (i32a [||]) in
  let read () = (flat (bump [])).(0) in
  let start = read () in
  Alcotest.(check (float 0.0)) "counts up" (start +. 1.0) (read ());
  let dead = function
    | [ x ] ->
        ignore (bump []);
        [ x ]
    | _ -> []
  in
  ignore (Api.jit_flat dead [ f32 [||] [| 1.0 |] ]);
  Alcotest.(check (float 0.0)) "dead call still ran" (start +. 3.0) (read ());
  let no_results = Ffi.ffi_call ~has_side_effect:true "ojax_bump" [] [] in
  Alcotest.(check int) "no results" 0 (List.length no_results);
  Alcotest.(check (float 0.0)) "ran" (start +. 5.0) (read ())

let ffi_layouts () =
  register_all ();
  let m = i32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let probe layouts =
    Ffi.ffi_call1 ?input_layouts:layouts "ojax_layout_probe" (i32a [| 6 |])
      [ m ]
  in
  check_floats "default" [| 0.; 1.; 2.; 3.; 4.; 5. |] (probe None);
  check_floats "row major"
    [| 0.; 1.; 2.; 3.; 4.; 5. |]
    (probe (Some [ Ffi.Major_to_minor [| 0; 1 |] ]));
  check_floats "column major"
    [| 0.; 3.; 1.; 4.; 2.; 5. |]
    (probe (Some [ Ffi.Major_to_minor [| 1; 0 |] ]));
  check_floats "none in list"
    [| 0.; 1.; 2.; 3.; 4.; 5. |]
    (probe (Some [ Ffi.Default ]));
  let fill layout =
    Ffi.ffi_call1 ?output_layout:layout "ojax_fill_raw" (i32a [| 2; 3 |]) []
  in
  check_floats "out default" [| 0.; 1.; 2.; 3.; 4.; 5. |] (fill None);
  check_floats "out row major"
    [| 0.; 1.; 2.; 3.; 4.; 5. |]
    (fill (Some (Ffi.Major_to_minor [| 0; 1 |])));
  check_floats "out column major"
    [| 0.; 2.; 4.; 1.; 3.; 5. |]
    (fill (Some (Ffi.Major_to_minor [| 1; 0 |])));
  raises_prefix "alias layout mismatch"
    "input_output_aliases contains the mapping '0:0' referring to an input \
     with layout (1, 0) and an output with a different layout (0, 1)."
    (fun () ->
      Ffi.ffi_call ~input_layouts:[ Ffi.Default ]
        ~output_layouts:[ Ffi.Major_to_minor [| 1; 0 |] ]
        ~input_output_aliases:[ (0, 0) ]
        "t"
        [ i32a [| 2; 3 |] ]
        [ m ])

let ffi_aliases () =
  register_all ();
  check_floats "no alias" [| 1.; 2.; 3. |]
    (Ffi.ffi_call1 "ojax_alias_check" (f32a [| 3 |]) [ xs ]);
  check_floats "alias" [| 101.; 102.; 103. |]
    (Ffi.ffi_call1
       ~input_output_aliases:[ (0, 0) ]
       "ojax_alias_check" (f32a [| 3 |]) [ xs ])

let ffi_attrs () =
  register_all ();
  let probe ?attributes () =
    Ffi.ffi_call1 "ojax_attr_probe" (f32a [| 6 |]) ?attributes []
  in
  check_floats "all kinds"
    [| 6.; 5.; 104.; 1.; 2.; 4. |]
    (probe
       ~attributes:
         [
           ("arr", Ffi.Ffi_array (i32_arr [| 3 |] [| 1.; 2.; 3. |]));
           ("name", Ffi.Ffi_string "hello");
           ("flag", Ffi.Ffi_bool true);
           ("d", Ffi.Ffi_dict [ ("a", Ffi.Ffi_int 1L); ("b", Ffi.Ffi_int 2L) ]);
         ]
       ());
  check_floats "none" [| 0.; -1.; -1.; -1.; -1.; 0. |] (probe ());
  raises_prefix "sequence attribute"
    "Unsupported attribute type for attribute: l" (fun () ->
      probe
        ~attributes:[ ("l", Ffi.Ffi_seq [ Ffi.Ffi_int 1L; Ffi.Ffi_int 2L ]) ]
        ())

let ffi_types () =
  register_all ();
  let lib = Lazy.force lib in
  let read () =
    (flat (Ffi.ffi_call1 "ojax_type_id_read" (aval D.I64 [||]) [])).(0)
  in
  let before = Config.with_value Config.enable_x64 true read in
  Alcotest.(check (float 0.0)) "unassigned" 0.0 before;
  Ffi.register_ffi_type "ojax_test_type"
    { type_id = Ffi.dlsym lib "ojax_test_type_id"; type_info = None };
  let assigned = Config.with_value Config.enable_x64 true read in
  Alcotest.(check bool) "assigned by the plugin" true (assigned > 0.0);
  Ffi.register_ffi_type "ojax_test_type_with_info"
    {
      type_id = Ffi.dlsym lib "ojax_test_type_id";
      type_info = Some (Ffi.dlsym lib "ojax_test_type_info");
    };
  let again = Config.with_value Config.enable_x64 true read in
  Alcotest.(check (float 0.0)) "known id is kept" assigned again;
  raises_prefix "name bound to another id"
    "Type name ojax_test_type already registered with type id" (fun () ->
      Ffi.register_ffi_type "ojax_test_type"
        { type_id = Ffi.dlsym lib "ojax_test_other_type_id"; type_info = None })

let ffi_vmap () =
  register_all ();
  let xb = f32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let y = f32 [| 3 |] [| 10.; 20.; 30. |] in
  let expected = [| 10.; 21.; 32.; 13.; 24.; 35. |] in
  let call m args =
    Ffi.ffi_call ~vmap_method:m "ojax_add_pair"
      [ f32a [| 3 |]; i32a [| 3 |] ]
      args
  in
  List.iter
    (fun (m, info) ->
      match X.Batching.vmap_flat (call m) [ Some 0; None ] [ xb; y ] with
      | [ a; b ] ->
          check_floats (m ^ " values") expected a;
          check_shape (m ^ " shape") [| 2; 3 |] a;
          check_floats (m ^ " info") info b
      | _ -> Alcotest.fail "arity")
    [
      ("sequential", [| 1.; 1.; 3.; 1.; 1.; 3. |]);
      ("sequential_unrolled", [| 1.; 1.; 3.; 1.; 1.; 3. |]);
      ("expand_dims", [| 2.; 2.; 1.; 2.; 2.; 1. |]);
      ("broadcast_all", [| 2.; 2.; 2.; 2.; 2.; 2. |]);
      ("legacy_vectorized", [| 2.; 1.; 3.; 2.; 1.; 3. |]);
    ];
  let xb_t = f32 [| 3; 2 |] [| 0.; 3.; 1.; 4.; 2.; 5. |] in
  (match
     X.Batching.vmap_flat (call "expand_dims") [ Some 1; None ] [ xb_t; y ]
   with
  | [ a; _ ] ->
      check_shape "moved axis" [| 2; 3 |] a;
      check_floats "moved axis values" expected a
  | _ -> Alcotest.fail "arity");
  raises_msg "no method"
    "vmap is only supported for the ffi_call primitive when vmap_method is one \
     of 'sequential', 'sequential_unrolled', 'expand_dims', 'broadcast_all', \
     or 'legacy_vectorized'. Got vmap_method=None." (fun () ->
      X.Batching.vmap_flat
        (fun args -> [ Ffi.ffi_call1 "t" (f32a [||]) args ])
        [ Some 0 ]
        [ f32 [| 2 |] [| 1.; 1. |] ])

let ffi_ad () =
  register_all ();
  let call args =
    [
      Ffi.ffi_call1 "ojax_scale" (f32a [||])
        ~attributes:[ ("scale", Ffi.Ffi_float 2.0) ]
        args;
    ]
  in
  let msg =
    "The FFI call to `ojax_scale` cannot be differentiated. You can use \
     `jax.custom_jvp` or `jax.custom_jvp` to add support."
  in
  let three = f32 [||] [| 3.0 |] in
  raises_msg "jvp" msg (fun () ->
      X.Ad.jvp call [ three ] [ f32 [||] [| 1.0 |] ]);
  raises_msg "vjp" msg (fun () -> X.Ad.vjp call [ three ]);
  let via_stop = function
    | [ c ] ->
        let stopped = Custom_derivatives.stop_gradient c in
        [
          Lax_api.mul c
            (Custom_derivatives.stop_gradient (List.hd (call [ stopped ])));
        ]
    | _ -> []
  in
  let _, pull = X.Ad.vjp via_stop [ three ] in
  check_floats "stop_gradient" [| 6.0 |] (List.hd (pull [ f32 [||] [| 1.0 |] ]))

let ffi_lowering () =
  let lowered attrs extra =
    let f args = extra ?attributes:attrs args in
    Stages.raw_text (Stages.lower f [ f32a [| 3 |] ])
  in
  let contains s sub =
    let n = String.length s and m = String.length sub in
    let rec go i = i + m <= n && (String.sub s i m = sub || go (i + 1)) in
    go 0
  in
  let check text fragment =
    if not (contains text fragment) then
      Alcotest.failf "missing %s in\n%s" fragment text
  in
  let text =
    lowered
      (Some
         [
           ("n", Ffi.Ffi_int 3L);
           ("s", Ffi.Ffi_string "hi");
           ("fl", Ffi.Ffi_float 1.5);
           ("b", Ffi.Ffi_bool true);
           ("arr", Ffi.Ffi_array (i32_arr [| 2 |] [| 1.; 2. |]));
           ("d", Ffi.Ffi_dict [ ("a", Ffi.Ffi_int 1L) ]);
           ("l", Ffi.Ffi_seq [ Ffi.Ffi_int 1L; Ffi.Ffi_int 2L ]);
         ])
      (Ffi.ffi_call ~has_side_effect:true ~vmap_method:"sequential"
         ~input_output_aliases:[ (0, 0) ]
         ~input_layouts:[ Ffi.Major_to_minor [| 0 |] ]
         "tgt"
         [ f32a [| 3 |] ])
  in
  check text
    "stablehlo.custom_call @tgt(%0) {backend_config = \"\", has_side_effect = \
     true, mhlo.backend_config = {arr = dense<[1, 2]> : tensor<2xi32>, b = \
     true, d = {a = 1 : i64}, fl = 1.500000e+00 : f64, l = [1 : i64, 2 : i64], \
     n = 3 : i64, s = \"hi\"}, operand_layouts = [dense<[0]> : \
     tensor<1xindex>], output_operand_aliases = \
     [#stablehlo.output_operand_alias<output_tuple_indices = [], operand_index \
     = 0, operand_tuple_indices = []>], result_layouts = [dense<[0]> : \
     tensor<1xindex>]} : (tensor<3xf32>) -> tensor<3xf32>";
  let legacy =
    lowered None
      (Ffi.ffi_call ~custom_call_api_version:2 ~legacy_backend_config:"xx" "tgt"
         [ f32a [| 3 |] ])
  in
  check legacy
    "{api_version = 2 : i32, backend_config = \"xx\", operand_layouts";
  let two =
    Stages.raw_text
      (Stages.lower
         (fun args ->
           Ffi.ffi_call
             ~output_layouts:[ Ffi.Major_to_minor [| 1; 0 |]; Ffi.Default ]
             "tgt2"
             [ f32a [| 2; 3 |]; i32a [||] ]
             args)
         [ f32a [| 2; 3 |] ])
  in
  check two
    "backend_config = \"\", mhlo.backend_config = {}, operand_layouts = \
     [dense<[1, 0]> : tensor<2xindex>], result_layouts = [dense<[0, 1]> : \
     tensor<2xindex>, dense<> : tensor<0xindex>]} : (tensor<2x3xf32>) -> \
     (tensor<2x3xf32>, tensor<i32>)"

let ffi_export () =
  register_all ();
  let scale args =
    [
      Ffi.ffi_call1 "ojax_scale" (f32a [| 3 |])
        ~attributes:[ ("scale", Ffi.Ffi_float 2.0) ]
        args;
    ]
  in
  raises_prefix "unstable target"
    "Cannot serialize code with custom calls whose targets have no \
     compatibility guarantees. See \
     https://docs.jax.dev/en/latest/export/export.html#compatibility-guarantees-for-custom-calls. \
     Examples are:\n" (fun () -> Export.export scale [ f32a [| 3 |] ]);
  let e =
    Export.export
      ~disabled_checks:[ Export.custom_call_check "ojax_scale" ]
      scale
      [ f32a [| 3 |] ]
  in
  Alcotest.(check (list string))
    "recorded"
    [ "custom_call:ojax_scale" ]
    (List.map Export.disabled_check_to_string e.disabled_safety_checks);
  (match Export.call e [ xs ] with
  | [ r ] -> check_floats "call" [| 2.; 4.; 6. |] r
  | _ -> Alcotest.fail "arity");
  match Export.call (Export.deserialize (Export.serialize e)) [ xs ] with
  | [ r ] -> check_floats "after deserialize" [| 2.; 4.; 6. |] r
  | _ -> Alcotest.fail "arity"

external make_tensor :
  int -> int -> int -> int array -> int array -> int -> int -> nativeint
  = "ojax_test_make_tensor_bytecode" "ojax_test_make_tensor"

external deleter_calls : unit -> int = "ojax_test_deleter_calls"
external data_pointer : nativeint -> nativeint = "ojax_test_data_pointer"
external write_i32 : nativeint -> int -> int -> unit = "ojax_test_write_i32"

external describe :
  nativeint ->
  int * int * int * int * int array * int array * int * int * int * bool
  = "ojax_test_describe"

external sum_f32 : nativeint -> float = "ojax_test_sum_f32"

let foreign ?(code = 0) ?(bits = 32) ?(lanes = 1) ?(strides = [||])
    ?(misalign = 0) ?(offset = 0) shape =
  let p = make_tensor code bits lanes shape strides misalign offset in
  let capsule = Dlpack.capsule_of_pointer p in
  ( { Dlpack.dlpack = (fun () -> capsule); dlpack_device = (fun () -> (1, 0)) },
    p )

let dlpack_dtypes () =
  List.iter
    (fun (name, expected) ->
      Alcotest.(check bool) name expected (Dlpack.is_supported_dtype_name name))
    [
      ("float32", true); ("f8", true); ("int4", false); ("uint2", false);
      ("object", false); ("U3", false); ("S2", false);
      ("float8_e4m3fn", false); ("bfloat16", true); ("datetime64[s]", false);
      ("int8", true); ("bool", true); ("int", true); ("float", true);
      ("complex", true);
    ];
  raises_msg "unknown name" "data type 'banana' not understood" (fun () ->
      Dlpack.is_supported_dtype_name "banana");
  List.iter
    (fun d ->
      Alcotest.(check bool) (D.numpy_name d) false (Dlpack.is_supported_dtype d))
    [ D.I2; D.I4; D.Uint2; D.Uint4; D.F4_e2m1fn; D.F8_e4m3; D.F8_e8m0fnu; D.Float0 ];
  List.iter
    (fun d ->
      Alcotest.(check bool) (D.numpy_name d) true (Dlpack.is_supported_dtype d))
    [
      D.F16; D.Bf16; D.F32; D.F64; D.I8; D.I16; D.I32; D.I64; D.Uint8;
      D.Uint16; D.Uint32; D.Uint64; D.Bool; D.Complex64; D.Complex128;
    ];
  Alcotest.(check (pair int int)) "version" (0, 8) Dlpack.dlpack_version;
  Alcotest.(check (pair int int)) "min version" (0, 5) Dlpack.min_dlpack_version
[@@ocamlformat "disable"]

let dlpack_export_fields () =
  let cases =
    [
      (D.I8, 0, 8); (D.I16, 0, 16); (D.I32, 0, 32); (D.Uint8, 1, 8);
      (D.Uint16, 1, 16); (D.Uint32, 1, 32); (D.F16, 2, 16); (D.Bf16, 4, 16);
      (D.F32, 2, 32); (D.Complex64, 5, 64); (D.Bool, 6, 8);
    ]
  in
  List.iter
    (fun (d, code, bits) ->
      let v =
        if d = D.Complex64 then
          Concrete (Ndarray.of_complex d [| 2; 3 |] (Array.make 6 Complex.zero))
        else Concrete (Ndarray.of_floats d [| 2; 3 |] (Array.make 6 0.0))
      in
      let c = Dlpack.to_dlpack v in
      let cd, bt, lanes, ndim, shape, strides, off, dev, id, aligned =
        describe (Dlpack.capsule_pointer c)
      in
      let name = D.numpy_name d in
      Alcotest.(check (list int)) (name ^ " fields")
        [ code; bits; 1; 2; 0; 1; 0 ] [ cd; bt; lanes; ndim; off; dev; id ];
      Alcotest.(check (array int)) (name ^ " shape") [| 2; 3 |] shape;
      Alcotest.(check (array int)) (name ^ " strides") [| 3; 1 |] strides;
      Alcotest.(check bool) (name ^ " aligned") true aligned)
    cases;
  let _, _, _, ndim, _, strides, _, _, _, _ =
    describe (Dlpack.capsule_pointer (Dlpack.to_dlpack (f32 [||] [| 1.5 |])))
  in
  Alcotest.(check int) "scalar rank" 0 ndim;
  Alcotest.(check (array int)) "scalar has no strides" [||] strides;
  raises_msg "old version"
    "JAX does not support any version below (0, 5) but version ((0, 4)) was \
     requested." (fun () -> Dlpack.to_dlpack ~max_version:(0, 4) xs);
  ignore (Dlpack.to_dlpack ~max_version:(0, 5) xs);
  raises_msg "bad device"
    "The device specification passed to to_dlpack contains an unsupported \
     device type (DLDeviceType: 99)" (fun () ->
      Dlpack.to_dlpack ~dl_device:(99, 0) xs);
  ignore (Dlpack.to_dlpack ~dl_device:(1, 0) xs);
  Alcotest.(check (float 0.0)) "foreign consumer reads the data" 6.0
    (sum_f32 (Dlpack.capsule_pointer (Dlpack.to_dlpack xs)))
[@@ocamlformat "disable"]

let dlpack_round_trip () =
  let device = Api.device_put (T.Leaf (f32 [| 3 |] [| 0.; 1.; 2. |])) in
  let v = match device with T.Leaf x -> x | _ -> Alcotest.fail "leaf" in
  let v =
    match v with Concrete nd -> Backend.of_host_value (Concrete nd) | x -> x
  in
  let back = Dlpack.from_dlpack (Dlpack.external_of_value v) in
  check_floats "values" [| 0.; 1.; 2. |] back;
  check_dtype "dtype" D.F32 back;
  Alcotest.(check bool)
    "shares memory" true
    (Dlpack.unsafe_buffer_pointer back = Dlpack.unsafe_buffer_pointer v);
  check_floats "numpy.from_dlpack" [| 0.; 1.; 2. |]
    (Numpy.Lax_numpy.from_dlpack (Dlpack.external_of_value v));
  check_floats "usable in a computation" [| 0.; 2.; 4. |]
    (Lax_api.add back back);
  let copied = Dlpack.from_dlpack ~copy:true (Dlpack.external_of_value v) in
  check_floats "copy values" [| 0.; 1.; 2. |] copied;
  Alcotest.(check bool)
    "copy=true is a real copy" false
    (Dlpack.unsafe_buffer_pointer copied = Dlpack.unsafe_buffer_pointer v);
  let c = Dlpack.to_dlpack v in
  let ext =
    { Dlpack.dlpack = (fun () -> c); dlpack_device = (fun () -> (1, 0)) }
  in
  check_floats "first import" [| 0.; 1.; 2. |] (Dlpack.from_dlpack ext);
  Alcotest.(check bool) "capsule marked used" true (Dlpack.capsule_used c);
  raises_msg "second import"
    "INVALID_ARGUMENT: DLPack tensor must be a capsule with name \"dltensor\", \
     got \"used_dltensor\". Note that a DLPack tensor may be consumed at most \
     once." (fun () -> Dlpack.from_dlpack ext);
  let kept = Dlpack.to_dlpack v in
  (match v with Device b -> Pjrt.Buffer.destroy b | _ -> ());
  Alcotest.(check (float 0.0))
    "tensor outlives the array" 3.0
    (sum_f32 (Dlpack.capsule_pointer kept));
  check_floats "import after the source is gone" [| 0.; 1.; 2. |]
    (Dlpack.from_dlpack
       { Dlpack.dlpack = (fun () -> kept); dlpack_device = (fun () -> (1, 0)) })

let device_buffer = function
  | Device b -> b
  | Concrete _ | Tracer _ -> Alcotest.fail "expected a device value"

let dlpack_import () =
  let ext, p = foreign [| 2; 3 |] in
  let before = deleter_calls () in
  let v = Dlpack.from_dlpack ext in
  check_floats "aligned import" [| 0.; 1.; 2.; 3.; 4.; 5. |] v;
  check_dtype "int32" D.I32 v;
  Alcotest.(check bool)
    "aligned is a view" true
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  write_i32 (data_pointer p) 0 42;
  check_floats "external writes show" [| 42.; 1.; 2.; 3.; 4.; 5. |] v;
  Alcotest.(check int) "view keeps the tensor" before (deleter_calls ());
  Pjrt.Buffer.destroy (device_buffer v);
  Alcotest.(check int)
    "view release runs the deleter" (before + 1) (deleter_calls ());
  let ext, p = foreign [| 2; 3 |] in
  let before = deleter_calls () in
  let v = Dlpack.from_dlpack ~copy:true ext in
  Alcotest.(check int)
    "copy releases immediately" (before + 1) (deleter_calls ());
  check_floats "copy values" [| 0.; 1.; 2.; 3.; 4.; 5. |] v;
  Alcotest.(check bool)
    "copy is not the source" false
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  let ext, p = foreign [| 2; 3 |] in
  let v = Dlpack.from_dlpack ~copy:false ext in
  Alcotest.(check bool)
    "copy=false view" true
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  let ext, p = foreign ~misalign:4 [| 2; 3 |] in
  let before = deleter_calls () in
  let v = Dlpack.from_dlpack ext in
  check_floats "misaligned copies" [| 0.; 1.; 2.; 3.; 4.; 5. |] v;
  Alcotest.(check bool)
    "misaligned is not a view" false
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  Alcotest.(check int) "misaligned released" (before + 1) (deleter_calls ());
  let ext, _ = foreign ~misalign:4 [| 2; 3 |] in
  raises_msg "misaligned copy=false"
    "Specified input which requires a copy since the source data buffer is not \
     aligned to 64 bytes.  However copy=False. Set copy=True or copy=None to \
     perform the requested operation." (fun () ->
      Dlpack.from_dlpack ~copy:false ext);
  let ext, p = foreign ~code:2 ~offset:64 [| 2 |] in
  let v = Dlpack.from_dlpack ext in
  check_floats "byte offset 64" [| 0.; 1. |] v;
  Alcotest.(check bool)
    "offset view" true
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  let ext, p = foreign ~code:2 ~offset:4 [| 2 |] in
  let v = Dlpack.from_dlpack ext in
  check_floats "byte offset 4" [| 0.; 1. |] v;
  Alcotest.(check bool)
    "offset 4 copies" false
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  let ext, p = foreign ~code:2 ~misalign:4 ~offset:60 [| 2 |] in
  let v = Dlpack.from_dlpack ext in
  Alcotest.(check bool)
    "offset restores alignment" true
    (Dlpack.unsafe_buffer_pointer v = data_pointer p)

let dlpack_unused_capsule () =
  Gc.full_major ();
  Gc.full_major ();
  let before = deleter_calls () in
  let drop () = ignore (foreign [| 2 |]) in
  drop ();
  Gc.full_major ();
  Gc.full_major ();
  Alcotest.(check int)
    "unused capsule runs the deleter" (before + 1) (deleter_calls ())

let dlpack_layouts () =
  let ext, _ = foreign ~strides:[| 1; 2 |] [| 2; 3 |] in
  check_floats "fortran order"
    [| 0.; 2.; 4.; 1.; 3.; 5. |]
    (Dlpack.from_dlpack ext);
  let ext, _ = foreign ~strides:[| 1; 2 |] [| 2; 3 |] in
  check_floats "fortran copy"
    [| 0.; 2.; 4.; 1.; 3.; 5. |]
    (Dlpack.from_dlpack ~copy:true ext);
  let ext, p = foreign ~strides:[| 3; 1 |] [| 2; 3 |] in
  let v = Dlpack.from_dlpack ~copy:false ext in
  check_floats "explicit c strides" [| 0.; 1.; 2.; 3.; 4.; 5. |] v;
  Alcotest.(check bool)
    "c strides view" true
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  let ext, p = foreign ~strides:[| 99; 1 |] [| 1; 3 |] in
  let v = Dlpack.from_dlpack ~copy:false ext in
  check_floats "unit dim stride ignored" [| 0.; 1.; 2. |] v;
  Alcotest.(check bool)
    "unit dim view" true
    (Dlpack.unsafe_buffer_pointer v = data_pointer p);
  let ext, _ = foreign ~strides:[| 6; 2 |] [| 2; 3 |] in
  raises_msg "strided"
    "UNIMPLEMENTED: Only DLPack tensors with trivial (compact) striding are \
     supported; i.e., tensors whose striding represents a transposition of the \
     underlying buffer but not broadcasting. Dimensions were: [2,3], strides \
     were [6,2]." (fun () -> Dlpack.from_dlpack ext);
  let ext, _ = foreign ~strides:[| -1 |] [| 4 |] in
  raises_msg "negative stride"
    "UNIMPLEMENTED: Only DLPack tensors with trivial (compact) striding are \
     supported; i.e., tensors whose striding represents a transposition of the \
     underlying buffer but not broadcasting. Dimensions were: [4], strides \
     were [-1]." (fun () -> Dlpack.from_dlpack ext);
  let ext, _ = foreign ~strides:[| 0; 1 |] [| 2; 0 |] in
  check_shape "zero size" [| 2; 0 |] (Dlpack.from_dlpack ext);
  let ext, _ = foreign ~code:2 [||] in
  check_floats "scalar" [| 0. |] (Dlpack.from_dlpack ext);
  let ext, _ = foreign ~code:6 ~bits:8 [| 2 |] in
  let b = Dlpack.from_dlpack ext in
  check_dtype "bool" D.Bool b;
  check_floats "bool values" [| 0.; 1. |] b;
  let ext, _ = foreign ~code:2 ~bits:64 [| 2 |] in
  let v = Dlpack.from_dlpack ext in
  check_dtype "f64 narrows without x64" D.F32 v;
  check_floats "f64 narrowed" [| 0.; 1. |] v;
  Config.with_value Config.enable_x64 true (fun () ->
      let ext, p = foreign ~code:2 ~bits:64 [| 2 |] in
      let v = Dlpack.from_dlpack ext in
      check_dtype "f64 kept with x64" D.F64 v;
      Alcotest.(check bool)
        "f64 view with x64" true
        (Dlpack.unsafe_buffer_pointer v = data_pointer p))

let dlpack_errors () =
  let err msg expected make =
    let ext, _ = make () in
    raises_msg msg expected (fun () -> Dlpack.from_dlpack ext)
  in
  err "lanes"
    "UNIMPLEMENTED: DLPack types with lanes != 1 not implemented, got 2"
    (fun () -> foreign ~code:2 ~lanes:2 [| 2 |]);
  err "float width"
    "UNIMPLEMENTED: Invalid or unsupported DLPack float width: 8 bits"
    (fun () -> foreign ~code:2 ~bits:8 [| 2 |]);
  err "bool width"
    "UNIMPLEMENTED: Only 8-bit DLPack booleans are supported, got 32 bits"
    (fun () -> foreign ~code:6 ~bits:32 [| 2 |]);
  err "int width"
    "UNIMPLEMENTED: Invalid or unsupported DLPack integer width: 24 bits"
    (fun () -> foreign ~code:0 ~bits:24 [| 2 |]);
  err "uint width"
    "UNIMPLEMENTED: Invalid or unsupported DLPack unsigned integer width: 24 \
     bits" (fun () -> foreign ~code:1 ~bits:24 [| 2 |]);
  err "bfloat width"
    "UNIMPLEMENTED: Invalid or unsupported DLPack bfloat width: 32 bits"
    (fun () -> foreign ~code:4 ~bits:32 [| 2 |]);
  err "complex width"
    "UNIMPLEMENTED: Invalid or unsupported DLPack complex width: 32 bits"
    (fun () -> foreign ~code:5 ~bits:32 [| 2 |]);
  let on device =
    {
      Dlpack.dlpack = (fun () -> Alcotest.fail "must not be reached");
      dlpack_device = (fun () -> device);
    }
  in
  raises_msg "unsupported device"
    "Array passed to from_dlpack is on unsupported device type (DLDeviceType: \
     99, array: <external>" (fun () -> Dlpack.from_dlpack (on (99, 0)));
  raises_msg "cuda" "Unknown backend cuda. Available backends are ['cpu']"
    (fun () -> Dlpack.from_dlpack (on (2, 0)));
  raises_msg "cuda host" "Unknown backend cuda. Available backends are ['cpu']"
    (fun () -> Dlpack.from_dlpack (on (3, 0)));
  raises_msg "rocm" "Unknown backend rocm. Available backends are ['cpu']"
    (fun () -> Dlpack.from_dlpack (on (10, 0)));
  raises_msg "device id"
    "INVALID_ARGUMENT: No matching device found for local_device_id 5"
    (fun () -> Dlpack.from_dlpack (on (1, 5)));
  raises_msg "many device sharding"
    "from_dlpack can only unpack a dlpack tensor onto a singular device, but a \
     Sharding with 2 devices was provided." (fun () ->
      Dlpack.from_dlpack
        ~device:
          (Dlpack.With_sharding
             (Sharding.named_sharding
                (Sharding.make_mesh [ 2 ] [ "a" ])
                [ Sharding.Named "a" ]))
        (on (1, 0)));
  let ext, _ = foreign ~code:2 [| 2 |] in
  check_floats "explicit device" [| 0.; 1. |]
    (Dlpack.from_dlpack ~device:(Dlpack.On_device "cpu:0") ~copy:false ext);
  let ext, _ = foreign ~code:2 [| 2 |] in
  raises_msg "other device without a copy"
    "Specified device=gpu:0 which requires a copy since the source device is \
     cpu:0, however copy=False. Set copy=True or copy=None to perform the \
     requested operation." (fun () ->
      Dlpack.from_dlpack ~device:(Dlpack.On_device "gpu:0") ~copy:false ext)

let () =
  let pjrt_cases =
    [
      ("export metadata", `Quick, export_metadata);
      ("export platforms", `Quick, export_platforms);
      ("export vjp", `Quick, export_vjp);
      ("export trees", `Quick, export_trees);
      ("symbolic export", `Quick, symbolic_export);
      ("ffi scale", `Quick, ffi_scale);
      ("ffi multiple results", `Quick, ffi_multi);
      ("ffi errors", `Quick, ffi_errors);
      ("ffi side effects", `Quick, ffi_side_effects);
      ("ffi layouts", `Quick, ffi_layouts);
      ("ffi aliases", `Quick, ffi_aliases);
      ("ffi attributes", `Quick, ffi_attrs);
      ("ffi types", `Quick, ffi_types);
      ("ffi vmap", `Quick, ffi_vmap);
      ("ffi ad", `Quick, ffi_ad);
      ("ffi lowering", `Quick, ffi_lowering);
      ("ffi export", `Quick, ffi_export);
      ("dlpack export fields", `Quick, dlpack_export_fields);
      ("dlpack round trip", `Quick, dlpack_round_trip);
      ("dlpack import", `Quick, dlpack_import);
      ("dlpack unused capsule", `Quick, dlpack_unused_capsule);
      ("dlpack layouts", `Quick, dlpack_layouts);
      ("dlpack errors", `Quick, dlpack_errors);
    ]
  in
  Alcotest.run "export_ffi"
    [
      ( "shape_poly",
        [
          ("expression strings", `Quick, expr_strings);
          ("arithmetic", `Quick, arithmetic);
          ("comparisons", `Quick, comparisons);
          ("bounds", `Quick, bounds_table);
          ("constraints", `Quick, constraints);
          ("parser", `Quick, parser_table);
          ("args specs", `Quick, args_specs);
        ] );
      ("dlpack dtypes", [ ("dtypes", `Quick, dlpack_dtypes) ]);
      ("pjrt", if Ojax.Pjrt.Discover.find () = None then [] else pjrt_cases);
    ]
