open Ojax
open Ojax.Types
module L = Lax_api
module Tree = Tree_util
module R = Random.Core
module S = Example_libraries.Stax
module O = Example_libraries.Optimizers
module P = Profiler

let () =
  Lax.install ();
  Random.Prng.install ();
  Random.Threefry.install ()

let f32 shape xs = Concrete (Ndarray.of_floats Dtype.F32 shape xs)
let scalar x = f32 [||] [| x |]
let plugin_present () = Sys.getenv_opt "OJAX_PJRT_PLUGIN" <> None

let host = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected a concrete value"

let floats v =
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

let ints64 v =
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
      Ndarray.get_i64 nd idx)

let leaf = function Tree.Leaf v -> v | _ -> Alcotest.fail "expected a leaf"
let check_floats ?(eps = 1e-6) name = Alcotest.(check (array (float eps))) name

let check_message name expected f =
  match f () with
  | _ -> Alcotest.fail (name ^ ": expected an exception")
  | exception Invalid_argument got -> Alcotest.(check string) name expected got
  | exception Failure got -> Alcotest.(check string) name expected got
  | exception Errors.Jax_runtime_error got ->
      Alcotest.(check string) name expected got
  | exception Errors.Jaxpr_type_error got ->
      Alcotest.(check string) name expected got
  | exception Errors.Inconclusive_dimension_operation got ->
      Alcotest.(check string) name expected got
  | exception Dtypes.Type_promotion_error got ->
      Alcotest.(check string) name expected got

let key_page =
  "\nSee https://docs.jax.dev/en/latest/errors.html#jax.errors.KeyReuseError"

let check_reuse name expected f =
  match f () with
  | _ -> Alcotest.fail (name ^ ": expected KeyReuseError")
  | exception Errors.Key_reuse_error got ->
      Alcotest.(check string) name (expected ^ key_page) got

let seed n = Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int n |])
let key n = R.key (seed n)
let raw_key n = R.prng_key (seed n)
let uniform1 k = R.uniform k ~shape:[||] ~minval:0.0 ~maxval:1.0

let reuse_eager =
  "Previously-consumed key passed to jit-compiled function at index 0"

let capture_log thunk =
  let lines = ref [] in
  let prev = Config.set_log_sink (fun m -> lines := m :: !lines) in
  Fun.protect
    ~finally:(fun () ->
      let (_ : string -> unit) = Config.set_log_sink prev in
      ())
    (fun () ->
      let r = thunk () in
      (r, List.rev !lines))

let setters_restore () =
  let bool_setter name flag setter =
    let before = Config.value flag in
    setter (not before) (fun () ->
        Alcotest.(check bool) name (not before) (Config.value flag));
    Alcotest.(check bool) (name ^ " restored") before (Config.value flag)
  in
  bool_setter "enable_checks" Config.enable_checks Api.enable_checks;
  bool_setter "debug_key_reuse" Config.debug_key_reuse Api.debug_key_reuse;
  bool_setter "check_tracer_leaks" Config.check_tracer_leaks
    Api.check_tracer_leaks;
  bool_setter "explain_cache_misses" Config.explain_cache_misses
    Api.explain_cache_misses;
  bool_setter "softmax_custom_jvp" Config.softmax_custom_jvp
    Api.softmax_custom_jvp;
  bool_setter "threefry_partitionable" Config.threefry_partitionable
    Api.threefry_partitionable;
  bool_setter "allow_f16_reductions" Config.allow_f16_reductions
    Api.allow_f16_reductions;
  bool_setter "thread_guard" Config.thread_guard Api.thread_guard;
  bool_setter "jax2tf" Config.jax2tf_associative_scan_reductions
    Api.jax2tf_associative_scan_reductions;
  Api.checking_leaks (fun () ->
      Alcotest.(check bool)
        "checking_leaks" true
        (Config.value Config.check_tracer_leaks));
  Api.enable_custom_prng (fun () ->
      Alcotest.(check bool)
        "custom prng" true
        (Config.value Config.enable_custom_prng));
  Alcotest.(check bool)
    "defaults" true
    (Config.value Config.threefry_partitionable
    && Config.value Config.allow_f16_reductions
    && (not (Config.value Config.enable_checks))
    && Config.value Config.numpy_dtype_promotion = "standard"
    && Config.value Config.legacy_prng_key = "allow"
    && Config.value Config.default_matmul_precision = None);
  Api.transfer_guard "log" (fun () ->
      List.iter
        (fun flag ->
          Alcotest.(check (option string))
            (Config.name flag) (Some "log") (Config.value flag))
        [
          Config.transfer_guard_host_to_device;
          Config.transfer_guard_device_to_device;
          Config.transfer_guard_device_to_host;
        ]);
  Alcotest.(check (option string))
    "guard restored" None
    (Config.value Config.transfer_guard_host_to_device);
  Api.transfer_guard_device_to_host "disallow" (fun () ->
      Alcotest.(check (option string))
        "d2h" (Some "disallow")
        (Config.value Config.transfer_guard_device_to_host));
  Api.transfer_guard_device_to_device "log" (fun () ->
      Alcotest.(check (option string))
        "d2d" (Some "log")
        (Config.value Config.transfer_guard_device_to_device))

let setters_validate () =
  check_message "transfer"
    "new enum value must be None or in ['allow', 'log', 'disallow', \
     'log_explicit', 'disallow_explicit'], got x of type <class 'str'>."
    (fun () -> Api.transfer_guard_host_to_device "x" Fun.id);
  check_message "gc"
    "new enum value must be None or in ['allow', 'log', 'fatal'], got x of \
     type <class 'str'>." (fun () ->
      Api.array_garbage_collection_guard "x" Fun.id);
  check_message "promotion" "'x' is not a valid NumpyDtypePromotion" (fun () ->
      Api.numpy_dtype_promotion "x" Fun.id);
  check_message "legacy" "'x' is not a valid LegacyPrngKeyState" (fun () ->
      Api.legacy_prng_key "x" Fun.id);
  match Api.default_matmul_precision "x" Fun.id with
  | () -> Alcotest.fail "matmul: expected an exception"
  | exception Invalid_argument m ->
      Alcotest.(check bool)
        "matmul" true
        (String.starts_with ~prefix:"new enum value must be None or in" m)

let no_effect_flags () =
  let x = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let run () =
    floats
      (Api.jit
         (function
           | [ Tree.Leaf a ] -> Tree.Leaf (L.cumsum a)
           | _ -> Alcotest.fail "arity")
         [ Tree.Leaf x ]
      |> leaf)
  in
  check_floats "cumsum unchanged" [| 1.; 3.; 6. |]
    (Api.jax2tf_associative_scan_reductions true run);
  check_floats "single process guard" [| 1.; 3.; 6. |]
    (Api.thread_guard true (fun () -> Api.thread_guard false run))

let leaked : value list ref = ref []

let leak_case name transform =
  leaked := [];
  let fn a =
    leaked := a :: !leaked;
    L.mul a a
  in
  match Api.checking_leaks (fun () -> transform fn) with
  | () -> Alcotest.fail (name ^ ": expected a leak")
  | exception Failure m ->
      Alcotest.(check bool)
        name true
        (String.starts_with ~prefix:"Leaked trace" m)

let tree_fn fn = function
  | [ Tree.Leaf a ] -> Tree.Leaf (fn a)
  | _ -> Alcotest.fail "arity"

let tracer_leaks () =
  leak_case "jit" (fun fn ->
      ignore (Api.jit (tree_fn fn) [ Tree.Leaf (scalar 1.) ]));
  leak_case "make_jaxpr" (fun fn ->
      ignore (Api.make_jaxpr (tree_fn fn) [ Tree.Leaf (scalar 1.) ]));
  leak_case "vmap" (fun fn ->
      ignore
        (Api.vmap (tree_fn fn) [ Some 0 ]
           [ Tree.Leaf (f32 [| 2 |] [| 1.; 2. |]) ]));
  leaked := [];
  let keep a =
    leaked := a :: !leaked;
    L.mul a a
  in
  let ok =
    Api.checking_leaks (fun () ->
        Api.grad (tree_fn keep) [ Tree.Leaf (scalar 2.) ])
  in
  check_floats "grad is not leak checked" [| 4. |] (floats (leaf ok));
  leaked := [];
  let clean a = L.mul a a in
  let sq = tree_fn clean in
  Api.checking_leaks (fun () ->
      let jitted =
        Api.jit
          (function
            | [ Tree.Leaf a ] -> Api.grad sq [ Tree.Leaf a ]
            | _ -> Alcotest.fail "arity")
          [ Tree.Leaf (scalar 2.) ]
      in
      check_floats "jit grad" [| 4. |] (floats (leaf jitted));
      let v = Api.vmap sq [ Some 0 ] [ Tree.Leaf (f32 [| 2 |] [| 1.; 2. |]) ] in
      check_floats "vmap" [| 1.; 4. |] (floats (leaf v));
      let y, _ = Api.linearize sq [ Tree.Leaf (scalar 2.) ] in
      check_floats "linearize" [| 4. |] (floats (leaf y));
      let y, _ = Api.vjp sq [ Tree.Leaf (scalar 2.) ] in
      check_floats "vjp" [| 4. |] (floats (leaf y));
      Api.softmax_custom_jvp true (fun () ->
          let soft =
            Api.jit
              (tree_fn (fun a -> Nn.Functions.softmax a))
              [ Tree.Leaf (f32 [| 2 |] [| 1.; 2. |]) ]
          in
          ignore (leaf soft)))

let sample_jaxpr () =
  (Api.make_jaxpr
     (tree_fn (fun a -> L.add (L.sin a) (L.full_like a 1.0)))
     [ Tree.Leaf (scalar 1.) ])
    .jaxpr

let jaxpr_errors () =
  let j = sample_jaxpr () in
  Jaxpr.typecheck_jaxpr j;
  check_message "unbound" "Variable 'b' not defined" (fun () ->
      Jaxpr.typecheck_jaxpr { j with in_binders = [] });
  let first = List.hd j.eqns in
  check_message "rebind" "Variable 'b' already bound" (fun () ->
      Jaxpr.typecheck_jaxpr { j with eqns = first :: j.eqns });
  let wrong =
    {
      vid = Core.fresh_id ();
      vaval = { shape = [| 2 |]; dtype = Dtype.F32; weak_type = false };
    }
  in
  check_message "typed"
    "Value for variable 'b' inconsistently typed as f32[] for let-binder of \
     type f32[2]" (fun () ->
      Jaxpr.typecheck_jaxpr
        { j with eqns = [ { first with outs = [ wrong ] } ] })

let enable_checks_effect () =
  let good = sample_jaxpr () in
  let bad = { good with in_binders = [] } in
  let lowered =
    {
      Stages.closed = { jid = Core.fresh_id (); jaxpr = bad; consts = [] };
      in_avals = [];
      out_avals = [];
    }
  in
  check_message "compile under checks" "Variable 'b' not defined" (fun () ->
      Api.enable_checks true (fun () -> Stages.compile lowered));
  ignore (Stages.compile lowered);
  let r =
    Api.enable_checks true (fun () ->
        Api.jit (tree_fn L.sin) [ Tree.Leaf (scalar 0.) ])
  in
  check_floats "valid programs still run" [| 0. |] (floats (leaf r))

let key_reuse_eager () =
  Api.debug_key_reuse true (fun () ->
      let k = key 0 in
      check_floats "first use" [| 0.947667 |] (floats (uniform1 k));
      check_reuse "second use" reuse_eager (fun () -> uniform1 k);
      let k = key 0 in
      ignore (R.split k 2);
      check_reuse "split twice"
        "Previously-consumed key passed to random_split at index 0" (fun () ->
          R.split k 2);
      let k = key 0 in
      ignore (R.bits k ~shape:[||]);
      check_reuse "bits twice"
        "Previously-consumed key passed to random_bits at index 0" (fun () ->
          R.bits k ~shape:[||]);
      let keys = R.split (key 0) 3 in
      let k0 = L.index_in_dim ~keepdims:false keys 0 in
      check_floats "subkey" [| 0.8423141 |] (floats (uniform1 k0));
      check_reuse "subkey twice" reuse_eager (fun () -> uniform1 k0);
      check_reuse "slice twice"
        "Previously-consumed key passed to slice at index 0" (fun () ->
          L.index_in_dim ~keepdims:false keys 0);
      ignore (L.index_in_dim ~keepdims:false keys 1);
      let k = key 0 in
      ignore (R.fold_in k (seed 1));
      ignore (R.fold_in k (seed 1));
      ignore (uniform1 k);
      ignore (R.fold_in k (seed 1));
      check_reuse "after fold_in" reuse_eager (fun () -> uniform1 k);
      let k = key 0 in
      let c = R.clone k in
      ignore (uniform1 k);
      check_floats "clone is fresh" [| 0.947667 |] (floats (uniform1 c));
      check_reuse "clone reuse" reuse_eager (fun () -> uniform1 c);
      let k = key 0 in
      ignore (R.key_data k);
      ignore (uniform1 k);
      ignore (R.key_data k);
      let d = R.key_data (key 0) in
      ignore (uniform1 (R.wrap_key_data d));
      ignore (uniform1 (R.wrap_key_data d));
      let raw = raw_key 0 in
      ignore (uniform1 raw);
      check_floats "raw keys are not tracked" [| 0.947667 |]
        (floats (uniform1 raw)));
  let k = key 0 in
  ignore (uniform1 k);
  ignore (uniform1 k);
  Api.debug_key_reuse true (fun () ->
      ignore (uniform1 k);
      check_reuse "tracked once enabled" reuse_eager (fun () -> uniform1 k))

let sampler_table =
  [
    ("uniform", fun k -> uniform1 k);
    ("normal", fun k -> R.normal k ~shape:[||]);
    ("exponential", fun k -> R.exponential k ~shape:[||]);
    ("randint", fun k -> R.randint k ~shape:[||] ~minval:0 ~maxval:3);
    ("bernoulli", fun k -> R.bernoulli k ~shape:[||] ~p:0.5);
    ("gamma", fun k -> R.gamma k ~shape:[||] ~a:2.0);
    ("permutation", fun k -> R.permutation k 3);
    ("choice", fun k -> R.choice k ~n:3 ~shape:[||] ~replace:true);
    ("beta", fun k -> R.beta k ~shape:[||] ~a:1.0 ~b:2.0);
    ("poisson", fun k -> R.poisson k ~shape:[||] ~lam:2.0);
  ]

let key_reuse_samplers () =
  Api.debug_key_reuse true (fun () ->
      List.iter
        (fun (name, sample) ->
          let k = key 3 in
          ignore (sample k);
          check_reuse name reuse_eager (fun () -> sample k))
        sampler_table)

let jit_key fn k = Api.jit (tree_fn fn) [ Tree.Leaf k ] |> leaf

let key_reuse_jit () =
  Api.debug_key_reuse true (fun () ->
      check_reuse "static reuse" "In jit, argument 0 is already consumed."
        (fun () -> jit_key (fun k -> L.add (uniform1 k) (uniform1 k)) (key 0));
      let single = Api.jit (tree_fn uniform1) in
      let k = key 0 in
      check_floats "jit first" [| 0.947667 |]
        (floats (leaf (single [ Tree.Leaf k ])));
      check_reuse "jit second" reuse_eager (fun () -> single [ Tree.Leaf k ]);
      let k = key 0 in
      let same = jit_key Fun.id k in
      check_floats "identity keeps data" [| 0.; 0. |] (floats same);
      ignore (uniform1 k);
      let out = jit_key Fun.id (key 0) in
      ignore (uniform1 out);
      check_reuse "forwarded output" reuse_eager (fun () -> uniform1 out);
      let split_use k =
        let keys = R.split k 2 in
        L.add
          (uniform1 (L.index_in_dim ~keepdims:false keys 0))
          (uniform1 (L.index_in_dim ~keepdims:false keys 1))
      in
      check_floats "split in jit" [| 0.84960794 |]
        (floats (jit_key split_use (key 0)));
      let k = key 0 in
      let pair =
        Api.jit
          (function
            | [ Tree.Leaf k ] ->
                let keys = R.split k 2 in
                Tree.Tuple
                  [
                    Tree.Leaf (L.index_in_dim ~keepdims:false keys 0);
                    Tree.Leaf (uniform1 (L.index_in_dim ~keepdims:false keys 1));
                  ]
            | _ -> Alcotest.fail "arity")
          [ Tree.Leaf k ]
      in
      let out =
        match pair with
        | Tree.Tuple [ Tree.Leaf a; _ ] -> a
        | _ -> Alcotest.fail "shape"
      in
      check_floats "sourced output" [| 0.8423141 |] (floats (uniform1 out));
      check_reuse "sourced output twice" reuse_eager (fun () -> uniform1 out);
      check_reuse "input was consumed" reuse_eager (fun () -> uniform1 k);
      let ks = R.split (key 1) 2 in
      let sampled =
        Api.vmap (tree_fn uniform1) [ Some 0 ] [ Tree.Leaf ks ] |> leaf
      in
      check_floats "vmap" [| 0.70261526; 0.40364635 |] (floats sampled);
      check_reuse "vmap twice" reuse_eager (fun () ->
          Api.vmap (tree_fn uniform1) [ Some 0 ] [ Tree.Leaf ks ]);
      let raw =
        jit_key (fun k -> L.add (uniform1 k) (uniform1 k)) (raw_key 0)
      in
      check_floats "raw reuse in jit passes" [| 1.895334 |] (floats raw))

let legacy_message =
  "Legacy uint32 key array passed as key to jax.random function. Please create \
   keys using jax.random.key(). If use of a raw key array was intended, set \
   jax_legacy_prng_key=\"allow\"."

let legacy_keys () =
  Api.legacy_prng_key "error" (fun () ->
      List.iter
        (fun (name, f) -> check_message name legacy_message f)
        [
          ("uniform", fun () -> ignore (uniform1 (raw_key 0)));
          ("split", fun () -> ignore (R.split (raw_key 0) 2));
          ("fold_in", fun () -> ignore (R.fold_in (raw_key 0) (seed 1)));
          ("key_data", fun () -> ignore (R.key_data (raw_key 0)));
          ("key_impl", fun () -> ignore (R.key_impl (raw_key 0)));
          ("jit", fun () -> ignore (jit_key uniform1 (raw_key 0)));
        ];
      check_floats "typed passes" [| 0.947667 |] (floats (uniform1 (key 0)));
      List.iter
        (fun (name, sample) ->
          ignore name;
          ignore (sample (key 3)))
        sampler_table;
      check_floats "typed in jit" [| 0.947667 |]
        (floats (jit_key uniform1 (key 0)));
      let ks = R.split (key 0) 2 in
      ignore (uniform1 (L.index_in_dim ~keepdims:false ks 0));
      ignore (Api.vmap (tree_fn uniform1) [ Some 0 ] [ Tree.Leaf ks ]));
  let v, logged =
    capture_log (fun () ->
        Api.legacy_prng_key "warn" (fun () -> uniform1 (raw_key 0)))
  in
  check_floats "warn still samples" [| 0.947667 |] (floats v);
  Alcotest.(check (list string)) "warned" [ legacy_message ] logged;
  let _, quiet = capture_log (fun () -> uniform1 (raw_key 0)) in
  Alcotest.(check (list string)) "allow is quiet" [] quiet

let custom_prng () =
  Alcotest.(check bool) "raw by default" false (R.is_key (raw_key 0));
  Alcotest.(check bool) "key is typed" true (R.is_key (key 0));
  Alcotest.(check bool) "split typed" true (R.is_key (R.split (key 0) 2));
  let raw = Concrete (Ndarray.of_floats Dtype.Uint32 [| 2 |] [| 0.; 0. |]) in
  Alcotest.(check bool) "raw split stays raw" false (R.is_key (R.split raw 2));
  let (), logged =
    capture_log (fun () ->
        Api.enable_custom_prng (fun () ->
            let k = raw_key 0 in
            Alcotest.(check bool) "PRNGKey typed" true (R.is_key k);
            check_floats "PRNGKey data" [| 0.; 0. |] (floats (R.key_data k));
            Alcotest.(check bool) "split typed" true (R.is_key (R.split k 2));
            Alcotest.(check bool)
              "raw split upgraded" true
              (R.is_key (R.split raw 2));
            Alcotest.(check bool)
              "raw fold_in upgraded" true
              (R.is_key (R.fold_in raw (seed 1)));
            check_floats "raw uniform" [| 0.947667 |] (floats (uniform1 raw))))
  in
  Alcotest.(check int) "deprecation warnings" 3 (List.length logged);
  Alcotest.(check string)
    "warning text"
    "Raw arrays as random keys to jax.random functions are deprecated. \
     Assuming valid threefry2x32 key for now."
    (List.hd logged)

let threefry_original () =
  let k = raw_key 42 in
  let check_u32 name expected v =
    Alcotest.(check (array int64)) name expected (ints64 v)
  in
  Api.threefry_partitionable false (fun () ->
      check_u32 "split3"
        [|
          3134548294L;
          3733159049L;
          3746501087L;
          894150801L;
          801545058L;
          2363201431L;
        |]
        (R.split k 3);
      check_u32 "bits5"
        [| 3134548294L; 3733159049L; 843694964L; 894150801L; 801545058L |]
        (R.bits k ~shape:[| 5 |]);
      check_u32 "bits2x2"
        [| 2465931498L; 3679230171L; 255383827L; 267815257L |]
        (R.bits k ~shape:[| 2; 2 |]);
      check_u32 "split nested"
        [|
          2954079971L;
          2575172198L;
          1139032965L;
          1035738655L;
          3276725750L;
          1223413151L;
          173471767L;
          13772463L;
        |]
        (Random.Prng.random_split k [| 2; 2 |]);
      check_floats "uniform"
        [| 0.5741443634033203; 0.1001582145690918; 0.059461116790771484 |]
        (floats (R.uniform k ~shape:[| 3 |] ~minval:0.0 ~maxval:1.0));
      check_floats "normal"
        [| 0.18693546950817108; -1.2806504964828491; -1.559313178062439 |]
        (floats (R.normal k ~shape:[| 3 |]));
      check_u32 "randint" [| 95L; 34L; 37L; 93L |]
        (R.randint k ~shape:[| 4 |] ~minval:0 ~maxval:100);
      check_u32 "fold_in unchanged"
        [| 2547012911L; 1371500959L |]
        (R.fold_in k (seed 7));
      check_floats "jit"
        [| 0.5741443634033203; 0.1001582145690918; 0.059461116790771484 |]
        (floats
           (jit_key
              (fun k -> R.uniform k ~shape:[| 3 |] ~minval:0.0 ~maxval:1.0)
              k));
      Config.with_value Config.enable_x64 true (fun () ->
          check_floats "uniform x64" ~eps:1e-14
            [| 0.7298188969046309; 0.8691938242843895; 0.8723002596642415 |]
            (floats (R.uniform k ~shape:[| 3 |] ~minval:0.0 ~maxval:1.0))));
  check_u32 "partitionable split"
    [|
      1832780943L; 270669613L; 64467757L; 2916123636L; 2465931498L; 255383827L;
    |]
    (R.split k 3);
  check_u32 "partitionable bits"
    [| 2098992034L; 2919706841L; 2646866425L; 2409546199L; 1935504149L |]
    (R.bits k ~shape:[| 5 |]);
  let same_trace = Api.jit (tree_fn (fun k -> R.bits k ~shape:[| 5 |])) in
  check_u32 "flag splits jit cache"
    [| 2098992034L; 2919706841L; 2646866425L; 2409546199L; 1935504149L |]
    (leaf (same_trace [ Tree.Leaf k ]));
  check_u32 "flag splits jit cache off"
    [| 3134548294L; 3733159049L; 843694964L; 894150801L; 801545058L |]
    (leaf
       (Api.threefry_partitionable false (fun () -> same_trace [ Tree.Leaf k ])));
  Api.threefry_partitionable false (fun () ->
      Api.debug_key_reuse true (fun () ->
          let k = key 0 in
          ignore (R.split k 2);
          check_reuse "original split consumes"
            "Previously-consumed key passed to random_split at index 0"
            (fun () -> R.split k 2)))

let softmax_jvp () =
  let x = f32 [| 3 |] [| 0.5; -0.25; 1.5 |] in
  let v = f32 [| 3 |] [| 1.; 0.; -1. |] in
  let soft = tree_fn (fun a -> Nn.Functions.softmax a) in
  let run () =
    let p, t = Api.jvp soft [ Tree.Leaf x ] [ Tree.Leaf v ] in
    (floats (leaf p), floats (leaf t))
  in
  let primal =
    [| 0.23862655460834503; 0.11271920055150986; 0.6486542224884033 |]
  in
  let eps = 1e-6 in
  let p, t = run () in
  check_floats ~eps "primal off" primal p;
  check_floats ~eps "tangent off"
    [| 0.3364700376987457; 0.04621799290180206; -0.382688045501709 |]
    t;
  let first =
    tree_fn (fun a -> L.index_in_dim ~keepdims:false (Nn.Functions.softmax a) 0)
  in
  check_floats ~eps "grad off"
    [| 0.18168392777442932; -0.026897795498371124; -0.1547861248254776 |]
    (floats (leaf (Api.grad first [ Tree.Leaf x ])));
  Api.softmax_custom_jvp true (fun () ->
      let p, t = run () in
      check_floats ~eps "primal on" primal p;
      check_floats ~eps "tangent on"
        [| 0.3364700675010681; 0.04621799290180206; -0.382688045501709 |]
        t;
      check_floats ~eps "grad on"
        [| 0.18168392777442932; -0.026897793635725975; -0.1547861248254776 |]
        (floats (leaf (Api.grad first [ Tree.Leaf x ])));
      let text =
        Pretty_printer.closed_jaxpr_to_string
          (Api.make_jaxpr soft [ Tree.Leaf x ])
      in
      let contains s sub =
        let n = String.length sub in
        let rec go i =
          i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
        in
        go 0
      in
      Alcotest.(check bool) "custom_jvp eqn" true (contains text "custom_jvp"))

let strict_text a b =
  Printf.sprintf
    "Input dtypes (%s, %s) have no available implicit dtype promotion path \
     when jax_numpy_dtype_promotion=strict. Try explicitly casting inputs to \
     the desired output type, or set jax_numpy_dtype_promotion=standard."
    a b

let dtype_t =
  Alcotest.testable
    (fun fmt d -> Format.pp_print_string fmt (Dtype.short_name d))
    ( = )

let dtype_promotion () =
  let check_rt name expected args =
    Alcotest.(check dtype_t) name expected (fst (Dtypes.result_type args))
  in
  Alcotest.(check dtype_t)
    "standard" Dtype.F32
    (Dtypes.promote_types Dtype.I32 Dtype.F32);
  Api.numpy_dtype_promotion "strict" (fun () ->
      List.iter
        (fun (a, an, b, bn) ->
          check_message (an ^ bn)
            (strict_text ("'" ^ an ^ "'") ("'" ^ bn ^ "'"))
            (fun () -> Dtypes.promote_types a b))
        [
          (Dtype.I32, "int32", Dtype.F32, "float32");
          (Dtype.F32, "float32", Dtype.F64, "float64");
          (Dtype.I32, "int32", Dtype.I64, "int64");
          (Dtype.I8, "int8", Dtype.Uint8, "uint8");
          (Dtype.F16, "float16", Dtype.Bf16, "bfloat16");
        ];
      Alcotest.(check dtype_t)
        "same" Dtype.F32
        (Dtypes.promote_types Dtype.F32 Dtype.F32);
      check_rt "strong and weak int" Dtype.I32
        [ (Dtype.I32, false); (Dtype.I32, true) ];
      check_rt "strong and weak float" Dtype.F32
        [ (Dtype.F32, false); (Dtype.F32, true) ];
      check_rt "weak int and weak float" Dtype.F32
        [ (Dtype.I32, true); (Dtype.F32, true) ];
      check_message "int and weak float"
        (strict_text "'int32'" "\"<class 'float'>\"") (fun () ->
          Dtypes.result_type [ (Dtype.I32, false); (Dtype.F32, true) ]);
      let ints = Concrete (Ndarray.of_floats Dtype.I32 [| 2 |] [| 1.; 1. |]) in
      let fls = f32 [| 2 |] [| 1.; 1. |] in
      check_message "ufunc add" (strict_text "'int32'" "'float32'") (fun () ->
          Numpy.Ufuncs.add ints fls))

let f16_reductions () =
  let ones shape dt =
    let n = Array.fold_left ( * ) 1 shape in
    Concrete (Ndarray.of_floats dt shape (Array.make n 1.0))
  in
  check_floats "allowed by default" [| 2.; 2.; 2. |]
    (floats (L.reduce_sum (ones [| 2; 3 |] Dtype.F16) [| 0 |]));
  Api.allow_f16_reductions false (fun () ->
      check_message "f16"
        "reduce_sum on operand f16[2,3] is not allowed when \
         jax_allow_f16_reductions=False." (fun () ->
          L.reduce_sum (ones [| 2; 3 |] Dtype.F16) [| 0 |]);
      check_message "bf16"
        "reduce_sum on operand bf16[2,3] is not allowed when \
         jax_allow_f16_reductions=False." (fun () ->
          L.reduce_sum (ones [| 2; 3 |] Dtype.Bf16) [| 0 |]);
      check_floats "singleton axis" [| 1.; 1.; 1. |]
        (floats (L.reduce_sum (ones [| 1; 3 |] Dtype.F16) [| 0 |]));
      check_floats "f32" [| 2.; 2.; 2. |]
        (floats (L.reduce_sum (ones [| 2; 3 |] Dtype.F32) [| 0 |]));
      check_floats "jnp.sum upcasts" [| 6. |]
        (floats (Numpy.Reductions.sum (ones [| 2; 3 |] Dtype.F16))))

let matrix = f32 [| 2; 2 |] [| 1.; 1.; 1.; 1. |]

let dot_text () =
  Stages.as_text
    (Stages.lower
       (function [ a; b ] -> [ L.dot a b ] | _ -> Alcotest.fail "arity")
       (let a = { shape = [| 2; 2 |]; dtype = Dtype.F32; weak_type = false } in
        [ a; a ]))

let contains s sub =
  let n = String.length sub in
  let rec go i =
    i + n <= String.length s && (String.sub s i n = sub || go (i + 1))
  in
  go 0

let matmul_precision () =
  List.iter
    (fun (name, level) ->
      Api.default_matmul_precision name (fun () ->
          let text = dot_text () in
          Alcotest.(check bool)
            name true
            (contains text (Printf.sprintf "precision = [%s, %s]" level level))))
    [
      ("default", "DEFAULT");
      ("high", "HIGH");
      ("highest", "HIGHEST");
      ("bfloat16", "DEFAULT");
      ("tensorfloat32", "HIGH");
      ("float32", "HIGHEST");
    ];
  Api.default_matmul_precision "BF16_BF16_F32_X3" (fun () ->
      Alcotest.(check bool)
        "x3" true
        (contains (dot_text ())
           "precision = [DEFAULT, DEFAULT], algorithm = <lhs_precision_type = \
            bf16, rhs_precision_type = bf16, accumulation_type = f32, \
            lhs_component_count = 1, rhs_component_count = 1, \
            num_primitive_operations = 3, allow_imprecise_accumulation = \
            false> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>"));
  Api.default_matmul_precision "F64_F64_F64" (fun () ->
      let text = dot_text () in
      Alcotest.(check bool)
        "f64 converts" true
        (contains text
           "stablehlo.convert %0 : (tensor<2x2xf32>) -> tensor<2x2xf64>"
        && contains text "-> tensor<2x2xf64>\n"
        && contains text ": (tensor<2x2xf64>) -> tensor<2x2xf32>"));
  let unsupported =
    "The precision 'F16_F16_F32' is not supported by dot_general on CPU"
  in
  Api.default_matmul_precision "F16_F16_F32" (fun () ->
      check_message "lowering" unsupported dot_text;
      check_message "eager" unsupported (fun () -> L.dot matrix matrix));
  Api.default_matmul_precision "high" (fun () ->
      let conv =
        Stages.as_text
          (Stages.lower
             (function
               | [ a; b ] ->
                   [
                     L.conv_general_dilated a b ~window_strides:[| 1 |] L.Valid;
                   ]
               | _ -> Alcotest.fail "arity")
             [
               { shape = [| 1; 1; 3 |]; dtype = Dtype.F32; weak_type = false };
               { shape = [| 1; 1; 2 |]; dtype = Dtype.F32; weak_type = false };
             ])
      in
      Alcotest.(check bool)
        "conv" true
        (contains conv
           "precision_config = [#stablehlo<precision HIGH>, \
            #stablehlo<precision HIGH>]"));
  let f =
    Api.jit (function
      | [ Tree.Leaf a; Tree.Leaf b ] -> Tree.Leaf (L.dot a b)
      | _ -> Alcotest.fail "arity")
  in
  List.iter
    (fun name ->
      check_floats name [| 2.; 2.; 2.; 2. |]
        (Api.default_matmul_precision name (fun () ->
             floats (leaf (f [ Tree.Leaf matrix; Tree.Leaf matrix ])))))
    [ "high"; "highest"; "F32_F32_F32"; "F64_F64_F64"; "F16_F16_F16" ]

let cache_misses () =
  let f = Api.jit (tree_fn (fun a -> L.add a a)) in
  let _, logged =
    capture_log (fun () ->
        Api.explain_cache_misses true (fun () ->
            ignore (f [ Tree.Leaf (scalar 1.) ]);
            ignore (f [ Tree.Leaf (scalar 2.) ]);
            ignore (f [ Tree.Leaf (f32 [| 2 |] [| 1.; 1. |]) ]);
            Config.with_value Config.enable_x64 true (fun () ->
                ignore (f [ Tree.Leaf (f32 [| 2 |] [| 1.; 1. |]) ]))))
  in
  Alcotest.(check (list string))
    "explanations"
    [
      "TRACING CACHE MISS at <ocaml>:\n  never seen function:\n    <fun>";
      "TRACING CACHE MISS at <ocaml>:\n\
      \  for <fun>\n\
      \  all previously seen cache keys differ. For the closest previous key:\n\
      \  different input types:\n\
      \    * at args[0], now f32[2] and before f32[]";
      "TRACING CACHE MISS at <ocaml>:\n\
      \  for <fun>\n\
      \  all previously seen cache keys differ. For the closest previous key:\n\
      \  Tracing context doesn't match, e.g. due to config or context manager.\n\
      \    * jax_enable_x64 differs: now True and before False";
    ]
    logged;
  let g = Api.jit (tree_fn (fun a -> L.add a a)) in
  let _, quiet = capture_log (fun () -> ignore (g [ Tree.Leaf (scalar 1.) ])) in
  Alcotest.(check (list string)) "off by default" [] quiet

let transfer_text aval =
  Printf.sprintf
    "host-to-device transfer: aval=ShapedArray(%s), \
     dst_sharding=SingleDeviceSharding(device=CpuDevice(id=0), \
     memory_kind=device)"
    aval

let transfer_guards () =
  let x = f32 [| 2 |] [| 1.; 1. |] in
  let double = Api.jit (tree_fn (fun a -> L.add a a)) in
  let disallowed =
    "INVALID_ARGUMENT: Disallowed " ^ transfer_text "float32[2]"
  in
  let scalar_disallowed =
    "INVALID_ARGUMENT: Disallowed " ^ transfer_text "float32[]"
  in
  Api.transfer_guard "disallow" (fun () ->
      check_message "jit" disallowed (fun () -> double [ Tree.Leaf x ]);
      check_message "scalar" scalar_disallowed (fun () ->
          Api.jit (tree_fn (fun a -> L.add a a)) [ Tree.Leaf (scalar 1.) ]);
      let on_device = Api.device_put (Tree.Leaf x) in
      check_floats "explicit put" [| 1.; 1. |] (floats (leaf on_device));
      if plugin_present () then
        check_floats "device argument" [| 2.; 2. |]
          (floats (leaf (double [ on_device ]))));
  Api.transfer_guard "disallow_explicit" (fun () ->
      check_message "explicit" disallowed (fun () ->
          Api.device_put (Tree.Leaf x));
      check_message "implicit" disallowed (fun () -> double [ Tree.Leaf x ]));
  let log_of level thunk =
    snd (capture_log (fun () -> Api.transfer_guard level thunk))
  in
  Alcotest.(check (list string))
    "log implicit"
    [ transfer_text "float32[2]" ]
    (log_of "log" (fun () -> ignore (double [ Tree.Leaf x ])));
  Alcotest.(check (list string))
    "log ignores explicit" []
    (log_of "log" (fun () -> ignore (Api.device_put (Tree.Leaf x))));
  Alcotest.(check (list string))
    "log_explicit"
    [ transfer_text "float32[2]" ]
    (log_of "log_explicit" (fun () -> ignore (Api.device_put (Tree.Leaf x))));
  Alcotest.(check (list string))
    "allow" []
    (log_of "allow" (fun () -> ignore (double [ Tree.Leaf x ])));
  if plugin_present () then
    let y = leaf (double [ Api.device_put (Tree.Leaf x) ]) in
    Api.transfer_guard "disallow_explicit" (fun () ->
        check_floats "device_get on cpu" [| 2.; 2. |]
          (floats (leaf (Api.device_get (Tree.Leaf y))));
        match Api.device_put (Tree.Leaf y) with
        | Tree.Leaf _ -> ()
        | _ -> Alcotest.fail "tree")

let gc_message =
  "`jax.Array` was garbage collected before it was explicitly destroyed"

let drop_device_array () =
  ignore (Api.device_put (Tree.Leaf (f32 [| 2 |] [| 1.; 1. |])))

let gc_guard () =
  if plugin_present () then (
    let (), logged =
      capture_log (fun () ->
          Api.array_garbage_collection_guard "log" (fun () ->
              drop_device_array ();
              Gc.full_major ();
              Gc.full_major ()))
    in
    Alcotest.(check bool) "logged" true (List.mem gc_message logged);
    let (), quiet =
      capture_log (fun () ->
          let kept = Api.device_put (Tree.Leaf (f32 [| 1 |] [| 1. |])) in
          Api.array_garbage_collection_guard "log" (fun () ->
              (match kept with
              | Tree.Leaf (Device b) -> Pjrt.Buffer.destroy b
              | _ -> Alcotest.fail "expected a device array");
              Gc.full_major ();
              Gc.full_major ()))
    in
    Alcotest.(check (list string)) "explicitly destroyed" [] quiet;
    let (), allowed =
      capture_log (fun () ->
          drop_device_array ();
          Gc.full_major ();
          Gc.full_major ())
    in
    Alcotest.(check (list string)) "allow" [] allowed;
    Unix.putenv "OJAX_GC_FATAL_CHILD" "1";
    let pid =
      Unix.create_process Sys.executable_name [| Sys.executable_name |]
        Unix.stdin Unix.stdout
        (Unix.descr_of_out_channel (open_out "/dev/null"))
    in
    let _, status = Unix.waitpid [] pid in
    Unix.putenv "OJAX_GC_FATAL_CHILD" "";
    Alcotest.(check bool)
      "fatal aborts" true
      (status = Unix.WSIGNALED Sys.sigabrt))

let () =
  if Sys.getenv_opt "OJAX_GC_FATAL_CHILD" = Some "1" then (
    Api.array_garbage_collection_guard "fatal" (fun () ->
        drop_device_array ();
        Gc.full_major ();
        Gc.full_major ());
    exit 0)

let error_types () =
  Alcotest.(check int) "even" 3 (Core.divide_shape_sizes [| 6 |] [| 2 |]);
  Alcotest.(check int) "empty" 1 (Core.divide_shape_sizes [| 0 |] [| 0 |]);
  check_message "uneven"
    "Cannot divide evenly the sizes of shapes (5,) and (2,). The remainder 1 \
     should be 0." (fun () -> Core.divide_shape_sizes [| 5 |] [| 2 |]);
  (match Core.divide_shape_sizes [| 6 |] [| 0 |] with
  | _ -> Alcotest.fail "expected division by zero"
  | exception Division_by_zero -> ());
  if plugin_present () then
    match Backend.run_text "this is not stablehlo" [ scalar 1. ] with
    | _ -> Alcotest.fail "expected a runtime error"
    | exception Errors.Jax_runtime_error _ -> ()

let monitoring_listeners () =
  Monitoring.clear_event_listeners ();
  let log = ref [] in
  let push x = log := x :: !log in
  let events e _ = push ("event:" ^ e) in
  let second e kwargs =
    match kwargs with
    | [ ("a", Monitoring.Number n) ] -> push (Printf.sprintf "second:%s:%d" e n)
    | _ -> push ("second:" ^ e)
  in
  let durations e d _ = push (Printf.sprintf "duration:%s:%g" e d) in
  let spans e s t _ = push (Printf.sprintf "span:%s:%g:%g" e s t) in
  let scalars e v _ = push (Printf.sprintf "scalar:%s:%g" e v) in
  Monitoring.register_event_listener events;
  Monitoring.register_event_listener second;
  Monitoring.register_event_duration_secs_listener durations;
  Monitoring.register_event_time_span_listener spans;
  Monitoring.register_scalar_listener scalars;
  Alcotest.(check int)
    "getter" 2
    (List.length (Monitoring.get_event_listeners ()));
  Alcotest.(check int)
    "duration getter" 1
    (List.length (Monitoring.get_event_duration_listeners ()));
  Alcotest.(check int)
    "span getter" 1
    (List.length (Monitoring.get_event_time_span_listeners ()));
  Alcotest.(check int)
    "scalar getter" 1
    (List.length (Monitoring.get_scalar_listeners ()));
  Monitoring.record_event ~kwargs:[ ("a", Monitoring.Number 7) ] "e1";
  Monitoring.record_event_duration_secs "d" 1.5;
  Monitoring.record_event_time_span "s" 1.0 2.0;
  Monitoring.record_scalar "v" 3.0;
  Alcotest.(check (list string))
    "order of registration"
    [ "event:e1"; "second:e1:7"; "duration:d:1.5"; "span:s:1:2"; "scalar:v:3" ]
    (List.rev !log);
  log := [];
  Monitoring.unregister_event_listener events;
  Monitoring.unregister_event_duration_listener durations;
  Monitoring.unregister_event_time_span_listener spans;
  Monitoring.record_event "e2";
  Monitoring.record_event_duration_secs "d" 1.0;
  Monitoring.record_event_time_span "s" 1.0 2.0;
  Alcotest.(check (list string)) "after unregister" [ "second:e2" ] !log;
  (match Monitoring.unregister_event_listener events with
  | () -> Alcotest.fail "unregistering twice must fail"
  | exception Assert_failure _ -> ());
  log := [];
  Monitoring.clear_event_listeners ();
  Monitoring.record_event "e3";
  Monitoring.record_scalar "v" 9.0;
  Alcotest.(check (list string))
    "clear leaves scalar listeners, as jax does" [ "scalar:v:9" ] !log;
  Monitoring.unregister_scalar_listener scalars;
  Alcotest.(check int)
    "scalar removed" 0
    (List.length (Monitoring.get_scalar_listeners ()));
  let seen = ref [] in
  let compile_listener e _ kwargs =
    match kwargs with
    | ("fun_name", Monitoring.Text _) :: _ -> seen := e :: !seen
    | _ -> ()
  in
  Monitoring.register_event_duration_secs_listener compile_listener;
  ignore (Api.jit (tree_fn (fun a -> L.add a a)) [ Tree.Leaf (scalar 1.) ]);
  Monitoring.unregister_event_duration_listener compile_listener;
  Alcotest.(check bool)
    "tracing is reported" true
    (List.mem "/jax/core/compile/jaxpr_trace_duration" !seen);
  if plugin_present () then
    Alcotest.(check bool)
      "backend compile is reported" true
      (List.mem "/jax/core/compile/backend_compile_duration" !seen)

let xspace_hex =
  "0a8f02080112092f686f73743a4350551a53080512046d61696e18e807223508021080897a18c08db701220408072003220908082a0568656c6c6f220b080911000000000000f83f2204080a18122204080b3808220a080310c09fab0318f40348c0a8a5041a1a08061203726177180522070804100018e8075a065365636f6e64220c080412080804120465765f63220c080312080803120465765f622215080212110802120465765f6122074576656e7420412a0c080b1208080b1204726566642a0d08081209080812056c6162656c2a100807120c08071208737465705f6e756d2a0d0809120908091205726174696f2a0d080a1209080a1205636f756e74320f08082a0b706c616e652d6c6576656c0a0d080212092f6465766963653a58"

let of_hex hex =
  String.init
    (String.length hex / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub hex (2 * i) 2)))

let profile_data_parsing () =
  let data = P.profile_data_from_serialized_xspace (of_hex xspace_hex) in
  Alcotest.(check (list string))
    "planes"
    [ "/host:CPU"; "/device:X" ]
    (List.map (fun p -> p.P.plane_name) data.P.planes);
  let plane =
    match P.find_plane_with_name data "/host:CPU" with
    | Some p -> p
    | None -> Alcotest.fail "plane"
  in
  Alcotest.(check bool)
    "absent plane" true
    (P.find_plane_with_name data "/nope" = None);
  Alcotest.(check bool)
    "plane stats" true
    (plane.P.plane_stats = [ ("label", P.Stat_text "plane-level") ]);
  Alcotest.(check (list string))
    "lines" [ "main"; "raw" ]
    (List.map (fun l -> l.P.line_name) plane.P.lines);
  let summary (e : P.profile_event) =
    (e.event_name, e.start_ns, e.end_ns, e.duration_ns)
  in
  let main = List.hd plane.P.lines and raw = List.nth plane.P.lines 1 in
  Alcotest.(
    check (list (pair string (pair (float 0.) (pair (float 0.) (float 0.))))))
    "events"
    [
      ("ev_a", (3000., (6000., 3000.)));
      ("ev_b", (8000., (8000., 0.)));
      ("ev_c", (5., (6., 1.)));
    ]
    (List.map
       (fun e ->
         let n, s, t, d = summary e in
         (n, (s, (t, d))))
       (main.P.events @ raw.P.events));
  Alcotest.(check bool)
    "event stats" true
    ((List.hd main.P.events).P.event_stats
    = [
        ("step_num", P.Stat_int 3L);
        ("label", P.Stat_text "hello");
        ("ratio", P.Stat_float 1.5);
        ("count", P.Stat_uint 18L);
        ("refd", P.Stat_text "label");
      ]);
  check_message "garbage" "Failed to parse XSpace from array" (fun () ->
      P.profile_data_from_serialized_xspace "\xff\xff");
  Alcotest.(check int)
    "empty" 0
    (List.length (P.profile_data_from_serialized_xspace "").P.planes);
  check_message "missing" "NOT_FOUND: /nonexistent; No such file or directory"
    (fun () -> P.profile_data_from_file "/nonexistent")

let temp_dir () =
  let path = Filename.temp_file "ojax_profile" "" in
  Sys.remove path;
  Sys.mkdir path 0o755;
  path

let rec find_pb dir =
  Array.fold_left
    (fun acc name ->
      let p = Filename.concat dir name in
      if Sys.is_directory p then find_pb p @ acc
      else if Filename.check_suffix name ".xplane.pb" then p :: acc
      else acc)
    [] (Sys.readdir dir)

let profiler_session () =
  check_message "stop none" "No profile started" P.stop_trace;
  let dir = temp_dir () in
  P.start_trace dir;
  check_message "double"
    "Profile has already been started. Only one profile may be run at a time."
    (fun () -> P.start_trace dir);
  let r =
    P.trace_annotation
      ~stats:[ ("k", P.Stat_int 2L) ]
      "myann"
      (fun () -> P.step_trace_annotation "stepx" ~step_num:3 (fun () -> 41 + 1))
  in
  Alcotest.(check int) "annotation result" 42 r;
  P.stop_trace ();
  Alcotest.(check int)
    "unrecorded outside a session" 7
    (P.trace_annotation "ignored" (fun () -> 7));
  let data =
    match find_pb dir with
    | [ path ] -> P.profile_data_from_file path
    | _ -> Alcotest.fail "expected one xplane file"
  in
  let plane =
    match P.find_plane_with_name data "/host:CPU" with
    | Some p -> p
    | None -> Alcotest.fail "plane"
  in
  let events = List.concat_map (fun l -> l.P.events) plane.P.lines in
  let by_name name =
    List.find (fun (e : P.profile_event) -> e.event_name = name) events
  in
  Alcotest.(check bool)
    "annotation stats" true
    ((by_name "myann").P.event_stats = [ ("k", P.Stat_int 2L) ]);
  Alcotest.(check bool)
    "step stats" true
    ((by_name "stepx").P.event_stats
    = [ ("_r", P.Stat_int 1L); ("step_num", P.Stat_int 3L) ]);
  let off = { (P.profile_options ()) with host_tracer_level = 0 } in
  let quiet_dir = temp_dir () in
  P.trace ~profiler_options:off quiet_dir (fun () ->
      P.trace_annotation "dropped" (fun () -> ()));
  let quiet =
    match find_pb quiet_dir with
    | [ path ] -> P.profile_data_from_file path
    | _ -> Alcotest.fail "expected one xplane file"
  in
  let quiet_events =
    List.concat_map
      (fun p -> List.concat_map (fun l -> l.P.events) p.P.lines)
      quiet.P.planes
  in
  Alcotest.(check int) "host tracer off" 0 (List.length quiet_events);
  let o = P.profile_options () in
  Alcotest.(check bool)
    "defaults" true
    (o.P.host_tracer_level = 2
    && o.P.python_tracer_level = 1
    && o.P.enable_hlo_proto && o.P.include_dataset_ops
    && (not o.P.raise_error_on_start_failure)
    && o.P.duration_ms = 0 && o.P.start_timestamp_ns = 0
    && o.P.repository_path = "" && o.P.session_id = ""
    && o.P.advanced_configuration = [])

let profiler_server () =
  check_message "stop none" "No active profiler server." P.stop_server;
  let server = P.start_server 0 in
  let port = P.server_port server in
  check_message "double" "Only one profiler server can be active at a time."
    (fun () -> P.start_server 0);
  let client = Unix.socket Unix.PF_INET Unix.SOCK_STREAM 0 in
  Unix.connect client (Unix.ADDR_INET (Unix.inet_addr_loopback, port));
  Unix.close client;
  check_message "same pid"
    "INVALID_ARGUMENT: Cannot register subprocess with the same pid as current \
     process." (fun () -> P.register_subprocess (Unix.getpid ()) port);
  let child = Unix.getpid () + 100000 in
  let unregister = P.register_subprocess child port in
  check_message "already registered"
    (Printf.sprintf "ALREADY_EXISTS: Subprocess %d is already registered." child)
    (fun () -> P.register_subprocess child port);
  unregister ();
  let unregister = P.register_subprocess child port in
  unregister ();
  P.stop_server ();
  check_message "stop twice" "No active profiler server." P.stop_server;
  check_message "refused"
    (Printf.sprintf
       "DEADLINE_EXCEEDED: Timeout while connecting to localhost:%d" port)
    (fun () -> P.register_subprocess child port);
  let again = P.start_server port in
  Alcotest.(check int) "port reusable" port (P.server_port again);
  P.stop_server ()

let stages_surface () =
  let f = function
    | [ Tree.Leaf x; Tree.Leaf y ] ->
        Tree.Dict [ ("a", Tree.Leaf (L.add x (L.reduce_sum y [| 0 |]))) ]
    | _ -> Alcotest.fail "arity"
  in
  let wrapped = Stages.jit ~name:"lam" f in
  let args = [ Tree.Leaf (scalar 1.); Tree.Leaf (f32 [| 2 |] [| 1.; 1. |]) ] in
  let traced = wrapped.Stages.trace args in
  Alcotest.(check string) "name" "lam" traced.Stages.fun_name;
  (match traced.Stages.args_info with
  | Tree.Tuple [ Tree.Leaf a; Tree.Leaf b ] ->
      Alcotest.(check (array int)) "shape0" [||] (Stages.arg_info_shape a);
      Alcotest.(check (array int)) "shape1" [| 2 |] (Stages.arg_info_shape b);
      Alcotest.(check dtype_t) "dtype" Dtype.F32 (Stages.arg_info_dtype b);
      Alcotest.(check bool) "donated" false b.Stages.donated
  | _ -> Alcotest.fail "args_info");
  (match traced.Stages.out_avals with
  | Tree.Dict [ ("a", Tree.Leaf a) ] ->
      Alcotest.(check (array int)) "out shape" [||] a.shape
  | _ -> Alcotest.fail "out_avals");
  Alcotest.(check bool)
    "jaxpr text" true
    (contains
       (Pretty_printer.closed_jaxpr_to_string traced.Stages.jaxpr)
       "reduce_sum");
  let scalar_of = function
    | Tree.Dict [ ("a", Tree.Leaf v) ] -> floats v
    | _ -> Alcotest.fail "dict"
  in
  check_floats "traced call" [| 3. |] (scalar_of (traced.Stages.call args));
  check_floats "wrapped call" [| 3. |] (scalar_of (wrapped.Stages.apply args));
  (match
     traced.Stages.call
       [ Tree.Leaf (scalar 1.); Tree.Leaf (f32 [| 3 |] [| 1.; 1.; 1. |]) ]
   with
  | _ -> Alcotest.fail "expected a type mismatch"
  | exception Invalid_argument _ -> ());
  let lowered = wrapped.Stages.wrapped_lower args in
  Alcotest.(check bool)
    "lowered text" true
    (contains (Stages.as_text lowered) "stablehlo.reduce");
  let flat = [ scalar 1.; f32 [| 2 |] [| 1.; 1. |] ] in
  check_floats "compiled" [| 3. |]
    (floats (List.hd (Stages.call (Stages.compile lowered) flat)));
  if plugin_present () then (
    let options =
      [ ("xla_cpu_enable_fast_math", Pjrt.Executable.Option_bool false) ]
    in
    check_floats "with options" [| 3. |]
      (floats
         (List.hd
            (Stages.call
               (Stages.compile ~compiler_options:options lowered)
               flat)));
    check_message "unknown option" "No such compile option: 'no_such'"
      (fun () ->
        Stages.compile
          ~compiler_options:[ ("no_such", Pjrt.Executable.Option_bool true) ]
          lowered);
    check_message "bad option value"
      "While setting option xla_cpu_enable_fast_math, 'x' is not a valid bool \
       value." (fun () ->
        Stages.compile
          ~compiler_options:
            [ ("xla_cpu_enable_fast_math", Pjrt.Executable.Option_string "x") ]
          lowered))

let params_of out =
  match out with
  | Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] -> (w, b)
  | _ -> Alcotest.fail "params"

let iota shape =
  let n = Array.fold_left ( * ) 1 shape in
  f32 shape (Array.init n (fun i -> float_of_int (i + 1)))

let stax_transpose () =
  let eps = 1e-5 in
  let k = key 0 in
  let check_layer name (layer : S.layer) input_shape ~out_shape ~w_shape
      ~b_shape ~w ~b ~out =
    let shape, params = layer.init k input_shape in
    Alcotest.(check (array int)) (name ^ " out shape") out_shape shape;
    let wv, bv = params_of params in
    Alcotest.(check (array int))
      (name ^ " w shape") w_shape (Core.get_aval wv).shape;
    Alcotest.(check (array int))
      (name ^ " b shape") b_shape (Core.get_aval bv).shape;
    check_floats ~eps (name ^ " w") w (floats wv);
    check_floats ~eps:1e-10 (name ^ " b") b (floats bv);
    check_floats ~eps (name ^ " apply") out
      (floats (layer.apply params (iota input_shape)))
  in
  check_layer "c1dt"
    (S.conv1d_transpose 2 [| 3 |] ~strides:[| 2 |] ~padding:L.Same)
    [| 1; 4; 3 |] ~out_shape:[| 1; 8; 2 |] ~w_shape:[| 3; 3; 2 |]
    ~b_shape:[| 1; 1; 2 |]
    ~w:
      [|
        0.3907737135887146;
        -0.3540942966938019;
        -0.2937445044517517;
        -0.45224085450172424;
        -0.34079301357269287;
        0.2320438176393509;
        0.2844080626964569;
        -0.3988063633441925;
        0.6162166595458984;
        -0.6798989772796631;
        -0.49410420656204224;
        0.052924975752830505;
        -0.44560790061950684;
        0.09473508596420288;
        0.6405243277549744;
        0.19931383430957794;
        0.23446401953697205;
        -0.18670788407325745;
      |]
    ~b:[| -2.442455752316164e-06; -2.0356806089694146e-06 |]
    ~out:
      [|
        1.5388303995132446;
        -0.06676293164491653;
        0.03452634811401367;
        -1.5998313426971436;
        1.607877492904663;
        -0.3071843981742859;
        1.254088044166565;
        -4.677172660827637;
        2.1647274494171143;
        -1.7080352306365967;
        2.473649263381958;
        -7.754514217376709;
        2.7215771675109863;
        -3.108886957168579;
        3.693211317062378;
        -10.831854820251465;
      |];
  check_layer "c1dt valid"
    (S.conv1d_transpose 1 [| 2 |])
    [| 1; 3; 1 |] ~out_shape:[| 1; 4; 1 |] ~w_shape:[| 2; 1; 1 |]
    ~b_shape:[| 1; 1; 1 |]
    ~w:[| 0.7567299008369446; -0.685700535774231 |]
    ~b:[| -2.442455752316164e-06 |]
    ~out:
      [|
        -0.6857029795646667;
        -0.6146736145019531;
        -0.5436442494392395;
        2.2701873779296875;
      |];
  check_layer "general nhwc"
    (S.general_conv_transpose ("NHWC", "HWIO", "NHWC") 2 [| 2; 2 |]
       ~strides:[| 2; 1 |])
    [| 1; 2; 3; 1 |] ~out_shape:[| 1; 4; 4; 2 |] ~w_shape:[| 2; 2; 1; 2 |]
    ~b_shape:[| 1; 1; 1; 2 |]
    ~w:
      [|
        0.43689826130867004;
        -0.39588943123817444;
        -0.3284163177013397;
        -0.5056205987930298;
        -0.38101816177368164;
        0.259432852268219;
        0.3179778456687927;
        -0.4458790421485901;
      |]
    ~b:[| -2.442455752316164e-06; -2.0356806089694146e-06 |]
    ~out:
      [|
        0.31797540187835693;
        -0.44588106870651245;
        0.254935085773468;
        -0.6323272585868835;
        0.1918947696685791;
        -0.8187735080718994;
        -1.143056869506836;
        0.7782965302467346;
        -0.3284187614917755;
        -0.5056226253509521;
        -0.2199368178844452;
        -1.407132625579834;
        -0.11145484447479248;
        -2.308642864227295;
        1.3106924295425415;
        -1.187670350074768;
        1.2719089984893799;
        -1.7835181951522827;
        0.06581419706344604;
        -1.191665768623352;
        0.0027738227508962154;
        -1.3781121969223022;
        -2.286111354827881;
        1.5565950870513916;
        -1.31366765499115;
        -2.022484540939331;
        0.10550898313522339;
        -4.1116623878479;
        0.21399110555648804;
        -5.013172626495361;
        2.621387243270874;
        -2.3753387928009033;
      |];
  check_layer "general channels first"
    (S.general_conv_transpose ("NCH", "OIH", "NCH") 2 [| 2 |])
    [| 1; 1; 3 |] ~out_shape:[| 1; 2; 4 |] ~w_shape:[| 2; 1; 2 |]
    ~b_shape:[| 1; 2; 1 |]
    ~w:
      [|
        0.6178674101829529;
        -0.5598721504211426;
        -0.46445077657699585;
        -0.7150554656982422;
      |]
    ~b:[| -2.442455752316164e-06; -2.0356806089694146e-06 |]
    ~out:
      [|
        -0.5598745942115784;
        -0.5018793344497681;
        -0.44388407468795776;
        1.8535997867584229;
        -0.7150574922561646;
        -1.8945637941360474;
        -3.0740702152252197;
        -1.3933542966842651;
      |]

let rec value_floats = function
  | S.Value v -> Array.to_list (floats v)
  | S.Values vs -> List.concat_map value_floats vs

let shape_list = function
  | S.Shapes ss ->
      List.map (function S.Shape s -> Array.to_list s | S.Shapes _ -> []) ss
  | S.Shape s -> [ Array.to_list s ]

let stax_fan () =
  let k = key 0 in
  let a = f32 [| 1; 2 |] [| 1.; 2. |] and b = f32 [| 1; 2 |] [| 3.; 4. |] in
  let out_shape, params = (S.fan_out 3).S.net_init k (S.Shape [| 1; 2 |]) in
  Alcotest.(check (list (list int)))
    "fan_out shapes"
    [ [ 1; 2 ]; [ 1; 2 ]; [ 1; 2 ] ]
    (shape_list out_shape);
  Alcotest.(check (list (float 0.)))
    "fan_out values" [ 1.; 2.; 1.; 2. ]
    (value_floats ((S.fan_out 2).S.net_apply params (S.Value a)));
  let shapes l = S.Shapes (List.map (fun s -> S.Shape s) l) in
  let sum_shape, _ =
    S.fan_in_sum.S.net_init k (shapes [ [| 1; 2 |]; [| 1; 2 |] ])
  in
  Alcotest.(check (list (list int)))
    "sum shape"
    [ [ 1; 2 ] ]
    (shape_list sum_shape);
  Alcotest.(check (list (float 0.)))
    "sum" [ 4.; 6. ]
    (value_floats
       (S.fan_in_sum.S.net_apply (Tree.Tuple [])
          (S.Values [ S.Value a; S.Value b ])));
  let concat_shape axis l =
    fst ((S.fan_in_concat ?axis ()).S.net_init k (shapes l))
  in
  Alcotest.(check (list (list int)))
    "concat default"
    [ [ 1; 5 ] ]
    (shape_list (concat_shape None [ [| 1; 2 |]; [| 1; 3 |] ]));
  Alcotest.(check (list (list int)))
    "concat 0"
    [ [ 5; 2 ] ]
    (shape_list (concat_shape (Some 0) [ [| 1; 2 |]; [| 4; 2 |] ]));
  Alcotest.(check (list (list int)))
    "concat -2"
    [ [ 5; 2 ] ]
    (shape_list (concat_shape (Some (-2)) [ [| 1; 2 |]; [| 4; 2 |] ]));
  let concat axis =
    (S.fan_in_concat ?axis ()).S.net_apply (Tree.Tuple [])
      (S.Values [ S.Value a; S.Value b ])
  in
  Alcotest.(check (list (float 0.)))
    "concat values" [ 1.; 2.; 3.; 4. ]
    (value_floats (concat None));
  (match concat (Some 0) with
  | S.Value v ->
      Alcotest.(check (array int))
        "axis 0 shape" [| 2; 2 |] (Core.get_aval v).shape
  | S.Values _ -> Alcotest.fail "concat output");
  let w =
    f32 [| 2; 2 |]
      [|
        0.6560734510421753;
        -0.0008375536999665201;
        0.8354882001876831;
        -0.7731967568397522;
      |]
  and bias = f32 [| 2 |] [| -0.0009129454265348613; 0.0013319568242877722 |] in
  let dense_params = Tree.Tuple [ Tree.Leaf w; Tree.Leaf bias ] in
  let net =
    S.serial_net
      [
        S.fan_out 2;
        S.parallel [ S.of_layer (S.dense 2); S.of_layer S.identity ];
        S.fan_in_sum;
      ]
  in
  let shape, tree = net.S.net_init k (S.Shape [| 1; 2 |]) in
  Alcotest.(check (list (list int))) "net shape" [ [ 1; 2 ] ] (shape_list shape);
  (match tree with
  | Tree.Tuple
      [
        Tree.Tuple [];
        Tree.Tuple [ Tree.Tuple [ Tree.Leaf _; Tree.Leaf _ ]; Tree.Tuple [] ];
        Tree.Tuple [];
      ] ->
      ()
  | _ -> Alcotest.fail "net parameter structure");
  let params =
    Tree.Tuple
      [
        Tree.Tuple []; Tree.Tuple [ dense_params; Tree.Tuple [] ]; Tree.Tuple [];
      ]
  in
  Alcotest.(check (list (float 1e-5)))
    "residual block" [ 3.3261368; 0.45410085 ]
    (value_floats (net.S.net_apply params (S.Value a)))

let schedules () =
  Alcotest.(check (float 0.)) "scalar" 0.5 (O.make_schedule (`Scalar 0.5) 3);
  Alcotest.(check (float 0.))
    "schedule" 6.0
    (O.make_schedule (`Schedule (fun i -> float_of_int i *. 2.)) 3);
  Alcotest.(check (float 0.))
    "rank 0" 2.0
    (O.make_schedule (`Value (scalar 2.)) 1);
  match
    let (_ : O.schedule) =
      O.make_schedule (`Value (f32 [| 2 |] [| 1.; 2. |]))
    in
    ()
  with
  | () -> Alcotest.fail "expected a type error"
  | exception Invalid_argument _ -> ()

let optimizer_state () =
  let opt = O.adam (O.make_schedule (`Scalar 0.1)) in
  let params =
    Tree.Dict
      [
        ("a", Tree.Leaf (f32 [| 2 |] [| 1.; 2. |]));
        ("b", Tree.Tuple [ Tree.Leaf (scalar 3.) ]);
      ]
  in
  let grads = Tree.tree_map (fun v -> L.mul v (L.full_like v 0.5)) params in
  let packed = O.optimizer_state_of (opt.O.init params) in
  Alcotest.(check int) "per leaf states" 2 (List.length packed.O.packed_state);
  Alcotest.(check int)
    "leaves" 6
    (List.length (O.optimizer_state_leaves packed));
  Alcotest.(check bool)
    "tree def" true
    (packed.O.tree_def = Tree.tree_structure params);
  Alcotest.(check int) "subtree defs" 2 (List.length packed.O.subtree_defs);
  let stepped = opt.O.update 0 grads (opt.O.init params) in
  let state = O.optimizer_state_of stepped in
  let marked = O.unpack_optimizer_state state in
  Alcotest.(check bool)
    "marked structure" true
    (Tree.tree_structure marked = Tree.tree_structure params);
  (match Tree.tree_leaves marked with
  | [
   O.Join_point (Tree.Tuple [ Tree.Leaf x; Tree.Leaf m; Tree.Leaf v ]);
   O.Join_point _;
  ] ->
      check_floats "x" [| 0.90000063; 1.9000007 |] (floats x);
      check_floats "m" [| 0.05; 0.1 |] (floats m);
      check_floats ~eps:1e-9 "v" [| 0.00025; 0.001 |] (floats v)
  | _ -> Alcotest.fail "join points");
  let round = O.pack_optimizer_state marked in
  Alcotest.(check bool)
    "round trip defs" true
    (round.O.tree_def = state.O.tree_def
    && round.O.subtree_defs = state.O.subtree_defs);
  let continued = opt.O.update 1 grads (O.leaf_state_of round) in
  match opt.O.get_params continued with
  | Tree.Dict [ ("a", Tree.Leaf a); ("b", Tree.Tuple [ Tree.Leaf b ]) ] ->
      check_floats ~eps:1e-5 "a"
        [| 0.8000016212463379; 1.8000017404556274 |]
        (floats a);
      check_floats ~eps:1e-5 "b" [| 2.800001621246338 |] (floats b)
  | _ -> Alcotest.fail "params"

let () =
  Alcotest.run "runtime_flags"
    [
      ( "config",
        [
          Alcotest.test_case "setters restore" `Quick setters_restore;
          Alcotest.test_case "setters validate" `Quick setters_validate;
          Alcotest.test_case "flags without an effect" `Quick no_effect_flags;
          Alcotest.test_case "tracer leaks" `Quick tracer_leaks;
          Alcotest.test_case "enable_checks" `Quick enable_checks_effect;
          Alcotest.test_case "softmax jvp" `Quick softmax_jvp;
          Alcotest.test_case "dtype promotion" `Quick dtype_promotion;
          Alcotest.test_case "f16 reductions" `Quick f16_reductions;
          Alcotest.test_case "matmul precision" `Quick matmul_precision;
          Alcotest.test_case "cache misses" `Quick cache_misses;
          Alcotest.test_case "transfer guards" `Quick transfer_guards;
          Alcotest.test_case "gc guard" `Quick gc_guard;
        ] );
      ( "prng",
        [
          Alcotest.test_case "key reuse eager" `Quick key_reuse_eager;
          Alcotest.test_case "key reuse samplers" `Quick key_reuse_samplers;
          Alcotest.test_case "key reuse jit" `Quick key_reuse_jit;
          Alcotest.test_case "legacy keys" `Quick legacy_keys;
          Alcotest.test_case "custom prng" `Quick custom_prng;
          Alcotest.test_case "threefry original" `Quick threefry_original;
        ] );
      ( "errors",
        [
          Alcotest.test_case "error types" `Quick error_types;
          Alcotest.test_case "jaxpr errors" `Quick jaxpr_errors;
        ] );
      ( "monitoring",
        [ Alcotest.test_case "listeners" `Quick monitoring_listeners ] );
      ( "profiler",
        [
          Alcotest.test_case "profile data" `Quick profile_data_parsing;
          Alcotest.test_case "session" `Quick profiler_session;
          Alcotest.test_case "server" `Quick profiler_server;
        ] );
      ("stages", [ Alcotest.test_case "surface" `Quick stages_surface ]);
      ( "example_libraries",
        [
          Alcotest.test_case "stax transpose" `Quick stax_transpose;
          Alcotest.test_case "stax fan" `Quick stax_fan;
          Alcotest.test_case "schedules" `Quick schedules;
          Alcotest.test_case "optimizer state" `Quick optimizer_state;
        ] );
    ]
