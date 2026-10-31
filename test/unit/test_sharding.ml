open Ojax
open Ojax.Types
module T = Tree_util
module S = Sharding
module P = Parallel

let child_mode = Array.length Sys.argv > 1 && Sys.argv.(1) = "--report-devices"

let () =
  if child_mode then begin
    print_string (String.concat "," (Api.devices ()));
    Printf.printf "|%d" (Api.device_count ());
    exit 0
  end

let () = Config.set Config.num_cpu_devices 4
let () = Lax.install ()
let leaf = function T.Leaf v -> v | _ -> Alcotest.fail "expected a leaf"
let f32 shape data = Concrete (Ndarray.of_floats Dtype.F32 shape data)
let arange n = f32 [| n |] (Array.init n float_of_int)
let matrix r c = f32 [| r; c |] (Array.init (r * c) float_of_int)
let host v = S.host_array v

let flat v =
  let nd = host v in
  let n = Array.fold_left ( * ) 1 (Ndarray.shape nd) in
  let shape = Ndarray.shape nd in
  Array.init n (fun i ->
      let idx = Array.make (Array.length shape) 0 in
      let rem = ref i in
      for d = Array.length shape - 1 downto 0 do
        idx.(d) <- !rem mod shape.(d);
        rem := !rem / shape.(d)
      done;
      Ndarray.get_f nd idx)

let values name expected v =
  Alcotest.(check (array (float 1e-5))) name expected (flat v)

let shape_of v = (Core.get_aval v).shape

let spec_of v =
  match S.array_sharding v with
  | S.Named_sharding (_, spec) -> S.spec_to_string spec
  | other -> S.sharding_to_string other

let check_spec name expected v =
  Alcotest.(check string) name expected (spec_of v)

let is_invalid name f =
  match f () with
  | _ -> Alcotest.failf "%s: expected Invalid_argument" name
  | exception Invalid_argument _ -> ()

let contains text piece =
  let n = String.length piece in
  let rec go i =
    i + n <= String.length text && (String.sub text i n = piece || go (i + 1))
  in
  go 0

let raises_with name piece f =
  match f () with
  | _ -> Alcotest.failf "%s: expected Invalid_argument" name
  | exception Invalid_argument msg ->
      if not (contains msg piece) then
        Alcotest.failf "%s: message %S lacks %S" name msg piece

let mesh4 = lazy (S.make_mesh [ 4 ] [ "i" ])
let mesh22 = lazy (S.make_mesh [ 2; 2 ] [ "a"; "b" ])
let auto4 = lazy (S.make_mesh ~axis_types:[ S.Auto ] [ 4 ] [ "i" ])

let auto22 =
  lazy (S.make_mesh ~axis_types:[ S.Auto; S.Auto ] [ 2; 2 ] [ "a"; "b" ])

let on_i x =
  leaf
    (S.device_put (T.Leaf x)
       (S.named_sharding (Lazy.force mesh4) [ S.Named "i" ]))

let on_ab x =
  leaf
    (S.device_put (T.Leaf x)
       (S.named_sharding (Lazy.force mesh22) [ S.Named "a"; S.Named "b" ]))

let unary f = function
  | [ T.Leaf a ] -> T.Leaf (f a)
  | _ -> Alcotest.fail "expected one leaf"

let binary f = function
  | [ T.Leaf a; T.Leaf b ] -> T.Leaf (f a b)
  | _ -> Alcotest.fail "expected two leaves"

let shard ?(mesh = mesh4) ?axis_names ~in_spec ~out_spec f x =
  leaf
    (S.shard_map ~mesh:(Lazy.force mesh) ?axis_names ~in_specs:[ in_spec ]
       ~out_specs:[ out_spec ] (unary f) [ T.Leaf x ])

let i_spec = [ S.Named "i" ]

let discovery () =
  Alcotest.(check int) "device_count" 4 (Api.device_count ());
  Alcotest.(check int) "local_device_count" 4 (Api.local_device_count ());
  Alcotest.(check (list string))
    "devices"
    [ "cpu:0"; "cpu:1"; "cpu:2"; "cpu:3" ]
    (Api.devices ());
  let ids =
    Array.to_list (Array.map (fun (d : S.device) -> d.id) (S.local_devices ()))
  in
  Alcotest.(check (list int)) "ids" [ 0; 1; 2; 3 ] ids

let child_report env =
  let tmp = Filename.temp_file "ojax_devices" ".txt" in
  let command =
    Printf.sprintf "%s %s --report-devices > %s 2>/dev/null" env
      (Filename.quote Sys.executable_name)
      (Filename.quote tmp)
  in
  let code = Sys.command command in
  let ic = open_in tmp in
  let text = really_input_string ic (in_channel_length ic) in
  close_in ic;
  Sys.remove tmp;
  Alcotest.(check int) "child exit" 0 code;
  text

let environment () =
  Alcotest.(check string)
    "no flag keeps one device" "cpu:0|1"
    (child_report "env -u XLA_FLAGS");
  Alcotest.(check string)
    "flag gives three" "cpu:0,cpu:1,cpu:2|3"
    (child_report "XLA_FLAGS=--xla_force_host_platform_device_count=3");
  Alcotest.(check string)
    "flag gives six" "cpu:0,cpu:1,cpu:2,cpu:3,cpu:4,cpu:5|6"
    (child_report "XLA_FLAGS=--xla_force_host_platform_device_count=6");
  Alcotest.(check string)
    "interpreter has one" "cpu:0|1"
    (child_report
       "OJAX_BACKEND=interpreter \
        XLA_FLAGS=--xla_force_host_platform_device_count=6")

let mesh_objects () =
  let m = Lazy.force mesh4 in
  Alcotest.(check string)
    "mesh repr" "Mesh('i': 4, axis_types=(Explicit,))" (S.mesh_to_string m);
  Alcotest.(check int) "size" 4 (S.mesh_size m);
  raises_with "too many devices" "must be >= the product" (fun () ->
      S.make_mesh [ 8 ] [ "i" ]);
  raises_with "names" "Mesh requires the ndim" (fun () ->
      S.make_mesh [ 2; 2 ] [ "i" ]);
  let sub = S.make_mesh [ 2 ] [ "x" ] in
  Alcotest.(check (list int))
    "first devices" [ 0; 1 ]
    (Array.to_list (Array.map (fun (d : S.device) -> d.id) sub.devices));
  Alcotest.(check string) "empty" "Mesh()" (S.mesh_to_string S.empty_mesh);
  Alcotest.(check string)
    "empty abstract" "AbstractMesh((), axis_types=())"
    (S.abstract_mesh_to_string S.empty_abstract_mesh)

let specs () =
  Alcotest.(check string) "single" "P('a',)" (S.spec_to_string [ S.Named "a" ]);
  Alcotest.(check string) "empty" "P()" (S.spec_to_string []);
  Alcotest.(check string)
    "none" "P('a', None)"
    (S.spec_to_string [ S.Named "a"; S.Unconstrained ]);
  Alcotest.(check string)
    "tuple" "P(('a', 'b'), 'c')"
    (S.spec_to_string [ S.Axes [ "a"; "b" ]; S.Named "c" ]);
  let m = Lazy.force mesh22 in
  raises_with "duplicate" "has duplicate entries for `a`" (fun () ->
      S.named_sharding m [ S.Named "a"; S.Named "a" ]);
  raises_with "duplicate tuple" "has duplicate entries for `a`" (fun () ->
      S.named_sharding m [ S.Axes [ "a"; "b" ]; S.Named "a" ]);
  raises_with "unknown" "Resource axis: zz of P('zz',) is not found" (fun () ->
      S.named_sharding (Lazy.force mesh4) [ S.Named "zz" ])

let named_properties () =
  let s = S.named_sharding (Lazy.force mesh4) i_spec in
  Alcotest.(check string)
    "repr"
    "NamedSharding(mesh=Mesh('i': 4, axis_types=(Explicit,)), spec=P('i',), \
     memory_kind=device)"
    (S.sharding_to_string s);
  Alcotest.(check int) "num_devices" 4 (S.num_devices s);
  Alcotest.(check (array int))
    "shard shape" [| 2; 2 |]
    (S.shard_shape s [| 8; 2 |]);
  Alcotest.(check bool) "addressable" true (S.is_fully_addressable s);
  Alcotest.(check bool) "replicated" false (S.is_fully_replicated s);
  raises_with "indivisible" "is partitioned 4 times" (fun () ->
      S.shard_shape s [| 7 |]);
  let m = Lazy.force mesh22 in
  Alcotest.(check bool)
    "P() replicated" true
    (S.is_fully_replicated (S.named_sharding m []));
  Alcotest.(check bool)
    "P(a) replicated" false
    (S.is_fully_replicated (S.named_sharding m [ S.Named "a" ]))

let index_list s shape =
  List.map
    (fun ((d : S.device), index) -> (d.id, Array.to_list index))
    (S.devices_indices_map s shape)

let indices_table = Alcotest.(list (pair int (list (pair int int))))

let indices_map () =
  let m = Lazy.force mesh22 in
  Alcotest.check indices_table "b, a"
    [
      (0, [ (0, 2); (0, 3) ]);
      (1, [ (2, 4); (0, 3) ]);
      (2, [ (0, 2); (3, 6) ]);
      (3, [ (2, 4); (3, 6) ]);
    ]
    (index_list (S.named_sharding m [ S.Named "b"; S.Named "a" ]) [| 4; 6 |]);
  Alcotest.check indices_table "tuple"
    [ (0, [ (0, 2) ]); (1, [ (2, 4) ]); (2, [ (4, 6) ]); (3, [ (6, 8) ]) ]
    (index_list (S.named_sharding m [ S.Axes [ "a"; "b" ] ]) [| 8 |]);
  Alcotest.check indices_table "partial"
    [ (0, [ (0, 2) ]); (1, [ (0, 2) ]); (2, [ (2, 4) ]); (3, [ (2, 4) ]) ]
    (index_list (S.named_sharding m [ S.Named "a" ]) [| 4 |])

let placement () =
  let x = on_i (arange 8) in
  values "global" (Array.init 8 float_of_int) x;
  Alcotest.(check (array int)) "shape" [| 8 |] (shape_of x);
  let shards = S.addressable_shards x in
  Alcotest.(check (list int))
    "shard devices" [ 0; 1; 2; 3 ]
    (List.map (fun (s : S.shard) -> s.device.id) shards);
  List.iteri
    (fun k (s : S.shard) ->
      let lo = float_of_int (2 * k) in
      values "shard data" [| lo; lo +. 1.0 |] s.data;
      Alcotest.(check (array (pair int int)))
        "shard index"
        [| (2 * k, (2 * k) + 2) |]
        s.index;
      Alcotest.(check int) "replica" 0 s.replica_id)
    shards;
  check_spec "spec" "P('i',)" x;
  let partial =
    leaf
      (S.device_put
         (T.Leaf (arange 8))
         (S.named_sharding (Lazy.force mesh22) [ S.Named "a" ]))
  in
  Alcotest.(check (list (pair int int)))
    "partial replica ids"
    [ (0, 0); (1, 1); (2, 0); (3, 1) ]
    (List.map
       (fun (s : S.shard) -> (s.device.id, s.replica_id))
       (S.addressable_shards partial));
  values "partial global" (Array.init 8 float_of_int) partial

let placement_dtypes () =
  let sharding = S.named_sharding (Lazy.force mesh4) i_spec in
  let ints =
    Concrete
      (Ndarray.of_int64s Dtype.I64 [| 8 |]
         (Array.init 8 (fun i -> Int64.add 0x1_0000_0000L (Int64.of_int i))))
  in
  let placed = leaf (S.device_put (T.Leaf ints) sharding) in
  Alcotest.(check int64)
    "wide integer"
    (Int64.add 0x1_0000_0000L 5L)
    (Ndarray.get_i64 (host placed) [| 5 |]);
  let bools =
    Concrete
      (Ndarray.of_floats Dtype.Bool [| 8 |]
         (Array.init 8 (fun i -> if i > 3 then 1.0 else 0.0)))
  in
  let flipped =
    leaf
      (Api.jit
         (unary (fun a -> Core.bind1 Not [ a ]))
         [ S.device_put (T.Leaf bools) sharding ])
  in
  values "bool" [| 1.; 1.; 1.; 1.; 0.; 0.; 0.; 0. |] flipped;
  check_spec "bool spec" "P('i',)" flipped;
  let complex_in =
    Concrete
      (Ndarray.of_complex Dtype.Complex64 [| 8 |]
         (Array.init 8 (fun i -> { Complex.re = float_of_int i; im = 0.0 })))
  in
  let factor =
    Concrete
      (Ndarray.of_complex Dtype.Complex64 [||]
         [| { Complex.re = 1.0; im = 1.0 } |])
  in
  let scaled =
    leaf
      (Api.jit
         (unary (fun a -> Lax_api.mul a factor))
         [ S.device_put (T.Leaf complex_in) sharding ])
  in
  let z = Ndarray.get_c (host scaled) [| 3 |] in
  Alcotest.(check (float 1e-6)) "complex re" 3.0 z.Complex.re;
  Alcotest.(check (float 1e-6)) "complex im" 3.0 z.Complex.im

let single_device_sharding () =
  let d = (S.local_devices ()).(2) in
  let s = S.make_single_device_sharding d in
  Alcotest.(check string)
    "repr" "SingleDeviceSharding(device=CpuDevice(id=2), memory_kind=device)"
    (S.sharding_to_string s);
  Alcotest.(check int) "num" 1 (S.num_devices s);
  Alcotest.(check bool) "replicated" true (S.is_fully_replicated s);
  let y = leaf (S.device_put (T.Leaf (arange 3)) s) in
  Alcotest.(check string)
    "array sharding"
    "SingleDeviceSharding(device=CpuDevice(id=2), memory_kind=device)"
    (S.sharding_to_string (S.array_sharding y));
  let r =
    leaf
      (Api.jit
         (unary (fun a -> Lax_api.add a (f32 [||] [| 1.0 |])))
         [ T.Leaf y ])
  in
  values "jit follows" [| 1.; 2.; 3. |] r;
  Alcotest.(check (list int))
    "still device 2" [ 2 ]
    (List.map (fun (s : S.shard) -> s.device.id) (S.addressable_shards r));
  let eager = Lax_api.add y (f32 [||] [| 1.0 |]) in
  values "eager follows" [| 1.; 2.; 3. |] eager;
  Alcotest.(check (list int))
    "eager device" [ 2 ]
    (List.map (fun (s : S.shard) -> s.device.id) (S.addressable_shards eager));
  let other =
    leaf
      (S.device_put
         (T.Leaf (arange 3))
         (S.make_single_device_sharding (S.local_devices ()).(1)))
  in
  raises_with "cross device" "Received incompatible devices" (fun () ->
      Api.jit (binary Lax_api.add) [ T.Leaf y; T.Leaf other ])

let jit_propagates () =
  let x = on_i (arange 8) in
  (match
     Api.jit
       (function
         | [ T.Leaf a ] ->
             T.Tuple
               [
                 T.Leaf (Lax_api.mul a (f32 [||] [| 2.0 |]));
                 T.Leaf (Numpy.Reductions.sum a);
               ]
         | _ -> Alcotest.fail "arity")
       [ T.Leaf x ]
   with
  | T.Tuple [ T.Leaf a; T.Leaf b ] ->
      values "doubled" (Array.init 8 (fun i -> float_of_int (2 * i))) a;
      check_spec "elementwise keeps" "P('i',)" a;
      values "sum" [| 28.0 |] b;
      check_spec "sum replicated" "P()" b
  | _ -> Alcotest.fail "tree");
  let shards =
    S.addressable_shards
      (leaf (Api.jit (unary (fun a -> Lax_api.mul a a)) [ T.Leaf x ]))
  in
  values "device 3 holds its piece" [| 36.0; 49.0 |] (List.nth shards 3).data;
  let xa = on_ab (matrix 4 4) in
  let t =
    leaf
      (Api.jit (unary (fun a -> Lax_api.transpose a [| 1; 0 |])) [ T.Leaf xa ])
  in
  check_spec "transpose swaps" "P('b', 'a')" t;
  values "transpose"
    [| 0.; 4.; 8.; 12.; 1.; 5.; 9.; 13.; 2.; 6.; 10.; 14.; 3.; 7.; 11.; 15. |]
    t;
  values "T device 1" [| 2.; 6.; 3.; 7. |]
    (List.nth (S.addressable_shards t) 1).data;
  values "T device 2" [| 8.; 12.; 9.; 13. |]
    (List.nth (S.addressable_shards t) 2).data;
  let summed =
    leaf
      (Api.jit
         (unary (fun a -> Numpy.Reductions.sum ~axis:[| 0 |] a))
         [ T.Leaf xa ])
  in
  check_spec "sum axis 0" "P('b',)" summed;
  values "sum axis 0" [| 24.; 28.; 32.; 36. |] summed;
  let mixed =
    leaf
      (Api.jit (binary Lax_api.add)
         [ T.Leaf x; T.Leaf (f32 [| 8 |] (Array.make 8 1.0)) ])
  in
  check_spec "mixed with host" "P('i',)" mixed;
  values "mixed" (Array.init 8 (fun i -> float_of_int (i + 1))) mixed

let jit_with_shardings () =
  let m = Lazy.force mesh4 in
  let sharded = S.named_sharding m i_spec in
  let replicated = S.named_sharding m [] in
  let double = unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])) in
  let expected = Array.init 8 (fun i -> float_of_int (2 * i)) in
  let out =
    leaf (S.jit ~in_shardings:[ sharded ] double [ T.Leaf (arange 8) ])
  in
  values "in_shardings" expected out;
  check_spec "in spec" "P('i',)" out;
  let out2 =
    leaf
      (S.jit ~out_shardings:[ replicated ] double [ T.Leaf (on_i (arange 8)) ])
  in
  values "out_shardings" expected out2;
  check_spec "out spec" "P()" out2;
  values "replicated shard on device 2" expected
    (List.nth (S.addressable_shards out2) 2).data;
  raises_with "mismatch" "does not match the sharding on the respective arg"
    (fun () ->
      S.jit ~in_shardings:[ replicated ] double [ T.Leaf (on_i (arange 8)) ]);
  let m22 = Lazy.force auto22 in
  let by_col = S.named_sharding m22 [ S.Unconstrained; S.Named "b" ] in
  let cols =
    leaf (S.jit ~in_shardings:[ by_col ] double [ T.Leaf (matrix 4 4) ])
  in
  check_spec "column spec" "P(None, 'b')" cols;
  Alcotest.(check (array (pair int int)))
    "column index"
    [| (0, 4); (0, 2) |]
    (List.hd (S.addressable_shards cols)).index;
  values "columns" (Array.init 16 (fun i -> float_of_int (2 * i))) cols

let eager_ops () =
  let x = on_i (arange 8) in
  let doubled = Lax_api.mul x (f32 [||] [| 2.0 |]) in
  check_spec "mul" "P('i',)" doubled;
  values "shard 1" [| 4.; 6. |] (List.nth (S.addressable_shards doubled) 1).data;
  let total = Numpy.Reductions.sum x in
  check_spec "sum" "P()" total;
  values "sum" [| 28.0 |] total;
  let s = Lax_api.sin x in
  check_spec "sin" "P('i',)" s

let reshard () =
  let x = on_i (arange 8) in
  let m = Lazy.force mesh4 in
  let r = leaf (S.reshard (T.Leaf x) (S.named_sharding m [])) in
  check_spec "replicated" "P()" r;
  values "shard 1 whole"
    (Array.init 8 float_of_int)
    (List.nth (S.addressable_shards r) 1).data;
  let xa = on_ab (matrix 4 4) in
  let swapped =
    leaf
      (S.reshard (T.Leaf xa)
         (S.named_sharding (Lazy.force mesh22) [ S.Named "b"; S.Named "a" ]))
  in
  check_spec "swap" "P('b', 'a')" swapped;
  values "device 1" [| 8.; 9.; 12.; 13. |]
    (List.nth (S.addressable_shards swapped) 1).data;
  values "device 2" [| 2.; 3.; 6.; 7. |]
    (List.nth (S.addressable_shards swapped) 2).data;
  let traced =
    leaf
      (Api.jit
         (unary (fun a -> leaf (S.reshard (T.Leaf a) (S.named_sharding m []))))
         [ T.Leaf x ])
  in
  check_spec "under jit" "P()" traced;
  values "under jit" (Array.init 8 float_of_int) traced;
  let auto = Lazy.force auto4 in
  let constrained =
    leaf
      (Api.jit
         (unary (fun a ->
              leaf
                (S.reshard
                   (T.Leaf (Lax_api.mul a (f32 [||] [| 2.0 |])))
                   (S.named_sharding auto i_spec))))
         [ T.Leaf (arange 8) ])
  in
  check_spec "constraint from host" "P('i',)" constrained;
  values "constraint" (Array.init 8 (fun i -> float_of_int (2 * i))) constrained;
  raises_with "spec without mesh" "not under a mesh context" (fun () ->
      S.reshard_spec (T.Leaf x) []);
  S.with_mesh m (fun () ->
      check_spec "spec in mesh" "P()" (leaf (S.reshard_spec (T.Leaf x) [])))

let shard_map_collectives () =
  let x = on_i (arange 8) in
  let run name f out_spec expected =
    let y = shard ~in_spec:i_spec ~out_spec f x in
    values name expected y
  in
  run "psum" (fun a -> P.psum a ~axis_name:"i") [] [| 12.; 16. |];
  run "pmean" (fun a -> P.pmean a ~axis_name:"i") [] [| 3.; 4. |];
  run "pmax" (fun a -> P.pmax a ~axis_name:"i") [] [| 6.; 7. |];
  run "pmin" (fun a -> P.pmin a ~axis_name:"i") [] [| 0.; 1. |];
  run "all_gather"
    (fun a -> P.all_gather a ~axis_name:"i")
    []
    (Array.init 8 float_of_int);
  run "all_gather axis 1"
    (fun a -> P.all_gather ~axis:1 a ~axis_name:"i")
    []
    [| 0.; 2.; 4.; 6.; 1.; 3.; 5.; 7. |];
  run "axis_index"
    (fun a ->
      Lax_api.add a
        (Core.bind1 (Convert_element_type Dtype.F32)
           [ P.axis_index ~axis_name:"i" ]))
    i_spec
    [| 0.; 1.; 3.; 4.; 6.; 7.; 9.; 10. |];
  let ring = [ (0, 1); (1, 2); (2, 3); (3, 0) ] in
  run "ppermute"
    (fun a -> P.ppermute a ~axis_name:"i" ~perm:ring)
    i_spec
    [| 6.; 7.; 0.; 1.; 2.; 3.; 4.; 5. |];
  run "ppermute partial"
    (fun a -> P.ppermute a ~axis_name:"i" ~perm:[ (0, 1) ])
    i_spec
    [| 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0. |];
  run "pshuffle"
    (fun a -> P.pshuffle a ~axis_name:"i" ~perm:[ 1; 2; 3; 0 ])
    i_spec
    [| 2.; 3.; 4.; 5.; 6.; 7.; 0.; 1. |];
  run "pbroadcast"
    (fun a -> P.pbroadcast ~source:2 a ~axis_name:"i")
    [] [| 4.; 5. |];
  let long = on_i (arange 16) in
  values "tiled all_to_all"
    [| 0.; 4.; 8.; 12.; 1.; 5.; 9.; 13.; 2.; 6.; 10.; 14.; 3.; 7.; 11.; 15. |]
    (shard ~in_spec:i_spec ~out_spec:i_spec
       (fun a ->
         P.all_to_all ~tiled:true a ~axis_name:"i" ~split_axis:0 ~concat_axis:0)
       long)

let compiled_for_devices () =
  let before, fallbacks_before = Backend.stats () in
  let x = on_i (arange 8) in
  let y =
    shard ~in_spec:i_spec ~out_spec:i_spec
      (fun a -> Lax_api.mul (P.psum a ~axis_name:"i") a)
      x
  in
  let after, fallbacks_after = Backend.stats () in
  (match y with
  | Device (Pjrt.Buffer.Placed p) ->
      Alcotest.(check (list int))
        "buffers live on their devices" [ 0; 1; 2; 3 ]
        (Array.to_list
           (Array.map
              (fun (s : Pjrt.Buffer.shard) -> Pjrt.Buffer.raw_device s.data)
              p.shards))
  | _ -> Alcotest.fail "expected device buffers");
  Alcotest.(check bool) "compiled on xla" true (after > before);
  Alcotest.(check int)
    "no host fallback"
    (List.length fallbacks_before)
    (List.length fallbacks_after)

let shard_map_two_axes () =
  let xa = on_ab (matrix 4 4) in
  let mesh = mesh22 in
  let in_spec = [ S.Named "a"; S.Named "b" ] in
  let index_f axis =
    Core.bind1 (Convert_element_type Dtype.F32) [ P.axis_index ~axis_name:axis ]
  in
  let combo =
    shard ~mesh ~in_spec ~out_spec:in_spec
      (fun a -> Lax_api.add (P.psum a ~axis_name:"a") (index_f "b"))
      xa
  in
  values "psum a + index b"
    [|
      8.;
      10.;
      13.;
      15.;
      16.;
      18.;
      21.;
      23.;
      8.;
      10.;
      13.;
      15.;
      16.;
      18.;
      21.;
      23.;
    |]
    combo;
  check_spec "combo spec" "P('a', 'b')" combo;
  values "psum b"
    [| 2.; 4.; 10.; 12.; 18.; 20.; 26.; 28. |]
    (shard ~mesh ~in_spec ~out_spec:[ S.Named "a" ]
       (fun a -> P.psum a ~axis_name:"b")
       xa);
  values "psum both" [| 20.; 24.; 36.; 40. |]
    (shard ~mesh ~in_spec ~out_spec:[]
       (fun a -> P.psum (P.psum a ~axis_name:"a") ~axis_name:"b")
       xa);
  let mesh = auto22 in
  values "tuple spec"
    (Array.init 8 (fun i -> float_of_int (2 * i)))
    (shard ~mesh
       ~in_spec:[ S.Axes [ "a"; "b" ] ]
       ~out_spec:[ S.Axes [ "a"; "b" ] ]
       (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |]))
       (arange 8));
  let reversed = [ S.Axes [ "b"; "a" ] ] in
  values "reversed tuple spec"
    [| 0.; 1.; 4.; 6.; 4.; 5.; 12.; 14. |]
    (shard ~mesh ~in_spec:reversed ~out_spec:reversed
       (fun a -> Lax_api.mul a (Lax_api.add (index_f "a") (f32 [||] [| 1.0 |])))
       (arange 8))

let shard_map_args () =
  let x = on_i (arange 8) in
  let m = Lazy.force mesh4 in
  values "static argument"
    (Array.init 8 (fun i -> float_of_int (3 * i)))
    (leaf
       (S.shard_map ~mesh:m ~in_specs:[ i_spec; [] ] ~out_specs:[ i_spec ]
          (binary Lax_api.mul)
          [ T.Leaf x; T.Leaf (f32 [||] [| 3.0 |]) ]));
  let tree =
    S.shard_map ~mesh:m ~in_specs:[ i_spec ] ~out_specs:[ i_spec ]
      (function
        | [ T.Leaf a ] ->
            T.Dict
              [
                ("p", T.Leaf (Lax_api.mul a (f32 [||] [| 2.0 |])));
                ( "q",
                  T.Leaf (Lax_api.expand_dims (Numpy.Reductions.sum a) [| 0 |])
                );
              ]
        | _ -> Alcotest.fail "arity")
      [ T.Leaf x ]
  in
  (match tree with
  | T.Dict [ ("p", T.Leaf p); ("q", T.Leaf q) ] ->
      values "tree p" (Array.init 8 (fun i -> float_of_int (2 * i))) p;
      values "tree q" [| 1.; 5.; 9.; 13. |] q;
      check_spec "tree q spec" "P('i',)" q
  | _ -> Alcotest.fail "tree shape");
  let inferred =
    S.shard_map ~mesh:m ~out_specs:[ i_spec ]
      (unary (fun a -> Lax_api.mul a a))
      [ T.Leaf x ]
  in
  values "inferred in_specs"
    (Array.init 8 (fun i -> float_of_int (i * i)))
    (leaf inferred);
  raises_with "explicit input must match"
    "does not match the specs of the input" (fun () ->
      shard ~in_spec:i_spec ~out_spec:i_spec Fun.id (arange 8));
  let am = auto4 in
  raises_with "divisibility" "is partitioned 4 times" (fun () ->
      shard ~mesh:am ~in_spec:i_spec ~out_spec:i_spec Fun.id (arange 7));
  raises_with "out refers to manual" "must refer to an axis marked as manual"
    (fun () ->
      shard ~mesh:auto22 ~axis_names:[ "a" ] ~in_spec:[ S.Named "a" ]
        ~out_spec:[ S.Named "b" ] Fun.id (arange 8));
  raises_with "in refers to manual" "must refer to an axis marked as manual"
    (fun () ->
      shard ~mesh:auto22 ~axis_names:[ "a" ] ~in_spec:[ S.Named "b" ]
        ~out_spec:[ S.Named "a" ] Fun.id (arange 8));
  raises_with "no mesh" "The context mesh cannot be empty" (fun () ->
      S.shard_map ~in_specs:[ i_spec ] ~out_specs:[ i_spec ] (unary Fun.id)
        [ T.Leaf (arange 8) ])

let shard_map_jit () =
  let x = on_i (arange 8) in
  let m = Lazy.force mesh4 in
  let squared_psum a = Lax_api.mul (P.psum a ~axis_name:"i") a in
  let expected = [| 0.; 16.; 24.; 48.; 48.; 80.; 72.; 112. |] in
  let call z = shard ~in_spec:i_spec ~out_spec:i_spec squared_psum z in
  let before, _ = Backend.stats () in
  let y = leaf (Api.jit (unary call) [ T.Leaf x ]) in
  values "jit over shard_map" expected y;
  check_spec "jit over shard_map spec" "P('i',)" y;
  let after, _ = Backend.stats () in
  Alcotest.(check bool) "compiled" true (after > before);
  let am = Lazy.force auto4 in
  ignore m;
  let call_auto z =
    leaf
      (S.shard_map ~mesh:am ~in_specs:[ i_spec ] ~out_specs:[ i_spec ]
         (unary squared_psum) [ T.Leaf z ])
  in
  let host_in = leaf (Api.jit (unary call_auto) [ T.Leaf (arange 8) ]) in
  values "jit with host input" expected host_in;
  check_spec "host input spec" "P('i',)" host_in;
  let inner_jit =
    shard ~in_spec:i_spec ~out_spec:[]
      (fun a ->
        leaf (Api.jit (unary (fun b -> P.psum b ~axis_name:"i")) [ T.Leaf a ]))
      x
  in
  values "jit under shard_map" [| 12.; 16. |] inner_jit;
  let scanned =
    shard ~in_spec:i_spec ~out_spec:[]
      (fun a ->
        let body = function
          | [ carry; elt ] ->
              [ Lax_api.add carry elt; P.psum elt ~axis_name:"i" ]
          | _ -> Alcotest.fail "scan arity"
        in
        match Lax_api.scan body [ f32 [||] [| 0.0 |] ] [ a ] with
        | [ _; ys ] -> ys
        | _ -> Alcotest.fail "scan outputs")
      x
  in
  values "scan under shard_map" [| 12.; 16. |] scanned

let shard_map_ad () =
  let am = auto4 in
  let x = arange 8 in
  let mapped f z = shard ~mesh:am ~in_spec:i_spec ~out_spec:i_spec f z in
  let grad_psum =
    leaf
      (Api.grad
         (unary (fun z ->
              Numpy.Reductions.sum
                (shard ~mesh:am ~in_spec:i_spec ~out_spec:[]
                   (fun a -> P.psum (Lax_api.mul a a) ~axis_name:"i")
                   z)))
         [ T.Leaf x ])
  in
  values "grad psum" (Array.init 8 (fun i -> float_of_int (2 * i))) grad_psum;
  let grad_sin =
    leaf
      (Api.grad
         (unary (fun z ->
              Numpy.Reductions.sum
                (mapped
                   (fun a -> Lax_api.mul (Lax_api.sin a) (f32 [||] [| 2.0 |]))
                   z)))
         [ T.Leaf x ])
  in
  values "grad sin"
    [|
      2.0;
      1.0806045532226562;
      -0.832293689250946;
      -1.979984998703003;
      -1.3072872161865234;
      0.5673243999481201;
      1.9203405380249023;
      1.5078045129776;
    |]
    grad_sin;
  let _, tangent =
    Api.jvp
      (unary (fun z -> mapped Lax_api.sin z))
      [ T.Leaf x ]
      [ T.Leaf (f32 [| 8 |] (Array.make 8 1.0)) ]
  in
  values "jvp"
    [|
      1.0;
      0.5403022766113281;
      -0.416146844625473;
      -0.9899924993515015;
      -0.6536436080932617;
      0.28366219997406006;
      0.9601702690124512;
      0.7539022564888;
    |]
    (leaf tangent);
  let batched =
    leaf
      (Api.vmap
         (unary (fun z ->
              shard ~mesh:am ~in_spec:i_spec ~out_spec:[]
                (fun a -> P.psum a ~axis_name:"i")
                z))
         [ Some 0 ]
         [ T.Leaf (f32 [| 2; 8 |] (Array.init 16 float_of_int)) ])
  in
  values "vmap over shard_map" [| 12.; 16.; 44.; 48. |] batched;
  let jit_grad =
    leaf
      (Api.jit
         (fun args ->
           Api.grad
             (unary (fun z ->
                  Numpy.Reductions.sum
                    (shard ~mesh:am ~in_spec:i_spec ~out_spec:[]
                       (fun a -> P.psum (Lax_api.mul a a) ~axis_name:"i")
                       z)))
             args)
         [ T.Leaf x ])
  in
  values "jit of grad" (Array.init 8 (fun i -> float_of_int (2 * i))) jit_grad;
  let closure =
    leaf
      (Api.jit
         (unary (fun z -> mapped (fun a -> Lax_api.mul a z) x))
         [ T.Leaf (f32 [||] [| 2.0 |]) ])
  in
  values "closure over outer tracer"
    (Array.init 8 (fun i -> float_of_int (2 * i)))
    closure

let pmap_real () =
  let double = unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])) in
  let y = leaf (Api.pmap double [ T.Leaf (arange 4) ]) in
  values "pmap" [| 0.; 2.; 4.; 6. |] y;
  check_spec "pmap spec" "P('_internal_pmap_axis_name',)" y;
  Alcotest.(check string)
    "pmap mesh"
    "NamedSharding(mesh=Mesh('_internal_pmap_axis_name': 4, \
     axis_types=(Auto,)), spec=P('_internal_pmap_axis_name',), \
     memory_kind=device)"
    (S.sharding_to_string (S.array_sharding y));
  let sum_k = unary (fun a -> P.psum a ~axis_name:"k") in
  values "pmap psum"
    [| 12.; 16.; 12.; 16.; 12.; 16.; 12.; 16. |]
    (leaf (Api.pmap ~axis_name:"k" sum_k [ T.Leaf (matrix 4 2) ]));
  let idx =
    leaf
      (Api.pmap ~axis_name:"k"
         (unary (fun a ->
              Lax_api.add a
                (Core.bind1 (Convert_element_type Dtype.F32)
                   [ P.axis_index ~axis_name:"k" ])))
         [ T.Leaf (f32 [| 4 |] (Array.make 4 0.0)) ])
  in
  values "pmap axis_index" [| 0.; 1.; 2.; 3. |] idx;
  let axes =
    leaf
      (Api.pmap ~in_axes:[ Some 1; None ] ~out_axes:1 (binary Lax_api.mul)
         [ T.Leaf (matrix 2 4); T.Leaf (f32 [||] [| 3.0 |]) ])
  in
  values "pmap in/out axes" [| 0.; 3.; 6.; 9.; 12.; 15.; 18.; 21. |] axes;
  Alcotest.(check (array int)) "axes shape" [| 2; 4 |] (shape_of axes);
  check_spec "out axis spec" "P(None, '_internal_pmap_axis_name')" axes;
  let less = leaf (Api.pmap ~axis_name:"k" sum_k [ T.Leaf (arange 2) ]) in
  values "fewer replicas" [| 1.; 1. |] less;
  Alcotest.(check int) "two devices" 2 (S.num_devices (S.array_sharding less));
  values "pmap all_gather"
    [| 0.; 1.; 2.; 3.; 0.; 1.; 2.; 3.; 0.; 1.; 2.; 3.; 0.; 1.; 2.; 3. |]
    (leaf
       (Api.pmap ~axis_name:"k"
          (unary (fun a -> P.all_gather a ~axis_name:"k"))
          [ T.Leaf (arange 4) ]));
  values "grad of pmap" [| 0.; 2.; 4.; 6. |]
    (leaf
       (Api.grad
          (unary (fun z ->
               Numpy.Reductions.sum
                 (leaf
                    (Api.pmap (unary (fun a -> Lax_api.mul a a)) [ T.Leaf z ]))))
          [ T.Leaf (arange 4) ]));
  values "jit of pmap" [| 6.; 6.; 6.; 6. |]
    (leaf
       (Api.jit
          (fun args -> Api.pmap ~axis_name:"k" sum_k args)
          [ T.Leaf (arange 4) ]));
  raises_with "too many" "more replicas" (fun () ->
      Api.pmap double [ T.Leaf (arange 5) ]);
  raises_with "scalar" "more replicas" (fun () ->
      Api.pmap double [ T.Leaf (f32 [||] [| 1.0 |]) ])

let make_arrays () =
  let m = Lazy.force mesh4 in
  let s = S.named_sharding m i_spec in
  let pieces =
    List.init 4 (fun k ->
        leaf
          (S.device_put
             (T.Leaf
                (f32 [| 2 |]
                   [| float_of_int (2 * k); float_of_int ((2 * k) + 1) |]))
             (S.make_single_device_sharding (S.local_devices ()).(k))))
  in
  let built = S.make_array_from_single_device_arrays [| 8 |] s pieces in
  values "single device arrays" (Array.init 8 float_of_int) built;
  check_spec "built spec" "P('i',)" built;
  values "built doubles under jit"
    (Array.init 8 (fun i -> float_of_int (2 * i)))
    (leaf
       (Api.jit
          (unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])))
          [ T.Leaf built ]));
  values "order of arrays does not matter"
    (Array.init 8 float_of_int)
    (S.make_array_from_single_device_arrays [| 8 |] s (List.rev pieces));
  raises_with "wrong count" "Expected 4 per-device arrays" (fun () ->
      S.make_array_from_single_device_arrays [| 8 |] s
        (List.filteri (fun i _ -> i < 2) pieces));
  let bad =
    List.init 4 (fun k ->
        leaf
          (S.device_put
             (T.Leaf (arange 3))
             (S.make_single_device_sharding (S.local_devices ()).(k))))
  in
  raises_with "wrong shape" "doesn't match the single device array shape"
    (fun () -> S.make_array_from_single_device_arrays [| 8 |] s bad);
  let local = S.make_array_from_process_local_data s (arange 8) in
  values "process local" (Array.init 8 float_of_int) local;
  check_spec "process local spec" "P('i',)" local;
  Alcotest.(check (array int))
    "explicit global shape" [| 8 |]
    (shape_of
       (S.make_array_from_process_local_data ~global_shape:[| 8 |] s (arange 8)));
  raises_with "bad global shape" "global_shape should be None or equal"
    (fun () ->
      S.make_array_from_process_local_data ~global_shape:[| 8 |] s (arange 7));
  let source = Array.init 8 float_of_int in
  let callback index =
    let lo, hi = index.(0) in
    f32 [| hi - lo |] (Array.sub source lo (hi - lo))
  in
  let from_cb = S.make_array_from_callback [| 8 |] s callback in
  values "callback" source from_cb;
  Alcotest.(check (array (pair int int)))
    "callback index"
    [| (4, 6) |]
    (List.nth (S.addressable_shards from_cb) 2).index

let contexts () =
  let m = Lazy.force mesh4 in
  Alcotest.(check string)
    "empty mesh" "Mesh()"
    (S.mesh_to_string (S.get_mesh ()));
  S.with_mesh m (fun () ->
      Alcotest.(check string)
        "in context" "Mesh('i': 4, axis_types=(Explicit,))"
        (S.mesh_to_string (S.get_mesh ()));
      Alcotest.(check string)
        "abstract"
        "AbstractMesh('i': 4, axis_types=(Explicit,), device_kind=cpu, \
         num_cores=None, platform=cpu)"
        (S.abstract_mesh_to_string (S.get_abstract_mesh ())));
  Alcotest.(check string) "restored" "Mesh()" (S.mesh_to_string (S.get_mesh ()));
  Alcotest.(check string)
    "restored abstract" "AbstractMesh((), axis_types=())"
    (S.abstract_mesh_to_string (S.get_abstract_mesh ()));
  let device =
    S.abstract_device ~num_cores:1 ~device_kind:"TPU v6 lite" ~platform:"tpu" ()
  in
  Alcotest.(check string)
    "abstract device"
    "AbstractDevice(device_kind=TPU v6 lite, num_cores=1, platform=tpu)"
    (S.abstract_device_to_string device);
  let am =
    S.abstract_mesh ~axis_types:[ S.Auto ] ~abstract_device:device [ 2 ] [ "x" ]
  in
  Alcotest.(check string)
    "abstract mesh"
    "AbstractMesh('x': 2, axis_types=(Auto,), device_kind=TPU v6 lite, \
     num_cores=1, platform=tpu)"
    (S.abstract_mesh_to_string am);
  Alcotest.(check string)
    "update types"
    "AbstractMesh('x': 2, axis_types=(Manual,), device_kind=TPU v6 lite, \
     num_cores=1, platform=tpu)"
    (S.abstract_mesh_to_string (S.update_axis_types am [ ("x", S.Manual) ]));
  let q = S.abstract_mesh [ 4 ] [ "q" ] in
  S.use_abstract_mesh q (fun () ->
      Alcotest.(check string)
        "use_abstract_mesh" "AbstractMesh('q': 4, axis_types=(Auto,))"
        (S.abstract_mesh_to_string (S.get_abstract_mesh ()));
      Alcotest.(check string)
        "concrete unchanged" "Mesh()"
        (S.mesh_to_string (S.get_mesh ()));
      raises_with "size change" "cannot change the size of the mesh" (fun () ->
          S.use_abstract_mesh (S.abstract_mesh [ 2 ] [ "r" ]) (fun () -> ())));
  Alcotest.(check string)
    "exit" "AbstractMesh((), axis_types=())"
    (S.abstract_mesh_to_string (S.get_abstract_mesh ()));
  let seen = ref "" in
  ignore
    (S.with_mesh (Lazy.force mesh22) (fun () ->
         S.shard_map ~mesh:(Lazy.force mesh22) ~axis_names:[ "a" ]
           ~in_specs:[ [ S.Named "a" ] ] ~out_specs:[ [ S.Named "a" ] ]
           (unary (fun a ->
                seen := S.abstract_mesh_to_string (S.get_abstract_mesh ());
                a))
           [ T.Leaf (arange 4) ]));
  Alcotest.(check string)
    "manual inside shard_map"
    "AbstractMesh('a': 2, 'b': 2, axis_types=(Manual, Explicit), \
     device_kind=cpu, num_cores=None, platform=cpu)"
    !seen;
  raises_with "set_mesh under jit" "can only be used outside" (fun () ->
      Api.jit
        (unary (fun a ->
             S.set_mesh m;
             a))
        [ T.Leaf (arange 8) ]);
  raises_with "get_mesh under jit" "can only be used outside" (fun () ->
      Api.jit
        (unary (fun a ->
             ignore (S.get_mesh ());
             a))
        [ T.Leaf (arange 8) ]);
  raises_with "set_mesh with manual" "contains manual axes" (fun () ->
      S.set_mesh (S.make_mesh ~axis_types:[ S.Manual ] [ 4 ] [ "i" ]));
  S.set_mesh m;
  Alcotest.(check string)
    "global set" "Mesh('i': 4, axis_types=(Explicit,))"
    (S.mesh_to_string (S.get_mesh ()));
  S.concrete_context := None;
  S.abstract_context := None

let axis_types () =
  Alcotest.(check (list string))
    "names"
    [ "Auto"; "Explicit"; "Manual" ]
    (List.map S.axis_type_to_string [ S.Auto; S.Explicit; S.Manual ]);
  let t = S.manual_axis_type ~varying:[ "a" ] ~reduced:[ "b" ] () in
  let ct = S.manual_axis_to_ct t in
  Alcotest.(check (list string)) "ct unreduced" [ "b" ] ct.unreduced;
  Alcotest.(check (list string)) "ct reduced" [] ct.reduced;
  Alcotest.(check (list string)) "ct varying" [ "a" ] ct.varying;
  raises_with "varying and unreduced" "varying and unreduced cannot have common"
    (fun () -> S.manual_axis_type ~varying:[ "a" ] ~unreduced:[ "a" ] ());
  raises_with "varying and reduced" "varying and reduced cannot have common"
    (fun () -> S.manual_axis_type ~varying:[ "a" ] ~reduced:[ "a" ] ())

let auto_explicit () =
  let m = Lazy.force mesh4 in
  let x = on_i (arange 8) in
  S.with_mesh m (fun () ->
      let seen = ref "" in
      let out =
        leaf
          (S.auto_axes
             ~out_sharding:[ S.named_sharding m [] ]
             (unary (fun a ->
                  seen := S.abstract_mesh_to_string (S.get_abstract_mesh ());
                  Numpy.Reductions.sum a))
             [ T.Leaf x ])
      in
      values "auto sum" [| 28.0 |] out;
      check_spec "auto sum spec" "P()" out;
      Alcotest.(check bool)
        "auto mode inside" true
        (contains !seen "axis_types=(Auto,)");
      Alcotest.(check bool)
        "restored" true
        (contains
           (S.abstract_mesh_to_string (S.get_abstract_mesh ()))
           "axis_types=(Explicit,)");
      let doubled =
        leaf
          (S.auto_axes
             ~out_sharding:[ S.named_sharding m i_spec ]
             (unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])))
             [ T.Leaf x ])
      in
      values "auto double"
        (Array.init 8 (fun i -> float_of_int (2 * i)))
        doubled;
      check_spec "auto double spec" "P('i',)" doubled);
  let am = Lazy.force auto4 in
  S.with_mesh am (fun () ->
      let seen = ref "" in
      let out =
        leaf
          (S.explicit_axes
             ~in_sharding:[ S.named_sharding am i_spec ]
             (unary (fun a ->
                  seen := spec_of a;
                  Lax_api.mul a (f32 [||] [| 2.0 |])))
             [ T.Leaf (arange 8) ])
      in
      Alcotest.(check string) "explicit input" "P('i',)" !seen;
      values "explicit" (Array.init 8 (fun i -> float_of_int (2 * i))) out);
  raises_with "no context" "Context mesh cannot be empty" (fun () ->
      S.explicit_axes
        ~in_sharding:[ S.named_sharding m i_spec ]
        (unary Fun.id)
        [ T.Leaf (arange 8) ]);
  values "auto with no mesh runs the function" [| 3.0 |]
    (leaf
       (S.auto_axes ~out_sharding:[] (unary Fun.id)
          [ T.Leaf (f32 [||] [| 3.0 |]) ]))

let abstract_context_mesh () =
  let q = S.abstract_mesh [ 4 ] [ "q" ] in
  let psum_q =
    S.shard_map ~in_specs:[ [ S.Named "q" ] ] ~out_specs:[ [] ]
      (unary (fun a -> P.psum a ~axis_name:"q"))
  in
  raises_with "abstract needs named arrays" "with a `NamedSharding` as input"
    (fun () -> S.use_abstract_mesh q (fun () -> psum_q [ T.Leaf (arange 8) ]));
  let concrete = S.make_mesh ~axis_types:[ S.Auto ] [ 4 ] [ "q" ] in
  let placed =
    leaf
      (S.device_put
         (T.Leaf (arange 8))
         (S.named_sharding concrete [ S.Named "q" ]))
  in
  let out = leaf (S.use_abstract_mesh q (fun () -> psum_q [ T.Leaf placed ])) in
  values "abstract mesh with named input" [| 12.; 16. |] out;
  check_spec "abstract mesh spec" "P()" out;
  let x = on_i (arange 8) in
  Alcotest.(check (array (pair int int)))
    "global shard index"
    [| (6, 8) |]
    (List.nth (S.global_shards x) 3).index;
  values "addressable data" [| 6.; 7. |] (S.addressable_data x 3);
  Alcotest.(check bool) "committed" true (S.is_committed x);
  Alcotest.(check bool)
    "uncommitted" false
    (S.is_committed (leaf (Api.device_put (T.Leaf (arange 8)))))

let device_orders () =
  let devices = S.local_devices () in
  let reversed =
    S.make_mesh ~axis_types:[ S.Auto ]
      ~devices:(Array.init 4 (fun k -> devices.(3 - k)))
      [ 4 ] [ "i" ]
  in
  let x =
    leaf (S.device_put (T.Leaf (arange 8)) (S.named_sharding reversed i_spec))
  in
  Alcotest.(check (list int))
    "reversed devices" [ 3; 2; 1; 0 ]
    (List.map (fun (s : S.shard) -> s.device.id) (S.addressable_shards x));
  values "device 3 holds the start" [| 0.; 1. |]
    (List.hd (S.addressable_shards x)).data;
  let indexed =
    leaf
      (S.shard_map ~mesh:reversed ~in_specs:[ i_spec ] ~out_specs:[ i_spec ]
         (unary (fun a ->
              Lax_api.add a
                (Core.bind1 (Convert_element_type Dtype.F32)
                   [ P.axis_index ~axis_name:"i" ])))
         [ T.Leaf x ])
  in
  values "index follows the mesh" [| 0.; 1.; 3.; 4.; 6.; 7.; 9.; 10. |] indexed;
  let last = List.nth (S.addressable_shards indexed) 3 in
  Alcotest.(check int) "last shard device" 0 last.device.id;
  values "last shard" [| 9.; 10. |] last.data;
  let upper =
    S.make_mesh ~axis_types:[ S.Auto ] ~devices:(Array.sub devices 2 2) [ 2 ]
      [ "x" ]
  in
  let z =
    leaf
      (S.device_put
         (T.Leaf (arange 4))
         (S.named_sharding upper [ S.Named "x" ]))
  in
  let w =
    leaf
      (Api.jit
         (unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])))
         [ T.Leaf z ])
  in
  values "subset" [| 0.; 2.; 4.; 6. |] w;
  check_spec "subset spec" "P('x',)" w;
  Alcotest.(check (list int))
    "subset devices" [ 2; 3 ]
    (List.map (fun (s : S.shard) -> s.device.id) (S.addressable_shards w))

let host_fallback_and_transforms () =
  let am = Lazy.force auto4 in
  let sharded = S.named_sharding am i_spec in
  let x = leaf (S.device_put (T.Leaf (arange 8)) sharded) in
  values "vmap over placed"
    (Array.init 8 (fun i -> float_of_int (2 * i)))
    (leaf
       (Api.vmap
          (unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])))
          [ Some 0 ] [ T.Leaf x ]));
  values "grad through placed"
    (Array.init 8 (fun i -> float_of_int (2 * i)))
    (leaf
       (Api.grad
          (unary (fun a -> Numpy.Reductions.sum (Lax_api.mul a a)))
          [ T.Leaf x ]));
  values "grad through jit with shardings"
    (Array.init 8 (fun i -> float_of_int (2 * i)))
    (leaf
       (Api.grad
          (S.jit ~in_shardings:[ sharded ]
             (unary (fun a -> Numpy.Reductions.sum (Lax_api.mul a a))))
          [ T.Leaf (arange 8) ]));
  values "jit reshape"
    (Array.init 8 float_of_int)
    (leaf
       (Api.jit (unary (fun a -> Lax_api.reshape a [| 2; 4 |])) [ T.Leaf x ]));
  let constrained =
    leaf
      (Api.jit
         (unary (fun a ->
              leaf
                (S.with_sharding_constraint
                   (T.Leaf (Lax_api.mul a (f32 [||] [| 2.0 |])))
                   sharded)))
         [ T.Leaf (arange 8) ])
  in
  check_spec "with_sharding_constraint" "P('i',)" constrained;
  let diag =
    leaf
      (S.device_put
         (T.Leaf
            (Lax_api.add (matrix 4 4)
               (f32 [| 4; 4 |]
                  (Array.init 16 (fun i -> if i mod 5 = 0 then 20.0 else 0.0)))))
         sharded)
  in
  values "linalg falls back to the host" [| 0.086956516 |]
    (Numpy.Reductions.sum (Numpy.Linalg.inv diag))

let smap () =
  let m = Lazy.force mesh4 in
  let x = on_i (arange 8) in
  raises_with "no context" "The context mesh cannot be empty" (fun () ->
      S.smap ~in_axes:[ Some 0 ] ~out_axes:(Some 0) ~axis_name:"i"
        (unary Fun.id) [ T.Leaf x ]);
  S.with_mesh m (fun () ->
      let psum_times a = Lax_api.mul (P.psum a ~axis_name:"i") a in
      let y =
        leaf
          (S.smap ~in_axes:[ Some 0 ] ~out_axes:(Some 0) ~axis_name:"i"
             (unary psum_times) [ T.Leaf x ])
      in
      values "smap" [| 0.; 16.; 24.; 48.; 48.; 80.; 72.; 112. |] y;
      check_spec "smap spec" "P('i',)" y;
      values "smap infer"
        (Array.init 8 (fun i -> float_of_int (2 * i)))
        (leaf
           (S.smap ~out_axes:(Some 0) ~axis_name:"i"
              (unary (fun a -> Lax_api.mul a (f32 [||] [| 2.0 |])))
              [ T.Leaf x ]));
      values "smap in none"
        (Array.init 8 (fun i -> float_of_int (2 * i)))
        (leaf
           (S.smap ~in_axes:[ Some 0; None ] ~out_axes:(Some 0) ~axis_name:"i"
              (binary Lax_api.mul)
              [ T.Leaf x; T.Leaf (f32 [||] [| 2.0 |]) ]));
      values "smap out none" [| 12.; 16. |]
        (leaf
           (S.smap ~in_axes:[ Some 0 ] ~out_axes:None ~axis_name:"i"
              (unary (fun a -> P.psum a ~axis_name:"i"))
              [ T.Leaf x ])));
  S.with_mesh (Lazy.force mesh22) (fun () ->
      values "partially manual smap" [| 4.; 6.; 8.; 10. |]
        (leaf
           (S.smap ~in_axes:[ Some 0 ] ~out_axes:None ~axis_name:"a"
              (unary (fun a -> P.psum a ~axis_name:"a"))
              [ T.Leaf (arange 8) ]));
      values "partially manual smap on the second axis" [| 4.; 6.; 8.; 10. |]
        (leaf
           (S.smap ~in_axes:[ Some 0 ] ~out_axes:None ~axis_name:"b"
              (unary (fun a -> P.psum a ~axis_name:"b"))
              [ T.Leaf (arange 8) ])))

let ds () =
  let module I = Numpy.Indexing in
  let int32 v = Some (Concrete (Ndarray.of_floats Dtype.I32 [||] [| v |])) in
  let x = arange 10 in
  let m = matrix 3 4 in
  values "static" [| 1.; 2.; 3. |]
    (I.dslice_index x [ I.ds ~size:3 (int32 1.) ]);
  values "dynamic" [| 2.; 3.; 4. |]
    (I.dslice_index x [ I.ds ~size:3 (int32 2.) ]);
  values "clamped" [| 7.; 8.; 9. |]
    (I.dslice_index x [ I.ds ~size:3 (int32 8.) ]);
  values "negative wraps" [| 7.; 8.; 9. |]
    (I.dslice_index x [ I.ds ~size:3 (int32 (-2.)) ]);
  values "stride"
    [| 1.; 3.; 5.; 7.; 9.; 9. |]
    (I.dslice_index x [ I.ds ~size:6 ~stride:2 (int32 1.) ]);
  values "stride clamps" [| 7.; 9.; 9. |]
    (I.dslice_index x [ I.ds ~size:3 ~stride:2 (int32 7.) ]);
  values "matrix" [| 6.; 7.; 10.; 11. |]
    (I.dslice_index m [ I.ds ~size:2 (int32 1.); I.ds ~size:2 (int32 2.) ]);
  values "full then slice"
    [| 1.; 2.; 5.; 6.; 9.; 10. |]
    (I.dslice_index m [ I.ds None; I.ds ~size:2 (int32 1.) ]);
  values "size only" [| 0.; 1.; 2. |] (I.dslice_index x [ I.ds (int32 3.) ]);
  values "none is everything"
    (Array.init 10 float_of_int)
    (I.dslice_index x [ I.ds None ]);
  values "under jit" [| 2.; 3.; 4. |]
    (leaf
       (Api.jit
          (unary (fun i -> I.dslice_index x [ I.ds ~size:3 (Some i) ]))
          [ T.Leaf (Concrete (Ndarray.of_floats Dtype.I32 [||] [| 2. |])) ]))

let visualize () =
  let m = Lazy.force mesh4 in
  let m2 = Lazy.force mesh22 in
  let check name expected shape s =
    Alcotest.(check string) name expected (S.visualize_sharding_string shape s)
  in
  check "row"
    "┌───────┬───────┬───────┬───────┐\n\
     │ CPU 0 │ CPU 1 │ CPU 2 │ CPU 3 │\n\
     └───────┴───────┴───────┴───────┘\n\n"
    [| 8 |]
    (S.named_sharding m i_spec);
  check "grid"
    "┌────────────────┬────────────────┐\n\
     │                │                │\n\
     │     CPU 0      │     CPU 1      │\n\
     │                │                │\n\
     │                │                │\n\
     ├────────────────┼────────────────┤\n\
     │                │                │\n\
     │     CPU 2      │     CPU 3      │\n\
     │                │                │\n\
     │                │                │\n\
     └────────────────┴────────────────┘\n\n"
    [| 4; 6 |]
    (S.named_sharding m2 [ S.Named "a"; S.Named "b" ]);
  check "partial" "┌───────┬───────┐\n│CPU 0,1│CPU 2,3│\n└───────┴───────┘\n\n"
    [| 4 |]
    (S.named_sharding m2 [ S.Named "a" ]);
  check "single" "┌───────┐\n│ CPU 1 │\n└───────┘\n\n" [| 4 |]
    (S.make_single_device_sharding (S.local_devices ()).(1));
  check "replicated" "┌───────────┐\n│CPU 0,1,2,3│\n└───────────┘\n\n" [| 8 |]
    (S.named_sharding m []);
  check "tall columns"
    "┌───────┬───────┐\n\
     │       │       │\n\
     │       │       │\n\
     │       │       │\n\
     │       │       │\n\
     │CPU 0,2│CPU 1,3│\n\
     │       │       │\n\
     │       │       │\n\
     │       │       │\n\
     │       │       │\n\
     └───────┴───────┘\n\n"
    [| 8; 4 |]
    (S.named_sharding m2 [ S.Unconstrained; S.Named "b" ]);
  check "rows"
    "┌───────┐\n\
     │       │\n\
     │CPU 0,1│\n\
     │       │\n\
     │       │\n\
     ├───────┤\n\
     │       │\n\
     │CPU 2,3│\n\
     │       │\n\
     │       │\n\
     └───────┘\n\n"
    [| 100; 4 |]
    (S.named_sharding m2 [ S.Named "a" ]);
  raises_with "rank" "only works for shapes with 1 and 2 dimensions" (fun () ->
      S.visualize_sharding_string [| 4; 6; 2 |] (S.named_sharding m i_spec));
  let seen = ref "" in
  Debugging.inspect_array_sharding
    (on_i (arange 8))
    ~callback:(fun s -> seen := s);
  Alcotest.(check string)
    "inspect"
    "NamedSharding(mesh=Mesh('i': 4, axis_types=(Explicit,)), spec=P('i',), \
     memory_kind=device)"
    !seen

let () =
  let case name f = Alcotest.test_case name `Quick f in
  Alcotest.run "sharding"
    [
      ( "devices",
        [
          case "discovery" discovery;
          case "environment" environment;
          case "mesh" mesh_objects;
          case "specs" specs;
          case "named sharding" named_properties;
          case "indices" indices_map;
        ] );
      ( "arrays",
        [
          case "placement" placement;
          case "dtypes" placement_dtypes;
          case "single device" single_device_sharding;
          case "make arrays" make_arrays;
          case "reshard" reshard;
        ] );
      ( "execution",
        [
          case "jit propagation" jit_propagates;
          case "jit shardings" jit_with_shardings;
          case "eager" eager_ops;
          case "shard_map collectives" shard_map_collectives;
          case "compiled" compiled_for_devices;
          case "shard_map axes" shard_map_two_axes;
          case "shard_map args" shard_map_args;
          case "shard_map jit" shard_map_jit;
          case "shard_map ad" shard_map_ad;
          case "pmap" pmap_real;
          case "abstract context" abstract_context_mesh;
          case "device orders" device_orders;
          case "fallback and transforms" host_fallback_and_transforms;
        ] );
      ( "context",
        [
          case "mesh contexts" contexts;
          case "axis types" axis_types;
          case "auto and explicit" auto_explicit;
          case "smap" smap;
        ] );
      ("misc", [ case "ds" ds; case "visualize" visualize ]);
    ]
