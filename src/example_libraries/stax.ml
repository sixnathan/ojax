open Types
module Tree = Tree_util
module L = Lax_api
module NN = Nn.Functions

type layer = {
  init : value -> int array -> int array * value Tree.t;
  apply : ?rng:value -> value Tree.t -> value -> value;
}

let dtype_default () = Dtypes.default_float_dtype ()

let glorot_scale in_size out_size =
  sqrt (2.0 /. float_of_int (in_size + out_size))

let split2 key =
  let keys = Random.Core.split key 2 in
  (L.index_in_dim ~keepdims:false keys 0, L.index_in_dim ~keepdims:false keys 1)

let glorot key shape =
  let n = Array.length shape in
  let receptive =
    Array.fold_left ( * ) 1 (Array.sub shape 0 (Stdlib.max 0 (n - 2)))
  in
  let fan_in = shape.(n - 2) * receptive
  and fan_out = shape.(n - 1) * receptive in
  let std = glorot_scale fan_in fan_out in
  let sample = Random.Core.normal key ~shape in
  L.mul sample (L.full_like sample std)

let normal_init key shape =
  let sample = Random.Core.normal key ~shape in
  L.mul sample (L.full_like sample 0.01)

let dense ?(w_init = glorot) ?(b_init = normal_init) out_dim =
  {
    init =
      (fun rng input_shape ->
        let n = Array.length input_shape in
        let k1, k2 = split2 rng in
        let w = w_init k1 [| input_shape.(n - 1); out_dim |] in
        let b = b_init k2 [| out_dim |] in
        let out = Array.copy input_shape in
        out.(n - 1) <- out_dim;
        (out, Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ]));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ->
            let product =
              let nx = Array.length (Core.get_aval x).shape in
              L.dot_general x w
                ~lhs_contract:[| nx - 1 |]
                ~rhs_contract:[| 0 |] ~lhs_batch:[||] ~rhs_batch:[||]
            in
            L.add product
              (L.broadcast_in_dim b (Core.get_aval product).shape
                 [| Array.length (Core.get_aval product).shape - 1 |])
        | _ -> invalid_arg "dense: bad parameters");
  }

let elementwise f =
  {
    init = (fun _ input_shape -> (input_shape, Tree.Tuple []));
    apply = (fun ?rng:_ _ x -> f x);
  }

let tanh = elementwise L.tanh
let relu = elementwise NN.relu
let exp = elementwise L.exp
let log_softmax = elementwise (fun x -> NN.log_softmax x)
let softmax = elementwise (fun x -> NN.softmax x)
let softplus = elementwise NN.softplus
let sigmoid = elementwise NN.sigmoid
let elu = elementwise (fun x -> NN.elu x)
let leaky_relu = elementwise (fun x -> NN.leaky_relu x)
let selu = elementwise NN.selu
let gelu = elementwise (fun x -> NN.gelu x)
let identity = elementwise Fun.id

let flatten =
  {
    init =
      (fun _ input_shape ->
        let rest =
          Array.fold_left ( * ) 1
            (Array.sub input_shape 1 (Array.length input_shape - 1))
        in
        ([| input_shape.(0); rest |], Tree.Tuple []));
    apply =
      (fun ?rng:_ _ x ->
        let sh = (Core.get_aval x).shape in
        let rest =
          Array.fold_left ( * ) 1 (Array.sub sh 1 (Array.length sh - 1))
        in
        L.reshape x [| sh.(0); rest |]);
  }

let conv_dims_nhwc rank =
  let spatial = Array.init (rank - 2) (fun i -> i + 1) in
  let spatial_k = Array.init (rank - 2) Fun.id in
  {
    lhs_spec = Array.append [| 0; rank - 1 |] spatial;
    rhs_spec = Array.append [| rank - 1; rank - 2 |] spatial_k;
    out_spec = Array.append [| 0; rank - 1 |] spatial;
  }

let general_conv ?strides ?(padding = L.Valid) ?(w_init = glorot)
    ?(b_init = normal_init) out_chan filter_shape =
  let k = Array.length filter_shape in
  let strides = Option.value strides ~default:(Array.make k 1) in
  let dn = conv_dims_nhwc (k + 2) in
  {
    init =
      (fun rng input_shape ->
        let in_chan = input_shape.(k + 1) in
        let k1, k2 = split2 rng in
        let w = w_init k1 (Array.append filter_shape [| in_chan; out_chan |]) in
        let spatial = Array.sub input_shape 1 k in
        let pads = L.padtype_to_pads spatial filter_shape strides padding in
        let out_spatial =
          Array.mapi
            (fun i s ->
              let lo, hi = pads.(i) in
              ((s + lo + hi - filter_shape.(i)) / strides.(i)) + 1)
            spatial
        in
        let b = b_init k2 [| out_chan |] in
        ( Array.concat [ [| input_shape.(0) |]; out_spatial; [| out_chan |] ],
          Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf w; Tree.Leaf b ] ->
            let out =
              L.conv_general_dilated ~dimension_numbers:dn x w
                ~window_strides:strides padding
            in
            let sh = (Core.get_aval out).shape in
            L.add out (L.broadcast_in_dim b sh [| Array.length sh - 1 |])
        | _ -> invalid_arg "conv: bad parameters");
  }

let conv = general_conv

let pool_layer ~reducer ~init_value ?(normalize = false) ?strides
    ?(padding = L.Valid) window_shape =
  let k = Array.length window_shape in
  let strides = Option.value strides ~default:(Array.make k 1) in
  let wrap a fill = Array.concat [ [| fill |]; a; [| fill |] ] in
  let window = wrap window_shape 1 and stride = wrap strides 1 in
  let run x =
    let sh = (Core.get_aval x).shape in
    let pads = L.padtype_to_pads sh window stride padding in
    L.reduce_window x
      (L.full [||] init_value (Core.get_aval x).dtype)
      reducer ~window_dimensions:window ~window_strides:stride pads
  in
  {
    init =
      (fun _ input_shape ->
        let pads = L.padtype_to_pads input_shape window stride padding in
        let out =
          Array.mapi
            (fun i s ->
              let lo, hi = pads.(i) in
              ((s + lo + hi - window.(i)) / stride.(i)) + 1)
            input_shape
        in
        (out, Tree.Tuple []));
    apply =
      (fun ?rng:_ _ x ->
        let summed = run x in
        if not normalize then summed
        else
          let size = float_of_int (Array.fold_left ( * ) 1 window_shape) in
          L.div summed (L.full_like summed size));
  }

let max_pool ?strides ?padding window =
  pool_layer ~reducer:L.max ~init_value:Float.neg_infinity ?strides ?padding
    window

let sum_pool ?strides ?padding window =
  pool_layer ~reducer:L.add ~init_value:0.0 ?strides ?padding window

let avg_pool ?strides ?padding window =
  pool_layer ~reducer:L.add ~init_value:0.0 ~normalize:true ?strides ?padding
    window

let dropout ?(mode = `Train) rate =
  {
    init = (fun _ input_shape -> (input_shape, Tree.Tuple []));
    apply =
      (fun ?rng _ x ->
        match mode with
        | `Test -> x
        | `Train -> (
            match rng with
            | None ->
                invalid_arg
                  "dropout requires an rng in training mode; pass it to apply"
            | Some key ->
                let sh = (Core.get_aval x).shape in
                let keep = Random.Core.bernoulli key ~shape:sh ~p:rate in
                L.select keep (L.div x (L.full_like x rate)) (L.full_like x 0.0)
            ));
  }

let serial layers =
  {
    init =
      (fun rng input_shape ->
        let _, _, params_rev, out =
          List.fold_left
            (fun (key, shape, acc, _) layer ->
              let k1, k2 = split2 key in
              let out_shape, p = layer.init k1 shape in
              (k2, out_shape, p :: acc, out_shape))
            (rng, input_shape, [], input_shape)
            layers
        in
        (out, Tree.Tuple (List.rev params_rev)));
    apply =
      (fun ?rng params x ->
        match params with
        | Tree.Tuple ps ->
            List.fold_left2 (fun v layer p -> layer.apply ?rng p v) x layers ps
        | _ -> invalid_arg "serial: bad parameters");
  }

let batch_norm ?(axis = [| 0 |]) ?(epsilon = 1e-5) () =
  {
    init =
      (fun rng input_shape ->
        let n = Array.length input_shape in
        let axes = Array.map (fun a -> if a < 0 then a + n else a) axis in
        let kept =
          Array.of_list
            (List.filter (fun d -> not (Array.mem d axes)) (List.init n Fun.id))
        in
        let sh = Array.map (fun d -> input_shape.(d)) kept in
        ignore rng;
        let beta = L.full sh 0.0 (dtype_default ()) in
        let gamma = L.full sh 1.0 (dtype_default ()) in
        (input_shape, Tree.Tuple [ Tree.Leaf beta; Tree.Leaf gamma ]));
    apply =
      (fun ?rng:_ params x ->
        match params with
        | Tree.Tuple [ Tree.Leaf beta; Tree.Leaf gamma ] ->
            let sh = (Core.get_aval x).shape in
            let n = Array.length sh in
            let axes = Array.map (fun a -> if a < 0 then a + n else a) axis in
            let kept =
              Array.of_list
                (List.filter
                   (fun d -> not (Array.mem d axes))
                   (List.init n Fun.id))
            in
            let z = NN.standardize ~axis:axes.(0) ~epsilon x in
            let lift v = L.broadcast_in_dim v sh kept in
            L.add (L.mul (lift gamma) z) (lift beta)
        | _ -> invalid_arg "batch_norm: bad parameters");
  }
