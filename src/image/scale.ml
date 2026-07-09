open Types
module Nd = Ndarray
module D = Dtype
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions
module TC = Numpy.Tensor_contractions
module IDX = Numpy.Indexing

let get_aval = Core.get_aval
let dtype v = (get_aval v).dtype
let shape v = (get_aval v).shape
let ndim v = Array.length (shape v)

type resize_method =
  | Nearest
  | Linear
  | Lanczos3
  | Lanczos5
  | Cubic
  | Cubic_pytorch
  | Area

type kernel = Triangle | Lanczos of float | Keys_cubic | Area_kernel

let from_string s =
  match s with
  | "nearest" -> Nearest
  | "linear" | "bilinear" | "trilinear" | "triangle" -> Linear
  | "lanczos3" -> Lanczos3
  | "lanczos5" -> Lanczos5
  | "cubic" | "bicubic" | "tricubic" -> Cubic
  | "cubic-pytorch" | "bicubic-pytorch" -> Cubic_pytorch
  | "area" -> Area
  | _ -> invalid_arg (Printf.sprintf "Unknown resize method \"%s\"" s)

let kernels = function
  | Linear -> (1, Triangle)
  | Lanczos3 -> (3, Lanczos 3.0)
  | Lanczos5 -> (5, Lanczos 5.0)
  | Cubic -> (2, Keys_cubic)
  | Cubic_pytorch -> (2, Keys_cubic)
  | Area -> (1, Area_kernel)
  | Nearest -> invalid_arg "image/scale: nearest has no interpolation kernel"

let dfloat () = Dtypes.default_float_dtype ()
let is_inexact = function D.F32 | D.F64 -> true | _ -> false
let astype v dt = if dtype v = dt then v else NL.astype v dt

let promote_inexact v =
  let dt = dtype v in
  astype v (if is_inexact dt then dt else dfloat ())

let scalar x = Concrete (Nd.of_floats (dfloat ()) [||] [| x |])
let add_s v k = UF.add v (scalar k)
let sub_s v k = UF.subtract v (scalar k)
let mul_s v k = UF.multiply v (scalar k)
let div_s v k = UF.divide v (scalar k)
let arange_df n = NL.arange ~dtype:(dfloat ()) (float_of_int n)
let eps_f32 = 1.1920928955078125e-07

let apply_kernel k x =
  match k with
  | Triangle -> UF.maximum (scalar 0.0) (UF.subtract (scalar 1.0) (UF.abs x))
  | Lanczos radius ->
      let pi = Float.pi in
      let sin_pix = UF.sin (mul_s x pi) in
      let sin_pixr = UF.sin (div_s (mul_s x pi) radius) in
      let y = UF.multiply (mul_s sin_pix radius) sin_pixr in
      let denom = mul_s (UF.multiply x x) (pi *. pi) in
      let denom = NL.where_ (UF.not_equal x (scalar 0.0)) denom (scalar 1.0) in
      let out =
        NL.where_ (UF.greater x (scalar 1e-3)) (UF.divide y denom) (scalar 1.0)
      in
      NL.where_ (UF.greater x (scalar radius)) (scalar 0.0) out
  | Keys_cubic ->
      let out =
        add_s (UF.multiply (UF.multiply (sub_s (mul_s x 1.5) 2.5) x) x) 1.0
      in
      let alt =
        add_s
          (UF.multiply
             (sub_s (UF.multiply (add_s (mul_s x (-0.5)) 2.5) x) 4.0)
             x)
          2.0
      in
      let out = NL.where_ (UF.greater_equal x (scalar 1.0)) alt out in
      NL.where_ (UF.greater_equal x (scalar 2.0)) (scalar 0.0) out
  | Area_kernel -> invalid_arg "image/scale: area kernel handled separately"

let normalize weights sample_f input_size =
  let total = RED.sum ~axis:[| 0 |] ~keepdims:true weights in
  let threshold = 1000.0 *. eps_f32 in
  let safe = NL.where_ (UF.not_equal total (scalar 0.0)) total (scalar 1.0) in
  let weights =
    NL.where_
      (UF.greater (UF.abs total) (scalar threshold))
      (UF.divide weights safe) (scalar 0.0)
  in
  let bound = float_of_int input_size -. 0.5 in
  let inside =
    UF.logical_and
      (UF.greater_equal sample_f (scalar (-0.5)))
      (UF.less_equal sample_f (scalar bound))
  in
  NL.where_ (NL.expand_dims inside [| 0 |]) weights (scalar 0.0)

let compute_weight_mat ~input_size ~output_size ~scale ~translation ~kernel
    ~antialias ~edge_padding ~radius =
  ignore radius;
  if edge_padding then
    failwith
      "image/scale: edge_padding (cubic-pytorch without antialias) is not \
       supported";
  let inv_scale = 1.0 /. scale in
  let kernel_scale = if antialias then Float.max inv_scale 1.0 else 1.0 in
  let sample_f =
    sub_s
      (sub_s
         (mul_s (add_s (arange_df output_size) 0.5) inv_scale)
         (translation *. inv_scale))
      0.5
  in
  let weights =
    match kernel with
    | Area_kernel ->
        let l_i =
          sub_s
            (mul_s (arange_df output_size) inv_scale)
            (translation *. inv_scale)
        in
        let r_i = add_s l_i inv_scale in
        let l_j = arange_df input_size in
        let r_j = add_s l_j 1.0 in
        let min_r =
          UF.minimum (NL.expand_dims r_i [| 0 |]) (NL.expand_dims r_j [| 1 |])
        in
        let max_l =
          UF.maximum (NL.expand_dims l_i [| 0 |]) (NL.expand_dims l_j [| 1 |])
        in
        UF.maximum (scalar 0.0) (UF.subtract min_r max_l)
    | _ ->
        let expanded = arange_df input_size in
        let x =
          UF.abs
            (UF.subtract
               (NL.expand_dims sample_f [| 0 |])
               (NL.expand_dims expanded [| 1 |]))
        in
        apply_kernel kernel (div_s x kernel_scale)
  in
  normalize weights sample_f input_size

let contract_axis a d w =
  let n = ndim a in
  let r = TC.tensordot ~axes:(TC.Ax_pair ([| d |], [| 0 |])) a w in
  NL.moveaxis [| n - 1 |] [| d |] r

let scale_and_translate_internal image ~output_shape ~spatial_dims ~scale
    ~translation ~kernel ~antialias =
  if Array.length spatial_dims = 0 then image
  else begin
    let input_shape = shape image in
    let dt = dtype image in
    let nd = Array.length output_shape in
    let result = ref image in
    Array.iteri
      (fun i d0 ->
        let d = if d0 < 0 then d0 + nd else d0 in
        let w =
          compute_weight_mat ~input_size:input_shape.(d)
            ~output_size:output_shape.(d) ~scale:scale.(i)
            ~translation:translation.(i) ~kernel ~antialias ~edge_padding:false
            ~radius:0
        in
        result := contract_axis !result d (astype w dt))
      spatial_dims;
    !result
  end

let resolve_kernel method_ antialias =
  let method_ =
    if method_ = Cubic_pytorch && antialias then Cubic else method_
  in
  if method_ = Cubic_pytorch && not antialias then
    failwith
      "image/scale: cubic-pytorch without antialias (edge padding) is not \
       supported";
  snd (kernels method_)

let scale_and_translate image ~shape:output_shape ~spatial_dims ~scale
    ~translation ~method_ ?(antialias = true) () =
  if Array.length output_shape <> ndim image then
    invalid_arg "shape must have length equal to the number of dimensions of x";
  if method_ = Nearest then
    failwith
      "Nearest neighbor resampling is not currently supported for \
       scale_and_translate.";
  let kernel = resolve_kernel method_ antialias in
  let image = promote_inexact image in
  scale_and_translate_internal image ~output_shape ~spatial_dims ~scale
    ~translation ~kernel ~antialias

let resize_nearest image ~output_shape =
  let input_shape = shape image in
  let result = ref image in
  Array.iteri
    (fun d n ->
      if input_shape.(d) <> n then begin
        let m = input_shape.(d) in
        let offsets =
          div_s
            (mul_s
               (add_s (NL.arange ~dtype:D.F32 (float_of_int n)) 0.5)
               (float_of_int m))
            (float_of_int n)
        in
        let offsets = NL.astype (UF.floor offsets) D.I32 in
        result := IDX.take ~axis:d !result offsets
      end)
    output_shape;
  !result

let resize image ~shape:output_shape ~method_ ?(antialias = true) () =
  if Array.length output_shape <> ndim image then
    invalid_arg "shape must have length equal to the number of dimensions of x";
  match method_ with
  | Nearest -> resize_nearest image ~output_shape
  | _ ->
      let image = promote_inexact image in
      let img_shape = shape image in
      let spatial_dims =
        Array.of_list
          (List.filter
             (fun i -> img_shape.(i) <> output_shape.(i))
             (List.init (Array.length output_shape) Fun.id))
      in
      let kernel = resolve_kernel method_ antialias in
      let scale =
        Array.map
          (fun d ->
            if output_shape.(d) = 0 then 1.0
            else float_of_int output_shape.(d) /. float_of_int img_shape.(d))
          spatial_dims
      in
      let translation = Array.make (Array.length spatial_dims) 0.0 in
      scale_and_translate_internal image ~output_shape ~spatial_dims ~scale
        ~translation ~kernel ~antialias
