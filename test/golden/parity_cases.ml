module U = Yojson.Safe.Util
module F = Ojax.Numpy.Fft

let opt_int name params =
  match U.member name params with `Null -> None | v -> Some (U.to_int v)

let opt_string name params =
  match U.member name params with `Null -> None | v -> Some (U.to_string v)

let opt_ints name params =
  match U.member name params with
  | `Null -> None
  | v -> Some (List.map U.to_int (U.to_list v))

let fft1 op a p =
  let n = opt_int "n" p in
  let axis = opt_int "axis" p in
  let norm = opt_string "norm" p in
  [ op ?n ?axis ?norm a ]

let fftnd op a p =
  [
    op ?s:(opt_ints "s" p) ?axes:(opt_ints "axes" p) ?norm:(opt_string "norm" p)
      a;
  ]

let run op params get =
  let a () = get "a" in
  match op with
  | "fft" -> fft1 F.fft (a ()) params
  | "ifft" -> fft1 F.ifft (a ()) params
  | "rfft" -> fft1 F.rfft (a ()) params
  | "irfft" -> fft1 F.irfft (a ()) params
  | "hfft" -> fft1 F.hfft (a ()) params
  | "ihfft" -> fft1 F.ihfft (a ()) params
  | "fftn" -> fftnd F.fftn (a ()) params
  | "ifftn" -> fftnd F.ifftn (a ()) params
  | "rfftn" -> fftnd F.rfftn (a ()) params
  | "irfftn" -> fftnd F.irfftn (a ()) params
  | "fft2" -> [ F.fft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "ifft2" -> [ F.ifft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "rfft2" -> [ F.rfft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "irfft2" -> [ F.irfft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "fftshift" -> [ F.fftshift ?axes:(opt_ints "axes" params) (a ()) ]
  | "ifftshift" -> [ F.ifftshift ?axes:(opt_ints "axes" params) (a ()) ]
  | other -> failwith ("parity: unknown op " ^ other)
