exception Error of string

let env_var = "OJAX_PJRT_PLUGIN"

let expected_sha256 =
  "a30b08a486bf4e80c64940ef70035ca2339f4a9badfc07cae2aab9901ed9d979"

let pjrt_api_minor = 81
let mask = 0xFFFFFFFF

let k =
  [|
    0x428a2f98;
    0x71374491;
    0xb5c0fbcf;
    0xe9b5dba5;
    0x3956c25b;
    0x59f111f1;
    0x923f82a4;
    0xab1c5ed5;
    0xd807aa98;
    0x12835b01;
    0x243185be;
    0x550c7dc3;
    0x72be5d74;
    0x80deb1fe;
    0x9bdc06a7;
    0xc19bf174;
    0xe49b69c1;
    0xefbe4786;
    0x0fc19dc6;
    0x240ca1cc;
    0x2de92c6f;
    0x4a7484aa;
    0x5cb0a9dc;
    0x76f988da;
    0x983e5152;
    0xa831c66d;
    0xb00327c8;
    0xbf597fc7;
    0xc6e00bf3;
    0xd5a79147;
    0x06ca6351;
    0x14292967;
    0x27b70a85;
    0x2e1b2138;
    0x4d2c6dfc;
    0x53380d13;
    0x650a7354;
    0x766a0abb;
    0x81c2c92e;
    0x92722c85;
    0xa2bfe8a1;
    0xa81a664b;
    0xc24b8b70;
    0xc76c51a3;
    0xd192e819;
    0xd6990624;
    0xf40e3585;
    0x106aa070;
    0x19a4c116;
    0x1e376c08;
    0x2748774c;
    0x34b0bcb5;
    0x391c0cb3;
    0x4ed8aa4a;
    0x5b9cca4f;
    0x682e6ff3;
    0x748f82ee;
    0x78a5636f;
    0x84c87814;
    0x8cc70208;
    0x90befffa;
    0xa4506ceb;
    0xbef9a3f7;
    0xc67178f2;
  |]

let rotr x n = (x lsr n) lor (x lsl (32 - n)) land mask

let sha256_hex msg =
  let h =
    [|
      0x6a09e667;
      0xbb67ae85;
      0x3c6ef372;
      0xa54ff53a;
      0x510e527f;
      0x9b05688c;
      0x1f83d9ab;
      0x5be0cd19;
    |]
  in
  let len = String.length msg in
  let bitlen = len * 8 in
  let rem = (len + 1) mod 64 in
  let padlen = if rem <= 56 then 56 - rem else 120 - rem in
  let total = len + 1 + padlen + 8 in
  let m = Bytes.make total '\000' in
  Bytes.blit_string msg 0 m 0 len;
  Bytes.set m len '\x80';
  for i = 0 to 7 do
    Bytes.set m (total - 1 - i) (Char.chr ((bitlen lsr (8 * i)) land 0xff))
  done;
  let w = Array.make 64 0 in
  let nblocks = total / 64 in
  for b = 0 to nblocks - 1 do
    let base = b * 64 in
    for t = 0 to 15 do
      let o = base + (t * 4) in
      w.(t) <-
        (Char.code (Bytes.get m o) lsl 24)
        lor (Char.code (Bytes.get m (o + 1)) lsl 16)
        lor (Char.code (Bytes.get m (o + 2)) lsl 8)
        lor Char.code (Bytes.get m (o + 3))
    done;
    for t = 16 to 63 do
      let x = w.(t - 15) in
      let y = w.(t - 2) in
      let s0 = rotr x 7 lxor rotr x 18 lxor (x lsr 3) in
      let s1 = rotr y 17 lxor rotr y 19 lxor (y lsr 10) in
      w.(t) <- (w.(t - 16) + s0 + w.(t - 7) + s1) land mask
    done;
    let a = ref h.(0)
    and b1 = ref h.(1)
    and c = ref h.(2)
    and d = ref h.(3)
    and e = ref h.(4)
    and f = ref h.(5)
    and g = ref h.(6)
    and hh = ref h.(7) in
    for t = 0 to 63 do
      let ss1 = rotr !e 6 lxor rotr !e 11 lxor rotr !e 25 in
      let ch = !e land !f lxor (mask lxor !e land !g) in
      let temp1 = (!hh + ss1 + ch + k.(t) + w.(t)) land mask in
      let ss0 = rotr !a 2 lxor rotr !a 13 lxor rotr !a 22 in
      let maj = !a land !b1 lxor (!a land !c) lxor (!b1 land !c) in
      let temp2 = (ss0 + maj) land mask in
      hh := !g;
      g := !f;
      f := !e;
      e := (!d + temp1) land mask;
      d := !c;
      c := !b1;
      b1 := !a;
      a := (temp1 + temp2) land mask
    done;
    h.(0) <- (h.(0) + !a) land mask;
    h.(1) <- (h.(1) + !b1) land mask;
    h.(2) <- (h.(2) + !c) land mask;
    h.(3) <- (h.(3) + !d) land mask;
    h.(4) <- (h.(4) + !e) land mask;
    h.(5) <- (h.(5) + !f) land mask;
    h.(6) <- (h.(6) + !g) land mask;
    h.(7) <- (h.(7) + !hh) land mask
  done;
  let buf = Stdlib.Buffer.create 64 in
  Array.iter (fun x -> Stdlib.Buffer.add_string buf (Printf.sprintf "%08x" x)) h;
  Stdlib.Buffer.contents buf

let sha256_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in_noerr ic)
    (fun () -> sha256_hex (In_channel.input_all ic))

let validate_path = function
  | None ->
      raise
        (Error
           (Printf.sprintf
              "%s is not set; point it at an absolute path to the pinned PJRT \
               plugin"
              env_var))
  | Some "" ->
      raise
        (Error
           (Printf.sprintf "%s is set but empty; expected an absolute path"
              env_var))
  | Some path ->
      if Filename.is_relative path then
        raise
          (Error
             (Printf.sprintf "%s must be an absolute path, got: %s" env_var path))
      else path

let resolve () = validate_path (Sys.getenv_opt env_var)

let verify_at path =
  if not (Sys.file_exists path) then
    raise (Error (Printf.sprintf "PJRT plugin not found: %s" path));
  if Sys.is_directory path then
    raise (Error (Printf.sprintf "PJRT plugin path is a directory: %s" path));
  let actual = sha256_file path in
  if not (String.equal actual expected_sha256) then
    raise
      (Error
         (Printf.sprintf
            "PJRT plugin sha256 mismatch at %s: expected %s, got %s" path
            expected_sha256 actual));
  path

let preflight () = verify_at (resolve ())
