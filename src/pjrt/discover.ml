exception Error of string

let env_var = "OJAX_PJRT_PLUGIN"
let gpu_env_var = "OJAX_PJRT_GPU_PLUGIN"

let pinned_sha256 =
  [
    "a30b08a486bf4e80c64940ef70035ca2339f4a9badfc07cae2aab9901ed9d979";
    "5c4ceb9f9b96562fe264ab55f6caa7a5e686a060fd53c55b0c39093100aca130";
    "6852a300d723a8cb93e40fac89a87c1ab6cdc6b03a1ea9fd7350721c115dfe08";
  ]

let pinned_gpu_sha256 =
  [ "bf07d68eafbf05fd25a5f8f692e57178755a5c33b80bc9779897ab2ee8b9d0e8" ]

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

let exact_sha256 =
  "6852a300d723a8cb93e40fac89a87c1ab6cdc6b03a1ea9fd7350721c115dfe08"

let conventional_names =
  [
    "pjrt_c_api_cpu_xla_5a9e73c_plugin.so";
    "pjrt_c_api_cpu_v0.83.1_plugin.so";
    "pjrt_c_api_cpu_plugin.so";
  ]

let conventional_dir () =
  match Sys.getenv_opt "HOME" with
  | Some home when home <> "" ->
      Some (Filename.concat (Filename.concat home ".local") "pjrt")
  | _ -> None

let conventional_path () =
  Option.bind (conventional_dir ()) (fun dir ->
      List.find_map
        (fun name ->
          let path = Filename.concat dir name in
          if Sys.file_exists path then Some path else None)
        conventional_names)

let validate_named name = function
  | None ->
      raise
        (Error
           (Printf.sprintf
              "%s is not set and no pinned PJRT plugin was found in \
               ~/.local/pjrt; point it at an absolute path to the pinned PJRT \
               plugin"
              name))
  | Some "" ->
      raise
        (Error
           (Printf.sprintf "%s is set but empty; expected an absolute path" name))
  | Some path ->
      if Filename.is_relative path then
        raise
          (Error
             (Printf.sprintf "%s must be an absolute path, got: %s" name path))
      else path

let validate_path = validate_named env_var

let resolve () =
  match Sys.getenv_opt env_var with
  | Some _ as configured -> validate_path configured
  | None -> validate_path (conventional_path ())

let find () = try Some (resolve ()) with Error _ -> None
let last_verified = ref None

let stamp_path path =
  Option.map
    (fun dir ->
      let stat = Unix.stat path in
      let identity =
        Printf.sprintf "%s\000%d\000%h" path stat.Unix.st_size
          stat.Unix.st_mtime
      in
      ( dir,
        Filename.concat dir ("plugin-" ^ Digest.to_hex (Digest.string identity))
      ))
    (Sys.getenv_opt "OJAX_TEST_COMPILE_CACHE")

let read_stamp path =
  try Some (In_channel.with_open_bin path In_channel.input_all)
  with Sys_error _ -> None

let write_stamp dir path hash =
  (try Sys.mkdir dir 0o755 with Sys_error _ -> ());
  let temp = Printf.sprintf "%s.%d.tmp" path (Unix.getpid ()) in
  Out_channel.with_open_bin temp (fun oc -> output_string oc hash);
  Sys.rename temp path

let stamped_sha256 path =
  match stamp_path path with
  | None -> sha256_file path
  | Some (dir, stamp) -> (
      match read_stamp stamp with
      | Some hash -> hash
      | None ->
          let hash = sha256_file path in
          write_stamp dir stamp hash;
          hash)

let verify_against pins path =
  if not (Sys.file_exists path) then
    raise (Error (Printf.sprintf "PJRT plugin not found: %s" path));
  if Sys.is_directory path then
    raise (Error (Printf.sprintf "PJRT plugin path is a directory: %s" path));
  let actual = stamped_sha256 path in
  if not (List.mem actual pins) then
    raise
      (Error
         (Printf.sprintf
            "PJRT plugin sha256 mismatch at %s: expected one of %s, got %s" path
            (String.concat ", " pins) actual));
  last_verified := Some actual;
  path

let verify_at = verify_against pinned_sha256
let verify_gpu_at = verify_against pinned_gpu_sha256
let cached = ref None

let preflight () =
  match !cached with
  | Some p -> p
  | None ->
      let p = verify_at (resolve ()) in
      cached := Some p;
      p

let available () =
  match preflight () with _ -> true | exception Error _ -> false

let verified_sha256 () = if available () then !last_verified else None
let is_exact () = available () && !last_verified = Some exact_sha256
let gpu_configured () = Sys.getenv_opt gpu_env_var <> None
let gpu_cached = ref None

let gpu_preflight () =
  match !gpu_cached with
  | Some p -> p
  | None ->
      let p =
        verify_gpu_at (validate_named gpu_env_var (Sys.getenv_opt gpu_env_var))
      in
      gpu_cached := Some p;
      p
