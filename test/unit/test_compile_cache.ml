module A = Ojax.Pjrt.Abi
module C = Ojax.Pjrt.Client
module E = Ojax.Pjrt.Executable
module D = Ojax.Pjrt.Discover

let program value =
  Printf.sprintf
    {|module @m {
  func.func public @main() -> (tensor<f32> {jax.result_info = "result"}) {
    %%c = stablehlo.constant dense<%s> : tensor<f32>
    return %%c : tensor<f32>
  }
}|}
    value

let directory =
  lazy
    (let dir = Filename.temp_file "ojax_compile_cache" "" in
     Sys.remove dir;
     Unix.putenv "OJAX_TEST_COMPILE_CACHE" dir;
     dir)

let entries dir =
  Sys.readdir dir |> Array.to_list
  |> List.filter (fun f -> not (String.starts_with ~prefix:"plugin-" f))

let client () =
  match D.find () with
  | Some path when Sys.file_exists path -> Some (C.create (A.open_plugin path))
  | _ -> None

let cleanup dir =
  Array.iter (fun f -> Sys.remove (Filename.concat dir f)) (Sys.readdir dir);
  Sys.rmdir dir

let test_store_and_load () =
  let dir = Lazy.force directory in
  match client () with
  | None -> ()
  | Some c ->
      let first = E.compile c (program "1.0") in
      Alcotest.(check int) "one entry" 1 (List.length (entries dir));
      let again = E.compile c (program "1.0") in
      Alcotest.(check int) "reused" 1 (List.length (entries dir));
      Alcotest.(check int) "outputs" (E.num_outputs first) (E.num_outputs again);
      ignore (E.compile c (program "2.0"));
      Alcotest.(check int) "new text" 2 (List.length (entries dir));
      let name = List.hd (entries dir) in
      let path = Filename.concat dir name in
      let oc = open_out_bin path in
      output_string oc "not the key";
      close_out oc;
      ignore (E.compile c (program "1.0"));
      ignore (E.compile c (program "2.0"));
      Alcotest.(check bool)
        "a damaged entry is replaced" true
        (In_channel.with_open_bin path In_channel.input_all <> "not the key");
      cleanup dir

let () =
  Alcotest.run "compile_cache"
    [
      ( "compile_cache",
        [ Alcotest.test_case "store and load" `Quick test_store_and_load ] );
    ]
