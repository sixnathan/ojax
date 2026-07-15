# PJRT CPU plugin bit-parity feasibility evaluation (MODULE_MAP row 113)

Date: 2026-07-15. Session: sess-m5-r113-supervisor. Status: INVESTIGATE ONLY — no plugin build performed or committed, per the row mandate ("defer the build unless trivial"; it is not trivial).

## Problem

The dlopen'd CPU PJRT plugin (gomlx/gopjrt v0.9.1 `pjrt_c_api_cpu_plugin.so`, darwin-arm64) is built from a NEWER XLA revision than jaxlib 0.10.2. M4 row 92 measured the consequence: 1-ULP f32 divergence on FMA-contraction-sensitive e2e programs (matmul chains); integer programs bit-identical. Question: feasibility + cost of self-building the plugin from the jaxlib-0.10.2-matched XLA revision to restore float bit-exactness.

## Findings

1. **Matching XLA revision (VERIFIED).** jax tag `jax-v0.10.2` (= pinned commit `990e6a0b84138346e6a38785412f36356e0e5dc3`) pins XLA in `third_party/xla/revision.bzl`: `XLA_COMMIT = 5a9e73cbd92530cac2ac36f4736a774b2412afe2`, `XLA_SHA256 = 08a52210a04cd68d38f6201d56273d04ac0f8b4e0da9f72677c74a48cc637422`. There is no `jaxlib-v0.10.2` tag. CRITICAL: `workspace.bzl` at that tag applies FOUR patches on top of the pin — `xla_1f3fbb74.patch`, `xla_4c9ae741.patch`, `xla_llvm_pr_203230.patch`, `xla_9b348c6b.patch` (all in `third_party/xla/` of the jax repo). A faithful rebuild must apply them.
2. **Toolchain (VERIFIED).** bazel `7.7.0` (`.bazelversion` identical in jax@jax-v0.10.2 and xla@5a9e73c); use bazelisk.
3. **Target (VERIFIED).** `//xla/pjrt/c:pjrt_c_api_cpu_plugin.so` exists at the pinned commit: `xla_cc_binary`, `linkshared = True`, exports `GetPjrtApi` on macOS via `-Wl,-exported_symbol,_GetPjrtApi`. So YES, `GetPjrtApi` is exposable from a jaxlib-matched CPU build. Recipe: `./configure.py --backend CPU --os DARWIN` then `bazel build -c opt //xla/pjrt/c:pjrt_c_api_cpu_plugin.so`; artifact at `bazel-bin/xla/pjrt/c/pjrt_c_api_cpu_plugin.so`.
4. **Cost (PARTIALLY VERIFIED).** Artifact ~150–200 MB unstripped (the vendored gomlx darwin-arm64 build of this exact target is 184 MB). Wall-clock INFERRED ~1–3 h cold on Apple Silicon (XLA builds its bundled LLVM; lczero wiki reports "some hours" for the analogous GPU plugin target); bazel cache ~10–20 GB (inferred from XLA/TF build norms).
5. **macOS arm64 viability (VERIFIED).** gopjrt ships darwin-arm64 builds of exactly this target (v0.8.4+; v0.9.1 in use here), and jaxlib 0.10.2 ships macosx_11_0_arm64 wheels, so the revision builds on/for Apple Silicon. Historical macOS segfault (openxla/xla#19152, Nov 2024) was resolved before gopjrt v0.9.1.
6. **Trivial path CLOSED (VERIFIED by wheel inspection).** `jaxlib-0.10.2-cp312-macosx_11_0_arm64.whl` contains one 260 MB `jaxlib/libjax_common.dylib` exporting ONLY 27 `_Wrapped_PyInit_*` symbols — no `GetPjrtApi`; CPU runtime reachable only through the Python bindings. No CPU PJRT plugin wheel exists on PyPI under any name (`jax_plugins/` at the tag has only cuda/oneapi/rocm; jaxlib wheel tooling defines no CPU plugin wheel). Re-confirms the M4 row-73 preflight.
7. **Residual risk even after a matched rebuild.** Host-compiler differences vs Google's release toolchain should have minimal numeric impact (XLA JIT-compiles kernels via its bundled LLVM at the pinned revision), but bit-exactness can only be proven by running the row-92 e2e corpus against the rebuilt plugin.

## Go/no-go

**NO-GO for building now; GO is available and fully pinned for the future.** The build is feasible and well-trodden (gopjrt does exactly this) but NOT trivial (~1–3 h + 10–20 GB + bazel/patch orchestration), and the row mandates deferral unless trivial. The divergence it would close is bounded and documented (1-ULP f32 on FMA-contraction-sensitive programs only; integers bit-identical; all e2e goldens pass at current tolerances). If bit-exactness becomes required, execute: clone openxla/xla @ `5a9e73cbd92530cac2ac36f4736a774b2412afe2`; apply jax-v0.10.2's four `third_party/xla/*.patch`; bazelisk (bazel 7.7.0); `./configure.py --backend CPU --os DARWIN`; `bazel build -c opt //xla/pjrt/c:pjrt_c_api_cpu_plugin.so`; point `tools/pjrt.lock`/`OJAX_PJRT_PLUGIN` at the new artifact; re-run the row-92 e2e corpus and require byte-identical npz compare on the previously divergent f32 matmul programs.

## Sources

- https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/third_party/xla/revision.bzl
- https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/third_party/xla/workspace.bzl
- https://raw.githubusercontent.com/jax-ml/jax/jax-v0.10.2/.bazelversion
- https://github.com/openxla/xla/blob/5a9e73cbd92530cac2ac36f4736a774b2412afe2/xla/pjrt/c/BUILD
- https://github.com/jax-ml/jax/blob/jax-v0.10.2/jaxlib/tools/BUILD.bazel
- https://github.com/gomlx/gopjrt/releases/tag/v0.9.1 and docs/devel.md
- https://github.com/openxla/xla/issues/19152
- https://lczero.org/dev/wiki/xla-backend/
- Local: `vendor/pjrt/pjrt_c_api_cpu_plugin.so` (184 MB, exports `_GetPjrtApi`); jaxlib wheel inspection via `nm -gU`
