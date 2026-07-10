#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>

#include "pjrt_c_api.h"
#include "pjrt_abi_assert.h"

#include <caml/alloc.h>
#include <caml/callback.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/memory.h>
#include <caml/mlvalues.h>
#include <caml/threads.h>

typedef const PJRT_Api *(*ojax_get_pjrt_api_t)(void);

struct ojax_pjrt_plugin {
  void *handle;
  const PJRT_Api *api;
};

#define Ojax_plugin_val(v) ((struct ojax_pjrt_plugin *)Data_custom_val(v))

static void ojax_pjrt_plugin_finalize(value v_plugin) {
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->handle != NULL) {
    dlclose(p->handle);
    p->handle = NULL;
    p->api = NULL;
  }
}

static struct custom_operations ojax_pjrt_plugin_ops = {
    "ojax.pjrt.plugin",
    ojax_pjrt_plugin_finalize,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static void ojax_pjrt_raise(const char *msg) {
  static const value *exn = NULL;
  if (exn == NULL) exn = caml_named_value("ojax.pjrt.abi.error");
  if (exn == NULL)
    caml_failwith(msg);
  else
    caml_raise_with_string(*exn, msg);
}

CAMLprim value ojax_pjrt_open(value v_path) {
  CAMLparam1(v_path);
  CAMLlocal1(v_plugin);
  char *path = caml_stat_strdup(String_val(v_path));
  char errbuf[512];
  errbuf[0] = '\0';
  void *handle = NULL;
  const PJRT_Api *api = NULL;
  caml_release_runtime_system();
  handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    const char *e = dlerror();
    snprintf(errbuf, sizeof(errbuf), "dlopen failed: %s", e ? e : "unknown");
  } else {
    ojax_get_pjrt_api_t get =
        (ojax_get_pjrt_api_t)dlsym(handle, "GetPjrtApi");
    if (get == NULL) {
      const char *e = dlerror();
      snprintf(errbuf, sizeof(errbuf), "dlsym GetPjrtApi failed: %s",
               e ? e : "unknown");
      dlclose(handle);
      handle = NULL;
    } else {
      api = get();
      if (api == NULL) {
        snprintf(errbuf, sizeof(errbuf), "GetPjrtApi returned NULL");
        dlclose(handle);
        handle = NULL;
      }
    }
  }
  caml_acquire_runtime_system();
  caml_stat_free(path);
  if (handle == NULL) ojax_pjrt_raise(errbuf);
  if (api->pjrt_api_version.major_version != PJRT_API_MAJOR) {
    snprintf(errbuf, sizeof(errbuf),
             "PJRT_API_MAJOR mismatch: plugin %d, expected %d",
             api->pjrt_api_version.major_version, PJRT_API_MAJOR);
    dlclose(handle);
    ojax_pjrt_raise(errbuf);
  }
  v_plugin = caml_alloc_custom(&ojax_pjrt_plugin_ops,
                               sizeof(struct ojax_pjrt_plugin), 0, 1);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  p->handle = handle;
  p->api = api;
  CAMLreturn(v_plugin);
}

CAMLprim value ojax_pjrt_api_version(value v_plugin) {
  CAMLparam1(v_plugin);
  CAMLlocal1(v_tuple);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->api == NULL) ojax_pjrt_raise("api_version on closed plugin");
  v_tuple = caml_alloc_tuple(2);
  Store_field(v_tuple, 0, Val_int(p->api->pjrt_api_version.major_version));
  Store_field(v_tuple, 1, Val_int(p->api->pjrt_api_version.minor_version));
  CAMLreturn(v_tuple);
}

CAMLprim value ojax_pjrt_struct_size(value v_plugin) {
  CAMLparam1(v_plugin);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->api == NULL) ojax_pjrt_raise("struct_size on closed plugin");
  CAMLreturn(Val_long((intnat)p->api->struct_size));
}

CAMLprim value ojax_pjrt_close(value v_plugin) {
  CAMLparam1(v_plugin);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->handle != NULL) {
    void *h = p->handle;
    p->handle = NULL;
    p->api = NULL;
    caml_release_runtime_system();
    dlclose(h);
    caml_acquire_runtime_system();
  }
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_pjrt_maxrss(value v_unit) {
  CAMLparam1(v_unit);
  struct rusage ru;
  intnat bytes = 0;
  if (getrusage(RUSAGE_SELF, &ru) == 0) bytes = (intnat)ru.ru_maxrss;
  CAMLreturn(Val_long(bytes));
}
