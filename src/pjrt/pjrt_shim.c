#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>

#include "pjrt_c_api.h"
#include "pjrt_abi_assert.h"

#include <caml/alloc.h>
#include <caml/bigarray.h>
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

static void ojax_disable_fast_min_max(void) {
  const char *flag = "--xla_cpu_enable_fast_min_max=false";
  const char *current = getenv("XLA_FLAGS");
  if (current == NULL) {
    setenv("XLA_FLAGS", flag, 1);
  } else if (strstr(current, "xla_cpu_enable_fast_min_max") == NULL) {
    size_t n = strlen(current) + strlen(flag) + 2;
    char *joined = malloc(n);
    if (joined != NULL) {
      snprintf(joined, n, "%s %s", current, flag);
      setenv("XLA_FLAGS", joined, 1);
      free(joined);
    }
  }
}

CAMLprim value ojax_pjrt_open(value v_path) {
  CAMLparam1(v_path);
  CAMLlocal1(v_plugin);
  char *path = caml_stat_strdup(String_val(v_path));
  char errbuf[512];
  errbuf[0] = '\0';
  void *handle = NULL;
  const PJRT_Api *api = NULL;
  ojax_disable_fast_min_max();
  caml_release_runtime_system();
  handle = dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
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
    caml_release_runtime_system();
    dlclose(handle);
    caml_acquire_runtime_system();
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

CAMLprim value ojax_pjrt_library_loadable(value v_name) {
  CAMLparam1(v_name);
  void *handle = dlopen(String_val(v_name), RTLD_LAZY | RTLD_LOCAL);
  if (handle != NULL) dlclose(handle);
  CAMLreturn(Val_bool(handle != NULL));
}

CAMLprim value ojax_pjrt_maxrss(value v_unit) {
  CAMLparam1(v_unit);
  struct rusage ru;
  intnat bytes = 0;
  if (getrusage(RUSAGE_SELF, &ru) == 0) bytes = (intnat)ru.ru_maxrss;
  CAMLreturn(Val_long(bytes));
}

static void ojax_pjrt_check(const PJRT_Api *api, PJRT_Error *err) {
  if (err == NULL) return;
  PJRT_Error_Message_Args ma;
  memset(&ma, 0, sizeof ma);
  ma.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
  ma.error = err;
  api->PJRT_Error_Message(&ma);
  char buf[1024];
  size_t n = ma.message_size < 1023 ? ma.message_size : 1023;
  if (ma.message != NULL && n > 0)
    memcpy(buf, ma.message, n);
  else
    n = 0;
  buf[n] = '\0';
  PJRT_Error_Destroy_Args da;
  memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
  da.error = err;
  api->PJRT_Error_Destroy(&da);
  ojax_pjrt_raise(buf);
}

static PJRT_Error *ojax_pjrt_await(const PJRT_Api *api, PJRT_Event *event) {
  PJRT_Error *awaited = NULL;
  if (event == NULL) return NULL;
  PJRT_Event_Await_Args ea;
  memset(&ea, 0, sizeof ea);
  ea.struct_size = PJRT_Event_Await_Args_STRUCT_SIZE;
  ea.event = event;
  awaited = api->PJRT_Event_Await(&ea);
  PJRT_Event_Destroy_Args de;
  memset(&de, 0, sizeof de);
  de.struct_size = PJRT_Event_Destroy_Args_STRUCT_SIZE;
  de.event = event;
  api->PJRT_Event_Destroy(&de);
  return awaited;
}

struct ojax_pjrt_client {
  const PJRT_Api *api;
  PJRT_Client *client;
  PJRT_Device *device;
  PJRT_Device *const *devices;
  size_t num_devices;
};

#define Ojax_client_val(v) ((struct ojax_pjrt_client *)Data_custom_val(v))

static void ojax_pjrt_client_finalize(value v_client) {
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client != NULL) {
    PJRT_Client_Destroy_Args a;
    memset(&a, 0, sizeof a);
    a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    a.client = c->client;
    PJRT_Error *err = c->api->PJRT_Client_Destroy(&a);
    if (err != NULL) {
      PJRT_Error_Destroy_Args da;
      memset(&da, 0, sizeof da);
      da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
      da.error = err;
      c->api->PJRT_Error_Destroy(&da);
    }
    c->client = NULL;
    c->device = NULL;
    c->devices = NULL;
    c->num_devices = 0;
  }
}

static struct custom_operations ojax_pjrt_client_ops = {
    "ojax.pjrt.client",
    ojax_pjrt_client_finalize,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

struct ojax_pjrt_buffer {
  const PJRT_Api *api;
  PJRT_Buffer *buffer;
  atomic_int *shares;
};

#define Ojax_buffer_val(v) ((struct ojax_pjrt_buffer *)Data_custom_val(v))

static PJRT_Error *ojax_pjrt_buffer_release(const PJRT_Api *api,
                                            PJRT_Buffer *buffer,
                                            atomic_int *shares) {
  if (shares != NULL) {
    if (atomic_fetch_sub(shares, 1) > 1) return NULL;
    free(shares);
  }
  PJRT_Buffer_Destroy_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
  a.buffer = buffer;
  return api->PJRT_Buffer_Destroy(&a);
}

static void ojax_pjrt_buffer_finalize(value v_buffer) {
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer != NULL) {
    PJRT_Error *err = ojax_pjrt_buffer_release(b->api, b->buffer, b->shares);
    if (err != NULL) {
      PJRT_Error_Destroy_Args da;
      memset(&da, 0, sizeof da);
      da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
      da.error = err;
      b->api->PJRT_Error_Destroy(&da);
    }
    b->buffer = NULL;
  }
}

static struct custom_operations ojax_pjrt_buffer_ops = {
    "ojax.pjrt.buffer",
    ojax_pjrt_buffer_finalize,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static void ojax_pjrt_client_drain_destroy(const PJRT_Api *api,
                                           PJRT_Client *client) {
  if (client == NULL) return;
  PJRT_Client_Destroy_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
  a.client = client;
  PJRT_Error *err = api->PJRT_Client_Destroy(&a);
  if (err != NULL) {
    PJRT_Error_Destroy_Args da;
    memset(&da, 0, sizeof da);
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
  }
}

static size_t ojax_pjrt_fill_options(value v_options, PJRT_NamedValue *out) {
  size_t n = (size_t)Wosize_val(v_options);
  for (size_t i = 0; i < n; i++) {
    value v_pair = Field(v_options, i);
    value v_opt = Field(v_pair, 1);
    PJRT_NamedValue *nv = &out[i];
    memset(nv, 0, sizeof *nv);
    nv->struct_size = PJRT_NamedValue_STRUCT_SIZE;
    nv->name = caml_stat_strdup(String_val(Field(v_pair, 0)));
    nv->name_size = caml_string_length(Field(v_pair, 0));
    nv->value_size = 1;
    switch (Tag_val(v_opt)) {
      case 0:
        nv->type = PJRT_NamedValue_kString;
        nv->string_value = caml_stat_strdup(String_val(Field(v_opt, 0)));
        nv->value_size = caml_string_length(Field(v_opt, 0));
        break;
      case 1:
        nv->type = PJRT_NamedValue_kInt64;
        nv->int64_value = Int64_val(Field(v_opt, 0));
        break;
      case 2:
        nv->type = PJRT_NamedValue_kFloat;
        nv->float_value = (float)Double_val(Field(v_opt, 0));
        break;
      default:
        nv->type = PJRT_NamedValue_kBool;
        nv->bool_value = Bool_val(Field(v_opt, 0));
        break;
    }
  }
  return n;
}

static void ojax_pjrt_free_options(PJRT_NamedValue *opts, size_t n) {
  for (size_t i = 0; i < n; i++) {
    caml_stat_free((void *)opts[i].name);
    if (opts[i].type == PJRT_NamedValue_kString)
      caml_stat_free((void *)opts[i].string_value);
  }
  caml_stat_free(opts);
}

CAMLprim value ojax_pjrt_client_create(value v_plugin, value v_options) {
  CAMLparam2(v_plugin, v_options);
  CAMLlocal1(v_client);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->api == NULL) ojax_pjrt_raise("client_create on closed plugin");
  const PJRT_Api *api = p->api;
  size_t num_options = (size_t)Wosize_val(v_options);
  PJRT_NamedValue *options =
      caml_stat_alloc((num_options > 0 ? num_options : 1) *
                      sizeof(PJRT_NamedValue));
  ojax_pjrt_fill_options(v_options, options);

  PJRT_Plugin_Initialize_Args ia;
  memset(&ia, 0, sizeof ia);
  ia.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  PJRT_Client_Create_Args ca;
  memset(&ca, 0, sizeof ca);
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
  ca.create_options = options;
  ca.num_options = num_options;
  PJRT_Client_AddressableDevices_Args da;
  memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Client_AddressableDevices_Args_STRUCT_SIZE;

  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_Plugin_Initialize(&ia);
  PJRT_Error *cerr = NULL;
  PJRT_Error *derr = NULL;
  if (err == NULL) {
    cerr = api->PJRT_Client_Create(&ca);
    if (cerr == NULL) {
      da.client = ca.client;
      derr = api->PJRT_Client_AddressableDevices(&da);
    }
  }
  caml_acquire_runtime_system();
  ojax_pjrt_free_options(options, num_options);
  ojax_pjrt_check(api, err);
  ojax_pjrt_check(api, cerr);
  if (derr != NULL) {
    ojax_pjrt_client_drain_destroy(api, ca.client);
    ojax_pjrt_check(api, derr);
  }
  if (da.num_addressable_devices == 0) {
    ojax_pjrt_client_drain_destroy(api, ca.client);
    ojax_pjrt_raise("PJRT client has no addressable devices");
  }

  v_client = caml_alloc_custom(&ojax_pjrt_client_ops,
                               sizeof(struct ojax_pjrt_client), 0, 1);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  c->api = api;
  c->client = ca.client;
  c->device = da.addressable_devices[0];
  c->devices = da.addressable_devices;
  c->num_devices = da.num_addressable_devices;
  CAMLreturn(v_client);
}

CAMLprim value ojax_pjrt_client_platform_name(value v_client) {
  CAMLparam1(v_client);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("platform_name on closed client");
  PJRT_Client_PlatformName_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_PlatformName_Args_STRUCT_SIZE;
  a.client = c->client;
  ojax_pjrt_check(c->api, c->api->PJRT_Client_PlatformName(&a));
  CAMLreturn(caml_alloc_initialized_string(a.platform_name_size,
                                           a.platform_name));
}

CAMLprim value ojax_pjrt_client_platform_version(value v_client) {
  CAMLparam1(v_client);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("platform_version on closed client");
  PJRT_Client_PlatformVersion_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_PlatformVersion_Args_STRUCT_SIZE;
  a.client = c->client;
  ojax_pjrt_check(c->api, c->api->PJRT_Client_PlatformVersion(&a));
  CAMLreturn(caml_alloc_initialized_string(a.platform_version_size,
                                           a.platform_version));
}

CAMLprim value ojax_pjrt_client_device_count(value v_client) {
  CAMLparam1(v_client);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("device_count on closed client");
  CAMLreturn(Val_int((int)c->num_devices));
}

CAMLprim value ojax_pjrt_client_device_kind(value v_client) {
  CAMLparam1(v_client);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("device_kind on closed client");
  PJRT_Device_GetDescription_Args g;
  memset(&g, 0, sizeof g);
  g.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  g.device = c->device;
  ojax_pjrt_check(c->api, c->api->PJRT_Device_GetDescription(&g));
  PJRT_DeviceDescription_Kind_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
  a.device_description = g.device_description;
  ojax_pjrt_check(c->api, c->api->PJRT_DeviceDescription_Kind(&a));
  CAMLreturn(caml_alloc_initialized_string(a.device_kind_size, a.device_kind));
}

static const PJRT_DeviceDescription *ojax_pjrt_description(
    const PJRT_Api *api, PJRT_Device *device) {
  PJRT_Device_GetDescription_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Device_GetDescription_Args_STRUCT_SIZE;
  a.device = device;
  ojax_pjrt_check(api, api->PJRT_Device_GetDescription(&a));
  return a.device_description;
}

static int ojax_pjrt_device_id(const PJRT_Api *api, PJRT_Device *device) {
  PJRT_DeviceDescription_Id_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
  a.device_description = (PJRT_DeviceDescription *)ojax_pjrt_description(api, device);
  ojax_pjrt_check(api, api->PJRT_DeviceDescription_Id(&a));
  return a.id;
}

CAMLprim value ojax_pjrt_client_devices(value v_client) {
  CAMLparam1(v_client);
  CAMLlocal3(v_out, v_entry, v_kind);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("devices on closed client");
  const PJRT_Api *api = c->api;
  v_out = caml_alloc((mlsize_t)c->num_devices, 0);
  for (size_t i = 0; i < c->num_devices; i++) {
    const PJRT_DeviceDescription *d = ojax_pjrt_description(api, c->devices[i]);
    PJRT_DeviceDescription_Id_Args ia;
    memset(&ia, 0, sizeof ia);
    ia.struct_size = PJRT_DeviceDescription_Id_Args_STRUCT_SIZE;
    ia.device_description = (PJRT_DeviceDescription *)d;
    ojax_pjrt_check(api, api->PJRT_DeviceDescription_Id(&ia));
    PJRT_DeviceDescription_ProcessIndex_Args pa;
    memset(&pa, 0, sizeof pa);
    pa.struct_size = PJRT_DeviceDescription_ProcessIndex_Args_STRUCT_SIZE;
    pa.device_description = (PJRT_DeviceDescription *)d;
    ojax_pjrt_check(api, api->PJRT_DeviceDescription_ProcessIndex(&pa));
    PJRT_DeviceDescription_Kind_Args ka;
    memset(&ka, 0, sizeof ka);
    ka.struct_size = PJRT_DeviceDescription_Kind_Args_STRUCT_SIZE;
    ka.device_description = (PJRT_DeviceDescription *)d;
    ojax_pjrt_check(api, api->PJRT_DeviceDescription_Kind(&ka));
    v_kind = caml_alloc_initialized_string(ka.device_kind_size, ka.device_kind);
    v_entry = caml_alloc_tuple(3);
    Store_field(v_entry, 0, Val_int(ia.id));
    Store_field(v_entry, 1, Val_int(pa.process_index));
    Store_field(v_entry, 2, v_kind);
    Store_field(v_out, i, v_entry);
  }
  CAMLreturn(v_out);
}

static PJRT_Device *ojax_pjrt_find_device(struct ojax_pjrt_client *c, int id) {
  for (size_t i = 0; i < c->num_devices; i++)
    if (ojax_pjrt_device_id(c->api, c->devices[i]) == id) return c->devices[i];
  ojax_pjrt_raise("unknown PJRT device id");
  return NULL;
}

CAMLprim value ojax_pjrt_client_destroy(value v_client) {
  CAMLparam1(v_client);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client != NULL) {
    const PJRT_Api *api = c->api;
    PJRT_Client *client = c->client;
    c->client = NULL;
    c->device = NULL;
    c->devices = NULL;
    c->num_devices = 0;
    PJRT_Client_Destroy_Args a;
    memset(&a, 0, sizeof a);
    a.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    a.client = client;
    caml_release_runtime_system();
    PJRT_Error *err = api->PJRT_Client_Destroy(&a);
    caml_acquire_runtime_system();
    ojax_pjrt_check(api, err);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_pjrt_buffer_from_host(value v_client, value v_data,
                                          value v_type, value v_dims,
                                          value v_device) {
  CAMLparam5(v_client, v_data, v_type, v_dims, v_device);
  CAMLlocal1(v_buffer);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("buffer_from_host on closed client");
  const PJRT_Api *api = c->api;
  PJRT_Client *client = c->client;
  PJRT_Device *device = Long_val(v_device) < 0
                            ? c->device
                            : ojax_pjrt_find_device(c, Long_val(v_device));
  const void *data = Caml_ba_data_val(v_data);
  uintnat nbytes = caml_ba_byte_size(Caml_ba_array_val(v_data));
  int btype = Int_val(v_type);
  int nd = (int)Wosize_val(v_dims);
  int64_t *dims = NULL;
  if (nd > 0) {
    dims = caml_stat_alloc((size_t)nd * sizeof(int64_t));
    for (int i = 0; i < nd; i++) dims[i] = (int64_t)Long_val(Field(v_dims, i));
  }

  PJRT_Client_BufferFromHostBuffer_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
  a.client = client;
  a.data = data;
  a.type = (PJRT_Buffer_Type)btype;
  a.dims = dims;
  a.num_dims = (size_t)nd;
  a.byte_strides = NULL;
  a.num_byte_strides = 0;
  a.host_buffer_semantics =
      PJRT_HostBufferSemantics_kImmutableUntilTransferCompletes;
  a.device = device;
  a.memory = NULL;
  a.device_layout = NULL;

  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_Client_BufferFromHostBuffer(&a);
  PJRT_Error *awaited = NULL;
  if (err == NULL) awaited = ojax_pjrt_await(api, a.done_with_host_buffer);
  caml_acquire_runtime_system();
  if (dims != NULL) caml_stat_free(dims);
  ojax_pjrt_check(api, err);
  if (awaited != NULL) {
    PJRT_Buffer_Destroy_Args bd;
    memset(&bd, 0, sizeof bd);
    bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    bd.buffer = a.buffer;
    PJRT_Error *be = api->PJRT_Buffer_Destroy(&bd);
    if (be != NULL) {
      PJRT_Error_Destroy_Args de;
      memset(&de, 0, sizeof de);
      de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
      de.error = be;
      api->PJRT_Error_Destroy(&de);
    }
    ojax_pjrt_check(api, awaited);
  }

  v_buffer = caml_alloc_custom_mem(
      &ojax_pjrt_buffer_ops, sizeof(struct ojax_pjrt_buffer), (mlsize_t)nbytes);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  b->api = api;
  b->buffer = a.buffer;
  b->shares = NULL;
  CAMLreturn(v_buffer);
}

CAMLprim value ojax_pjrt_buffer_to_host(value v_buffer, value v_dst) {
  CAMLparam2(v_buffer, v_dst);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer == NULL) ojax_pjrt_raise("buffer_to_host on destroyed buffer");
  const PJRT_Api *api = b->api;
  PJRT_Buffer *buffer = b->buffer;
  void *dst = Caml_ba_data_val(v_dst);
  size_t dst_size = caml_ba_byte_size(Caml_ba_array_val(v_dst));

  PJRT_Buffer_ToHostBuffer_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
  a.src = buffer;
  a.host_layout = NULL;
  a.dst = dst;
  a.dst_size = dst_size;

  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_Buffer_ToHostBuffer(&a);
  PJRT_Error *awaited = NULL;
  if (err == NULL) awaited = ojax_pjrt_await(api, a.event);
  caml_acquire_runtime_system();
  ojax_pjrt_check(api, err);
  ojax_pjrt_check(api, awaited);
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_pjrt_buffer_belongs_to(value v_buffer, value v_client) {
  CAMLparam2(v_buffer, v_client);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  CAMLreturn(Val_bool(b->api == c->api));
}

CAMLprim value ojax_pjrt_buffer_element_type(value v_buffer) {
  CAMLparam1(v_buffer);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer == NULL) ojax_pjrt_raise("element_type on destroyed buffer");
  PJRT_Buffer_ElementType_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  a.buffer = b->buffer;
  PJRT_Error *err = b->api->PJRT_Buffer_ElementType(&a);
  ojax_pjrt_check(b->api, err);
  CAMLreturn(Val_int((int)a.type));
}

CAMLprim value ojax_pjrt_buffer_dimensions(value v_buffer) {
  CAMLparam1(v_buffer);
  CAMLlocal1(v_arr);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer == NULL) ojax_pjrt_raise("dimensions on destroyed buffer");
  PJRT_Buffer_Dimensions_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  a.buffer = b->buffer;
  PJRT_Error *err = b->api->PJRT_Buffer_Dimensions(&a);
  ojax_pjrt_check(b->api, err);
  int nd = (int)a.num_dims;
  v_arr = caml_alloc(nd, 0);
  for (int i = 0; i < nd; i++)
    Store_field(v_arr, i, Val_long((intnat)a.dims[i]));
  CAMLreturn(v_arr);
}

CAMLprim value ojax_pjrt_buffer_destroy(value v_buffer) {
  CAMLparam1(v_buffer);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer != NULL) {
    const PJRT_Api *api = b->api;
    PJRT_Buffer *buffer = b->buffer;
    atomic_int *shares = b->shares;
    b->buffer = NULL;
    b->shares = NULL;
    caml_release_runtime_system();
    PJRT_Error *err = ojax_pjrt_buffer_release(api, buffer, shares);
    caml_acquire_runtime_system();
    ojax_pjrt_check(api, err);
  }
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_pjrt_buffer_device(value v_buffer) {
  CAMLparam1(v_buffer);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer == NULL) ojax_pjrt_raise("device on destroyed buffer");
  PJRT_Buffer_Device_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_Device_Args_STRUCT_SIZE;
  a.buffer = b->buffer;
  ojax_pjrt_check(b->api, b->api->PJRT_Buffer_Device(&a));
  CAMLreturn(Val_int(ojax_pjrt_device_id(b->api, a.device)));
}

struct ojax_pjrt_executable {
  const PJRT_Api *api;
  PJRT_LoadedExecutable *executable;
  int num_outputs;
};

#define Ojax_executable_val(v) \
  ((struct ojax_pjrt_executable *)Data_custom_val(v))

static void ojax_pjrt_loaded_destroy(const PJRT_Api *api,
                                     PJRT_LoadedExecutable *exec) {
  PJRT_LoadedExecutable_Destroy_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
  a.executable = exec;
  PJRT_Error *err = api->PJRT_LoadedExecutable_Destroy(&a);
  if (err != NULL) {
    PJRT_Error_Destroy_Args da;
    memset(&da, 0, sizeof da);
    da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    da.error = err;
    api->PJRT_Error_Destroy(&da);
  }
}

static void ojax_pjrt_executable_finalize(value v_exec) {
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable != NULL) {
    ojax_pjrt_loaded_destroy(e->api, e->executable);
    e->executable = NULL;
  }
}

static struct custom_operations ojax_pjrt_executable_ops = {
    "ojax.pjrt.executable",
    ojax_pjrt_executable_finalize,
    custom_compare_default,
    custom_hash_default,
    custom_serialize_default,
    custom_deserialize_default,
    custom_compare_ext_default,
    custom_fixed_length_default};

static value ojax_pjrt_wrap_loaded(const PJRT_Api *api,
                                   PJRT_LoadedExecutable *loaded) {
  CAMLparam0();
  CAMLlocal1(v_exec);
  PJRT_LoadedExecutable_GetExecutable_Args ga;
  memset(&ga, 0, sizeof ga);
  ga.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ga.loaded_executable = loaded;
  PJRT_Error *gerr = api->PJRT_LoadedExecutable_GetExecutable(&ga);
  if (gerr != NULL) {
    ojax_pjrt_loaded_destroy(api, loaded);
    ojax_pjrt_check(api, gerr);
  }
  PJRT_Executable_NumOutputs_Args na;
  memset(&na, 0, sizeof na);
  na.struct_size = PJRT_Executable_NumOutputs_Args_STRUCT_SIZE;
  na.executable = ga.executable;
  PJRT_Error *nerr = api->PJRT_Executable_NumOutputs(&na);
  int num_outputs = (int)na.num_outputs;
  PJRT_Executable_Destroy_Args da;
  memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  da.executable = ga.executable;
  PJRT_Error *derr = api->PJRT_Executable_Destroy(&da);
  if (derr != NULL) {
    PJRT_Error_Destroy_Args de;
    memset(&de, 0, sizeof de);
    de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    de.error = derr;
    api->PJRT_Error_Destroy(&de);
  }
  if (nerr != NULL) {
    ojax_pjrt_loaded_destroy(api, loaded);
    ojax_pjrt_check(api, nerr);
  }

  v_exec = caml_alloc_custom(&ojax_pjrt_executable_ops,
                             sizeof(struct ojax_pjrt_executable), 0, 1);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  e->api = api;
  e->executable = loaded;
  e->num_outputs = num_outputs;
  CAMLreturn(v_exec);
}


CAMLprim value ojax_pjrt_compile(value v_client, value v_code,
                                 value v_options) {
  CAMLparam3(v_client, v_code, v_options);
  CAMLlocal1(v_exec);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("compile on closed client");
  const PJRT_Api *api = c->api;
  mlsize_t code_len = caml_string_length(v_code);
  mlsize_t opt_len = caml_string_length(v_options);
  char *code = caml_stat_alloc(code_len > 0 ? code_len : 1);
  memcpy(code, String_val(v_code), code_len);
  char *opts = caml_stat_alloc(opt_len > 0 ? opt_len : 1);
  memcpy(opts, String_val(v_options), opt_len);
  static const char fmt[] = "mlir";

  PJRT_Program program;
  memset(&program, 0, sizeof program);
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  program.code = code;
  program.code_size = (size_t)code_len;
  program.format = fmt;
  program.format_size = sizeof(fmt) - 1;

  PJRT_Client_Compile_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  a.client = c->client;
  a.program = &program;
  a.compile_options = opts;
  a.compile_options_size = (size_t)opt_len;

  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_Client_Compile(&a);
  caml_acquire_runtime_system();
  caml_stat_free(code);
  caml_stat_free(opts);
  ojax_pjrt_check(api, err);

  CAMLreturn(ojax_pjrt_wrap_loaded(api, a.executable));
}

CAMLprim value ojax_pjrt_executable_serialize(value v_exec) {
  CAMLparam1(v_exec);
  CAMLlocal1(v_out);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable == NULL) ojax_pjrt_raise("serialize on destroyed executable");
  const PJRT_Api *api = e->api;
  PJRT_LoadedExecutable_GetExecutable_Args ga;
  memset(&ga, 0, sizeof ga);
  ga.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ga.loaded_executable = e->executable;
  ojax_pjrt_check(api, api->PJRT_LoadedExecutable_GetExecutable(&ga));
  PJRT_Executable_Serialize_Args sa;
  memset(&sa, 0, sizeof sa);
  sa.struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE;
  sa.executable = ga.executable;
  caml_release_runtime_system();
  PJRT_Error *serr = api->PJRT_Executable_Serialize(&sa);
  caml_acquire_runtime_system();
  PJRT_Executable_Destroy_Args da;
  memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  da.executable = ga.executable;
  PJRT_Error *derr = api->PJRT_Executable_Destroy(&da);
  if (derr != NULL) {
    PJRT_Error_Destroy_Args de;
    memset(&de, 0, sizeof de);
    de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    de.error = derr;
    api->PJRT_Error_Destroy(&de);
  }
  ojax_pjrt_check(api, serr);
  v_out = caml_alloc_initialized_string(sa.serialized_bytes_size,
                                        sa.serialized_bytes);
  sa.serialized_executable_deleter(sa.serialized_executable);
  CAMLreturn(v_out);
}

CAMLprim value ojax_pjrt_executable_deserialize(value v_client, value v_bytes) {
  CAMLparam2(v_client, v_bytes);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("deserialize on closed client");
  const PJRT_Api *api = c->api;
  mlsize_t len = caml_string_length(v_bytes);
  char *bytes = caml_stat_alloc(len > 0 ? len : 1);
  memcpy(bytes, String_val(v_bytes), len);
  PJRT_Executable_DeserializeAndLoad_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
  a.client = c->client;
  a.serialized_executable = bytes;
  a.serialized_executable_size = (size_t)len;
  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_Executable_DeserializeAndLoad(&a);
  caml_acquire_runtime_system();
  caml_stat_free(bytes);
  ojax_pjrt_check(api, err);
  CAMLreturn(ojax_pjrt_wrap_loaded(api, a.loaded_executable));
}

CAMLprim value ojax_pjrt_executable_num_outputs(value v_exec) {
  CAMLparam1(v_exec);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable == NULL) ojax_pjrt_raise("num_outputs on destroyed executable");
  CAMLreturn(Val_int(e->num_outputs));
}

CAMLprim value ojax_pjrt_executable_destroy(value v_exec) {
  CAMLparam1(v_exec);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable != NULL) {
    const PJRT_Api *api = e->api;
    PJRT_LoadedExecutable *exec = e->executable;
    e->executable = NULL;
    caml_release_runtime_system();
    ojax_pjrt_loaded_destroy(api, exec);
    caml_acquire_runtime_system();
  }
  CAMLreturn(Val_unit);
}

CAMLprim value ojax_pjrt_execute(value v_exec, value v_args) {
  CAMLparam2(v_exec, v_args);
  CAMLlocal2(v_out, v_buf);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable == NULL) ojax_pjrt_raise("execute on destroyed executable");
  const PJRT_Api *api = e->api;
  int num_args = (int)Wosize_val(v_args);
  int num_outputs = e->num_outputs;

  PJRT_Buffer **args = NULL;
  if (num_args > 0)
    args = caml_stat_alloc((size_t)num_args * sizeof(PJRT_Buffer *));
  for (int i = 0; i < num_args; i++) {
    struct ojax_pjrt_buffer *b = Ojax_buffer_val(Field(v_args, i));
    if (b->buffer == NULL) {
      if (args != NULL) caml_stat_free(args);
      ojax_pjrt_raise("execute given destroyed input buffer");
    }
    args[i] = b->buffer;
  }
  PJRT_Buffer **outputs =
      caml_stat_alloc((size_t)(num_outputs > 0 ? num_outputs : 1) *
                      sizeof(PJRT_Buffer *));
  for (int i = 0; i < num_outputs; i++) outputs[i] = NULL;

  PJRT_Buffer *const *argument_lists[1];
  argument_lists[0] = args;
  PJRT_Buffer **output_lists[1];
  output_lists[0] = outputs;
  PJRT_Event *events[1];
  events[0] = NULL;

  PJRT_ExecuteOptions opts;
  memset(&opts, 0, sizeof opts);
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;

  PJRT_LoadedExecutable_Execute_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  a.executable = e->executable;
  a.options = &opts;
  a.argument_lists = argument_lists;
  a.num_devices = 1;
  a.num_args = (size_t)num_args;
  a.output_lists = output_lists;
  a.device_complete_events = events;
  a.execute_device = NULL;

  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_LoadedExecutable_Execute(&a);
  PJRT_Error *awaited = NULL;
  if (err == NULL) awaited = ojax_pjrt_await(api, events[0]);
  caml_acquire_runtime_system();

  if (args != NULL) caml_stat_free(args);
  if (err != NULL || awaited != NULL) {
    for (int i = 0; i < num_outputs; i++) {
      if (outputs[i] != NULL) {
        PJRT_Buffer_Destroy_Args bd;
        memset(&bd, 0, sizeof bd);
        bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        bd.buffer = outputs[i];
        PJRT_Error *be = api->PJRT_Buffer_Destroy(&bd);
        if (be != NULL) {
          PJRT_Error_Destroy_Args de;
          memset(&de, 0, sizeof de);
          de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
          de.error = be;
          api->PJRT_Error_Destroy(&de);
        }
      }
    }
    caml_stat_free(outputs);
    ojax_pjrt_check(api, err);
    ojax_pjrt_check(api, awaited);
  }

  v_out = caml_alloc(num_outputs, 0);
  for (int i = 0; i < num_outputs; i++) {
    mlsize_t nbytes = 0;
    PJRT_Buffer_OnDeviceSizeInBytes_Args sa;
    memset(&sa, 0, sizeof sa);
    sa.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
    sa.buffer = outputs[i];
    PJRT_Error *serr = api->PJRT_Buffer_OnDeviceSizeInBytes(&sa);
    if (serr == NULL) {
      nbytes = (mlsize_t)sa.on_device_size_in_bytes;
    } else {
      PJRT_Error_Destroy_Args de;
      memset(&de, 0, sizeof de);
      de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
      de.error = serr;
      api->PJRT_Error_Destroy(&de);
    }
    v_buf = caml_alloc_custom_mem(&ojax_pjrt_buffer_ops,
                                  sizeof(struct ojax_pjrt_buffer), nbytes);
    struct ojax_pjrt_buffer *ob = Ojax_buffer_val(v_buf);
    ob->api = api;
    ob->buffer = outputs[i];
    ob->shares = NULL;
    Store_field(v_out, i, v_buf);
  }
  caml_stat_free(outputs);
  CAMLreturn(v_out);
}

CAMLprim value ojax_ffi_dlopen(value v_path) {
  CAMLparam1(v_path);
  void *handle = dlopen(String_val(v_path), RTLD_NOW | RTLD_LOCAL);
  if (handle == NULL) {
    char errbuf[512];
    const char *e = dlerror();
    snprintf(errbuf, sizeof errbuf, "dlopen failed: %s", e ? e : "unknown");
    ojax_pjrt_raise(errbuf);
  }
  CAMLreturn(caml_copy_nativeint((intnat)handle));
}

CAMLprim value ojax_ffi_dlsym(value v_handle, value v_name) {
  CAMLparam2(v_handle, v_name);
  dlerror();
  void *sym = dlsym((void *)Nativeint_val(v_handle), String_val(v_name));
  if (sym == NULL) {
    char errbuf[512];
    const char *e = dlerror();
    snprintf(errbuf, sizeof errbuf, "dlsym %s failed: %s", String_val(v_name),
             e ? e : "symbol is NULL");
    ojax_pjrt_raise(errbuf);
  }
  CAMLreturn(caml_copy_nativeint((intnat)sym));
}

typedef struct {
  size_t struct_size;
  PJRT_Extension_Base *extension_start;
  const char *type_name;
  size_t type_name_size;
  int64_t type_id;
  const void *const *type_info;
} ojax_pjrt_ffi_type_args;

typedef struct {
  size_t struct_size;
  const char *target_name;
  size_t target_name_size;
  void *handler;
  const char *platform_name;
  size_t platform_name_size;
  int traits;
} ojax_pjrt_ffi_handler_args;

typedef struct {
  PJRT_Extension_Base base;
  PJRT_Error *(*type_register)(ojax_pjrt_ffi_type_args *);
  PJRT_Error *(*user_data_add)(void *);
  PJRT_Error *(*register_handler)(ojax_pjrt_ffi_handler_args *);
} ojax_pjrt_ffi_extension;

_Static_assert(offsetof(ojax_pjrt_ffi_type_args, type_info) + sizeof(void *) ==
                   48,
               "PJRT_FFI_Type_Register_Args size drift");
_Static_assert(offsetof(ojax_pjrt_ffi_handler_args, traits) + sizeof(int) == 52,
               "PJRT_FFI_Register_Handler_Args size drift");

static const ojax_pjrt_ffi_extension *ojax_pjrt_find_ffi(const PJRT_Api *api) {
  for (PJRT_Extension_Base *e = api->extension_start; e != NULL; e = e->next)
    if (e->type == PJRT_Extension_Type_FFI)
      return (const ojax_pjrt_ffi_extension *)e;
  ojax_pjrt_raise("PJRT plugin does not provide the FFI extension");
  return NULL;
}

CAMLprim value ojax_pjrt_ffi_register_handler(value v_plugin, value v_name,
                                              value v_platform,
                                              value v_handler) {
  CAMLparam4(v_plugin, v_name, v_platform, v_handler);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->api == NULL) ojax_pjrt_raise("register_handler on closed plugin");
  const ojax_pjrt_ffi_extension *ext = ojax_pjrt_find_ffi(p->api);
  ojax_pjrt_ffi_handler_args a;
  memset(&a, 0, sizeof a);
  a.struct_size = 52;
  a.target_name = String_val(v_name);
  a.target_name_size = caml_string_length(v_name);
  a.handler = (void *)Nativeint_val(v_handler);
  a.platform_name = String_val(v_platform);
  a.platform_name_size = caml_string_length(v_platform);
  ojax_pjrt_check(p->api, ext->register_handler(&a));
  CAMLreturn(Val_unit);
}

typedef struct {
  size_t struct_size;
  void *extension_start;
  void (*deleter)(void *);
} ojax_xla_ffi_type_info;

CAMLprim value ojax_pjrt_ffi_register_type(value v_plugin, value v_name,
                                           value v_type_id, value v_type_info) {
  CAMLparam4(v_plugin, v_name, v_type_id, v_type_info);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->api == NULL) ojax_pjrt_raise("register_type on closed plugin");
  const ojax_pjrt_ffi_extension *ext = ojax_pjrt_find_ffi(p->api);
  int64_t *type_id = (int64_t *)Nativeint_val(v_type_id);
  const ojax_xla_ffi_type_info *info =
      (const ojax_xla_ffi_type_info *)Nativeint_val(v_type_info);
  const void *pjrt_info[3] = {info != NULL ? (const void *)info->deleter : NULL,
                              NULL, NULL};
  ojax_pjrt_ffi_type_args a;
  memset(&a, 0, sizeof a);
  a.struct_size = sizeof a;
  a.type_name = String_val(v_name);
  a.type_name_size = caml_string_length(v_name);
  a.type_id = *type_id;
  a.type_info = pjrt_info;
  ojax_pjrt_check(p->api, ext->type_register(&a));
  *type_id = a.type_id;
  CAMLreturn(Val_unit);
}

typedef struct {
  int32_t device_type;
  int32_t device_id;
} ojax_dl_device;

typedef struct {
  uint8_t code;
  uint8_t bits;
  uint16_t lanes;
} ojax_dl_data_type;

typedef struct {
  void *data;
  ojax_dl_device device;
  int32_t ndim;
  ojax_dl_data_type dtype;
  int64_t *shape;
  int64_t *strides;
  uint64_t byte_offset;
} ojax_dl_tensor;

typedef struct ojax_dl_managed_tensor {
  ojax_dl_tensor dl_tensor;
  void *manager_ctx;
  void (*deleter)(struct ojax_dl_managed_tensor *);
} ojax_dl_managed_tensor;

struct ojax_dlpack_holder {
  ojax_dl_managed_tensor tensor;
  const PJRT_Api *api;
  PJRT_Buffer *buffer;
  atomic_int *shares;
  int64_t dims[];
};

static void ojax_dlpack_holder_delete(ojax_dl_managed_tensor *t) {
  struct ojax_dlpack_holder *h = (struct ojax_dlpack_holder *)t->manager_ctx;
  PJRT_Buffer_DecreaseExternalReferenceCount_Args d;
  memset(&d, 0, sizeof d);
  d.struct_size = PJRT_Buffer_DecreaseExternalReferenceCount_Args_STRUCT_SIZE;
  d.buffer = h->buffer;
  PJRT_Error *err = h->api->PJRT_Buffer_DecreaseExternalReferenceCount(&d);
  PJRT_Error *rerr = ojax_pjrt_buffer_release(h->api, h->buffer, h->shares);
  PJRT_Error *errs[2] = {err, rerr};
  for (int i = 0; i < 2; i++) {
    if (errs[i] != NULL) {
      PJRT_Error_Destroy_Args da;
      memset(&da, 0, sizeof da);
      da.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
      da.error = errs[i];
      h->api->PJRT_Error_Destroy(&da);
    }
  }
  free(h);
}

static int ojax_dlpack_of_buffer_type(PJRT_Buffer_Type t, uint8_t *code,
                                      uint8_t *bits) {
  switch (t) {
    case PJRT_Buffer_Type_PRED: *code = 6; *bits = 8; return 1;
    case PJRT_Buffer_Type_S8: *code = 0; *bits = 8; return 1;
    case PJRT_Buffer_Type_S16: *code = 0; *bits = 16; return 1;
    case PJRT_Buffer_Type_S32: *code = 0; *bits = 32; return 1;
    case PJRT_Buffer_Type_S64: *code = 0; *bits = 64; return 1;
    case PJRT_Buffer_Type_U8: *code = 1; *bits = 8; return 1;
    case PJRT_Buffer_Type_U16: *code = 1; *bits = 16; return 1;
    case PJRT_Buffer_Type_U32: *code = 1; *bits = 32; return 1;
    case PJRT_Buffer_Type_U64: *code = 1; *bits = 64; return 1;
    case PJRT_Buffer_Type_F16: *code = 2; *bits = 16; return 1;
    case PJRT_Buffer_Type_F32: *code = 2; *bits = 32; return 1;
    case PJRT_Buffer_Type_F64: *code = 2; *bits = 64; return 1;
    case PJRT_Buffer_Type_BF16: *code = 4; *bits = 16; return 1;
    case PJRT_Buffer_Type_C64: *code = 5; *bits = 64; return 1;
    case PJRT_Buffer_Type_C128: *code = 5; *bits = 128; return 1;
    default: return 0;
  }
}

CAMLprim value ojax_pjrt_buffer_unsafe_pointer(value v_buffer) {
  CAMLparam1(v_buffer);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer == NULL) ojax_pjrt_raise("unsafe_pointer on destroyed buffer");
  PJRT_Buffer_UnsafePointer_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE;
  a.buffer = b->buffer;
  ojax_pjrt_check(b->api, b->api->PJRT_Buffer_UnsafePointer(&a));
  CAMLreturn(caml_copy_nativeint((intnat)a.buffer_pointer));
}

CAMLprim value ojax_pjrt_buffer_to_dlpack(value v_buffer) {
  CAMLparam1(v_buffer);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer == NULL) ojax_pjrt_raise("to_dlpack on destroyed buffer");
  const PJRT_Api *api = b->api;

  PJRT_Buffer_ReadyEvent_Args ra;
  memset(&ra, 0, sizeof ra);
  ra.struct_size = PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
  ra.buffer = b->buffer;
  ojax_pjrt_check(api, api->PJRT_Buffer_ReadyEvent(&ra));
  caml_release_runtime_system();
  PJRT_Error *awaited = ojax_pjrt_await(api, ra.event);
  caml_acquire_runtime_system();
  ojax_pjrt_check(api, awaited);

  PJRT_Buffer_ElementType_Args ta;
  memset(&ta, 0, sizeof ta);
  ta.struct_size = PJRT_Buffer_ElementType_Args_STRUCT_SIZE;
  ta.buffer = b->buffer;
  ojax_pjrt_check(api, api->PJRT_Buffer_ElementType(&ta));
  uint8_t code = 0, bits = 0;
  if (!ojax_dlpack_of_buffer_type(ta.type, &code, &bits))
    ojax_pjrt_raise("to_dlpack: unsupported element type");

  PJRT_Buffer_Dimensions_Args da;
  memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
  da.buffer = b->buffer;
  ojax_pjrt_check(api, api->PJRT_Buffer_Dimensions(&da));

  PJRT_Buffer_UnsafePointer_Args pa;
  memset(&pa, 0, sizeof pa);
  pa.struct_size = PJRT_Buffer_UnsafePointer_Args_STRUCT_SIZE;
  pa.buffer = b->buffer;
  ojax_pjrt_check(api, api->PJRT_Buffer_UnsafePointer(&pa));

  PJRT_Buffer_IncreaseExternalReferenceCount_Args ia;
  memset(&ia, 0, sizeof ia);
  ia.struct_size = PJRT_Buffer_IncreaseExternalReferenceCount_Args_STRUCT_SIZE;
  ia.buffer = b->buffer;
  ojax_pjrt_check(api, api->PJRT_Buffer_IncreaseExternalReferenceCount(&ia));

  size_t nd = da.num_dims;
  struct ojax_dlpack_holder *h =
      malloc(sizeof *h + (2 * nd + 1) * sizeof(int64_t));
  if (h == NULL) ojax_pjrt_raise("to_dlpack: out of memory");
  if (b->shares == NULL) {
    b->shares = malloc(sizeof(atomic_int));
    if (b->shares == NULL) ojax_pjrt_raise("to_dlpack: out of memory");
    atomic_init(b->shares, 1);
  }
  atomic_fetch_add(b->shares, 1);
  h->api = api;
  h->buffer = b->buffer;
  h->shares = b->shares;
  int64_t stride = 1;
  for (size_t i = nd; i > 0; i--) {
    h->dims[i - 1] = da.dims[i - 1];
    h->dims[nd + i - 1] = stride;
    stride *= da.dims[i - 1];
  }
  memset(&h->tensor, 0, sizeof h->tensor);
  h->tensor.dl_tensor.data = (void *)pa.buffer_pointer;
  h->tensor.dl_tensor.device.device_type = 1;
  h->tensor.dl_tensor.device.device_id = 0;
  h->tensor.dl_tensor.ndim = (int32_t)nd;
  h->tensor.dl_tensor.dtype.code = code;
  h->tensor.dl_tensor.dtype.bits = bits;
  h->tensor.dl_tensor.dtype.lanes = 1;
  h->tensor.dl_tensor.shape = h->dims;
  h->tensor.dl_tensor.strides = nd > 0 ? h->dims + nd : NULL;
  h->tensor.dl_tensor.byte_offset = 0;
  h->tensor.manager_ctx = h;
  h->tensor.deleter = ojax_dlpack_holder_delete;
  CAMLreturn(caml_copy_nativeint((intnat)&h->tensor));
}

CAMLprim value ojax_pjrt_dlpack_delete(value v_ptr) {
  CAMLparam1(v_ptr);
  ojax_dl_managed_tensor *t = (ojax_dl_managed_tensor *)Nativeint_val(v_ptr);
  if (t != NULL && t->deleter != NULL) t->deleter(t);
  CAMLreturn(Val_unit);
}

static int ojax_dlpack_to_buffer_type(ojax_dl_data_type d, PJRT_Buffer_Type *out,
                                      char *errbuf, size_t n) {
  if (d.lanes != 1) {
    snprintf(errbuf, n,
             "UNIMPLEMENTED: DLPack types with lanes != 1 not implemented, got "
             "%d",
             d.lanes);
    return 0;
  }
  const char *kind = NULL;
  switch (d.code) {
    case 6:
      if (d.bits == 8) {
        *out = PJRT_Buffer_Type_PRED;
        return 1;
      }
      snprintf(errbuf, n,
               "UNIMPLEMENTED: Only 8-bit DLPack booleans are supported, got "
               "%d bits",
               d.bits);
      return 0;
    case 0:
      kind = "integer";
      if (d.bits == 8) *out = PJRT_Buffer_Type_S8;
      else if (d.bits == 16) *out = PJRT_Buffer_Type_S16;
      else if (d.bits == 32) *out = PJRT_Buffer_Type_S32;
      else if (d.bits == 64) *out = PJRT_Buffer_Type_S64;
      else break;
      return 1;
    case 1:
      kind = "unsigned integer";
      if (d.bits == 8) *out = PJRT_Buffer_Type_U8;
      else if (d.bits == 16) *out = PJRT_Buffer_Type_U16;
      else if (d.bits == 32) *out = PJRT_Buffer_Type_U32;
      else if (d.bits == 64) *out = PJRT_Buffer_Type_U64;
      else break;
      return 1;
    case 2:
      kind = "float";
      if (d.bits == 16) *out = PJRT_Buffer_Type_F16;
      else if (d.bits == 32) *out = PJRT_Buffer_Type_F32;
      else if (d.bits == 64) *out = PJRT_Buffer_Type_F64;
      else break;
      return 1;
    case 4:
      kind = "bfloat";
      if (d.bits == 16) *out = PJRT_Buffer_Type_BF16;
      else break;
      return 1;
    case 5:
      kind = "complex";
      if (d.bits == 64) *out = PJRT_Buffer_Type_C64;
      else if (d.bits == 128) *out = PJRT_Buffer_Type_C128;
      else break;
      return 1;
    default:
      snprintf(errbuf, n, "UNIMPLEMENTED: Unknown or invalid DLPack type code %d",
               d.code);
      return 0;
  }
  snprintf(errbuf, n,
           "UNIMPLEMENTED: Invalid or unsupported DLPack %s width: %d bits",
           kind, d.bits);
  return 0;
}

static void ojax_dlpack_view_deleted(void *ptr, void *arg) {
  (void)ptr;
  ojax_dl_managed_tensor *t = (ojax_dl_managed_tensor *)arg;
  if (t->deleter != NULL) t->deleter(t);
}

static int ojax_dlpack_layout(const ojax_dl_tensor *t, int *dense_row_major,
                              int64_t *byte_strides, int elt_bytes,
                              char *errbuf, size_t n) {
  int nd = t->ndim;
  *dense_row_major = 1;
  if (t->strides == NULL) {
    int64_t s = elt_bytes;
    for (int i = nd - 1; i >= 0; i--) {
      byte_strides[i] = s;
      s *= t->shape[i];
    }
    return 1;
  }
  int empty = 0;
  for (int i = 0; i < nd; i++) {
    if (t->shape[i] == 0) empty = 1;
    byte_strides[i] = t->strides[i] * elt_bytes;
  }
  if (empty) return 1;
  int order[nd > 0 ? nd : 1];
  for (int i = 0; i < nd; i++) order[i] = i;
  for (int i = 1; i < nd; i++) {
    int x = order[i];
    int j = i;
    while (j > 0) {
      int y = order[j - 1];
      int before = t->strides[x] < t->strides[y] ||
                   (t->strides[x] == t->strides[y] && y < x);
      if (!before) break;
      order[j] = y;
      j--;
    }
    order[j] = x;
  }
  int64_t stride = 1;
  int compact = 1;
  for (int i = 0; i < nd; i++) {
    int d = order[i];
    if (t->shape[d] > 1 && t->strides[d] != stride) compact = 0;
    stride *= t->shape[d];
  }
  if (!compact) {
    int off = snprintf(errbuf, n,
                       "UNIMPLEMENTED: Only DLPack tensors with trivial "
                       "(compact) striding are supported; i.e., tensors whose "
                       "striding represents a transposition of the underlying "
                       "buffer but not broadcasting. Dimensions were: [");
    for (int i = 0; i < nd && (size_t)off < n; i++)
      off += snprintf(errbuf + off, n - off, "%s%lld", i ? "," : "",
                      (long long)t->shape[i]);
    if ((size_t)off < n) off += snprintf(errbuf + off, n - off, "], strides were [");
    for (int i = 0; i < nd && (size_t)off < n; i++)
      off += snprintf(errbuf + off, n - off, "%s%lld", i ? "," : "",
                      (long long)t->strides[i]);
    if ((size_t)off < n) snprintf(errbuf + off, n - off, "].");
    return 0;
  }
  int64_t expected = 1;
  for (int i = nd - 1; i >= 0; i--) {
    byte_strides[i] = t->strides[i] * elt_bytes;
    if (t->shape[i] > 1 && t->strides[i] != expected) *dense_row_major = 0;
    expected *= t->shape[i];
  }
  return 1;
}

#define OJAX_DLPACK_COPY_NEVER 0
#define OJAX_DLPACK_COPY_ALWAYS 1

CAMLprim value ojax_pjrt_buffer_of_dlpack(value v_client, value v_ptr,
                                          value v_copy) {
  CAMLparam3(v_client, v_ptr, v_copy);
  CAMLlocal1(v_buffer);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("of_dlpack on closed client");
  const PJRT_Api *api = c->api;
  ojax_dl_managed_tensor *m = (ojax_dl_managed_tensor *)Nativeint_val(v_ptr);
  const ojax_dl_tensor *t = &m->dl_tensor;
  int copy = Int_val(v_copy);
  char errbuf[1024];
  errbuf[0] = '\0';

  PJRT_Buffer_Type btype = PJRT_Buffer_Type_INVALID;
  if (!ojax_dlpack_to_buffer_type(t->dtype, &btype, errbuf, sizeof errbuf))
    ojax_pjrt_raise(errbuf);
  int elt_bytes = t->dtype.bits / 8;
  int nd = t->ndim;
  int64_t byte_strides[nd > 0 ? nd : 1];
  int row_major = 1;
  if (!ojax_dlpack_layout(t, &row_major, byte_strides, elt_bytes, errbuf,
                          sizeof errbuf))
    ojax_pjrt_raise(errbuf);
  char *data = (char *)t->data + t->byte_offset;
  int aligned = ((uintptr_t)data % 64) == 0;
  if (copy == OJAX_DLPACK_COPY_NEVER && !aligned)
    ojax_pjrt_raise(
        "ojax.dlpack.value_error:Specified input which requires a copy since "
        "the source data buffer is not aligned to 64 bytes.  However "
        "copy=False. Set copy=True or copy=None to perform the requested "
        "operation.");
  if (copy == OJAX_DLPACK_COPY_NEVER && !row_major)
    ojax_pjrt_raise(
        "ojax.dlpack.value_error:from_dlpack cannot view a transposed tensor "
        "without a copy in ojax. Set copy=True or copy=None to perform the "
        "requested operation.");

  PJRT_Buffer *buffer = NULL;
  size_t nbytes = (size_t)elt_bytes;
  for (int i = 0; i < nd; i++) nbytes *= (size_t)t->shape[i];

  if (copy != OJAX_DLPACK_COPY_ALWAYS && aligned && row_major) {
    PJRT_Client_CreateViewOfDeviceBuffer_Args a;
    memset(&a, 0, sizeof a);
    a.struct_size = PJRT_Client_CreateViewOfDeviceBuffer_Args_STRUCT_SIZE;
    a.client = c->client;
    a.device_buffer_ptr = data;
    a.dims = t->shape;
    a.num_dims = (size_t)nd;
    a.element_type = btype;
    a.layout = NULL;
    a.device = c->device;
    a.on_delete_callback = ojax_dlpack_view_deleted;
    a.on_delete_callback_arg = m;
    ojax_pjrt_check(api, api->PJRT_Client_CreateViewOfDeviceBuffer(&a));
    buffer = a.buffer;
  } else {
    PJRT_Client_BufferFromHostBuffer_Args a;
    memset(&a, 0, sizeof a);
    a.struct_size = PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    a.client = c->client;
    a.data = data;
    a.type = btype;
    a.dims = t->shape;
    a.num_dims = (size_t)nd;
    a.byte_strides = byte_strides;
    a.num_byte_strides = (size_t)nd;
    a.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    a.device = c->device;
    caml_release_runtime_system();
    PJRT_Error *err = api->PJRT_Client_BufferFromHostBuffer(&a);
    PJRT_Error *awaited = NULL;
    if (err == NULL) awaited = ojax_pjrt_await(api, a.done_with_host_buffer);
    caml_acquire_runtime_system();
    ojax_pjrt_check(api, err);
    ojax_pjrt_check(api, awaited);
    buffer = a.buffer;
    if (m->deleter != NULL) m->deleter(m);
  }

  v_buffer = caml_alloc_custom_mem(&ojax_pjrt_buffer_ops,
                                   sizeof(struct ojax_pjrt_buffer),
                                   (mlsize_t)nbytes);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  b->api = api;
  b->buffer = buffer;
  b->shares = NULL;
  CAMLreturn(v_buffer);
}

CAMLprim value ojax_pjrt_executable_optimized(value v_exec) {
  CAMLparam1(v_exec);
  CAMLlocal1(v_out);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable == NULL) ojax_pjrt_raise("optimized on destroyed executable");
  const PJRT_Api *api = e->api;
  PJRT_LoadedExecutable_GetExecutable_Args ga;
  memset(&ga, 0, sizeof ga);
  ga.struct_size = PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
  ga.loaded_executable = e->executable;
  ojax_pjrt_check(api, api->PJRT_LoadedExecutable_GetExecutable(&ga));
  PJRT_Program program;
  memset(&program, 0, sizeof program);
  program.struct_size = PJRT_Program_STRUCT_SIZE;
  PJRT_Executable_OptimizedProgram_Args oa;
  memset(&oa, 0, sizeof oa);
  oa.struct_size = PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE;
  oa.executable = ga.executable;
  oa.program = &program;
  PJRT_Error *err = api->PJRT_Executable_OptimizedProgram(&oa);
  char *code = NULL;
  if (err == NULL) {
    code = caml_stat_alloc(program.code_size > 0 ? program.code_size : 1);
    program.code = code;
    err = api->PJRT_Executable_OptimizedProgram(&oa);
  }
  PJRT_Executable_Destroy_Args da;
  memset(&da, 0, sizeof da);
  da.struct_size = PJRT_Executable_Destroy_Args_STRUCT_SIZE;
  da.executable = ga.executable;
  PJRT_Error *derr = api->PJRT_Executable_Destroy(&da);
  if (derr != NULL) {
    PJRT_Error_Destroy_Args de;
    memset(&de, 0, sizeof de);
    de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    de.error = derr;
    api->PJRT_Error_Destroy(&de);
  }
  if (err != NULL) {
    if (code != NULL) caml_stat_free(code);
    ojax_pjrt_check(api, err);
  }
  v_out = caml_alloc_initialized_string(program.code_size, code);
  caml_stat_free(code);
  CAMLreturn(v_out);
}

CAMLprim value ojax_pjrt_execute_devices(value v_exec, value v_lists) {
  CAMLparam2(v_exec, v_lists);
  CAMLlocal3(v_out, v_row, v_buf);
  struct ojax_pjrt_executable *e = Ojax_executable_val(v_exec);
  if (e->executable == NULL) ojax_pjrt_raise("execute on destroyed executable");
  const PJRT_Api *api = e->api;
  size_t num_devices = Wosize_val(v_lists);
  if (num_devices == 0) ojax_pjrt_raise("execute needs at least one device");
  size_t num_args = Wosize_val(Field(v_lists, 0));
  size_t num_outputs = (size_t)e->num_outputs;
  size_t arg_cells = num_args > 0 ? num_args : 1;
  size_t out_cells = num_outputs > 0 ? num_outputs : 1;
  PJRT_Buffer **arg_store =
      caml_stat_alloc(num_devices * arg_cells * sizeof(PJRT_Buffer *));
  PJRT_Buffer **out_store =
      caml_stat_alloc(num_devices * out_cells * sizeof(PJRT_Buffer *));
  PJRT_Buffer *const **arg_lists =
      caml_stat_alloc(num_devices * sizeof(PJRT_Buffer *const *));
  PJRT_Buffer ***out_lists =
      caml_stat_alloc(num_devices * sizeof(PJRT_Buffer **));
  PJRT_Event **events = caml_stat_alloc(num_devices * sizeof(PJRT_Event *));
  const char *bad = NULL;
  for (size_t d = 0; d < num_devices; d++) {
    value row = Field(v_lists, d);
    if (Wosize_val(row) != num_args) bad = "execute argument lists differ";
    for (size_t i = 0; bad == NULL && i < num_args; i++) {
      struct ojax_pjrt_buffer *b = Ojax_buffer_val(Field(row, i));
      if (b->buffer == NULL) bad = "execute given destroyed input buffer";
      arg_store[d * arg_cells + i] = b->buffer;
    }
    for (size_t i = 0; i < out_cells; i++) out_store[d * out_cells + i] = NULL;
    arg_lists[d] = arg_store + d * arg_cells;
    out_lists[d] = out_store + d * out_cells;
    events[d] = NULL;
  }
  if (bad != NULL) {
    caml_stat_free(arg_store);
    caml_stat_free(out_store);
    caml_stat_free(arg_lists);
    caml_stat_free(out_lists);
    caml_stat_free(events);
    ojax_pjrt_raise(bad);
  }

  PJRT_ExecuteOptions opts;
  memset(&opts, 0, sizeof opts);
  opts.struct_size = PJRT_ExecuteOptions_STRUCT_SIZE;
  PJRT_LoadedExecutable_Execute_Args a;
  memset(&a, 0, sizeof a);
  a.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
  a.executable = e->executable;
  a.options = &opts;
  a.argument_lists = arg_lists;
  a.num_devices = num_devices;
  a.num_args = num_args;
  a.output_lists = out_lists;
  a.device_complete_events = events;
  a.execute_device = NULL;

  caml_release_runtime_system();
  PJRT_Error *err = api->PJRT_LoadedExecutable_Execute(&a);
  PJRT_Error *awaited = NULL;
  if (err == NULL) {
    for (size_t d = 0; d < num_devices; d++) {
      PJRT_Error *one = ojax_pjrt_await(api, events[d]);
      if (one != NULL && awaited == NULL)
        awaited = one;
      else if (one != NULL) {
        PJRT_Error_Destroy_Args de;
        memset(&de, 0, sizeof de);
        de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        de.error = one;
        api->PJRT_Error_Destroy(&de);
      }
    }
  }
  caml_acquire_runtime_system();
  caml_stat_free(arg_store);
  caml_stat_free(arg_lists);
  caml_stat_free(out_lists);
  caml_stat_free(events);
  if (err != NULL || awaited != NULL) {
    for (size_t i = 0; i < num_devices * out_cells; i++) {
      if (out_store[i] != NULL) {
        PJRT_Buffer_Destroy_Args bd;
        memset(&bd, 0, sizeof bd);
        bd.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        bd.buffer = out_store[i];
        PJRT_Error *be = api->PJRT_Buffer_Destroy(&bd);
        if (be != NULL) {
          PJRT_Error_Destroy_Args de;
          memset(&de, 0, sizeof de);
          de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
          de.error = be;
          api->PJRT_Error_Destroy(&de);
        }
      }
    }
    caml_stat_free(out_store);
    ojax_pjrt_check(api, err);
    ojax_pjrt_check(api, awaited);
  }

  v_out = caml_alloc(num_devices, 0);
  for (size_t d = 0; d < num_devices; d++) {
    v_row = caml_alloc(num_outputs, 0);
    Store_field(v_out, d, v_row);
    for (size_t i = 0; i < num_outputs; i++) {
      mlsize_t nbytes = 0;
      PJRT_Buffer_OnDeviceSizeInBytes_Args sa;
      memset(&sa, 0, sizeof sa);
      sa.struct_size = PJRT_Buffer_OnDeviceSizeInBytes_Args_STRUCT_SIZE;
      sa.buffer = out_store[d * out_cells + i];
      PJRT_Error *serr = api->PJRT_Buffer_OnDeviceSizeInBytes(&sa);
      if (serr == NULL) {
        nbytes = (mlsize_t)sa.on_device_size_in_bytes;
      } else {
        PJRT_Error_Destroy_Args de;
        memset(&de, 0, sizeof de);
        de.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        de.error = serr;
        api->PJRT_Error_Destroy(&de);
      }
      v_buf = caml_alloc_custom_mem(&ojax_pjrt_buffer_ops,
                                    sizeof(struct ojax_pjrt_buffer), nbytes);
      struct ojax_pjrt_buffer *ob = Ojax_buffer_val(v_buf);
      ob->api = api;
      ob->buffer = out_store[d * out_cells + i];
      ob->shares = NULL;
      Store_field(v_row, i, v_buf);
    }
  }
  caml_stat_free(out_store);
  CAMLreturn(v_out);
}
