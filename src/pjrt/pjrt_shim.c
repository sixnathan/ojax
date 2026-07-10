#include <dlfcn.h>
#include <stdio.h>
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
};

#define Ojax_buffer_val(v) ((struct ojax_pjrt_buffer *)Data_custom_val(v))

static void ojax_pjrt_buffer_finalize(value v_buffer) {
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  if (b->buffer != NULL) {
    PJRT_Buffer_Destroy_Args a;
    memset(&a, 0, sizeof a);
    a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    a.buffer = b->buffer;
    PJRT_Error *err = b->api->PJRT_Buffer_Destroy(&a);
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

CAMLprim value ojax_pjrt_client_create(value v_plugin) {
  CAMLparam1(v_plugin);
  CAMLlocal1(v_client);
  struct ojax_pjrt_plugin *p = Ojax_plugin_val(v_plugin);
  if (p->api == NULL) ojax_pjrt_raise("client_create on closed plugin");
  const PJRT_Api *api = p->api;

  PJRT_Plugin_Initialize_Args ia;
  memset(&ia, 0, sizeof ia);
  ia.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
  PJRT_Client_Create_Args ca;
  memset(&ca, 0, sizeof ca);
  ca.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
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
  ojax_pjrt_check(api, err);
  ojax_pjrt_check(api, cerr);
  ojax_pjrt_check(api, derr);
  if (da.num_addressable_devices == 0) {
    PJRT_Client_Destroy_Args dea;
    memset(&dea, 0, sizeof dea);
    dea.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    dea.client = ca.client;
    api->PJRT_Client_Destroy(&dea);
    ojax_pjrt_raise("PJRT client has no addressable devices");
  }

  v_client = caml_alloc_custom(&ojax_pjrt_client_ops,
                               sizeof(struct ojax_pjrt_client), 0, 1);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  c->api = api;
  c->client = ca.client;
  c->device = da.addressable_devices[0];
  CAMLreturn(v_client);
}

CAMLprim value ojax_pjrt_client_destroy(value v_client) {
  CAMLparam1(v_client);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client != NULL) {
    const PJRT_Api *api = c->api;
    PJRT_Client *client = c->client;
    c->client = NULL;
    c->device = NULL;
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
                                          value v_type, value v_dims) {
  CAMLparam4(v_client, v_data, v_type, v_dims);
  CAMLlocal1(v_buffer);
  struct ojax_pjrt_client *c = Ojax_client_val(v_client);
  if (c->client == NULL) ojax_pjrt_raise("buffer_from_host on closed client");
  const PJRT_Api *api = c->api;
  PJRT_Client *client = c->client;
  PJRT_Device *device = c->device;
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
  ojax_pjrt_check(api, awaited);

  v_buffer = caml_alloc_custom_mem(
      &ojax_pjrt_buffer_ops, sizeof(struct ojax_pjrt_buffer), (mlsize_t)nbytes);
  struct ojax_pjrt_buffer *b = Ojax_buffer_val(v_buffer);
  b->api = api;
  b->buffer = a.buffer;
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
    b->buffer = NULL;
    PJRT_Buffer_Destroy_Args a;
    memset(&a, 0, sizeof a);
    a.struct_size = PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
    a.buffer = buffer;
    caml_release_runtime_system();
    PJRT_Error *err = api->PJRT_Buffer_Destroy(&a);
    caml_acquire_runtime_system();
    ojax_pjrt_check(api, err);
  }
  CAMLreturn(Val_unit);
}
