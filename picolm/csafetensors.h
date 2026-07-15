/* SPDX-License-Identifier: MIT */
/* Copyright 2023 - Present, Syoyo Fujita. */
/* Pure C11 implementation of safetensors loader */
#ifndef CSAFETENSORS_H_
#define CSAFETENSORS_H_

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CSAFETENSORS_MAX_DIM 8
#define CSAFETENSORS_MAX_TENSORS 65536
#define CSAFETENSORS_MAX_METADATA 1024
#define CSAFETENSORS_MAX_STRING_LEN 4096

typedef enum csafetensors_dtype {
    CSAFETENSORS_DTYPE_BOOL = 0,
    CSAFETENSORS_DTYPE_UINT8,
    CSAFETENSORS_DTYPE_INT8,
    CSAFETENSORS_DTYPE_INT16,
    CSAFETENSORS_DTYPE_UINT16,
    CSAFETENSORS_DTYPE_FLOAT16,
    CSAFETENSORS_DTYPE_BFLOAT16,
    CSAFETENSORS_DTYPE_INT32,
    CSAFETENSORS_DTYPE_UINT32,
    CSAFETENSORS_DTYPE_FLOAT32,
    CSAFETENSORS_DTYPE_FLOAT64,
    CSAFETENSORS_DTYPE_INT64,
    CSAFETENSORS_DTYPE_UINT64
} csafetensors_dtype_t;

typedef enum csafetensors_error {
    CSAFETENSORS_SUCCESS = 0,
    CSAFETENSORS_ERROR_INVALID_ARGUMENT,
    CSAFETENSORS_ERROR_FILE_NOT_FOUND,
    CSAFETENSORS_ERROR_FILE_READ,
    CSAFETENSORS_ERROR_INVALID_HEADER,
    CSAFETENSORS_ERROR_JSON_PARSE,
    CSAFETENSORS_ERROR_MEMORY_ALLOCATION,
    CSAFETENSORS_ERROR_MMAP_FAILED,
    CSAFETENSORS_ERROR_INVALID_TENSOR,
    CSAFETENSORS_ERROR_BUFFER_TOO_SMALL
} csafetensors_error_t;

typedef struct csafetensors_tensor {
    char name[CSAFETENSORS_MAX_STRING_LEN];
    csafetensors_dtype_t dtype;
    size_t shape[CSAFETENSORS_MAX_DIM];
    size_t n_dims;
    size_t data_offset_begin;
    size_t data_offset_end;
} csafetensors_tensor_t;

typedef struct csafetensors_metadata {
    char key[CSAFETENSORS_MAX_STRING_LEN];
    char value[CSAFETENSORS_MAX_STRING_LEN];
} csafetensors_metadata_t;

typedef struct csafetensors {
    csafetensors_tensor_t *tensors;
    size_t n_tensors;
    csafetensors_metadata_t *metadata;
    size_t n_metadata;
    size_t header_size;
    uint8_t *storage;
    size_t storage_size;
    bool mmaped;
    const uint8_t *mmap_addr;
    size_t mmap_size;
    const uint8_t *databuffer_addr;
    size_t databuffer_size;
    void *_internal_file;
    void *_internal_mmap;
    char error_msg[1024];
} csafetensors_t;

void csafetensors_init(csafetensors_t *st);
void csafetensors_free(csafetensors_t *st);
csafetensors_error_t csafetensors_load_from_file(const char *filename, csafetensors_t *st);
csafetensors_error_t csafetensors_load_from_memory(const uint8_t *data, size_t size, csafetensors_t *st);
csafetensors_error_t csafetensors_mmap_from_file(const char *filename, csafetensors_t *st);
csafetensors_error_t csafetensors_mmap_from_memory(const uint8_t *data, size_t size, csafetensors_t *st);
const csafetensors_tensor_t *csafetensors_get_tensor(const csafetensors_t *st, const char *name);
const csafetensors_tensor_t *csafetensors_get_tensor_by_index(const csafetensors_t *st, size_t index);
const uint8_t *csafetensors_get_tensor_data(const csafetensors_t *st, const csafetensors_tensor_t *tensor);
const char *csafetensors_get_metadata(const csafetensors_t *st, const char *key);
size_t csafetensors_dtype_size(csafetensors_dtype_t dtype);
const char *csafetensors_dtype_name(csafetensors_dtype_t dtype);
size_t csafetensors_shape_size(const csafetensors_tensor_t *tensor);
bool csafetensors_validate(const csafetensors_t *st);
float csafetensors_bf16_to_f32(uint16_t x);
uint16_t csafetensors_f32_to_bf16(float x);
float csafetensors_f16_to_f32(uint16_t x);
uint16_t csafetensors_f32_to_f16(float x);
const char *csafetensors_get_error(const csafetensors_t *st);

#ifdef __cplusplus
}
#endif

#endif /* CSAFETENSORS_H_ */
