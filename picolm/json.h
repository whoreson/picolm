#ifndef JSON_H_
#define JSON_H_

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

typedef struct {
    char *key;
    JsonValue *value;
} JsonPair;

typedef struct {
    JsonValue **items;
    size_t count;
} JsonArray;

typedef struct {
    JsonPair *pairs;
    size_t count;
} JsonObject;

struct JsonValue {
    JsonType type;
    union {
        bool boolean;
        double number;
        char *string;
        JsonArray array;
        JsonObject object;
    } data;
};

JsonValue *json_parse(const char *input, size_t len, char *error, size_t error_size);
void json_free(JsonValue *v);
JsonValue *json_object_get(const JsonValue *obj, const char *key);
JsonValue *json_array_get(const JsonValue *arr, size_t index);
int json_get_int(const JsonValue *v, int default_val);
double json_get_double(const JsonValue *v, double default_val);
const char *json_get_string(const JsonValue *v, const char *default_val);
bool json_get_bool(const JsonValue *v, bool default_val);

#ifdef __cplusplus
}
#endif

#endif
