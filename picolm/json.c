#include "json.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct {
    const char *input;
    const char *pos;
    const char *end;
    char *error;
    size_t error_size;
} Parser;

static JsonValue *parse_value(Parser *p);

static void skip_whitespace(Parser *p) {
    while (p->pos < p->end && (*p->pos == ' ' || *p->pos == '\t' || *p->pos == '\n' || *p->pos == '\r')) {
        p->pos++;
    }
}

static int match(Parser *p, const char *str) {
    size_t len = strlen(str);
    if ((size_t)(p->end - p->pos) < len) return 0;
    if (memcmp(p->pos, str, len) == 0) {
        p->pos += len;
        return 1;
    }
    return 0;
}

static JsonValue *alloc_value(JsonType type) {
    JsonValue *v = (JsonValue *)calloc(1, sizeof(JsonValue));
    if (v) v->type = type;
    return v;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static int parse_unicode_escape(Parser *p, char **out, size_t *out_cap, size_t *out_len) {
    if (p->end - p->pos < 4) return 0;

    int codepoint = 0;
    for (int i = 0; i < 4; i++) {
        int digit = hex_digit(p->pos[i]);
        if (digit < 0) return 0;
        codepoint = (codepoint << 4) | digit;
    }
    p->pos += 4;

    if (codepoint >= 0xD800 && codepoint <= 0xDBFF) {
        if (p->end - p->pos < 6) return 0;
        if (p->pos[0] != '\\' || p->pos[1] != 'u') return 0;
        p->pos += 2;

        int low = 0;
        for (int i = 0; i < 4; i++) {
            int digit = hex_digit(p->pos[i]);
            if (digit < 0) return 0;
            low = (low << 4) | digit;
        }
        p->pos += 4;

        if (low < 0xDC00 || low > 0xDFFF) return 0;
        codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00);
    }

    char utf8[4];
    size_t utf8_len = 0;

    if (codepoint < 0x80) {
        utf8[0] = (char)codepoint;
        utf8_len = 1;
    } else if (codepoint < 0x800) {
        utf8[0] = (char)(0xC0 | (codepoint >> 6));
        utf8[1] = (char)(0x80 | (codepoint & 0x3F));
        utf8_len = 2;
    } else if (codepoint < 0x10000) {
        utf8[0] = (char)(0xE0 | (codepoint >> 12));
        utf8[1] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        utf8[2] = (char)(0x80 | (codepoint & 0x3F));
        utf8_len = 3;
    } else {
        utf8[0] = (char)(0xF0 | (codepoint >> 18));
        utf8[1] = (char)(0x80 | ((codepoint >> 12) & 0x3F));
        utf8[2] = (char)(0x80 | ((codepoint >> 6) & 0x3F));
        utf8[3] = (char)(0x80 | (codepoint & 0x3F));
        utf8_len = 4;
    }

    while (*out_len + utf8_len >= *out_cap) {
        *out_cap = *out_cap == 0 ? 64 : *out_cap * 2;
        *out = (char *)realloc(*out, *out_cap);
        if (!*out) return 0;
    }
    memcpy(*out + *out_len, utf8, utf8_len);
    *out_len += utf8_len;

    return 1;
}

static JsonValue *parse_string(Parser *p) {
    if (*p->pos != '"') {
        snprintf(p->error, p->error_size, "Expected '\"' at position %zu", (size_t)(p->pos - p->input));
        return NULL;
    }
    p->pos++;

    size_t cap = 64;
    size_t len = 0;
    char *str = (char *)malloc(cap);
    if (!str) {
        snprintf(p->error, p->error_size, "Memory allocation failed");
        return NULL;
    }

    while (p->pos < p->end && *p->pos != '"') {
        char c = *p->pos;

        if ((unsigned char)c < 0x20) {
            snprintf(p->error, p->error_size, "Invalid control character in string");
            free(str);
            return NULL;
        }

        if (c == '\\') {
            p->pos++;
            if (p->pos >= p->end) {
                snprintf(p->error, p->error_size, "Unexpected end of string");
                free(str);
                return NULL;
            }

            switch (*p->pos) {
                case '"':  c = '"'; break;
                case '\\': c = '\\'; break;
                case '/':  c = '/'; break;
                case 'b':  c = '\b'; break;
                case 'f':  c = '\f'; break;
                case 'n':  c = '\n'; break;
                case 'r':  c = '\r'; break;
                case 't':  c = '\t'; break;
                case 'u':
                    p->pos++;
                    if (!parse_unicode_escape(p, &str, &cap, &len)) {
                        snprintf(p->error, p->error_size, "Invalid unicode escape");
                        free(str);
                        return NULL;
                    }
                    continue;
                default:
                    snprintf(p->error, p->error_size, "Invalid escape character");
                    free(str);
                    return NULL;
            }
            p->pos++;
        } else {
            p->pos++;
        }

        if (len + 1 >= cap) {
            cap *= 2;
            str = (char *)realloc(str, cap);
            if (!str) {
                snprintf(p->error, p->error_size, "Memory allocation failed");
                return NULL;
            }
        }
        str[len++] = c;
    }

    if (p->pos >= p->end || *p->pos != '"') {
        snprintf(p->error, p->error_size, "Unterminated string");
        free(str);
        return NULL;
    }
    p->pos++;

    str[len] = '\0';

    JsonValue *v = alloc_value(JSON_STRING);
    if (!v) {
        free(str);
        snprintf(p->error, p->error_size, "Memory allocation failed");
        return NULL;
    }
    v->data.string = str;
    return v;
}

static JsonValue *parse_number(Parser *p) {
    const char *start = p->pos;

    if (*p->pos == '-') p->pos++;

    if (p->pos >= p->end) {
        snprintf(p->error, p->error_size, "Unexpected end of number");
        return NULL;
    }

    if (*p->pos == '0') {
        p->pos++;
    } else if (*p->pos >= '1' && *p->pos <= '9') {
        while (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') p->pos++;
    } else {
        snprintf(p->error, p->error_size, "Invalid number");
        return NULL;
    }

    if (p->pos < p->end && *p->pos == '.') {
        p->pos++;
        if (p->pos >= p->end || *p->pos < '0' || *p->pos > '9') {
            snprintf(p->error, p->error_size, "Invalid number");
            return NULL;
        }
        while (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') p->pos++;
    }

    if (p->pos < p->end && (*p->pos == 'e' || *p->pos == 'E')) {
        p->pos++;
        if (p->pos < p->end && (*p->pos == '+' || *p->pos == '-')) p->pos++;
        if (p->pos >= p->end || *p->pos < '0' || *p->pos > '9') {
            snprintf(p->error, p->error_size, "Invalid number exponent");
            return NULL;
        }
        while (p->pos < p->end && *p->pos >= '0' && *p->pos <= '9') p->pos++;
    }

    size_t len = (size_t)(p->pos - start);
    char *numstr = (char *)malloc(len + 1);
    if (!numstr) {
        snprintf(p->error, p->error_size, "Memory allocation failed");
        return NULL;
    }
    memcpy(numstr, start, len);
    numstr[len] = '\0';

    double val = strtod(numstr, NULL);
    free(numstr);

    JsonValue *v = alloc_value(JSON_NUMBER);
    if (!v) {
        snprintf(p->error, p->error_size, "Memory allocation failed");
        return NULL;
    }
    v->data.number = val;
    return v;
}

static JsonValue *parse_array(Parser *p) {
    if (*p->pos != '[') {
        snprintf(p->error, p->error_size, "Expected '['");
        return NULL;
    }
    p->pos++;

    JsonValue *v = alloc_value(JSON_ARRAY);
    if (!v) {
        snprintf(p->error, p->error_size, "Memory allocation failed");
        return NULL;
    }
    v->data.array.items = NULL;
    v->data.array.count = 0;

    skip_whitespace(p);

    if (p->pos < p->end && *p->pos == ']') {
        p->pos++;
        return v;
    }

    size_t capacity = 0;
    for (;;) {
        JsonValue *item = parse_value(p);
        if (!item) {
            json_free(v);
            return NULL;
        }

        if (v->data.array.count >= capacity) {
            size_t new_cap = capacity == 0 ? 8 : capacity * 2;
            JsonValue **new_items = (JsonValue **)realloc(v->data.array.items, new_cap * sizeof(JsonValue *));
            if (!new_items) {
                json_free(item);
                json_free(v);
                snprintf(p->error, p->error_size, "Memory allocation failed");
                return NULL;
            }
            v->data.array.items = new_items;
            capacity = new_cap;
        }
        v->data.array.items[v->data.array.count++] = item;

        skip_whitespace(p);

        if (p->pos >= p->end) {
            snprintf(p->error, p->error_size, "Unexpected end of array");
            json_free(v);
            return NULL;
        }

        if (*p->pos == ']') {
            p->pos++;
            return v;
        }

        if (*p->pos != ',') {
            snprintf(p->error, p->error_size, "Expected ',' or ']' in array");
            json_free(v);
            return NULL;
        }
        p->pos++;
        skip_whitespace(p);
    }
}

static JsonValue *parse_object(Parser *p) {
    if (*p->pos != '{') {
        snprintf(p->error, p->error_size, "Expected '{'");
        return NULL;
    }
    p->pos++;

    JsonValue *v = alloc_value(JSON_OBJECT);
    if (!v) {
        snprintf(p->error, p->error_size, "Memory allocation failed");
        return NULL;
    }
    v->data.object.pairs = NULL;
    v->data.object.count = 0;

    skip_whitespace(p);

    if (p->pos < p->end && *p->pos == '}') {
        p->pos++;
        return v;
    }

    size_t capacity = 0;
    for (;;) {
        if (p->pos >= p->end || *p->pos != '"') {
            snprintf(p->error, p->error_size, "Expected string key in object");
            json_free(v);
            return NULL;
        }

        JsonValue *key_val = parse_string(p);
        if (!key_val) {
            json_free(v);
            return NULL;
        }
        char *key = key_val->data.string;
        key_val->data.string = NULL;
        json_free(key_val);

        skip_whitespace(p);

        if (p->pos >= p->end || *p->pos != ':') {
            free(key);
            snprintf(p->error, p->error_size, "Expected ':' after key");
            json_free(v);
            return NULL;
        }
        p->pos++;

        JsonValue *val = parse_value(p);
        if (!val) {
            free(key);
            json_free(v);
            return NULL;
        }

        if (v->data.object.count >= capacity) {
            size_t new_cap = capacity == 0 ? 8 : capacity * 2;
            JsonPair *new_pairs = (JsonPair *)realloc(v->data.object.pairs, new_cap * sizeof(JsonPair));
            if (!new_pairs) {
                free(key);
                json_free(val);
                json_free(v);
                snprintf(p->error, p->error_size, "Memory allocation failed");
                return NULL;
            }
            v->data.object.pairs = new_pairs;
            capacity = new_cap;
        }
        v->data.object.pairs[v->data.object.count].key = key;
        v->data.object.pairs[v->data.object.count].value = val;
        v->data.object.count++;

        skip_whitespace(p);

        if (p->pos >= p->end) {
            snprintf(p->error, p->error_size, "Unexpected end of object");
            json_free(v);
            return NULL;
        }

        if (*p->pos == '}') {
            p->pos++;
            return v;
        }

        if (*p->pos != ',') {
            snprintf(p->error, p->error_size, "Expected ',' or '}' in object");
            json_free(v);
            return NULL;
        }
        p->pos++;
        skip_whitespace(p);
    }
}

static JsonValue *parse_value(Parser *p) {
    skip_whitespace(p);

    if (p->pos >= p->end) {
        snprintf(p->error, p->error_size, "Unexpected end of input");
        return NULL;
    }

    switch (*p->pos) {
        case 'n':
            if (match(p, "null")) {
                return alloc_value(JSON_NULL);
            }
            break;
        case 't':
            if (match(p, "true")) {
                JsonValue *v = alloc_value(JSON_BOOL);
                if (v) v->data.boolean = true;
                return v;
            }
            break;
        case 'f':
            if (match(p, "false")) {
                JsonValue *v = alloc_value(JSON_BOOL);
                if (v) v->data.boolean = false;
                return v;
            }
            break;
        case '"':
            return parse_string(p);
        case '[':
            return parse_array(p);
        case '{':
            return parse_object(p);
        default:
            if (*p->pos == '-' || (*p->pos >= '0' && *p->pos <= '9')) {
                return parse_number(p);
            }
    }

    snprintf(p->error, p->error_size, "Invalid JSON value at position %zu", (size_t)(p->pos - p->input));
    return NULL;
}

JsonValue *json_parse(const char *input, size_t len, char *error, size_t error_size) {
    Parser p;
    p.input = input;
    p.pos = input;
    p.end = input + len;
    p.error = error;
    p.error_size = error_size;

    return parse_value(&p);
}

void json_free(JsonValue *v) {
    if (!v) return;

    switch (v->type) {
        case JSON_STRING:
            free(v->data.string);
            break;
        case JSON_ARRAY:
            for (size_t i = 0; i < v->data.array.count; i++) {
                json_free(v->data.array.items[i]);
            }
            free(v->data.array.items);
            break;
        case JSON_OBJECT:
            for (size_t i = 0; i < v->data.object.count; i++) {
                free(v->data.object.pairs[i].key);
                json_free(v->data.object.pairs[i].value);
            }
            free(v->data.object.pairs);
            break;
        default:
            break;
    }
    free(v);
}

JsonValue *json_object_get(const JsonValue *obj, const char *key) {
    if (!obj || obj->type != JSON_OBJECT) return NULL;
    for (size_t i = 0; i < obj->data.object.count; i++) {
        if (strcmp(obj->data.object.pairs[i].key, key) == 0) {
            return obj->data.object.pairs[i].value;
        }
    }
    return NULL;
}

JsonValue *json_array_get(const JsonValue *arr, size_t index) {
    if (!arr || arr->type != JSON_ARRAY || index >= arr->data.array.count) return NULL;
    return arr->data.array.items[index];
}

int json_get_int(const JsonValue *v, int default_val) {
    if (!v || v->type != JSON_NUMBER) return default_val;
    return (int)v->data.number;
}

double json_get_double(const JsonValue *v, double default_val) {
    if (!v || v->type != JSON_NUMBER) return default_val;
    return v->data.number;
}

const char *json_get_string(const JsonValue *v, const char *default_val) {
    if (!v || v->type != JSON_STRING) return default_val;
    return v->data.string;
}

bool json_get_bool(const JsonValue *v, bool default_val) {
    if (!v || v->type != JSON_BOOL) return default_val;
    return v->data.boolean;
}
