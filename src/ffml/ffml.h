#pragma once

#ifndef FFML_H
#define FFML_H

#include <stdint.h>
#include <stddef.h>
#include <random>

#include <set>
#include <stack>
#include <string>
#include <cassert>
#include <unordered_map>

#define KB (1024)
#define MB (1024 * KB)
#define GB (1024 * MB)

#define FFML_MAX_DIMS           4
#define FFML_MAX_NODES         1000000

#define FFML_TYPE float

#define SEED 12345

#define MAX_TENSOR_NAME_BYTES 64

extern std::default_random_engine gen;
extern FFML_TYPE ffml_rand_uniform();
extern FFML_TYPE ffml_rand_normal();

struct Coord {
    uint64_t dim[FFML_MAX_DIMS];

    Coord() {
        dim[0] = 0;
        dim[1] = 0;
        dim[2] = 0;
        dim[3] = 0;
    }

    Coord(uint64_t input_array[FFML_MAX_DIMS]) {
        for (int i = 0; i < FFML_MAX_DIMS; i++) {
            dim[i] = input_array[i];
        }
    }

    // Coord(uint64_t x, uint64_t y, uint64_t z, uint64_t t) { dim[0] = x; dim[1] = y; dim[2] = z; dim[3] = t; }

    Coord(int x, int y, int z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, int y, int z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, int y, uint64_t z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, int y, uint64_t z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, uint64_t y, int z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, uint64_t y, int z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, uint64_t y, uint64_t z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(int x, uint64_t y, uint64_t z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, int y, int z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, int y, int z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, int y, uint64_t z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, int y, uint64_t z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, uint64_t y, int z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, uint64_t y, int z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, uint64_t y, uint64_t z, int t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }
    Coord(uint64_t x, uint64_t y, uint64_t z, uint64_t t) { dim[0] = (uint64_t) x; dim[1] = (uint64_t) y; dim[2] = (uint64_t) z; dim[3] = (uint64_t) t; }

};

// available tensor operations:
enum ffml_op_type {
    FFML_OP_NONE = 0,

    FFML_OP_LOOKUP,

    // FFML_OP_DUP,
    FFML_OP_ADD,
    FFML_OP_SUB,
    FFML_OP_MUL,
    FFML_OP_DIV,

    // unary
    FFML_OP_POW,
    FFML_OP_SQUARE,
    FFML_OP_NEG,
    FFML_OP_EXP,
    FFML_OP_ABS,

    FFML_OP_MATMUL,

    // reduce
    FFML_OP_MEAN,
    FFML_OP_MEAN_BATCHED,
    FFML_OP_SUM,

    // activation functions
    FFML_OP_TANH,
    FFML_OP_SIGMOID,
    FFML_OP_RELU,
    FFML_OP_GELU,
    FFML_OP_GELU_APPROX_TANH,
    // FFML_OP_GELU_NEW,
    FFML_OP_LEAKY_RELU,

    // views
    FFML_OP_TRANSPOSE,
    FFML_OP_SQUEEZE,
    FFML_OP_UNSQUEEZE,

    // losses
    FFML_OP_SOFTMAX,
    FFML_OP_SOFTMAX_CROSS_ENTROPY,

    // init
    FFML_OP_INIT_ZEROES,
    FFML_OP_INIT_ONES,
    FFML_OP_INIT_FILL,
    FFML_OP_INIT_RND_UNIFORM,
    FFML_OP_INIT_RND_NORMAL,
    FFML_OP_INIT_RND_NORMAL_KAIMING,
    FFML_OP_INIT_CONSTANT,
    FFML_OP_INIT_ARANGE,

    // conv
    FFML_OP_CONV2D,
    FFML_OP_MAXPOOL2D,

    FFML_OP_SELECT,
    FFML_OP_REPEAT,

    // FFML_OP_SQR,
    // FFML_OP_SQRT,
    // FFML_OP_REPEAT,
    // FFML_OP_ABS,
    // FFML_OP_SGN,
    // FFML_OP_STEP,
    // FFML_OP_RELU,
    // FFML_OP_GELU,
    // FFML_OP_SILU,
    // FFML_OP_NORM, // normalize
    FFML_OP_RMS_NORM,

};

// n-dimensional tensor
struct ffml_tensor {
    int     n_dims;
    uint64_t ne[FFML_MAX_DIMS]; // number of elements
    size_t  nb[FFML_MAX_DIMS]; // stride in bytes:
                                // nb[0] = sizeof(type)
                                // nb[1] = nb[0]   * ne[0] + padding
                                // nb[i] = nb[i-1] * ne[i-1]

    // compute data
    enum ffml_op_type op;

    bool is_view = false;

    bool init_ran = false;

    struct ffml_tensor * src0;
    struct ffml_tensor * src1;

    // cached values
    int broadcast_dim_src0[FFML_MAX_DIMS]; // important that it's signed!
    int broadcast_dim_src1[FFML_MAX_DIMS]; // important that it's signed!
    bool broadcast_to_src0_enabled[FFML_MAX_DIMS];
    bool broadcast_to_src1_enabled[FFML_MAX_DIMS];
    uint64_t size_bytes;
    uint64_t nelem;

    // // thread scheduling
    // int n_tasks;

    // // performance
    // int     perf_runs;
    // int64_t perf_cycles;
    // int64_t perf_time_us;

    void * data;
    void * grad;

    char name[64];

    uint64_t key;

    // char padding[8]; // TODO: remove and add padding to name? // TODO: calc how much padding you need and reenable

    std::unordered_map<std::string, FFML_TYPE> * op_metadata;
};

// computation graph
struct ffml_cgraph {
    int n_nodes;
    int n_leafs;
    // int n_threads;

    // size_t work_size;
    // struct ffml_tensor * work;

    struct ffml_tensor * nodes[FFML_MAX_NODES];
    // struct ffml_tensor * grads[FFML_MAX_NODES];
    struct ffml_tensor * leafs[FFML_MAX_NODES];

    std::set<ffml_tensor *> visited;

    // // performance
    // int     perf_runs;
    // int64_t perf_cycles;
    // int64_t perf_time_us;
};

struct ffml_memory_pool {
    uint64_t n_bytes;
    uint64_t n_used;
    void * data;
};

ffml_memory_pool * ffml_memory_pool_create(uint64_t n_bytes);

void ffml_memory_pool_destroy(ffml_memory_pool * pool);

void * ffml_memory_pool_alloc(ffml_memory_pool * pool, uint64_t n_bytes);

struct ffml_tensor * ffml_tensor_create(int n_dims, uint64_t ne[], const char * name);
struct ffml_tensor * ffml_tensor_create(int n_dims, uint64_t ne[]);
struct ffml_tensor * ffml_tensor_create(int n_dims, Coord ne, const char * name);
struct ffml_tensor * ffml_tensor_create(int n_dims, uint64_t ne[], const char * name);
struct ffml_tensor * ffml_tensor_create(int n_dims, Coord ne);

void ffml_tensor_destroy(struct ffml_tensor * t);

struct ffml_tensor * ffml_op(enum ffml_op_type op, struct ffml_tensor * src0, struct ffml_tensor * src1, const char * name = nullptr);
struct ffml_tensor * ffml_unary_op(enum ffml_op_type op, struct ffml_tensor * src0, const char * name = nullptr);

struct ffml_cgraph * ffml_cgraph_create(struct ffml_tensor * t);

// void ffml_cgraph_destroy(struct ffml_cgraph * cgraph); // todo

void ffml_cgraph_alloc(struct ffml_cgraph * cgraph, ffml_memory_pool * pool, const bool zero_out = false);

void ffml_cgraph_forward(struct ffml_cgraph * cgraph);

void ffml_calc_cached_(struct ffml_tensor * t, bool calc_strides = true);

void ffml_zerograd(struct ffml_cgraph * cgraph);
void ffml_cgraph_backward(struct ffml_cgraph * cgraph);
void ffml_set_name(struct ffml_tensor * t, const char * name);
void ffml_set_name(struct ffml_tensor * t, std::string name);

// void ffml_save(struct ffml_cgraph * cgraph, const char * filename);
// struct ffml_cgraph * ffml_load(const char * filename, ffml_memory_pool * pool);

struct ffml_tensor * ffml_get_tensor_by_name(struct ffml_cgraph * cgraph, const char * name);
struct ffml_tensor * ffml_get_tensor_by_key(struct ffml_cgraph * cgraph, uint64_t key);

inline void ffml_set_no_srcs_broadcasting(struct ffml_tensor * t) {
    for (int i = 0; i < FFML_MAX_DIMS; i++) {
        t->broadcast_dim_src0[i] = -1;
        t->broadcast_dim_src1[i] = -1;
        t->broadcast_to_src0_enabled[i] = false;
        t->broadcast_to_src1_enabled[i] = false;
    }
}

inline uint64_t ffml_offset_bytes(ffml_tensor * t, Coord coord) {
    assert(FFML_MAX_DIMS == 4);

    // printf("ffml_offset_bytes name: %s\n", t->name);
    // printf("ffml_offset_bytes: %d %d %d %d\n", coord.dim[0], coord.dim[1], coord.dim[2], coord.dim[3]);
    // printf("ffml_offset_bytes t ne: %d %d %d %d\n", t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
    // printf("ffml_offset_bytes t nb: %d %d %d %d\n", t->nb[0], t->nb[1], t->nb[2], t->nb[3]);

    assert(coord.dim[0] < t->ne[0] || (coord.dim[0] == 0 && t->ne[0] == 0));
    assert(coord.dim[1] < t->ne[1] || (coord.dim[1] == 0 && t->ne[1] == 0)); 
    assert(coord.dim[2] < t->ne[2] || (coord.dim[2] == 0 && t->ne[2] == 0));
    assert(coord.dim[3] < t->ne[3] || (coord.dim[3] == 0 && t->ne[3] == 0));

    return coord.dim[0] * t->nb[0] + coord.dim[1] * t->nb[1] + coord.dim[2]*t->nb[2] + coord.dim[3]*t->nb[3];
}

inline void ffml_set_data(ffml_tensor * t, Coord coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->data)[ ffml_offset_bytes(t, coord) / sizeof(FFML_TYPE) ] = value;
}

inline void ffml_inc_data(ffml_tensor * t, Coord coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->data)[ ffml_offset_bytes(t, coord) / sizeof(FFML_TYPE) ] += value;
}

inline FFML_TYPE ffml_get_data(ffml_tensor * t, Coord coord) {
    return ((FFML_TYPE *) t->data)[ ffml_offset_bytes(t, coord) / sizeof(FFML_TYPE) ];
}

// inline uint64_t ffml_offset_flat_bytes(ffml_tensor * t, uint64_t coord) { <-- old implementation that wasn't using strides but was very simple
//     return coord * t->nb[t->n_dims - 1];
// }
inline uint64_t ffml_offset_flat_bytes(ffml_tensor * t, uint64_t coord) { // todo: check how performance degraded here and maybe reimplement
    uint64_t offset = 0;
    for (int d = t->n_dims - 1; d >= 0; --d) {
        offset += (coord % t->ne[d]) * t->nb[d];
        coord /= t->ne[d];
    }
    return offset;
}

inline void ffml_set_data_flat(ffml_tensor * t, uint64_t coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->data)[ ffml_offset_flat_bytes(t, coord) / sizeof(FFML_TYPE) ] = value;
}

inline void ffml_inc_data_flat(ffml_tensor * t, uint64_t coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->data)[ ffml_offset_flat_bytes(t, coord) / sizeof(FFML_TYPE) ] += value;
}

inline FFML_TYPE ffml_get_data_flat(ffml_tensor * t, uint64_t coord) {
    return ((FFML_TYPE *) t->data)[ ffml_offset_flat_bytes(t, coord) / sizeof(FFML_TYPE) ];
}

inline void ffml_set_grad(ffml_tensor * t, Coord coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->grad)[ ffml_offset_bytes(t, coord) / sizeof(FFML_TYPE) ] = value;
}

inline void ffml_inc_grad(ffml_tensor * t, Coord coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->grad)[ ffml_offset_bytes(t, coord) / sizeof(FFML_TYPE) ] += value;
}

inline FFML_TYPE ffml_get_grad(ffml_tensor * t, Coord coord) {
    return ((FFML_TYPE *) t->grad)[ ffml_offset_bytes(t, coord) / sizeof(FFML_TYPE) ];
}

inline void ffml_set_grad_flat(ffml_tensor * t, uint64_t coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->grad)[ ffml_offset_flat_bytes(t, coord) / sizeof(FFML_TYPE) ] = value;
}

inline void ffml_inc_grad_flat(ffml_tensor * t, uint64_t coord, FFML_TYPE value) {
    ((FFML_TYPE *) t->grad)[ ffml_offset_flat_bytes(t, coord) / sizeof(FFML_TYPE) ] += value;
}

inline FFML_TYPE ffml_get_grad_flat(ffml_tensor * t, uint64_t coord) {
    return ((FFML_TYPE *) t->grad)[ ffml_offset_flat_bytes(t, coord) / sizeof(FFML_TYPE) ];
}

struct ffml_tensor * ffml_reshape(struct ffml_tensor * src, int n_dims, Coord dims);
struct ffml_tensor * ffml_flatten(struct ffml_tensor * src);

void ffml_debug_print_tensor_metadata(ffml_tensor * t);
void ffml_debug_print_tensor_data(ffml_tensor * t);
void ffml_debug_print_cgraph_data(struct ffml_cgraph * cgraph);
void ffml_debug_print_cgraph_shapes(struct ffml_cgraph * cgraph);

#endif