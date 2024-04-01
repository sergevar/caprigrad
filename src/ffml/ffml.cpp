#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <functional>

#include "ffml.h"

#include <string.h>

#include <thread>

std::default_random_engine gen(SEED);
FFML_TYPE ffml_rand_uniform(float min, float max) {
    std::uniform_real_distribution<FFML_TYPE> distribution(min, max);
    return distribution(gen);
}
FFML_TYPE ffml_rand_normal(float mean, float stddev) {
    std::normal_distribution<FFML_TYPE> distribution(mean, stddev);
    return distribution(gen);
}
uint64_t ffml_rand_uint64t(uint64_t min, uint64_t max) {
    std::uniform_int_distribution<uint64_t> distribution(min, max);
    return distribution(gen);
}
FFML_TYPE ffml_rand_normal_kaiming(float gain, float fan_in) {
    // Kaiming He initialization
    // https://arxiv.org/pdf/1502.01852.pdf

    const float mean = 0.0f;
    const float stddev = 1.0f;

    FFML_TYPE result = ffml_rand_normal(mean, stddev) * gain / sqrt(fan_in);
    // printf("fan_in: %f\n", fan_in);
    // printf("gain: %f\n", gain);
    // printf("kaiming: %f\n", result);
    return result;
}

ffml_memory_pool * ffml_memory_pool_create(uint64_t n_bytes) {
    ffml_memory_pool * pool = (ffml_memory_pool *) malloc(sizeof(ffml_memory_pool));
    pool->n_bytes = n_bytes;
    pool->n_used = 0;
    pool->data = malloc(n_bytes);
    return pool;
}

void ffml_memory_pool_destroy(ffml_memory_pool * pool) {
    free(pool->data);
    free(pool);
}

void * ffml_memory_pool_alloc(ffml_memory_pool * pool, uint64_t n_bytes) {
    // if (pool->n_used + n_bytes > pool->n_bytes) {
    //     return NULL;
    // }

    assert(pool->n_used + n_bytes <= pool->n_bytes);

    void * data = (void *) ((char *) pool->data + pool->n_used);
    pool->n_used += n_bytes;
    return data;
}

void ffml_calc_cached_(struct ffml_tensor * t, bool calc_strides) {
    // size in bytes
    t->size_bytes = sizeof(FFML_TYPE);
    for (int i = 0; i < t->n_dims; i++) {
        // printf("ne[%d] = %lu\n", i, t->ne[i]);
        t->size_bytes *= t->ne[i];
    }

    // number of elements
    t->nelem = 1;
    for (int i = 0; i < t->n_dims; i++) {
        t->nelem *= t->ne[i];
    }

    if (calc_strides) {
        // strides, contiguous
        t->nb[t->n_dims - 1] = sizeof(FFML_TYPE);
        for (int i = t->n_dims - 2; i >= 0; --i) {
            t->nb[i] = t->nb[i + 1] * t->ne[i + 1];
        }
        // make sure the rest is 0
        for (int i = t->n_dims; i < FFML_MAX_DIMS; i++) {
            t->nb[i] = 0;
        }
    }
}

struct ffml_tensor * ffml_tensor_create(int n_dims, uint64_t ne[], const char * name) {
    // todo: use pool for that perhaps? or separate from computation like here?

    struct ffml_tensor * t = (struct ffml_tensor *) malloc(sizeof(struct ffml_tensor)); // todo: don't forget to free
    
    t->op = FFML_OP_NONE;
    
    // name
    for (int i = 0; i < 32; i++) {
        t->name[i] = '\0';
    }
    char * name_ptr = (char *) name;
    char * t_name_ptr = (char *) t->name;
    while (*name_ptr != '\0') {
        *t_name_ptr = *name_ptr;
        name_ptr++;
        t_name_ptr++;
    }
    
    // dimensions
    t->n_dims = n_dims;
    
    // number of elements
    for (int i = 0; i < n_dims; i++) {
        t->ne[i] = ne[i];
    }
    for (int i = n_dims; i < FFML_MAX_DIMS; i++) {
        t->ne[i] = 1;
    }

    t->is_view = false;
    t->init_ran = false;

    // calc cached
    ffml_calc_cached_(t);

    t->src0 = NULL;
    t->src1 = NULL;
    t->data = NULL;

    t->op_metadata = new std::unordered_map<std::string, FFML_TYPE>();

    return t;
}

struct ffml_tensor * ffml_tensor_create(int n_dims, Coord ne, const char * name) {
    uint64_t ne_arr[FFML_MAX_DIMS];
    for (int i = 0; i < FFML_MAX_DIMS; i++) {
        ne_arr[i] = ne.dim[i];
    }
    for (int i = n_dims; i < FFML_MAX_DIMS; i++) {
        if (ne_arr[i] != 0) {
            printf("Error: the unused dimensions in new vector must be 0\n");
            printf("n_dims = %d, ne_arr[%d] = %lu\n", n_dims, i, ne_arr[i]);
            assert(false);
        }
    }
    return ffml_tensor_create(n_dims, ne_arr, name);
}

struct ffml_tensor * ffml_tensor_create(int n_dims, uint64_t ne[]) {
    return ffml_tensor_create(n_dims, ne, "");
}

struct ffml_tensor * ffml_tensor_create(int n_dims, Coord ne) {
    return ffml_tensor_create(n_dims, ne, "");
}

void ffml_tensor_destroy(struct ffml_tensor * t) {
    free(t);
}

struct ffml_tensor * ffml_op(enum ffml_op_type op, struct ffml_tensor * src0, struct ffml_tensor * src1, const char * name) {
    struct ffml_tensor * t = ffml_tensor_create(src0->n_dims, src0->ne);

    t->op = op;

    t->src0 = src0;
    t->src1 = src1;

    t->is_view = false;
    t->init_ran = false;

    // calculate broadcasting etc.
    if (op == FFML_OP_LOOKUP) {
        assert(false); // todo: For now you can implement it as a matrix multiplication

    } else if (op == FFML_OP_CONV2D) {
        assert(src0->n_dims == 3);
        assert(src1->n_dims == 4);

        // src0 is the input, dimensions: [in_channels, in_x, in_y]
        // src1 is the kernels, dimensions: [n_filters(out_channels), in_channels, kernel_x, kernel_y]
        // result is the output, dimensions: [n_filters(out_channels), out_x, out_y]

        t->n_dims = 3;
        t->ne[0] = src1->ne[0];
        t->ne[1] = src0->ne[1] - src1->ne[2] + 1;
        t->ne[2] = src0->ne[2] - src1->ne[3] + 1;

        // printf("src0: %d %d %d\n", src0->ne[0], src0->ne[1], src0->ne[2]);
        // printf("src1: %d %d %d %d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
        // printf("conv2d: %d %d %d\n", t->ne[0], t->ne[1], t->ne[2]);

        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else if (op == FFML_OP_SOFTMAX_CROSS_ENTROPY) {
        assert(src0->n_dims == src1->n_dims);

        assert(src0->ne[0] == src1->ne[0]);
        assert(src0->ne[1] == src1->ne[1]);

        t->n_dims = src0->n_dims;

        if (src0->n_dims == 1) {
            // unbatched
            t->ne[0] = 1;
        } else if (src0->n_dims == 2) {
            // batched
            t->ne[0] = src0->ne[0];
            t->ne[1] = 1;
        } else {
            assert(false);
        }
        
        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else if (op == FFML_OP_MATMUL) {
        assert(src0->n_dims >= 2);
        assert(src1->n_dims >= 2);
        assert(src0->n_dims == src1->n_dims);

        t->n_dims = src0->n_dims;

        if (src0->n_dims == 2) {
            assert(src0->ne[1] == src1->ne[0]); // matrix side, last 2 dims
            t->ne[0] = src0->ne[0];
            t->ne[1] = src1->ne[1];
        } else if (src0->n_dims == 3) {
            assert(src0->ne[2] == src1->ne[1]); // matrix side, last 2 dims
            assert(src0->ne[0] == src1->ne[0]); // stack
            t->ne[0] = src0->ne[0]; // stack
            t->ne[1] = src0->ne[1];
            t->ne[2] = src1->ne[2];
        } else if (src0->n_dims == 4) {
            assert(src0->ne[3] == src1->ne[2]); // matrix side, last 2 dims
            assert(src0->ne[0] == src1->ne[0]); // stack
            assert(src0->ne[1] == src1->ne[1]); // stack
            t->ne[0] = src0->ne[0]; // stack
            t->ne[1] = src0->ne[1]; // stack
            t->ne[2] = src0->ne[2];
            t->ne[3] = src1->ne[3];
        } else {
            assert(false);
        }

        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else if (op == FFML_OP_SELECT) {
        assert(src0->n_dims <= 3);
        assert(src1->n_dims == 2);

        t->n_dims = src0->n_dims + 1;

        for(int i = 0; i < src0->n_dims; i++) {
            t->ne[i] = src0->ne[i]; // same dimensions as src0
        }
        t->ne[src0->n_dims] = src1->ne[1]; // embedding size

        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else if (op == FFML_OP_REPEAT) {
        assert(src1->ne[0] % src0->ne[0] == 0);
        assert(src1->ne[1] % src0->ne[1] == 0);
        assert(src1->ne[2] % src0->ne[2] == 0);
        assert(src1->ne[3] % src0->ne[3] == 0);

        t->n_dims = src1->n_dims;
        t->ne[0] = src1->ne[0];
        t->ne[1] = src1->ne[1];
        t->ne[2] = src1->ne[2];
        t->ne[3] = src1->ne[3];

        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else {
        t->n_dims = std::max(src0->n_dims, src1->n_dims);

        assert(src0->n_dims > 0); // todo: what about scalarschrome?
        assert(src1->n_dims > 0);

        // print both shapes
        // printf("brbr src0: ");
        // for (int i = 0; i < src0->n_dims; i++) {
        //     printf("%lu ", src0->ne[i]);
        // }
        // printf("\n");
        // printf("brbr src1: ");
        // for (int i = 0; i < src1->n_dims; i++) {
        //     printf("%lu ", src1->ne[i]);
        // }
        // printf("\n");

        int src0_dim = src0->n_dims - 1;
        int src1_dim = src1->n_dims - 1;
        for (int i = t->n_dims-1; i >= 0; i--) {
            // if we're going into negative territory (dimension doesn't exist), we don't try to fetch from negative srcX_dim
            int src0_ne = (src0_dim >= 0) ? src0->ne[src0_dim] : 1;
            int src1_ne = (src1_dim >= 0) ? src1->ne[src1_dim] : 1;

            t->broadcast_dim_src0[i] = src0_dim;
            t->broadcast_dim_src1[i] = src1_dim;

            if (src0_ne == src1_ne) {
                // no broadcasting for this dimension, they're equal (including the case when both are 1)
                t->ne[i] = src0_ne;
                t->broadcast_to_src0_enabled[i] = false;
                t->broadcast_to_src1_enabled[i] = false;
            } else if (src0_ne == 1) {
                // broadcast to src0 along this dimension
                t->ne[i] = src1_ne; // use the longest one as target capacity on this dimension
                t->broadcast_to_src0_enabled[i] = true;
                t->broadcast_to_src1_enabled[i] = false;
            } else if (src1_ne == 1) {
                // broadcast to src1 along this dimension
                t->ne[i] = src0_ne; // use the longest one as target capacity on this dimension
                t->broadcast_to_src0_enabled[i] = false;
                t->broadcast_to_src1_enabled[i] = true;
            } else {
                printf("Error: cannot broadcast dimensions %d and %d (ne: %d and %d)\n", src0_dim, src1_dim, src0_ne, src1_ne);
                // print both shapes
                printf("src0: ");
                for (int j = 0; j < src0->n_dims; j++) {
                    printf("%lu ", src0->ne[j]);
                }
                printf("\n");
                printf("src1: ");
                for (int j = 0; j < src1->n_dims; j++) {
                    printf("%lu ", src1->ne[j]);
                }
                printf("\n");
                assert(false);
            }

            src0_dim--;
            src1_dim--;
        }
    }

    if (name != NULL) {
        ffml_set_name(t, name);
    }

    return t;
}

struct ffml_tensor * ffml_unary_op(enum ffml_op_type op, struct ffml_tensor * src0, const char * name) {
    struct ffml_tensor * t = ffml_tensor_create(src0->n_dims, src0->ne);

    t->n_dims = src0->n_dims;

    if (op == FFML_OP_TRANSPOSE) {
        assert(src0->n_dims == 2);
        for (int i = 0; i < src0->n_dims; i++) {
            t->ne[i] = src0->ne[src0->n_dims - i - 1];
        }
        // swap nb's
        t->nb[0] = src0->nb[1];
        t->nb[1] = src0->nb[0];

        t->is_view = true;

    } else if (op == FFML_OP_UNSQUEEZE) {
        // add new dimension on the left
        t->n_dims = src0->n_dims + 1;
        assert(t->n_dims <= FFML_MAX_DIMS);
        assert(src0->n_dims > 0);
        for (int i = 0; i < src0->n_dims; i++) {
            t->ne[i+1] = src0->ne[i];
        }
        t->ne[0] = 1;
        
        // clone nbs
        for (int i = 0; i < src0->n_dims; i++) {
            t->nb[i+1] = src0->nb[i];
        }

        uint64_t product_of_src_dims = 1;
        for(int i = 0; i < src0->n_dims; i++) {
            product_of_src_dims *= src0->ne[i];
        }
        t->nb[0] = product_of_src_dims;

        t->is_view = true;
    
    } else if (op == FFML_OP_SQUEEZE) {
        // remove dimension on the left if it's 1
        assert(t->n_dims > 1);
        assert(src0->ne[0] == 1);

        t->n_dims = src0->n_dims - 1;
        assert(t->n_dims > 0);
        for (int i = 1; i < src0->n_dims; i++) {
            t->ne[i-1] = src0->ne[i];
        }
        
        // clone nbs
        for (int i = 1; i < src0->n_dims; i++) {
            t->nb[i-1] = src0->nb[i];
        }

        t->is_view = true;

    } else if (op == FFML_OP_MEAN || op == FFML_OP_SUM) {
        // reducing

        t->n_dims = 1;
        t->ne[0] = 1;

        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else if (op == FFML_OP_MEAN_BATCHED) {
        assert(src0->n_dims == 2);
        t->n_dims = 2; // keep dims
        t->ne[0] = src0->ne[0];
        t->ne[1] = 1;

        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);

    } else if (op == FFML_OP_MAXPOOL2D) {
        assert(src0->n_dims == 3);
        t->n_dims = 3;

        // todo: right now hardcoded, should always be 2x2 pool
        t->ne[0] = src0->ne[0];
        t->ne[1] = ceil(src0->ne[1]*1.0 / 2.0);
        t->ne[2] = ceil(src0->ne[2]*1.0 / 2.0);
        
        ffml_calc_cached_(t);

        ffml_set_no_srcs_broadcasting(t);
        
    } else {
        for (int i = 0; i < src0->n_dims; i++) {
            t->ne[i] = src0->ne[i];
        }
    }

    t->op = op;
    t->src0 = src0;
    t->src1 = NULL;

    ffml_calc_cached_(t, false);

    if (name != NULL) {
        ffml_set_name(t, name);
    }

    return t;
}

void ffml_add_node(struct ffml_cgraph * cgraph, struct ffml_tensor * t) {
    cgraph->visited.insert(t);

    if (t->src0 != NULL && cgraph->visited.find(t->src0) == cgraph->visited.end()) {
        ffml_add_node(cgraph, t->src0);
    }
    if (t->src1 != NULL && cgraph->visited.find(t->src1) == cgraph->visited.end()) {
        ffml_add_node(cgraph, t->src1);
    }

    if (t->src0 == NULL && t->src1 == NULL) {
        cgraph->leafs[cgraph->n_leafs++] = t;
    }

    cgraph->nodes[cgraph->n_nodes++] = t;

    t->key = cgraph->n_nodes - 1;
}

struct ffml_cgraph * ffml_cgraph_create(struct ffml_tensor * t) {
    struct ffml_cgraph * cgraph = new ffml_cgraph();
    cgraph->n_nodes = 0;
    cgraph->n_leafs = 0;

    // cgraph->n_threads = 0;
    // cgraph->work_size = 0;
    // cgraph->work = NULL;
    // cgraph->perf_runs = 0;
    // cgraph->perf_cycles = 0;
    // cgraph->perf_time_us = 0;

    ffml_add_node(cgraph, t);

    return cgraph;
}

void ffml_cgraph_alloc(struct ffml_cgraph * cgraph, ffml_memory_pool * pool, const bool zero_out /* default= false */) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ffml_tensor * t = cgraph->nodes[i];

        if (t->data != NULL && t->grad != NULL) {
            // already allocated
            continue;
        }

        if (t->is_view) {
            t->data = t->src0->data;
            t->grad = t->src0->grad;
        } else {
            t->data = ffml_memory_pool_alloc(pool, t->size_bytes);
            t->grad = ffml_memory_pool_alloc(pool, t->size_bytes);

            if (zero_out) {
                memset(t->data, 0, t->size_bytes);
                memset(t->grad, 0, t->size_bytes);
            }
        }
    }
}

inline Coord ffml_broadcast_coord_to_src0(struct ffml_tensor* t, Coord coord) {
    Coord src_coord = {0,0,0,0};

    // compute coords
    for(int out_i = t->n_dims - 1; out_i >= 0; out_i--) {
        if (t->broadcast_dim_src0[out_i] >= 0) {
            // source dimension exists
            if (! t->broadcast_to_src0_enabled[out_i]) {
                // normal case, no broadcast, but might have to shift
                src_coord.dim[t->broadcast_dim_src0[out_i]] = coord.dim[out_i];
            } else {
                // do broadcast, so just set coordinate to 0 (the only element in this dimension)
                src_coord.dim[t->broadcast_dim_src0[out_i]] = 0;
            }
        } else {
            // the source dimension doesn't exist, we implicitly broadcast throughout all of it
        }
    }

    return src_coord;
}
inline Coord ffml_broadcast_coord_to_src1(struct ffml_tensor* t, Coord coord) {
    Coord src_coord = {0,0,0,0};

    // compute coords
    for(int out_i = t->n_dims - 1; out_i >= 0; out_i--) {
        if (t->broadcast_dim_src1[out_i] >= 0) {
            // source dimension exists
            if (! t->broadcast_to_src1_enabled[out_i]) {
                // normal case, no broadcast, but might have to shift
                src_coord.dim[t->broadcast_dim_src1[out_i]] = coord.dim[out_i];
            } else {
                // do broadcast, so just set coordinate to 0 (the only element in this dimension)
                src_coord.dim[t->broadcast_dim_src1[out_i]] = 0;
            }
        } else {
            // the source dimension doesn't exist, we implicitly broadcast throughout all of it
        }
    }

    return src_coord;
}

inline void _ffml_select(struct ffml_tensor* t, struct ffml_tensor* src0, struct ffml_tensor* src1, bool grad_mode = false) {
    assert(FFML_MAX_DIMS == 4);
    assert(src0->n_dims <= 3);
    assert(src1->n_dims == 2);

    assert(t->n_dims == src0->n_dims + 1);

    for (int i = 0; i < src0->n_dims; i++) {
        assert(t->ne[i] == src0->ne[i]);
    }
    assert(t->ne[src0->n_dims] == src1->ne[1]);

    Coord src0_coord;

    // init coords
    for(int src0_i = 0; src0_i < FFML_MAX_DIMS; src0_i++) {
        src0_coord.dim[src0_i] = (src0_i >= src0->n_dims) ? 0 : src0->ne[src0_i] - 1; // going backwards so that we can exit the while loop when we reach 0
    }

    while(true) {
        // compute coords
        FFML_TYPE src0_item = ffml_get_data(src0, src0_coord);

        // turn into int
        int src0_item_int = (int) src0_item;

        // make sure it's within embedding bounds
        assert(src0_item_int >= 0);
        assert(src0_item_int < src1->ne[0]);

        Coord out_coord = {0,0,0,0};
        for(int inside = 0; inside < src1->ne[1]; inside++) {
            out_coord.dim[0] = src0_coord.dim[0];
            out_coord.dim[1] = src0_coord.dim[1];
            out_coord.dim[2] = src0_coord.dim[2];
            out_coord.dim[3] = src0_coord.dim[3];

            out_coord.dim[src0->n_dims] = inside;

            Coord src1_coord = {src0_item_int, inside, 0, 0};

            // do the operation
            if (! grad_mode) {
                ffml_set_data(t, out_coord, ffml_get_data(src1, src1_coord));
            } else {
                ffml_inc_grad(src1, src1_coord, ffml_get_grad(t, out_coord));
            }
        }

        if (src0_coord.dim[0] == 0 && src0_coord.dim[1] == 0 && src0_coord.dim[2] == 0 && src0_coord.dim[3] == 0) {
            // exit the cycle here
            break;
        }

        // decrement coord
        for(int src0_i = src0->n_dims - 1; src0_i >= 0; src0_i--) {
            if (src0_coord.dim[src0_i] != 0) {
                src0_coord.dim[src0_i]--;
                break;
            } else {
                src0_coord.dim[src0_i] = src0->ne[src0_i] - 1;
            }
        }
    }
}

// Function object for two-operand operations
template <typename F>
inline void ffml_compute_binary_op(struct ffml_tensor* t, F f) {
    Coord out_coord;

    // init coords
    for(int out_i = 0; out_i < FFML_MAX_DIMS; out_i++) {
        out_coord.dim[out_i] = (out_i >= t->n_dims) ? 0 : t->ne[out_i] - 1; // going backwards so that we can exit the while loop when we reach 0
    }

    assert(FFML_MAX_DIMS == 4);
    while(true) {
        // compute coords
        Coord src0_coord = ffml_broadcast_coord_to_src0(t, out_coord);
        Coord src1_coord = ffml_broadcast_coord_to_src1(t, out_coord);

        // do the operation
        ffml_set_data(t, out_coord, f(ffml_get_data(t->src0, src0_coord), ffml_get_data(t->src1, src1_coord)));

        if (out_coord.dim[0] == 0 && out_coord.dim[1] == 0 && out_coord.dim[2] == 0 && out_coord.dim[3] == 0) {
            // exit the cycle here
            break;
        }

        // decrement coord
        for(int out_i = t->n_dims - 1; out_i >= 0; out_i--) {
            if (out_coord.dim[out_i] != 0) {
                out_coord.dim[out_i]--;
                break;
            } else {
                out_coord.dim[out_i] = t->ne[out_i] - 1;
            }
        }
    }
}

// Function object for single-operand operations
template <typename F>
inline void ffml_compute_unary_op(struct ffml_tensor* t, F f) {
    // there is no broadcasting for unary operations

    for(uint64_t i = 0; i < t->nelem; i++) {
        ffml_set_data_flat(t, i, f(ffml_get_data_flat(t->src0, i)));
    }
}

#define MATMUL_THREADING 1
#if MATMUL_THREADING
void _ffml_matmul(ffml_tensor* dst, ffml_tensor* arg0, ffml_tensor* arg1,
                           bool accum = false,
                           bool transpose_arg0 = false, bool transpose_arg1 = false,
                           bool dst_grad = false, bool arg0_grad = false, bool arg1_grad = false) {

    assert(dst->n_dims >= 2);
    assert(arg0->n_dims >= 2);
    assert(arg1->n_dims >= 2);

    // this is for the case when we want to transpose the arguments
    // argX_comp0 is the dimension of the argument that corresponds to the first dimension of the matrix
    int arg0_comp0 = 0;
    int arg0_comp1 = 1;
    int arg1_comp0 = 0;
    int arg1_comp1 = 1;
    if (transpose_arg0) {
        arg0_comp0 = 1;
        arg0_comp1 = 0;
    }
    if (transpose_arg1) {
        arg1_comp0 = 1;
        arg1_comp1 = 0;
    }
    // shift if we have more than 2 dimensions
    int extra_dimensions = dst->n_dims - 2;
    if (dst->n_dims > 2) {
        arg0_comp0 += extra_dimensions;
        arg0_comp1 += extra_dimensions;
        arg1_comp0 += extra_dimensions;
        arg1_comp1 += extra_dimensions;
    }

    // assert side dimensions
    assert(arg0->ne[ arg0_comp1 ] == arg1->ne[ arg1_comp0 ]);

    // assert output shape
    assert(dst->ne[0+extra_dimensions] == arg0->ne[ arg0_comp0 ]);
    assert(dst->ne[1+extra_dimensions] == arg1->ne[ arg1_comp1 ]);

    // for extra dimensions, they must match
    for (int i = 0; i < extra_dimensions; i++) {
        assert(dst->ne[i] == arg0->ne[i]);
        assert(dst->ne[i] == arg1->ne[i]);
    }

    assert(FFML_MAX_DIMS == 4);

    Coord coord0, coord1, coord_dst;

    if (!dst_grad) {
        // zero dst
        for (uint64_t i = 0; i < dst->nelem; i++) {
            ffml_set_data_flat(dst, i, 0.0f);
        }
    }

    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    int rows_per_thread = dst->ne[0] / num_threads;

    // printf("THREADS: %d\n", num_threads);
    // printf("ROWS PER THREAD: %d\n", rows_per_thread);

    // only break up work when it makes sense
    const int min_rows_per_thread = 100;
    if (rows_per_thread < min_rows_per_thread) {
        num_threads = 1;
        rows_per_thread = dst->ne[0];
    }

    // F
    auto _ffml_matmul_thread = [&](int start, int end, ffml_tensor* dst, ffml_tensor* arg0, ffml_tensor* arg1,
                           bool accum = false,
                           bool transpose_arg0 = false, bool transpose_arg1 = false,
                           bool dst_grad = false, bool arg0_grad = false, bool arg1_grad = false) {

        for (uint64_t extra_i = (extra_dimensions > 0 ? start : 0); extra_i < ((extra_dimensions > 0) ? end : 1); extra_i++) { // extra_i - iterating over extra dimension 1 if any
            for (uint64_t extra_j = 0; extra_j < (extra_dimensions > 1 ? dst->ne[1] : 1); extra_j++) { // extra_j - iterating over extra dimension 2 if any
                coord0.dim[0] = extra_i;
                coord0.dim[1] = extra_j;
                coord1.dim[0] = extra_i;
                coord1.dim[1] = extra_j;
                coord_dst.dim[0] = extra_i;
                coord_dst.dim[1] = extra_j;
        
                for (uint64_t i = (extra_dimensions > 0) ? 0 : start; i < ((extra_dimensions > 0) ? dst->ne[0+extra_dimensions] : end); i++) { // i - iterating over rows of dst
                    for (uint64_t j = 0; j < dst->ne[1+extra_dimensions]; j++) { // j - iterating over columns of dst
                        FFML_TYPE sum = 0.0f;
                        for (uint64_t k = 0; k < arg0->ne[ arg0_comp1 ]; k++) { // k - iterating over inner side: columns of arg0 and rows of arg1
                            if (transpose_arg0) {
                                coord0.dim[0+extra_dimensions] = k;
                                coord0.dim[1+extra_dimensions] = i;
                            } else {
                                coord0.dim[0+extra_dimensions] = i;
                                coord0.dim[1+extra_dimensions] = k;
                            }

                            if (transpose_arg1) {
                                coord1.dim[0+extra_dimensions] = j;
                                coord1.dim[1+extra_dimensions] = k;
                            } else {
                                coord1.dim[0+extra_dimensions] = k;
                                coord1.dim[1+extra_dimensions] = j;
                            }

                            auto multiplier1 = (arg0_grad) ? ffml_get_grad(arg0, coord0) : ffml_get_data(arg0, coord0);
                            auto multiplier2 = (arg1_grad) ? ffml_get_grad(arg1, coord1) : ffml_get_data(arg1, coord1);
                            
                            sum += multiplier1 * multiplier2;
                        }

                        coord_dst.dim[0+extra_dimensions] = i;
                        coord_dst.dim[1+extra_dimensions] = j;

                        if (accum) {
                            if (dst_grad) {
                                ffml_inc_grad(dst, coord_dst, sum);
                            } else {
                                ffml_inc_data(dst, coord_dst, sum);
                            }
                        } else {
                            if (dst_grad) {
                                ffml_set_grad(dst, coord_dst, sum);
                            } else {
                                ffml_set_data(dst, coord_dst, sum);
                            }
                        }
                    }
                }
            }
        }

    }; // end of lambda

    for(uint64_t i=0; i<num_threads; i++) {
        int start = i * rows_per_thread;
        int end = (i == num_threads - 1) ? dst->ne[0] : (i + 1) * rows_per_thread;
        threads.push_back(std::thread(_ffml_matmul_thread, start, end, dst, arg0, arg1, accum, transpose_arg0, transpose_arg1, dst_grad, arg0_grad, arg1_grad));
    }

    for(auto& thread: threads) {
        thread.join();
    }

}
#else
void _ffml_matmul(ffml_tensor* dst, ffml_tensor* arg0, ffml_tensor* arg1,
                           bool accum = false,
                           bool transpose_arg0 = false, bool transpose_arg1 = false,
                           bool dst_grad = false, bool arg0_grad = false, bool arg1_grad = false) {

    assert(dst->n_dims >= 2);
    assert(arg0->n_dims >= 2);
    assert(arg1->n_dims >= 2);

    // this is for the case when we want to transpose the arguments
    // argX_comp0 is the dimension of the argument that corresponds to the first dimension of the matrix
    int arg0_comp0 = 0;
    int arg0_comp1 = 1;
    int arg1_comp0 = 0;
    int arg1_comp1 = 1;
    if (transpose_arg0) {
        arg0_comp0 = 1;
        arg0_comp1 = 0;
    }
    if (transpose_arg1) {
        arg1_comp0 = 1;
        arg1_comp1 = 0;
    }
    // shift if we have more than 2 dimensions
    int extra_dimensions = dst->n_dims - 2;
    if (dst->n_dims > 2) {
        arg0_comp0 += extra_dimensions;
        arg0_comp1 += extra_dimensions;
        arg1_comp0 += extra_dimensions;
        arg1_comp1 += extra_dimensions;
    }

    // assert side dimensions
    assert(arg0->ne[ arg0_comp1 ] == arg1->ne[ arg1_comp0 ]);

    // assert output shape
    assert(dst->ne[0+extra_dimensions] == arg0->ne[ arg0_comp0 ]);
    assert(dst->ne[1+extra_dimensions] == arg1->ne[ arg1_comp1 ]);

    // for extra dimensions, they must match
    for (int i = 0; i < extra_dimensions; i++) {
        assert(dst->ne[i] == arg0->ne[i]);
        assert(dst->ne[i] == arg1->ne[i]);
    }

    assert(FFML_MAX_DIMS == 4);

    Coord coord0, coord1, coord_dst;

    if (!dst_grad) {
        // zero dst
        for (uint64_t i = 0; i < dst->nelem; i++) {
            ffml_set_data_flat(dst, i, 0.0f);
        }
    }


    for (uint64_t extra_i = 0; extra_i < (extra_dimensions > 0 ? dst->ne[0] : 1); extra_i++) { // extra_i - iterating over extra dimension 1 if any
        for (uint64_t extra_j = 0; extra_j < (extra_dimensions > 1 ? dst->ne[1] : 1); extra_j++) { // extra_j - iterating over extra dimension 2 if any
            coord0.dim[0] = extra_i;
            coord0.dim[1] = extra_j;
            coord1.dim[0] = extra_i;
            coord1.dim[1] = extra_j;
            coord_dst.dim[0] = extra_i;
            coord_dst.dim[1] = extra_j;
    
            for (uint64_t i = 0; i < dst->ne[0+extra_dimensions]; i++) { // i - iterating over rows of dst
                for (uint64_t j = 0; j < dst->ne[1+extra_dimensions]; j++) { // j - iterating over columns of dst
                    FFML_TYPE sum = 0.0f;
                    for (uint64_t k = 0; k < arg0->ne[ arg0_comp1 ]; k++) { // k - iterating over inner side: columns of arg0 and rows of arg1
                        if (transpose_arg0) {
                            coord0.dim[0+extra_dimensions] = k;
                            coord0.dim[1+extra_dimensions] = i;
                        } else {
                            coord0.dim[0+extra_dimensions] = i;
                            coord0.dim[1+extra_dimensions] = k;
                        }

                        if (transpose_arg1) {
                            coord1.dim[0+extra_dimensions] = j;
                            coord1.dim[1+extra_dimensions] = k;
                        } else {
                            coord1.dim[0+extra_dimensions] = k;
                            coord1.dim[1+extra_dimensions] = j;
                        }

                        auto multiplier1 = (arg0_grad) ? ffml_get_grad(arg0, coord0) : ffml_get_data(arg0, coord0);
                        auto multiplier2 = (arg1_grad) ? ffml_get_grad(arg1, coord1) : ffml_get_data(arg1, coord1);
                        
                        sum += multiplier1 * multiplier2;
                    }

                    coord_dst.dim[0+extra_dimensions] = i;
                    coord_dst.dim[1+extra_dimensions] = j;

                    if (accum) {
                        if (dst_grad) {
                            ffml_inc_grad(dst, coord_dst, sum);
                        } else {
                            ffml_inc_data(dst, coord_dst, sum);
                        }
                    } else {
                        if (dst_grad) {
                            ffml_set_grad(dst, coord_dst, sum);
                        } else {
                            ffml_set_data(dst, coord_dst, sum);
                        }
                    }
                }
            }
        }
    }
}
#endif

void ffml_cgraph_forward(struct ffml_cgraph * cgraph) {
    // go through all nodes in the graph
    // and compute their values

    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ffml_tensor * t = cgraph->nodes[i];

        if (t->is_view) {
            // nothing to do, just pass the data
            continue;
        }

        switch (t->op) {
            case FFML_OP_NONE: {
                // nothing to do
                break;
            }

            case FFML_OP_TRANSPOSE: {
                // nothing to do
                break;
            }

            case FFML_OP_INIT_FILL: {
                if (!t->init_ran) {
                    FFML_TYPE val = t->op_metadata->at("init_fill.value");
                    for (int j = 0; j < t->nelem; j++) ffml_set_data_flat(t, j, val);
                    t->init_ran = true;
                }
            }

            case FFML_OP_INIT_ARANGE: {
                if (!t->init_ran) {
                    FFML_TYPE val_start = t->op_metadata->at("init_arange.value_start");
                    FFML_TYPE val_end = t->op_metadata->at("init_arange.value_end");
                    FFML_TYPE nelem = t->nelem;

                    for (int j = 0; j < t->nelem; j++) {
                        FFML_TYPE val = val_start + (val_end - val_start) * (j / nelem);
                        ffml_set_data_flat(t, j, val);
                    }

                    t->init_ran = true;
                }
            }

            case FFML_OP_INIT_ZEROES: {
                if (!t->init_ran) {
                    for (int j = 0; j < t->nelem; j++) ffml_set_data_flat(t, j, 0.0f);
                    t->init_ran = true;
                }
            }

            case FFML_OP_INIT_ONES: {
                if (!t->init_ran) {
                    for (int j = 0; j < t->nelem; j++) ffml_set_data_flat(t, j, 1.0f);
                    t->init_ran = true;
                }
            }

            case FFML_OP_INIT_RND_UNIFORM: {
                if (!t->init_ran) {
                    for (int j = 0; j < t->nelem; j++) {
                        ffml_set_data_flat(t, j, ffml_rand_uniform(-1, 1)); // from -1 to 1
                    }
                    t->init_ran = true;
                }
                break;
            }

            case FFML_OP_INIT_RND_NORMAL: {
                if (!t->init_ran) {
                    for (int j = 0; j < t->nelem; j++) {
                        ffml_set_data_flat(t, j, ffml_rand_normal(0, 1)); // mean 0, stddev 1
                    }
                    t->init_ran = true;
                }
                break;
            }

            case FFML_OP_INIT_RND_NORMAL_KAIMING: {
                if (!t->init_ran) {
                    for (int j = 0; j < t->nelem; j++) {
                        FFML_TYPE val = ffml_rand_normal_kaiming(t->op_metadata->at("init_kaiming.gain"), t->op_metadata->at("init_kaiming.fan_in"));
                        ffml_set_data_flat(t, j, val);
                    }
                    t->init_ran = true;
                }
                break;
            }

            case FFML_OP_ADD: {
                ffml_compute_binary_op(t, std::plus<FFML_TYPE>());
                break;
            }

            case FFML_OP_SUB: {
                ffml_compute_binary_op(t, std::minus<FFML_TYPE>());
                break;
            }

            case FFML_OP_MUL: {
                ffml_compute_binary_op(t, std::multiplies<FFML_TYPE>());
                break;
            }

            case FFML_OP_DIV: {
                ffml_compute_binary_op(t, std::divides<FFML_TYPE>());
                break;
            }

            case FFML_OP_TANH: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return std::tanh(n); });
                break;
            }

            case FFML_OP_MAXPOOL2D: {
                assert(t->n_dims == 3);

                // todo: right now hardcoded, should always be 2x2 pool

                // iterate over all output channels (filters)
                for (uint64_t out_channel = 0; out_channel < t->ne[0]; out_channel++) {
                    // iterate over all output pixels
                    for (uint64_t out_x = 0; out_x < t->ne[1]; out_x++) {
                        for (uint64_t out_y = 0; out_y < t->ne[2]; out_y++) {
                            FFML_TYPE minus_inf = -std::numeric_limits<FFML_TYPE>::infinity();
                            FFML_TYPE datapoint0 = ffml_get_data(t->src0, {out_channel, out_x*2, out_y*2, 0});
                            FFML_TYPE datapoint1 = (out_x*2+1 < t->src0->ne[1]) ? ffml_get_data(t->src0, {out_channel, out_x*2+1, out_y*2, 0}) : minus_inf;
                            FFML_TYPE datapoint2 = (out_y*2+1 < t->src0->ne[2]) ? ffml_get_data(t->src0, {out_channel, out_x*2, out_y*2+1, 0}) : minus_inf;
                            FFML_TYPE datapoint3 = (out_x*2+1 < t->src0->ne[1] && out_y*2+1 < t->src0->ne[2]) ? ffml_get_data(t->src0, {out_channel, out_x*2+1, out_y*2+1, 0}) : minus_inf;

                            ffml_set_data(t, {out_channel, out_x, out_y, 0}, std::max(std::max(datapoint0, datapoint1), std::max(datapoint2, datapoint3)));
                        }
                    }
                }

                break;
            }

            case FFML_OP_POW: {
                ffml_compute_binary_op(t, [](FFML_TYPE n, FFML_TYPE p){ return std::pow(n, p); });
                break;
            }

            case FFML_OP_MEAN: {
                assert(t->n_dims == 1);

                // use cell 0 as accumulator
                ffml_set_data_flat(t, 0, 0);

                for (int j = 0; j < t->src0->nelem; j++) {
                    ffml_inc_data_flat(t, 0, ffml_get_data_flat(t->src0, j));
                }

                // divide by number of elements
                ffml_set_data_flat(t, 0, ffml_get_data_flat(t, 0) / t->src0->nelem);

                break;
            }

            case FFML_OP_MEAN_BATCHED: {
                assert(t->n_dims == 2);

                printf("finish this");
                exit(1);

                for(uint64_t i = 0; i < t->ne[0]; i++) {
                    // // use cell 0 as accumulator
                    // ffml_set_data(t, {i, 0}, 0);

                    // for (int j = 0; j < t->src0->ne[1]; j++) {
                    //     ffml_inc_data(t, {i, 0}, ffml_get_data(t->src0, {i, j}));
                    // }

                    // // divide by number of elements
                    // ffml_set_data(t, {i, 0}, ffml_get_data(t, {i, 0}) / t->src0->ne[1]);
                }

                // use cell 0 as accumulator
                ffml_set_data_flat(t, 0, 0);

                for (int j = 0; j < t->src0->nelem; j++) {
                    ffml_inc_data_flat(t, 0, ffml_get_data_flat(t->src0, j));
                }

                // divide by number of elements
                ffml_set_data_flat(t, 0, ffml_get_data_flat(t, 0) / t->src0->nelem);

                break;
            }

            case FFML_OP_SUM: {
                assert(t->n_dims == 1);

                ffml_set_data_flat(t, 0, 0);

                for (int j = 0; j < t->src0->nelem; j++) {
                    ffml_inc_data_flat(t, 0, ffml_get_data_flat(t->src0, j));
                }

                break;
            }

            case FFML_OP_RMS_NORM: {
                // assert shapes are the same
                for(int i=0; i<t->src0->n_dims; i++) {
                    assert(t->src0->ne[i] == t->ne[i]);
                }

                const float eps = 1e-6f; // TODO: make this a parameter

                FFML_TYPE sum = 0.0f;
                for(int64_t i = 0; i < t->src0->nelem; i++) {
                    FFML_TYPE data = ffml_get_data_flat(t->src0, i);
                    sum += data * data;
                }

                FFML_TYPE mean = sum / t->src0->nelem;

                FFML_TYPE scale = 1.0f/sqrtf(mean + eps);

                for(int64_t i = 0; i < t->src0->nelem; i++) {
                    FFML_TYPE data = ffml_get_data_flat(t->src0, i);
                    ffml_set_data_flat(t, i, data * scale);
                }

                break;
            }

            case FFML_OP_SQUARE: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return n * n; });
                break;
            }

            case FFML_OP_NEG: {
                ffml_compute_unary_op(t, std::negate<FFML_TYPE>());
                break;
            }

            case FFML_OP_EXP: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return std::exp(n); });
                break;
            }

            case FFML_OP_ABS: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return std::fabs(n); });
                break;
            }

            case FFML_OP_MATMUL: {
                // use matmul function
                _ffml_matmul(t, t->src0, t->src1);
                break;
            }

            case FFML_OP_SELECT: {
                _ffml_select(t, t->src0, t->src1);
                break;
            }

            case FFML_OP_RELU: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return std::max(n, (FFML_TYPE) 0); });
                break;
            }

            case FFML_OP_LEAKY_RELU: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return std::max(n, (FFML_TYPE) 0.01 * n); });
                break;
            }

            case FFML_OP_GELU: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return 0.5f * n * (1.0f + std::erff(n / sqrtf(2.0f))); });
                break;
            }

            case FFML_OP_GELU_APPROX_TANH: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return 0.5f * n * (1.0f + std::tanh(sqrtf(2.0 / M_PI) * (n + 0.044715f * n * n * n))); });
                break;
            }

            case FFML_OP_SIGMOID: {
                ffml_compute_unary_op(t, [](FFML_TYPE n){ return 1.0f / (1.0f + std::exp(-n)); });
                break;
            }

            case FFML_OP_REPEAT: {
                // of course, more efficient would be to implement as a custom view

                // TODO: implement support for rank > 2 tensors
                assert(t->n_dims == 2);
                assert(t->src0->n_dims == 2);
                assert(t->src1->n_dims == 2);

                const int nc  = t->ne[0];
                const int nr  = t->ne[1];
                const int nc0 = t->src0->ne[0];
                const int nr0 = t->src0->ne[1];
                const int ncnc0 = nc/nc0; // guaranteed to be an integer due to the check at creation
                const int nrnr0 = nr/nr0; // guaranteed to be an integer due to the check at creation

                for (int i = 0; i < ncnc0; i++) {
                    for (int j = 0; j < nrnr0; j++) {
                        for (int k = 0; k < nc0; k++) {
                            for (int l = 0; l < nr0; l++) {
                                ffml_set_data(t, {i*nc0+k, j*nr0+l, 0, 0}, ffml_get_data(t->src0, {k, l, 0, 0}));
                            }
                        }
                    }
                }

                break;
            }

            case FFML_OP_CONV2D: {
                // src0 is the input, dimensions: [in_channels, in_x, in_y]
                // src1 is the kernels, dimensions: [n_filters(out_channels), in_channels, kernel_x, kernel_y]
                // result is the output, dimensions: [n_filters(out_channels), out_x, out_y]

                // zero dst
                for (uint64_t i = 0; i < t->nelem; i++) {
                    ffml_set_data_flat(t, i, 0.0f);
                }

                // iterate over all output channels (filters)
                for (uint64_t out_channel = 0; out_channel < t->ne[0]; out_channel++) {
                    // iterate over all output pixels
                    for (uint64_t out_x = 0; out_x < t->ne[1]; out_x++) {
                        for (uint64_t out_y = 0; out_y < t->ne[2]; out_y++) {
                            // iterate over all input channels
                            for (uint64_t in_channel = 0; in_channel < t->src0->ne[0]; in_channel++) {
                                // iterate over all kernel pixels
                                for (uint64_t kernel_x = 0; kernel_x < t->src1->ne[2]; kernel_x++) {
                                    for (int kernel_y = 0; kernel_y < t->src1->ne[3]; kernel_y++) {
                                        // compute input pixel coordinates
                                        uint64_t in_x = out_x + kernel_x;
                                        uint64_t in_y = out_y + kernel_y;

                                        // fetch input pixel value
                                        FFML_TYPE in_pixel = 0;
                                        if (in_x >= 0 && in_x < t->src0->ne[1] && in_y >= 0 && in_y < t->src0->ne[2]) {
                                            Coord coord;
                                            coord.dim[0] = in_channel;
                                            coord.dim[1] = in_x;
                                            coord.dim[2] = in_y;
                                            in_pixel = ffml_get_data(t->src0, coord);
                                        }

                                        // fetch kernel pixel value
                                        Coord coord;
                                        coord.dim[0] = out_channel;
                                        coord.dim[1] = in_channel;
                                        coord.dim[2] = kernel_x;
                                        coord.dim[3] = kernel_y;
                                        FFML_TYPE kernel_pixel = ffml_get_data(t->src1, coord);

                                        // accumulate
                                        ffml_inc_data(t, {out_channel, out_x, out_y, 0}, in_pixel * kernel_pixel);

                                        // printf("accumulating %f * %f, already %f\n", in_pixel, kernel_pixel, ffml_get_data(t, {out_channel, out_x, out_y, 0}));
                                    }
                                }
                            }
                        }
                    }
                }

                break;
            }

            case FFML_OP_SOFTMAX: {
                assert(t->n_dims == 1);

                //Calculate the maximum of the tensor
                FFML_TYPE max_val = -std::numeric_limits<FFML_TYPE>::infinity();
                for (int j = 0; j < t->nelem; j++) {
                    max_val = std::max(max_val, ffml_get_data_flat(t->src0, j));
                }
                
                //Subtract max_val from all elements in the tensor and take the exponential of that
                //Also calculate the sum while we are here
                FFML_TYPE sum_exp = 0;
                for (int j = 0; j < t->nelem; j++) {
                    FFML_TYPE val = std::exp(ffml_get_data_flat(t->src0, j) - max_val);
                    ffml_set_data_flat(t, j, val);
                    sum_exp += val;
                }
                
                //Finally divide each element by sum_exp to normalize the tensor
                for (int j = 0; j < t->nelem; j++) {
                    FFML_TYPE val = ffml_get_data_flat(t, j) / sum_exp;
                    ffml_set_data_flat(t, j, val);
                }

                break;
            }

            case FFML_OP_SOFTMAX_CROSS_ENTROPY: {
                uint64_t batch_size;
                uint64_t elems;

                assert(t->n_dims == 1 || t->n_dims == 2);
                if (t->n_dims == 1) {
                    // unbatched
                    batch_size = 1;
                    elems = t->src0->nelem;
                } else {
                    batch_size = t->src0->ne[0];
                    elems = t->src0->ne[1];
                }

                // Allocate memory for softmax output
                FFML_TYPE* softmax_output = new FFML_TYPE[elems];

                for(uint64_t batchidx = 0; batchidx < batch_size; batchidx++) {
                    FFML_TYPE max_val = -std::numeric_limits<FFML_TYPE>::infinity();

                    uint64_t batchoffset = batchidx * elems;

                    // Calculate maximum value for stabilization
                    for (int j = 0; j < elems; j++) {
                        max_val = std::max(max_val, ffml_get_data_flat(t->src0, batchoffset + j));
                    }

                    // Subtract max_val from values, calculate exponentials, and sum them up
                    FFML_TYPE sum_exp_minus_max = 0;
                    for (int j = 0; j < elems; j++) {
                        sum_exp_minus_max += std::exp(ffml_get_data_flat(t->src0, batchoffset + j) - max_val);
                    }

                    FFML_TYPE log_sum_exp_minus_max = std::log(sum_exp_minus_max);

                    // Calculate log softmax by taking exp(val - max_val - log_sum_exp_minus_max).
                    // This will be log(e^val/sum(e^val)), which finally equals to val - max_val - log_sum_exp_minus_max.

                    for (int j = 0; j < elems; j++) {
                        FFML_TYPE val = ffml_get_data_flat(t->src0, batchoffset + j);
                        softmax_output[j] = val - max_val - log_sum_exp_minus_max;
                    }

                    FFML_TYPE loss = 0;
                    for (int j = 0; j < elems; j++) {
                        FFML_TYPE true_val = ffml_get_data_flat(t->src1, batchoffset + j);
                        if (true_val > 0.0f) { // Assuming true_val is the one-hot encoded vector.
                            FFML_TYPE log_softmax_val = softmax_output[j];
                            loss -= true_val * log_softmax_val;
                        }
                    }


                    // // Calculate softmax first
                    // FFML_TYPE max_val = -std::numeric_limits<FFML_TYPE>::infinity();
                    // for (int j = 0; j < elems; j++) {
                    //     max_val = std::max(max_val, ffml_get_data_flat(t->src0, j));
                    // }

                    // //Subtract max_val from all elements in the tensor and take the exponential of that
                    // //Also calculate the sum while we are here
                    // FFML_TYPE sum_exp = 0;
                    // for (int j = 0; j < elems; j++) {
                    //     FFML_TYPE val = std::exp(ffml_get_data_flat(t->src0, j) - max_val);
                    //     softmax_output[j] = val;
                    //     sum_exp += val;
                    // }

                    // //Finally divide each element by sum_exp to normalize the tensor
                    // for (int j = 0; j < elems; j++) {
                    //     softmax_output[j] /= sum_exp;
                    // }

                    // // Calculate cross entropy loss
                    // FFML_TYPE loss = 0;
                    // for (int j = 0; j < elems; j++) {
                    //     FFML_TYPE true_val = ffml_get_data_flat(t->src1, j);
                    //     if (true_val != 0) {
                    //         FFML_TYPE softmax_val = softmax_output[j];
                    //         FFML_TYPE _epsilon = 1e-7;
                    //         FFML_TYPE softmax_val_log = std::log(softmax_val + _epsilon);
                    //         loss -= true_val * softmax_val_log;
                    //     }
                    // }

                    // Put the final result in 't'
                    ffml_set_data_flat(t, batchidx + 0, loss);

                }


                // Free allocated memory
                delete[] softmax_output;

                break;
            }
            default: {
                assert(false);
            }
        }
    }
}

void ffml_debug_print_tensor_data(struct ffml_tensor * t) {
    printf("%s: ", t->name);

    // print address
    printf("%p | ", t->data);

    // print shape
    printf("shape[");
    for (int j = 0; j < t->n_dims; j++) {
        printf("%lu ", t->ne[j]);
    }
    printf("]=");

    // print size
    printf("%lu | ", t->nelem);

    // print strides
    printf("strides[");
    for (int j = 0; j < t->n_dims; j++) {
        printf("%lu ", t->nb[j]);
    }
    printf("]=");

    if (t->is_view) {
        printf("view of %s | ", t->src0->name);
    }

    // broadcasting
    bool br_src0 = (t->broadcast_to_src0_enabled[0] || t->broadcast_to_src0_enabled[1] || t->broadcast_to_src0_enabled[2] || t->broadcast_to_src0_enabled[3]);
    bool br_src1 = (t->broadcast_to_src1_enabled[0] || t->broadcast_to_src1_enabled[1] || t->broadcast_to_src1_enabled[2] || t->broadcast_to_src1_enabled[3]);
    if (true || br_src0 || br_src1) {
        printf("broadcasting: ");
        if (true || br_src0) {
            printf("src0[");
            for (int j = 0; j < t->n_dims; j++) {
                printf("%d", t->broadcast_dim_src0[j]);
                if (t->broadcast_to_src0_enabled[j]) {
                    printf("#");
                } else {
                    printf(" ");
                }
                printf(" ");
            }
            printf("] ");
        }
        if (true || br_src1) {
            printf("src1[");
            for (int j = 0; j < t->n_dims; j++) {
                printf("%d", t->broadcast_dim_src1[j]);
                if (t->broadcast_to_src1_enabled[j]) {
                    printf("#");
                } else {
                    printf(" ");
                }
                printf(" ");
            }
            printf("] ");
        }
        printf("| ");
    }

    printf("data: ");

    for (int j = 0; j < t->nelem; j++) {
        // printf("(%lu) %f ", j, ffml_get_data_flat(t, j));
        printf("%f ", ffml_get_data_flat(t, j));
    }

    printf(" | grad: ");

    for (int j = 0; j < t->nelem; j++) {
        printf("%f ", ffml_get_grad_flat(t, j));
    }

    printf("\n");
    printf("\n");
}

void ffml_debug_print_cgraph_data(struct ffml_cgraph * cgraph) {
    for (uint64_t i = 0; i < cgraph->n_nodes; i++) {
        struct ffml_tensor * t = cgraph->nodes[i];
        ffml_debug_print_tensor_data(t);
    }

    printf("\n");
}


void ffml_debug_print_cgraph_shapes(struct ffml_cgraph * cgraph) {
    for (uint64_t i = 0; i < cgraph->n_nodes; i++) {
        struct ffml_tensor * t = cgraph->nodes[i];
        printf("%s: ", t->name);

        // print shape
        printf("shape[");
        for (int j = 0; j < t->n_dims; j++) {
            printf("%lu ", t->ne[j]);
        }
        printf("]=");

        // print size
        // printf("size %lu | ", t->size_bytes);
        printf("%lu | ", t->nelem);

        printf("\n");
    }

    printf("\n");
}

void ffml_zerograd(struct ffml_cgraph * cgraph) {
    for (uint64_t i = 0; i < cgraph->n_nodes; i++) {
        struct ffml_tensor * t = cgraph->nodes[i];
        for (int j = 0; j < t->nelem; j++) {
            ffml_set_grad_flat(t, j, 0.0f);
        }
    }
}

void ffml_compute_binary_grad(struct ffml_tensor * t, std::function<void(struct ffml_tensor * t, struct ffml_tensor * src0, struct ffml_tensor * src1, Coord out_coord, Coord src0_coord, Coord src1_coord)> f) {
    Coord out_coord;

    // init coords
    for(int out_i = 0; out_i < FFML_MAX_DIMS; out_i++) {
        out_coord.dim[out_i] = (out_i >= t->n_dims) ? 0 : t->ne[out_i] - 1; // going backwards so that we can exit the while loop when we reach 0
    }

    assert(FFML_MAX_DIMS == 4);
    while(true) {
        // compute coords
        Coord src0_coord = ffml_broadcast_coord_to_src0(t, out_coord);
        Coord src1_coord = ffml_broadcast_coord_to_src1(t, out_coord);

        // do the operation
        f(t, t->src0, t->src1, out_coord, src0_coord, src1_coord);

        if (out_coord.dim[0] == 0 && out_coord.dim[1] == 0 && out_coord.dim[2] == 0 && out_coord.dim[3] == 0) {
            // exit the cycle here
            break;
        }

        // decrement coord
        for(int out_i = t->n_dims - 1; out_i >= 0; out_i--) {
            if (out_coord.dim[out_i] != 0) {
                out_coord.dim[out_i]--;
                break;
            } else {
                out_coord.dim[out_i] = t->ne[out_i] - 1;
            }
        }
    }
}

void ffml_cgraph_backward(struct ffml_cgraph * cgraph) {
    const uint64_t last_node_index = cgraph->n_nodes - 1;

    // set gradient of last node to 1
    struct ffml_tensor * t = cgraph->nodes[last_node_index];
    for(uint64_t i = 0; i < t->nelem; i++) {
        ffml_set_grad_flat(t, i, 1.0f);
    }

    // go through all nodes in the graph
    // and compute the parents' gradients

    for(uint64_t i = last_node_index; i > 0; i--) {
        struct ffml_tensor * t = cgraph->nodes[i];

        if (t->is_view) {
            // nothing to do, just pass the gradient
            // presumably, the higher operation already passed the gradient to our source
            continue;
        }

        switch (t->op) {
            // nothing to do
            case FFML_OP_NONE: {
                break;
            }
            case FFML_OP_INIT_ZEROES: {
                break;
            }
            case FFML_OP_INIT_FILL: {
                break;
            }
            case FFML_OP_INIT_ARANGE: {
                break;
            }
            case FFML_OP_INIT_ONES: {
                break;
            }
            case FFML_OP_INIT_RND_UNIFORM: {
                break;
            }
            case FFML_OP_INIT_RND_NORMAL: {
                break;
            }
            case FFML_OP_INIT_RND_NORMAL_KAIMING: {
                break;
            }

            case FFML_OP_ADD: {
                ffml_compute_binary_grad(t, [] (struct ffml_tensor * t, struct ffml_tensor * src0, struct ffml_tensor * src1, Coord out_coord, Coord src0_coord, Coord src1_coord) -> void {
                    ffml_inc_grad(src0, src0_coord, ffml_get_grad(t, out_coord));
                    ffml_inc_grad(src1, src1_coord, ffml_get_grad(t, out_coord));
                });
                break;
            }

            case FFML_OP_SUB: {
                ffml_compute_binary_grad(t, [] (struct ffml_tensor * t, struct ffml_tensor * src0, struct ffml_tensor * src1, Coord out_coord, Coord src0_coord, Coord src1_coord) -> void {
                    ffml_inc_grad(src0, src0_coord, ffml_get_grad(t, out_coord));
                    ffml_inc_grad(src1, src1_coord, -ffml_get_grad(t, out_coord));
                });
                break;
            }

            case FFML_OP_MUL: {
                ffml_compute_binary_grad(t, [] (struct ffml_tensor * t, struct ffml_tensor * src0, struct ffml_tensor * src1, Coord out_coord, Coord src0_coord, Coord src1_coord) -> void {
                    ffml_inc_grad(src0, src0_coord, ffml_get_grad(t, out_coord) * ffml_get_data(src1, src1_coord));
                    ffml_inc_grad(src1, src1_coord, ffml_get_grad(t, out_coord) * ffml_get_data(src0, src0_coord));
                });
                break;
            }

            case FFML_OP_DIV: {
                // todo: maybe replacing with inverse mul is faster?
                ffml_compute_binary_grad(t, [] (struct ffml_tensor * t, struct ffml_tensor * src0, struct ffml_tensor * src1, Coord out_coord, Coord src0_coord, Coord src1_coord) -> void {
                    ffml_inc_grad(src0, src0_coord, ffml_get_grad(t, out_coord) / ffml_get_data(src1, src1_coord));
                    ffml_inc_grad(src1, src1_coord, -ffml_get_grad(t, out_coord) * ffml_get_data(src0, src0_coord) / (ffml_get_data(src1, src1_coord) * ffml_get_data(src1, src1_coord)));
                });
                break;
            }

            case FFML_OP_TANH: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    FFML_TYPE tanh = ffml_get_data_flat(t, i);
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * (1.0f - tanh * tanh));
                }
                break;
            }

            case FFML_OP_POW: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    ffml_compute_binary_grad(t, [] (struct ffml_tensor * t, struct ffml_tensor * src0, struct ffml_tensor * src1, Coord out_coord, Coord src0_coord, Coord src1_coord) -> void {
                        FFML_TYPE n = ffml_get_data(src0, src0_coord);
                        FFML_TYPE p = ffml_get_data(src1, src1_coord);
                        assert(src1->op == FFML_OP_NONE);
                        ffml_inc_grad(src0, src0_coord, ffml_get_grad(t, out_coord) * p * std::pow(n, p - 1.0f));
                        ffml_inc_grad(src1, src1_coord, 0.0f);
                    });
                }
                break;
            }

            case FFML_OP_SQUARE: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * 2.0f * ffml_get_data_flat(t, i));
                }
                break;
            }

            case FFML_OP_NEG: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    ffml_inc_grad_flat(t->src0, i, -ffml_get_grad_flat(t, i));
                }
                break;
            }

            case FFML_OP_EXP: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * ffml_get_data_flat(t, i));
                }
                break;
            }

            case FFML_OP_ABS: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * (ffml_get_data_flat(t, i) > 0 ? 1.0f : -1.0f));
                }
                break;
            }

            case FFML_OP_MEAN: {
                assert(t->n_dims == 1);

                for (int j = 0; j < t->src0->nelem; j++) {
                    ffml_inc_grad_flat(t->src0, j, ffml_get_grad_flat(t, 0) / t->src0->nelem);
                }

                break;
            }

            case FFML_OP_SUM: {
                assert(t->n_dims == 1);

                for (int j = 0; j < t->src0->nelem; j++) {
                    ffml_inc_grad_flat(t->src0, j, ffml_get_grad_flat(t, 0));
                }

                break;
            }

            case FFML_OP_RMS_NORM: {
                for(int i=0; i<t->src0->n_dims; i++) {
                    assert(t->src0->ne[i] == t->ne[i]);
                }

                const float eps = 1e-6f; 

                FFML_TYPE sum = 0.0f;
                for(int64_t i = 0; i < t->src0->nelem; i++) {
                    FFML_TYPE data = ffml_get_data_flat(t->src0, i);
                    sum += data * data;
                }

                FFML_TYPE mean = sum / t->src0->nelem;
                FFML_TYPE scale = 1.0f/sqrtf(mean + eps);
                FFML_TYPE grad_scale= 0.0f;
                for(int64_t i = 0; i < t->src0->nelem; i++) {
                    FFML_TYPE data = ffml_get_grad_flat(t, i) * ffml_get_data_flat(t->src0, i);
                    grad_scale += data;
                }
                grad_scale *= powf(scale, 3) / 2;
                
                for(int64_t i = 0; i < t->src0->nelem; i++) {
                    FFML_TYPE grad = ffml_get_grad_flat(t, i) * scale - 2 * ffml_get_data_flat(t->src0, i) * grad_scale / t->src0->nelem;
                    ffml_inc_grad_flat(t->src0, i, grad);
                }
                break;
            }

            case FFML_OP_MATMUL: {
                // use matmul function
                _ffml_matmul(t->src0, t, t->src1, true, false, true, true, true, false);
                _ffml_matmul(t->src1, t->src0, t, true, true, false, true, false, true);
                break;
            }

            case FFML_OP_SELECT: {
                _ffml_select(t, t->src0, t->src1, true);
                break;
            }

            case FFML_OP_MAXPOOL2D: {
                assert(t->n_dims == 3);

                // todo: right now hardcoded, should always be 2x2 pool

                // iterate over all output channels (filters)
                for (uint64_t out_channel = 0; out_channel < t->ne[0]; out_channel++) {
                    // iterate over all output pixels
                    for (uint64_t out_x = 0; out_x < t->ne[1]; out_x++) {
                        for (uint64_t out_y = 0; out_y < t->ne[2]; out_y++) {

                            // input coords are:
                            // [out_channel, out_x*2, out_y*2, 0]
                            // [out_channel, out_x*2+1, out_y*2, 0]
                            // [out_channel, out_x*2, out_y*2+1, 0]
                            // [out_channel, out_x*2+1, out_y*2+1, 0]

                            // to backpropagate, we need to find the argmax coordinate
                            // and affect the gradient there

                            uint64_t in_channel = out_channel;

                            uint64_t max_x = 0;
                            uint64_t max_y = 0;
                            FFML_TYPE max_val = -std::numeric_limits<FFML_TYPE>::infinity();

                            // iterate over all pool pixels in the input
                            uint64_t x_howmany = (out_x*2+1 < t->src0->ne[1]) ? 2 : 1;
                            uint64_t y_howmany = (out_y*2+1 < t->src0->ne[2]) ? 2 : 1;
                            for (uint64_t pool_x = 0; pool_x < x_howmany; pool_x++) {
                                for (uint64_t pool_y = 0; pool_y < y_howmany; pool_y++) {
                                    // compute input pixel coordinates
                                    uint64_t in_x = out_x * 2 + pool_x;
                                    uint64_t in_y = out_y * 2 + pool_y;

                                    // fetch input pixel value
                                    FFML_TYPE in_pixel = 0;
                                    assert(in_x >= 0);
                                    assert(in_y >= 0);
                                    assert(in_x < t->src0->ne[1]);
                                    assert(in_y < t->src0->ne[2]);

                                    in_pixel = ffml_get_data(t->src0, {in_channel, in_x, in_y, 0});

                                    if (in_pixel > max_val) {
                                        max_val = in_pixel;
                                        max_x = in_x;
                                        max_y = in_y;
                                    }
                                }
                            }

                            // affect the gradient
                            ffml_inc_grad(t->src0, {in_channel, max_x, max_y, 0}, ffml_get_grad(t, {out_channel, out_x, out_y, 0}));

                        }
                    }
                }

                break;
            }

            case FFML_OP_CONV2D: {
                // src0 is the input, dimensions: [in_channels, in_x, in_y]
                // src1 is the kernels, dimensions: [n_filters(out_channels), in_channels, kernel_x, kernel_y]
                // result is the output, dimensions: [n_filters(out_channels), out_x, out_y]

                // zero grad should take care of making values 0s

                // iterate over all output channels (filters)
                for (uint64_t out_channel = 0; out_channel < t->ne[0]; out_channel++) {
                    // iterate over all output pixels
                    for (uint64_t out_x = 0; out_x < t->ne[1]; out_x++) {
                        for (uint64_t out_y = 0; out_y < t->ne[2]; out_y++) {
                            // iterate over all input channels
                            for (uint64_t in_channel = 0; in_channel < t->src0->ne[0]; in_channel++) {
                                // iterate over all kernel pixels
                                for (uint64_t kernel_x = 0; kernel_x < t->src1->ne[2]; kernel_x++) {
                                    for (uint64_t kernel_y = 0; kernel_y < t->src1->ne[3]; kernel_y++) {
                                        // compute input pixel coordinates
                                        uint64_t in_x = out_x + kernel_x;
                                        uint64_t in_y = out_y + kernel_y;

                                        // fetch input pixel value
                                        FFML_TYPE in_pixel = 0;
                                        if (in_x >= 0 && in_x < t->src0->ne[1] && in_y >= 0 && in_y < t->src0->ne[2]) {
                                            Coord coord;
                                            coord.dim[0] = in_channel;
                                            coord.dim[1] = in_x;
                                            coord.dim[2] = in_y;
                                            in_pixel = ffml_get_data(t->src0, coord);
                                        }

                                        // fetch kernel pixel value
                                        Coord coord;
                                        coord.dim[0] = out_channel;
                                        coord.dim[1] = in_channel;
                                        coord.dim[2] = kernel_x;
                                        coord.dim[3] = kernel_y;
                                        FFML_TYPE kernel_pixel = ffml_get_data(t->src1, coord);

                                        // accumulate
                                        ffml_inc_grad(t->src0, {in_channel, in_x, in_y, 0}, ffml_get_grad(t, {out_channel, out_x, out_y, 0}) * kernel_pixel);
                                        ffml_inc_grad(t->src1, coord, ffml_get_grad(t, {out_channel, out_x, out_y, 0}) * in_pixel);
                                    }
                                }
                            }
                        }
                    }
                }

                break;
            }

            case FFML_OP_SOFTMAX: {
                // Allocate memory to store intermidiate gradients
                FFML_TYPE* gradients = new FFML_TYPE[t->nelem * t->nelem]();
                
                // First calculate intermediate gradients, for each class 'i', applying the derivative rules, 
                // considering all other classes 'j'.
                for (int i = 0; i < t->nelem; ++i) {
                    for (int j = 0; j < t->nelem; ++j) {
                        FFML_TYPE si = ffml_get_data_flat(t, i);    // softmax of i-th class
                        FFML_TYPE sj = ffml_get_data_flat(t, j);    // softmax of j-th class
                        gradients[i * t->nelem + j] = (i == j ? 1 : 0) * si - si * sj;
                    }
                }
                
                // Then multiply these intermediate gradients by the upstream gradients to get the gradients 
                // we need to backpropagate to the previous layer.
                for (int i = 0; i < t->nelem; ++i) {
                    FFML_TYPE grad = 0;
                    for (int j = 0; j < t->nelem; ++j) {
                        grad += gradients[i * t->nelem + j] * ffml_get_grad_flat(t, j);
                    }
                    // Add the calculated gradient contribution to the source tensor gradient.
                    ffml_inc_grad_flat(t->src0, i, grad);
                }
                
                // free the memory allocated for gradients
                delete[] gradients; // todo: maybe rewrite or delete the op. it needs extra unplanned memory
                break;
            }

            case FFML_OP_SOFTMAX_CROSS_ENTROPY: {
                uint64_t batch_size;
                uint64_t elems;

                if (t->n_dims == 1) {
                    // unbatched
                    batch_size = 1;
                    elems = t->src0->nelem;
                } else {
                    batch_size = t->src0->ne[0];
                    elems = t->src0->ne[1];
                }

                // Allocate memory for softmax output
                FFML_TYPE* softmax_output = new FFML_TYPE[elems];

                for(uint64_t batchidx = 0; batchidx < batch_size; batchidx++) {
                    uint64_t batchoffset = batchidx * elems;

                    // Calculate softmax first
                    FFML_TYPE max_val = -std::numeric_limits<FFML_TYPE>::infinity();
                    for (int j = 0; j < elems; j++) {
                        max_val = std::max(max_val, ffml_get_data_flat(t->src0, batchoffset + j));
                    }

                    //Subtract max_val from all elements in the tensor and take the exponential of that
                    //Also calculate the sum while we are here
                    FFML_TYPE sum_exp = 0;
                    for (int j = 0; j < elems; j++) {
                        FFML_TYPE val = std::exp(ffml_get_data_flat(t->src0, batchoffset + j) - max_val);
                        softmax_output[j] = val;
                        sum_exp += val;
                    }

                    //Finally divide each element by sum_exp to normalize the tensor
                    for (int j = 0; j < elems; j++) {
                        softmax_output[j] /= sum_exp;
                    }

                    // Backward pass for softmax with cross entropy loss is simple
                    // The gradient of the loss with respect to the input of the softmax operation
                    // is just (softmax(x) - y)
                    for (int j = 0; j < elems; j++) {
                        FFML_TYPE y = ffml_get_data_flat(t->src1, batchoffset + j); // one-hot encoded true label, either 0 or 1
                        FFML_TYPE grad = softmax_output[j] - y;
                        ffml_inc_grad_flat(t->src0, batchoffset + j, grad * ffml_get_grad_flat(t, batchidx + 0));
                    }
                }

                // Free allocated memory
                delete[] softmax_output;

                break;
            }

            case FFML_OP_RELU: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    FFML_TYPE relu = ffml_get_data_flat(t, i);
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * (relu > 0.0f ? 1.0f : 0.0f));
                }
                break;
            }

            case FFML_OP_LEAKY_RELU: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    FFML_TYPE leaky_relu = ffml_get_data_flat(t, i);
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * (leaky_relu > 0.0f ? 1.0f : 0.01f));
                }
                break;
            }

            case FFML_OP_GELU: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    FFML_TYPE x = ffml_get_data_flat(t->src0, i);
                    FFML_TYPE grad = ffml_get_grad_flat(t, i);

                    float kAlpha = sqrtf(0.5f);
                    float kBeta = (2 / sqrtf(M_PI)) * sqrtf(0.5f) * 0.5f;
                    float cdf = 0.5f * (1 + std::erff(x * kAlpha));
                    float pdf = kBeta * expf(-0.5f * x * x);
                    float derivative = (cdf + x * pdf);

                    ffml_inc_grad_flat(t->src0, i, grad * derivative);
                }
                break;
            }

            case FFML_OP_GELU_APPROX_TANH: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    FFML_TYPE x = ffml_get_data_flat(t->src0, i);
                    FFML_TYPE grad = ffml_get_grad_flat(t, i);

                    float kBeta = sqrtf(2.0f) * (2.0f / sqrtf(M_PI)) * 0.5f;
                    float kKappa = 0.044715f;
                    float x_sq = x * x;
                    float x_cube = x_sq * x;
                    float inner = kBeta * (x + kKappa * x_cube);
                    float tanh_inner = tanhf(inner);

                    float left = 0.5f * x;
                    float right = 1.0f + tanh_inner;

                    float left_derivative = 0.5f * right;
                    
                    float tanh_derivative = 1.0f - tanh_inner * tanh_inner;
                    float inner_derivative = kBeta * (1.0f + 3.0f * kKappa * x_sq);
                    float right_derivative = left * tanh_derivative * inner_derivative;

                    float derivative = left_derivative + right_derivative;

                    ffml_inc_grad_flat(t->src0, i, grad * derivative);
                }
                break;
            }

            case FFML_OP_SIGMOID: {
                for(uint64_t i = 0; i < t->nelem; i++) {
                    FFML_TYPE sigmoid = ffml_get_data_flat(t, i);
                    ffml_inc_grad_flat(t->src0, i, ffml_get_grad_flat(t, i) * sigmoid * (1.0f - sigmoid));
                }
                break;
            }

            case FFML_OP_REPEAT: {
                assert(t->n_dims == 2);
                assert(t->src0->n_dims == 2);
                assert(t->src1->n_dims == 2);

                const int nc  = t->ne[0];
                const int nr  = t->ne[1];
                const int nc0 = t->src0->ne[0];
                const int nr0 = t->src0->ne[1];
                const int ncnc0 = nc/nc0; // guaranteed to be an integer due to the check at creation
                const int nrnr0 = nr/nr0; // guaranteed to be an integer due to the check at creation

                for (int i = 0; i < ncnc0; i++) {
                    for (int j = 0; j < nrnr0; j++) {
                        for (int k = 0; k < nc0; k++) {
                            for (int l = 0; l < nr0; l++) {
                                ffml_inc_grad(t->src0, {k, l, 0, 0}, ffml_get_grad(t, {i*nc0 + k, j*nr0 + l, 0, 0}));
                            }
                        }
                    }
                }

                break;
            }

            default: {
                assert(false);
            }
        }
    }
}

// void ffml_save(struct ffml_cgraph * cgraph, const char * filename) {
//     FILE * f = fopen(filename, "wb");
//     assert(f != NULL);

//     // write number of nodes
//     fwrite(&cgraph->n_nodes, sizeof(uint64_t), 1, f);

//     // write nodes
//     for (uint64_t i = 0; i < cgraph->n_nodes; i++) {
//         struct ffml_tensor * t = cgraph->nodes[i];

//         // write tensor data
//         fwrite(t, sizeof(struct ffml_tensor), 1, f);

//         // write relative pointers
//         struct ffml_tensor * root = (struct ffml_tensor *) cgraph->nodes[0]->data;
//         uint64_t root_addr = (uint64_t) root;
//         void * t_data_addr = t->data;
//         // struct ffml_tensor * t_src0 = t->src0;
//         // struct ffml_tensor * t_src1 = t->src1;
//         void * t_grad_addr = t->grad;
//         uint64_t t_data_rel_addr = (uint64_t) t_data_addr - root_addr;
//         // uint64_t t_src0_rel_addr = (uint64_t) t_src0 - root_addr;
//         // uint64_t t_src1_rel_addr = (uint64_t) t_src1 - root_addr;
//         uint64_t t_grad_rel_addr = (uint64_t) t_grad_addr - root_addr;
//         fwrite(&t_data_rel_addr, sizeof(uint64_t), 1, f);
//         // fwrite(&t_src0_rel_addr, sizeof(uint64_t), 1, f);
//         // fwrite(&t_src1_rel_addr, sizeof(uint64_t), 1, f);
//         fwrite(&t_grad_rel_addr, sizeof(uint64_t), 1, f);

//         // find src0 and src1
//         uint64_t src0_idx = UINT64_MAX;
//         uint64_t src1_idx = UINT64_MAX;
//         for(uint64_t q = 0; q < i; q++) {
//             struct ffml_tensor * t2 = cgraph->nodes[q];
//             if (t2 == t->src0) {
//                 src0_idx = q;
//             }
//             if (t2 == t->src1) {
//                 src1_idx = q;
//             }
//         }
//         fwrite(&src0_idx, sizeof(uint64_t), 1, f);
//         fwrite(&src1_idx, sizeof(uint64_t), 1, f);

//         // printf("root addr: %lu\n", root_addr);
//         // printf("root addr hex: %p\n", root);
//         // printf("data addr: %lu\n", (uint64_t) t_data_addr);
//         // printf("data addr hex: %p\n", t->data);
//         // printf("relative data addr: %lu\n", t_data_rel_addr);

//         // write data
//         if (! t->is_view) {
//             fwrite(t->data, sizeof(FFML_TYPE) * t->nelem, 1, f);
//         }

//         // // write grad
//         // fwrite(t->grad, sizeof(FFML_TYPE), t->nelem, f);
//     }

//     fclose(f);

//     // get inputs tensor
//     struct ffml_tensor * inputs = ffml_get_tensor_by_name(cgraph, "inputs");
//     // print metadata
//     ffml_debug_print_tensor_metadata(inputs);
// }

// struct ffml_cgraph * ffml_load(const char * filename, ffml_memory_pool * pool) {
//     FILE * f = fopen(filename, "rb");
//     assert(f != NULL);

//     struct ffml_cgraph * cgraph = new ffml_cgraph();
//     cgraph->n_nodes = 0;
//     cgraph->n_leafs = 0;

//     int n_nodes = 0;

//     // read number of nodes
//     fread(&n_nodes, sizeof(uint64_t), 1, f);

//     // printf("n_nodes: %lu\n", n_nodes);

//     // read nodes
//     for (uint64_t i = 0; i < n_nodes; i++) {
//         struct ffml_tensor * t = new ffml_tensor();

//         // read node
//         fread(t, sizeof(struct ffml_tensor), 1, f);

//         // read relative pointers
//         uint64_t t_data_rel_addr = 0;
//         uint64_t t_grad_rel_addr = 0;
//         fread(&t_data_rel_addr, sizeof(uint64_t), 1, f);
//         fread(&t_grad_rel_addr, sizeof(uint64_t), 1, f);

//         struct ffml_tensor * root = (struct ffml_tensor *) pool->data;

//         t->data = (void *) ((uint64_t) root + t_data_rel_addr);
//         t->grad = (void *) ((uint64_t) root + t_grad_rel_addr);

//         t->init_ran = true;

//         // read src0 and src1 indices
//         uint64_t src0_idx = 0;
//         uint64_t src1_idx = 0;
//         fread(&src0_idx, sizeof(uint64_t), 1, f);
//         fread(&src1_idx, sizeof(uint64_t), 1, f);

//         // point them
//         if (src0_idx == UINT64_MAX) {
//             t->src0 = NULL;
//         } else {
//             t->src0 = cgraph->nodes[src0_idx];
//             // printf("src0: %s\n", t->src0->name);
//         }
//         if (src1_idx == UINT64_MAX) {
//             t->src1 = NULL;
//         } else {
//             t->src1 = cgraph->nodes[src1_idx];
//             // printf("src1: %s\n", t->src1->name);
//         }

//         // ffml_debug_print_tensor_metadata(t);

//         // todo: make sure it doesn't go beyond the pool, and adjust pool used bytes too

//         // read data
//         if (! t->is_view) {
//             fread(t->data, sizeof(FFML_TYPE) * t->nelem, 1, f);
//         }

//         // // read grad
//         // t->grad = new FFML_TYPE[t->nelem];
//         // fread(t->grad, sizeof(FFML_TYPE), t->nelem, f);

//         cgraph->nodes[cgraph->n_nodes++] = t;

//         // if (cgraph->n_nodes % 10000 == 0) {
//         //     printf("loaded %lu nodes\n", cgraph->n_nodes);
//         // }

//         if (i > 10 * 1000 * 1000) {
//             printf("tried to load more than 10M nodes, something is wrong\n");
//             exit(1);
//         }
//     }

//     fclose(f);

//     return cgraph;
// }

struct ffml_tensor * ffml_get_tensor_by_name(struct ffml_cgraph * cgraph, const char * name) {
    for (uint64_t i = 0; i < cgraph->n_nodes; i++) {
        struct ffml_tensor * t = cgraph->nodes[i];
        if (strncmp(t->name, name, MAX_TENSOR_NAME_BYTES) == 0) {
            return t;
        }
    }

    return NULL;
}

struct ffml_tensor * ffml_get_tensor_by_key(struct ffml_cgraph * cgraph, uint64_t key) {
    if (key >= cgraph->n_nodes) {
        return NULL;
    }
    return cgraph->nodes[key];
}

void ffml_set_name(struct ffml_tensor * t, const char * name) {
    for (int i = 0; i < MAX_TENSOR_NAME_BYTES; i++) {
        t->name[i] = '\0';
    }
    char * name_ptr = (char *) name;
    char * t_name_ptr = (char *) t->name;
    while (*name_ptr != '\0') {
        *t_name_ptr = *name_ptr;
        name_ptr++;
        t_name_ptr++;

        if (t_name_ptr - t->name >= MAX_TENSOR_NAME_BYTES) {
            break;
        }
    }
}

void ffml_set_name(struct ffml_tensor * t, std::string name) {
    ffml_set_name(t, name.c_str());
}

struct ffml_tensor * ffml_reshape(struct ffml_tensor * src, int n_dims, Coord dims) {
    assert(n_dims <= FFML_MAX_DIMS);
    
    ffml_calc_cached_(src, false);

    // check if the number of elements is the same
    uint64_t nelem = 1;
    for (int i = 0; i < n_dims; i++) {
        nelem *= dims.dim[i];
    }
    assert(nelem == src->nelem);

    struct ffml_tensor * t = ffml_tensor_create(n_dims, dims, (std::string(src->name) + "_reshaped").c_str());

    t->src0 = src;
    t->is_view = true;
    t->op = FFML_OP_NONE;
    
    for (int i = 0; i < n_dims; i++) {
        t->ne[i] = dims.dim[i];
    }

    ffml_calc_cached_(t, true); // true will automatically calculate strides

    return t;
}

struct ffml_tensor * ffml_flatten(struct ffml_tensor * src) {
    ffml_calc_cached_(src, false);
    return ffml_reshape(src, 1, {src->nelem,0,0,0});
}

void ffml_debug_print_tensor_metadata(ffml_tensor * t) {
    printf("tensor: %s\n", t->name);
    printf("n_dims: %d\n", t->n_dims);
    printf("nelem: %lu\n", t->nelem);
    printf("ne: ");
    for(int i = 0; i < t->n_dims; i++) {
        printf("%lu ", t->ne[i]);
    }
    printf("\n");
    printf("nb: ");
    for(int i = 0; i < t->n_dims; i++) {
        printf("%lu ", t->nb[i]);
    }
    printf("\n");
    printf("data: %p\n", t->data);
    printf("grad: %p\n", t->grad);
    printf("op: %d\n", t->op);

    // is view?
    printf("is_view: %d\n", t->is_view);

    // init ran?
    printf("init_ran: %d\n", t->init_ran);

    // printf("requires_grad: %d\n", t->requires_grad);
    // printf("is_leaf: %d\n", t->is_leaf);
    // printf("is_parameter: %d\n", t->is_parameter);
    // printf("is_constant: %d\n", t->is_constant);
    // printf("is_variable: %d\n", t->is_variable);
    // printf("is_placeholder: %d\n", t->is_placeholder);
    // printf("is_computed: %d\n", t->is_computed);
    // printf("is_grad_computed: %d\n", t->is_grad_computed);
    // printf("is_grad_accumulated: %d\n", t->is_grad_accumulated);
    // printf("is_grad_zero: %d\n", t->is_grad_zero);
    // printf("is_grad_constant: %d\n", t->is_grad_constant);
    // printf("is_grad_variable: %d\n", t->is_grad_variable);
    // printf("is_grad_placeholder: %d\n", t->is_grad_placeholder);
    // printf("is_grad_computed: %d\n", t->is_grad_computed);
    // printf("is_grad_accumulated: %d\n", t->is_grad_accumulated);
    // printf("is_grad_zero: %d\n", t->is_grad_zero);
    // printf("is_grad_constant: %d\n", t->is_grad_constant);
    // printf("is_grad_variable: %d\n", t->is_grad_variable);
    // printf("is_grad_placeholder: %d\n", t->is_grad_placeholder);
    // printf("is_grad_computed: %d
}
