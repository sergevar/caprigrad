#pragma once

// Most of this file is taken from llama.cpp by ggerganov

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <inttypes.h>
#include <queue>
#include <map>

#define DEBUG_MODE true

#include "../../ffml/ffml.h"

#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#else
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>

// memcpy
#include <string.h>

#endif

#define Min(X, Y) ((Y) > (X) ? (X) : (Y))
#define Max(X, Y) ((Y) < (X) ? (X) : (Y))

#define LLAMA_FILE_VERSION 1
#define LLAMA_FILE_MAGIC 0x67676a74 // 'ggjt' in hex
#define LLAMA_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

// FP16 <-> FP32
// ref: https://github.com/Maratyszcza/FP16

static inline float fp32_from_bits(uint32_t w) {
    union {
        uint32_t as_bits;
        float as_value;
    } fp32;
    fp32.as_bits = w;
    return fp32.as_value;
}

static inline uint32_t fp32_to_bits(float f) {
	union {
		float as_value;
		uint32_t as_bits;
	} fp32;
	fp32.as_value = f;
	return fp32.as_bits;
}

static inline float compute_fp16_to_fp32(uint16_t h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L) || defined(__GNUC__) && !defined(__STRICT_ANSI__)
    const float exp_scale = 0x1.0p-112f;
#else
    const float exp_scale = fp32_from_bits(UINT32_C(0x7800000));
#endif
    const float normalized_value = fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value) : fp32_to_bits(normalized_value));
    return fp32_from_bits(result);
}

static void *mmap_file(const char *fname, uint64_t *mm_length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    HANDLE hFile = CreateFileA(fname,
                               GENERIC_READ,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                               NULL,
                               OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
                               NULL);
    if (hFile == INVALID_HANDLE_VALUE) return 0;
    LARGE_INTEGER fileSize;
    fileSize.QuadPart = -1;
    GetFileSizeEx(hFile, &fileSize);
    int64_t length = fileSize.QuadPart;
    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    CloseHandle(hFile);
    if (!hMapping) return 0;
    void *addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMapping);
    if (!addr) return 0;
#else
    int fd = open(fname, O_RDONLY);
    if (fd == -1) return 0;
    int64_t length = lseek(fd, 0, SEEK_END);
    void *addr = mmap(NULL, length, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (addr == MAP_FAILED) return 0;
#endif
    *mm_length = length;
    return addr;
}

static void munmap_file(void * addr, size_t length) {
#if defined(_WIN32) && !defined(_POSIX_MAPPED_FILES)
    UnmapViewOfFile(addr);
#else
    munmap(addr, length);
#endif
}

// determine number of model parts based on the dimension
static const std::unordered_map<int, int> LLAMA_N_PARTS = {
    { 4096, 1 },
    { 5120, 2 },
    { 6656, 4 },
    { 8192, 8 },
};

// available llama models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_7B,
    MODEL_13B,
    MODEL_30B,
    MODEL_65B,
};

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
//       needs modifications in ggml

static const std::map<e_model, size_t> MEM_REQ_SCRATCH0 = {
    { MODEL_7B,    512ull*MB },
    { MODEL_13B,   512ull*MB },
    { MODEL_30B,   512ull*MB },
    { MODEL_65B,   512ull*MB },
};

static const std::map<e_model, size_t> MEM_REQ_SCRATCH1 = {
    { MODEL_7B,    512ull*MB },
    { MODEL_13B,   512ull*MB },
    { MODEL_30B,   512ull*MB },
    { MODEL_65B,   512ull*MB },
};

// 2*n_embd*n_ctx*n_layer*sizeof(float16)
static const std::map<e_model, size_t> MEM_REQ_KV_SELF = {
    { MODEL_7B,   1026ull*MB },
    { MODEL_13B,  1608ull*MB },
    { MODEL_30B,  3124ull*MB },
    { MODEL_65B,  5120ull*MB },
};

// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t> MEM_REQ_EVAL = {
    { MODEL_7B,   768ull*MB },
    { MODEL_13B, 1024ull*MB },
    { MODEL_30B, 1280ull*MB },
    { MODEL_65B, 1536ull*MB },
};

struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

// default hparams (LLaMA 7B)
struct llama_hparams {
    int32_t n_vocab = 32000;
    int32_t n_ctx   = 512;   // this is provided as user input?
    int32_t n_embd  = 4096;
    int32_t n_mult  = 256;
    int32_t n_head  = 32;
    int32_t n_layer = 32;
    int32_t n_rot   = 64;
    int32_t f16     = 1;
};

//
// tokenizer
//

typedef int llama_token;

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct llama_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct llama_sp_bigram {
    struct comparator {
        bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
            return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<llama_sp_bigram>;
    using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
    llama_sp_symbol::index left;
    llama_sp_symbol::index right;
    float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct llama_tokenizer {
    llama_tokenizer(const llama_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            llama_sp_symbol sym;
            size_t char_len = Min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        llama_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const llama_vocab & vocab_;
    std::vector<llama_sp_symbol> symbols_;
    llama_sp_bigram::queue work_queue_;
};

static std::vector<llama_vocab::id> llama_tokenize(const llama_vocab & vocab, const std::string & text, bool bos) {
    llama_tokenizer tokenizer(vocab);
    std::vector<llama_vocab::id> output;

    if (text.size() == 0) {
        return output;
    }

    if (bos) {
        output.push_back(1);
    }

    tokenizer.tokenize(text, output);
    return output;
}

struct llama_layer {
    // normalization
    struct ffml_tensor * attention_norm;

    // attention
    struct ffml_tensor * wq;
    struct ffml_tensor * wk;
    struct ffml_tensor * wv;
    struct ffml_tensor * wo;

    // normalization
    struct ffml_tensor * ffn_norm;

    // ff
    struct ffml_tensor * w1;
    struct ffml_tensor * w2;
    struct ffml_tensor * w3;
};

struct llama_kv_cache {
    struct ffml_tensor * k;
    struct ffml_tensor * v;

    struct ffml_context * ctx;

    std::vector<uint8_t> buf;

    int n; // number of tokens currently in the cache
};

class LlamaModel {
public:
    llama_hparams hparams;
    llama_vocab vocab;
    e_model type = e_model::MODEL_UNKNOWN;
    llama_tokenizer * tokenizer;
    std::vector<llama_layer> layers;
    ffml_cgraph * cgraph;

    ffml_memory_pool * pool;

    std::unordered_map<std::string, struct ffml_tensor *> tensors;

    struct llama_kv_cache kv_self;

    struct ffml_tensor * tok_embeddings;
    struct ffml_tensor * norm;
    struct ffml_tensor * output;

    int n_loaded = 0;

    // model memory mapped file
    void * model_mm_addr = NULL;
    uint64_t mm_length = 0;

    LlamaModel(const std::string filename) {
        this->load(filename);
    }

    void load(const std::string fname, int n_ctx = 512, int n_parts = 1) {
        auto fin = std::ifstream(fname, std::ios::binary);
        if (!fin) {
            fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
            exit(1);
        }

        std::vector<char> f_buf(1024*1024);
        fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());

        fin.seekg(0, fin.end);
        const size_t file_size = fin.tellg();
        fin.seekg(0);

        // verify magic
        {
            uint32_t magic;
            fin.read((char *) &magic, sizeof(magic));
            if (magic == LLAMA_FILE_MAGIC_UNVERSIONED) {
                fprintf(stderr, "%s: invalid model file '%s' (too old, regenerate your model files or convert them with convert-unversioned-ggml-to-ggml.py!)\n",
                        __func__, fname.c_str());
                exit(1);
            }
            if (magic != LLAMA_FILE_MAGIC) {
                // return report_bad_magic(fname.c_str(), magic, LLAMA_FILE_MAGIC);
                fprintf(stderr, "%s: invalid model file '%s' (bad magic number 0x%x, expected 0x%x)\n",
                        __func__, fname.c_str(), magic, LLAMA_FILE_MAGIC);
                exit(1);
            }

            uint32_t format_version;
            fin.read((char *) &format_version, sizeof(format_version));

            if (format_version != LLAMA_FILE_VERSION) {
                fprintf(stderr, "%s: invalid model file '%s' (unsupported format version %" PRIu32 ", expected %d)\n",
                        __func__, fname.c_str(), format_version, LLAMA_FILE_VERSION);
                exit(1);
            }
        }

        int n_ff = 0;

        // load hparams
        {
            fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
            //fin.read((char *) &hparams.n_ctx,   sizeof(hparams.n_ctx));
            fin.read((char *) &hparams.n_embd,  sizeof(hparams.n_embd));
            fin.read((char *) &hparams.n_mult,  sizeof(hparams.n_mult));
            fin.read((char *) &hparams.n_head,  sizeof(hparams.n_head));
            fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
            fin.read((char *) &hparams.n_rot,   sizeof(hparams.n_rot));
            fin.read((char *) &hparams.f16,     sizeof(hparams.f16));

            fprintf(stdout, "%s: n_vocab=%d n_ctx=%d n_embd=%d n_mult=%d n_head=%d n_layer=%d n_rot=%d f16=%d\n",
                    __func__, hparams.n_vocab, hparams.n_ctx, hparams.n_embd, hparams.n_mult, hparams.n_head, hparams.n_layer, hparams.n_rot, hparams.f16);

            hparams.n_ctx = n_ctx;

            n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;

            if (n_parts < 1) {
                n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
            }

            // temp warning to tell the user to use "--n_parts"
            if (hparams.f16 == 4 && n_parts != 1) {
                fprintf(stderr, "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
                fprintf(stderr, "%s: use '--n_parts 1' if necessary\n", __func__);
            }

            if (hparams.n_layer == 32) {
                type = e_model::MODEL_7B;
            }

            if (hparams.n_layer == 40) {
                type = e_model::MODEL_13B;
            }

            if (hparams.n_layer == 60) {
                type = e_model::MODEL_30B;
            }

            if (hparams.n_layer == 80) {
                type = e_model::MODEL_65B;
            }

            fprintf(stderr, "%s: n_vocab = %d\n", __func__, hparams.n_vocab);
            fprintf(stderr, "%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
            fprintf(stderr, "%s: n_embd  = %d\n", __func__, hparams.n_embd);
            fprintf(stderr, "%s: n_mult  = %d\n", __func__, hparams.n_mult);
            fprintf(stderr, "%s: n_head  = %d\n", __func__, hparams.n_head);
            fprintf(stderr, "%s: n_layer = %d\n", __func__, hparams.n_layer);
            fprintf(stderr, "%s: n_rot   = %d\n", __func__, hparams.n_rot);
            fprintf(stderr, "%s: f16     = %d\n", __func__, hparams.f16);
            fprintf(stderr, "%s: n_ff    = %d\n", __func__, n_ff);
            fprintf(stderr, "%s: n_parts = %d\n", __func__, n_parts);
            fprintf(stderr, "%s: type    = %d\n", __func__, type);
        }

        // load vocab
        {
            std::string word;
            vocab.id_to_token.resize(hparams.n_vocab);
            std::vector<char> tmp(64);

            for (int i = 0; i < hparams.n_vocab; i++) {
                uint32_t len;
                fin.read((char *) &len, sizeof(len));

                word.resize(len);
                if (len > 0) {
                    tmp.resize(len);
                    fin.read(tmp.data(), len);
                    word.assign(tmp.data(), len);
                } else {
                    word.clear();
                }

                float score;
                fin.read((char *) &score, sizeof(score));

                vocab.token_to_id[word] = i;

                auto &tok_score = vocab.id_to_token[i];
                tok_score.tok = word;
                tok_score.score = score;
            }
        }

        printf("%s: vocab size: %lu\n", __func__, vocab.id_to_token.size());

        // // test: print the first 100 tokens
        // for (int i = 0; i < 100; i++) {
        //     printf("%s: %d: %s\n", __func__, i, vocab.id_to_token[i].tok.c_str());
        // }

        // init tokenizer
        {
            tokenizer = new llama_tokenizer(vocab);
        }

        // load the model
        // for the big tensors, we have the option to store the data in 16-bit floats or quantized
        // in order to save memory and also to speed up the computation
        // wtype is for per-layer weights, while vtype is for other weights

        // map model into memory
        char *mm_addr = NULL;
        model_mm_addr = mmap_file(fname.c_str(), &mm_length);
        if (model_mm_addr == NULL) {
            fprintf(stderr, "%s: failed to mmap '%s'\n", __func__, fname.c_str());
            exit(1);
        }
        mm_addr = (char *)model_mm_addr;
        fprintf(stderr, "%s: ggml map size = %6.2f MB\n", __func__, mm_length/(1024.0*1024.0));

        // auto & ctx = this->ctx;

        size_t ctx_size = 0;
        {
            const int n_layer = hparams.n_layer;
            ctx_size += (5 + 10*n_layer)*256; // object overhead
            fprintf(stderr, "%s: ggml ctx size = %6.2f KB\n", __func__, ctx_size/1024.0);
        }

        // print memory requirements
        {
            const size_t scale = 1; // float16

            // this is the total memory required to run the inference
            const size_t mem_required =
                ctx_size +
                mm_length +
                MEM_REQ_SCRATCH0.at(type) +
                MEM_REQ_SCRATCH1.at(type) +
                MEM_REQ_EVAL.at    (type);

            // this is the memory required by one llama_state
            const size_t mem_required_state =
                scale*MEM_REQ_KV_SELF.at(type);

            fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                    mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);

            // create the context
            this->pool = ffml_memory_pool_create(mem_required * 2 * 2); // 2 for grad, 2 for just in case

            const int n_embd  = hparams.n_embd;
            const int n_layer = (DEBUG_MODE) ? 1 : hparams.n_layer;
            const int n_vocab = hparams.n_vocab;

            layers.resize(n_layer);

            this->tok_embeddings = ffml_tensor_create(2, {n_embd, n_vocab,0,0});
            this->norm   = ffml_tensor_create(1, {n_embd,0,0,0});
            this->output = ffml_tensor_create(2, {n_embd, n_vocab,0,0});

            // map by name
            this->tensors["tok_embeddings.weight"] = this->tok_embeddings;
            this->tensors["norm.weight"]   = this->norm;
            this->tensors["output.weight"] = this->output;

            for (int i = 0; i < n_layer; ++i) {
                auto & layer = this->layers[i];

                layer.attention_norm = ffml_tensor_create(1, {n_embd,0,0,0});

                layer.wq = ffml_tensor_create(2, {n_embd, n_embd,0,0});
                layer.wk = ffml_tensor_create(2, {n_embd, n_embd,0,0});
                layer.wv = ffml_tensor_create(2, {n_embd, n_embd,0,0});
                layer.wo = ffml_tensor_create(2, {n_embd, n_embd,0,0});

                layer.ffn_norm = ffml_tensor_create(1, {n_embd,0,0,0});

                layer.w1 = ffml_tensor_create(2, {n_embd,   n_ff,0,0});
                layer.w2 = ffml_tensor_create(2, {  n_ff, n_embd,0,0});
                layer.w3 = ffml_tensor_create(2, {n_embd,   n_ff,0,0});

                // map by name
                this->tensors["layers." + std::to_string(i) + ".attention_norm.weight"] = layer.attention_norm;

                this->tensors["layers." + std::to_string(i) + ".attention.wq.weight"] = layer.wq;
                this->tensors["layers." + std::to_string(i) + ".attention.wk.weight"] = layer.wk;
                this->tensors["layers." + std::to_string(i) + ".attention.wv.weight"] = layer.wv;
                this->tensors["layers." + std::to_string(i) + ".attention.wo.weight"] = layer.wo;

                this->tensors["layers." + std::to_string(i) + ".ffn_norm.weight"] = layer.ffn_norm;

                this->tensors["layers." + std::to_string(i) + ".feed_forward.w1.weight"] = layer.w1;
                this->tensors["layers." + std::to_string(i) + ".feed_forward.w2.weight"] = layer.w2;
                this->tensors["layers." + std::to_string(i) + ".feed_forward.w3.weight"] = layer.w3;
            }

        }

        // alloc
        for (auto & tensor : this->tensors) {
            ffml_calc_cached_(tensor.second);
            tensor.second->data = ffml_memory_pool_alloc(pool, tensor.second->size_bytes);
            tensor.second->grad = ffml_memory_pool_alloc(pool, tensor.second->size_bytes);
        }

        std::vector<uint8_t> tmp;

        fprintf(stderr, "%s: loading tensors from '%s'\n", __func__, fname.c_str());

        // load weights
        {
            size_t total_size = 0;
            this->n_loaded = 0;

            while (true) {
                int32_t n_dims;
                int32_t length;
                int32_t ftype;

                fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
                fin.read(reinterpret_cast<char *>(&length), sizeof(length));
                fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

                if (fin.eof()) {
                    break;
                }

                int32_t nelements = 1;
                int32_t ne[2] = { 1, 1 };
                for (int i = 0; i < n_dims; ++i) {
                    fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                    nelements *= ne[i];
                }

                std::string name(length, 0);
                fin.read(&name[0], length);

                if (this->tensors.find(name.data()) == this->tensors.end()) {
                    if (DEBUG_MODE) {
                        break;
                    }
                    fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                    exit(1);
                }

                auto tensor = this->tensors[name.data()];

                if (tensor->nelem != nelements) {
                    fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                    exit(1);
                }
                if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                    fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%" PRId64 ", %" PRId64 "], expected [%d, %d]\n",
                            __func__, name.data(), tensor->ne[0], tensor->ne[1], ne[0], ne[1]);
                    exit(1);
                }
                if (0) {
                    static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
                    fprintf(stderr, "%24s - [%5d, %5d], type = %6s\n", name.data(), ne[0], ne[1], ftype_str[ftype]);
                }

                switch (ftype) {
                    case 0:  // f32
                    case 1:  // f16
                        break;
                    case 2:  // q4_0
                    case 3:  // q4_1
                        printf("quantized weights not supported yet\n");
                        exit(1);
                        // assert(ne[0] % 64 == 0);
                        break;
                    default:
                        fprintf(stderr, "%s: unknown ftype %d in model file\n", __func__, ftype);
                        exit(1);
                };

                // load the tensor data into memory without copying or reading it
                size_t offset = fin.tellg();
                size_t tensor_data_size = (ftype == 0) ? tensor->size_bytes : tensor->size_bytes / 2;
                offset = (offset + 31) & -32; // align to 32 bytes
                
                void * tmp_data = NULL;
                tmp_data = mm_addr + offset; // point to the data in the mmap'ed file

                // copy the data into the tensor
                if (ftype == 0) {
                    // copy directly
                    memcpy(tensor->data, tmp_data, tensor_data_size);
                } else {
                    for (uint64_t i = 0; i < tensor->nelem; i++) {
                        uint16_t d = ((uint16_t *) tmp_data)[i];
                        ffml_set_data_flat(tensor, i, compute_fp16_to_fp32(d));
                    }
                }

                tensor->init_ran = true;

                fin.seekg(offset + tensor_data_size); // skip the data
                total_size += tensor_data_size;
                this->n_loaded++;

                // progress
                printf("tensor loaded: %s, ftype %d\n", name.data(), ftype);
                printf("total size: %zu\n", total_size);
            }

            fin.close();

            fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, this->n_loaded);
            if (this->n_loaded == 0) {
                fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
            } else if (this->n_loaded != (int) this->tensors.size()) {
                fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, this->tensors.size(), this->n_loaded);
                exit(1);
            }

            // ggml strides
            set_ggml_strides(this->tok_embeddings);

        }


    }

    void set_ggml_strides(struct ffml_tensor * result) {
        result->nb[0] = sizeof(FFML_TYPE);
        result->nb[1] = result->nb[0]*(result->ne[0]/1);
        for (int i = 2; i < FFML_MAX_DIMS; i++) {
            result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
        }
    }

    void inference(std::vector<llama_vocab::id> & embd, std::vector<llama_vocab::id> & output, int how_many) {
        int n_past = 0;
        int n_remain = how_many;
        int n_consumed = 0;

        int n_ctx = hparams.n_ctx;

        // prepend 1 to embd
        embd.insert(embd.begin(), 1);

        int n_keep = 0; // # of tokens to keep from the original prompt

        while(n_remain > 0) {
            if (embd.size() > 0) {
                // infinite text generation via context swapping
                // if we run out of context:
                // - take the n_keep first tokens from the original prompt (via n_past)
                // - take half of the last (n_ctx - n_keep) tokens and recompute the logits in a batch
                if (n_past + (int) embd.size() > n_ctx) {
                    const int n_left = n_past - n_keep;

                    n_past = n_keep;

                    // insert n_left/2 tokens at the start of embd from last_n_tokens
                    // todo
                }

                // eval
                eval(embd, embd.size(), n_past);
            }

            // update counters
            n_past += 1;
            n_remain -= 1;
            n_consumed += 1;

            // todo: sample and out
        }
    }

    void eval(std::vector<llama_vocab::id> & tokens,
                         int   n_tokens,
                         int   n_past) {

        printf("Eval:\n");

        const int N = n_tokens;

        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx   = hparams.n_ctx;
        const int n_head  = hparams.n_head;
        const int n_vocab = hparams.n_vocab;
        const int n_rot   = hparams.n_embd/hparams.n_head;

        // print whatever is in embd
        for (int i = 0; i < N; i++) {
            printf("# %d: %d\n", i, tokens[i]);
        }

        // ---------- wire cgraph ------------



        struct ffml_tensor * embd = ffml_tensor_create(2, {1,N,0,0});

        struct ffml_tensor * tok_embeddings_transposed = ffml_unary_op(FFML_OP_TRANSPOSE, this->tok_embeddings);

        struct ffml_tensor * inpL = ffml_op(FFML_OP_SELECT, embd, tok_embeddings_transposed);

        struct ffml_tensor * out;

        struct ffml_tensor * delete_attention_norm;
        struct ffml_tensor * delete_attention_norm_repeated;

        for (int il = 0; il < n_layer; ++il) {
            struct ffml_tensor * inpSA = inpL;

            struct ffml_tensor * cur;

            // norm
            {
                cur = ffml_unary_op(FFML_OP_RMS_NORM, inpL);

                delete_attention_norm = this->layers[il].attention_norm;

                delete_attention_norm_repeated = ffml_op(
                            FFML_OP_REPEAT,
                            delete_attention_norm,
                            cur);

                // cur = attention_norm*cur
                cur = ffml_op(FFML_OP_MUL, 
                        delete_attention_norm_repeated,
                        cur);

                out = cur;
            }

            break;

        }

        // ----- set data ----------

        this->cgraph = ffml_cgraph_create(out);
        ffml_cgraph_alloc(this->cgraph, pool, false);

        for (int i = 0; i < N; i++) {
            ffml_set_data_flat(embd, i, (float)tokens[i]);
        }

        // ------ run ---------

        ffml_cgraph_forward(this->cgraph);

        // ------ get output -------

        ffml_debug_print_tensor_data(delete_attention_norm);
        ffml_debug_print_tensor_data(delete_attention_norm_repeated);
        ffml_debug_print_tensor_data(out);

        assert(out->n_dims == 3);

        exit(0);
        
    }

};
