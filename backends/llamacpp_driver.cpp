// Minimal llama.cpp driver: raw logits over a line protocol on stdin/stdout.
//   R\n            -> clear KV memory, reply "OK\n"
//   E n t1..tn\n   -> decode tokens, reply V float32 logits (raw bytes) for the last token
//   Q\n            -> quit
// usage: llamacpp_driver MODEL.gguf NGL THREADS KV(f16|f32) FLASH(auto|on|off) OPOFFLOAD(0|1)
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void quiet(enum ggml_log_level, const char *, void *) {}

int main(int argc, char **argv) {
    if (argc < 7) { fprintf(stderr, "args\n"); return 1; }
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = atoi(argv[2]);
    llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "load failed\n"); return 1; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 512;
    cp.n_threads = cp.n_threads_batch = atoi(argv[3]);
    ggml_type kv = strcmp(argv[4], "f32") == 0 ? GGML_TYPE_F32 : GGML_TYPE_F16;
    cp.type_k = kv; cp.type_v = kv;
    cp.flash_attn_type = strcmp(argv[5], "on") == 0 ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                       : strcmp(argv[5], "off") == 0 ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                       : LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.op_offload = atoi(argv[6]) != 0;
    cp.no_perf = true;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx failed\n"); return 1; }
    const int V = llama_vocab_n_tokens(llama_model_get_vocab(model));
    fprintf(stderr, "READY V=%d\n", V);
    std::string line;
    std::vector<llama_token> toks;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        if (line[0] == 'Q') break;
        if (line[0] == 'R') {
            llama_memory_clear(llama_get_memory(ctx), true);
            fwrite("OK\n", 1, 3, stdout); fflush(stdout);
            continue;
        }
        std::istringstream ss(line.substr(1));
        int n; ss >> n; toks.resize(n);
        for (int i = 0; i < n; i++) ss >> toks[i];
        if (llama_decode(ctx, llama_batch_get_one(toks.data(), n)) != 0) {
            fprintf(stderr, "decode failed\n"); return 2;
        }
        const float *lg = llama_get_logits_ith(ctx, -1);
        fwrite(lg, sizeof(float), V, stdout); fflush(stdout);
    }
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
