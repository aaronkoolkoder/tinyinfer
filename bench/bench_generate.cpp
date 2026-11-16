#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "generate.h"
#include "model.h"

using namespace tinyinfer;

// Greedy generation with and without the KV cache on synthetic prompts.
// usage: bench_generate [model_dir] [n_new]
int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "models/gpt2";
    const int n_new = argc > 2 ? std::atoi(argv[2]) : 32;
    std::unique_ptr<Model> model = Model::load(dir);

    std::printf("| prompt | new | cache | time to first token (s) | decode tok/s | total (s) | cache MiB |\n");
    std::printf("|---:|---:|:---:|---:|---:|---:|---:|\n");
    for (int prompt_len : {16, 128}) {
        std::vector<int32_t> prompt(prompt_len);
        for (int i = 0; i < prompt_len; ++i) prompt[i] = 1000 + (i * 37) % 20000;
        for (bool use_cache : {false, true}) {
            GenerateOptions opts;
            opts.max_new_tokens = n_new;
            opts.use_cache = use_cache;
            GenerateStats st;
            generate_greedy(*model, prompt, opts, {}, &st);
            const double tps = st.generated > 1 ? (st.generated - 1) / st.decode_seconds : 0.0;
            std::printf("| %d | %lld | %s | %.2f | %.2f | %.2f | %.1f |\n", prompt_len, (long long)st.generated,
                        use_cache ? "on" : "off", st.prefill_seconds, tps, st.prefill_seconds + st.decode_seconds,
                        st.cache_bytes / 1048576.0);
            std::fflush(stdout);
        }
    }
}
