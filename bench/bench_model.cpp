#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "model.h"
#include "ops.h"

using namespace tinyinfer;

// Mirrors llama-bench: ppN processes N prompt tokens in one batch, tgN decodes
// N tokens one at a time from an empty context. Reports mean +- stddev tok/s.
// usage: bench_model <model_dir> [-t 5,10,15] [-p 128] [-n 64] [-r 5]

namespace {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::vector<int> parse_list(const char* s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::atoi(item.c_str()));
    return out;
}

struct Stat {
    double mean, sd;
};

Stat stats(const std::vector<double>& v) {
    double m = 0;
    for (double x : v) m += x;
    m /= v.size();
    double var = 0;
    for (double x : v) var += (x - m) * (x - m);
    return {m, v.size() > 1 ? std::sqrt(var / (v.size() - 1)) : 0.0};
}

double run_pp(const Model& model, int n) {
    std::vector<int32_t> tokens(n);
    for (int i = 0; i < n; ++i) tokens[i] = 100 + (i * 37) % 10000;
    KVCache cache = model.make_cache(n);
    const double t0 = now();
    model.forward(cache, tokens.data(), n, true);
    return n / (now() - t0);
}

double run_tg(const Model& model, int n) {
    KVCache cache = model.make_cache(n);
    int32_t tok = 100;
    const double t0 = now();
    for (int i = 0; i < n; ++i) {
        model.forward(cache, &tok, 1, true);
        tok = 100 + (i * 37) % 10000;
    }
    return n / (now() - t0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: bench_model <model_dir> [-t 5,10,15] [-p 128] [-n 64] [-r 5]\n");
        return 2;
    }
    const std::string dir = argv[1];
    std::vector<int> threads = {num_threads()};
    int pp = 128, tg = 64, reps = 5;
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "-t")) threads = parse_list(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-p")) pp = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-n")) tg = std::atoi(argv[i + 1]);
        else if (!std::strcmp(argv[i], "-r")) reps = std::atoi(argv[i + 1]);
    }

    std::unique_ptr<Model> model = Model::load(dir);
    std::printf("| model | weights GiB | threads | test | t/s |\n|---|---:|---:|---:|---:|\n");
    for (int t : threads) {
        set_num_threads(t);
        run_tg(*model, 8);  // warm up: page in weights, spin up threads
        for (int test = 0; test < 2; ++test) {
            if ((test == 0 && pp <= 0) || (test == 1 && tg <= 0)) continue;
            std::vector<double> r;
            for (int i = 0; i < reps; ++i) r.push_back(test == 0 ? run_pp(*model, pp) : run_tg(*model, tg));
            Stat s = stats(r);
            std::printf("| %s | %.2f | %d | %s%d | %.2f +- %.2f |\n", model->arch_name(),
                        model->weight_bytes() / 1073741824.0, t, test == 0 ? "pp" : "tg", test == 0 ? pp : tg, s.mean,
                        s.sd);
            std::fflush(stdout);
        }
    }
}
