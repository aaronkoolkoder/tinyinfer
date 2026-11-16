#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include "generate.h"
#include "gpt2.h"
#include "tokenizer.h"
#include "version.h"

using namespace tinyinfer;

namespace {

struct Args {
    std::string model_dir = "models/gpt2";
    std::string prompt = "Hello, my name is";
    int n = 32;
    bool stop_at_eos = true;
    bool seed_given = false;
    SamplingParams sampling;
};

void usage() {
    std::printf(
        "tinyinfer %s\n"
        "usage: tinyinfer [options]\n"
        "  -m, --model DIR     exported model directory (default models/gpt2)\n"
        "  -p, --prompt TEXT   prompt (default \"Hello, my name is\")\n"
        "  -n, --n-tokens N    tokens to generate (default 32)\n"
        "  -t, --temp T        temperature, 0 = greedy (default 1.0)\n"
        "  -k, --top-k K       keep the K most likely tokens, 0 = off (default 0)\n"
        "      --top-p P       nucleus sampling threshold, 1 = off (default 1.0)\n"
        "  -s, --seed S        random seed (default: time-based, printed at the end)\n"
        "      --greedy        same as --temp 0\n"
        "      --ignore-eos    keep generating past <|endoftext|>\n"
        "  -h, --help\n",
        version());
}

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", arg.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "-m" || arg == "--model") a.model_dir = value();
        else if (arg == "-p" || arg == "--prompt") a.prompt = value();
        else if (arg == "-n" || arg == "--n-tokens") a.n = std::stoi(value());
        else if (arg == "-t" || arg == "--temp") a.sampling.temperature = std::stof(value());
        else if (arg == "-k" || arg == "--top-k") a.sampling.top_k = std::stoi(value());
        else if (arg == "--top-p") a.sampling.top_p = std::stof(value());
        else if (arg == "-s" || arg == "--seed") {
            a.sampling.seed = std::stoull(value());
            a.seed_given = true;
        } else if (arg == "--greedy") a.sampling.temperature = 0.0f;
        else if (arg == "--ignore-eos") a.stop_at_eos = false;
        else if (arg == "-h" || arg == "--help") {
            usage();
            std::exit(0);
        } else {
            std::fprintf(stderr, "unknown argument %s\n", arg.c_str());
            usage();
            std::exit(2);
        }
    }
    if (a.sampling.temperature < 0 || a.sampling.top_k < 0 || a.sampling.top_p <= 0 || a.sampling.top_p > 1) {
        std::fprintf(stderr, "invalid sampling parameters\n");
        std::exit(2);
    }
    if (!a.seed_given)
        a.sampling.seed = static_cast<uint64_t>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return a;
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args = parse(argc, argv);
        GPT2Weights w = GPT2Weights::load(args.model_dir + "/model.tinf");
        Tokenizer tok = Tokenizer::load(args.model_dir);

        std::vector<int32_t> prompt = tok.encode(args.prompt);
        GenerateOptions opts;
        opts.max_new_tokens = args.n;
        opts.stop_token = args.stop_at_eos ? tok.eos() : -1;

        std::printf("%s", args.prompt.c_str());
        std::fflush(stdout);
        auto t0 = std::chrono::steady_clock::now();
        Sampler sampler(args.sampling);
        std::vector<int32_t> out = generate(w, prompt, opts, sampler, [&](int32_t id) {
            if (id == tok.eos()) return;
            std::printf("%s", tok.token_bytes(id).c_str());
            std::fflush(stdout);
        });
        double secs = seconds_since(t0);

        const SamplingParams& sp = args.sampling;
        std::fprintf(stderr, "\n\n[%zu prompt tokens, %zu generated in %.2fs, %.2f tok/s, no KV cache]\n",
                     prompt.size(), out.size(), secs, out.size() / secs);
        std::fprintf(stderr, "[temp %.2f, top-k %d, top-p %.2f, seed %" PRIu64 "]\n", sp.temperature, sp.top_k,
                     sp.top_p, sp.seed);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
