#include <chrono>
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
};

void usage() {
    std::printf(
        "tinyinfer %s\n"
        "usage: tinyinfer [options]\n"
        "  -m, --model DIR     exported model directory (default models/gpt2)\n"
        "  -p, --prompt TEXT   prompt (default \"Hello, my name is\")\n"
        "  -n, --n-tokens N    tokens to generate (default 32)\n"
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
        std::vector<int32_t> out = generate_greedy(w, prompt, opts, [&](int32_t id) {
            if (id == tok.eos()) return;
            std::printf("%s", tok.token_bytes(id).c_str());
            std::fflush(stdout);
        });
        double secs = seconds_since(t0);

        std::fprintf(stderr, "\n\n[%zu prompt tokens, %zu generated in %.2fs, %.2f tok/s, no KV cache]\n",
                     prompt.size(), out.size(), secs, out.size() / secs);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
