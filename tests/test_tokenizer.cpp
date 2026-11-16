#include <cstdio>
#include <string>
#include <vector>

#include "test.h"
#include "tinf.h"
#include "tokenizer.h"

using namespace tinyinfer;

namespace {

const std::string kDir = TINYINFER_MODEL_DIR;

std::string as_text(const Tensor& bytes) {
    std::string s;
    for (int64_t i = 0; i < bytes.numel(); ++i) s += static_cast<char>(bytes.i32()[i]);
    return s;
}

std::vector<int32_t> as_ids(const Tensor& t) { return std::vector<int32_t>(t.i32(), t.i32() + t.numel()); }

void print_ids(const char* label, const std::vector<int32_t>& ids) {
    std::printf("    %s:", label);
    for (int32_t id : ids) std::printf(" %d", id);
    std::printf("\n");
}

struct Fixture {
    Tokenizer tok = Tokenizer::load(kDir);
    TinfFile ref = TinfFile::open(kDir + "/reference.tinf");
};

Fixture& fixture() {
    static Fixture f;
    return f;
}

}  // namespace

TEST(vocab_loads) {
    Fixture& f = fixture();
    CHECK(f.tok.vocab_size() == 50257);
    CHECK(f.tok.eos() == 50256);
    CHECK(f.tok.token_bytes(50256) == "<|endoftext|>");
}

TEST(pretokenize_follows_gpt2_regex) {
    using V = std::vector<std::string>;
    CHECK((Tokenizer::pretokenize("a  b") == V{"a", " ", " b"}));
    CHECK((Tokenizer::pretokenize("It's 42!") == V{"It", "'s", " 42", "!"}));
    CHECK((Tokenizer::pretokenize("x\n\ny") == V{"x", "\n", "\n", "y"}));
    CHECK((Tokenizer::pretokenize("end   ") == V{"end", "   "}));
}

TEST(encode_matches_hugging_face) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_tok_cases"); ++i) {
        const std::string p = "tok" + std::to_string(i) + ".";
        const std::string text = as_text(f.ref.get(p + "text"));
        const std::vector<int32_t> want = as_ids(f.ref.get(p + "ids"));
        const std::vector<int32_t> got = f.tok.encode(text);
        if (got != want) {
            std::printf("  case %lld: \"%s\"\n", (long long)i, text.c_str());
            print_ids("want", want);
            print_ids("got ", got);
        }
        CHECK(got == want);
    }
}

TEST(prompts_encode_like_reference) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_prompts"); ++i) {
        const std::string p = "p" + std::to_string(i) + ".";
        CHECK(f.tok.encode(as_text(f.ref.get(p + "text"))) == as_ids(f.ref.get(p + "tokens")));
    }
}

TEST(decode_roundtrips) {
    Fixture& f = fixture();
    for (int64_t i = 0; i < f.ref.config_int("n_tok_cases"); ++i) {
        const std::string p = "tok" + std::to_string(i) + ".";
        const std::string text = as_text(f.ref.get(p + "text"));
        CHECK(f.tok.decode(as_ids(f.ref.get(p + "ids"))) == text);
        CHECK(f.tok.decode(f.tok.encode(text)) == text);
    }
}

TEST_MAIN()
