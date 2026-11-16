#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace tinyinfer {

enum class Pretokenizer { GPT2, Qwen2 };

// Byte-level BPE (GPT-2, Qwen2). Loads vocab.json, merges.txt, and optionally
// pretokenizer.txt and special_tokens.txt from a model directory.
class Tokenizer {
public:
    static Tokenizer load(const std::string& dir);

    std::vector<int32_t> encode(const std::string& text) const;
    std::string decode(const std::vector<int32_t>& ids) const;
    const std::string& token_bytes(int32_t id) const;

    int32_t eos() const { return eos_; }
    std::vector<int32_t> stop_tokens() const;
    int32_t special_id(const std::string& content) const;  // -1 if absent
    size_t vocab_size() const { return id_to_bytes_.size(); }

    // Splits text into pre-tokens the way the model's regex does. Exposed for tests.
    static std::vector<std::string> pretokenize(const std::string& text, Pretokenizer mode = Pretokenizer::GPT2);

private:
    void bpe(const std::string& word, std::vector<int32_t>& out) const;
    void encode_ordinary(const std::string& text, std::vector<int32_t>& out) const;

    Pretokenizer mode_ = Pretokenizer::GPT2;
    std::vector<std::pair<std::string, int32_t>> specials_;  // longest first

    std::unordered_map<std::string, int32_t> vocab_;       // byte-mapped token string -> id
    std::unordered_map<std::string, int32_t> merge_rank_;  // "left right" -> rank
    std::vector<std::string> id_to_bytes_;                 // id -> raw bytes
    std::string byte_to_sym_[256];                         // byte -> UTF-8 of its mapped symbol
    int32_t eos_ = -1;
    mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

}  // namespace tinyinfer
