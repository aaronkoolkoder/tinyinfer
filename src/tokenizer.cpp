#include "tokenizer.h"

#include <algorithm>
#include <cctype>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tinyinfer {

namespace {

std::string read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path + " (run scripts/export.py first)");
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

void append_utf8(std::string& s, uint32_t cp) {
    if (cp < 0x80) {
        s += static_cast<char>(cp);
    } else if (cp < 0x800) {
        s += static_cast<char>(0xC0 | (cp >> 6));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        s += static_cast<char>(0xE0 | (cp >> 12));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    } else {
        s += static_cast<char>(0xF0 | (cp >> 18));
        s += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        s += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        s += static_cast<char>(0x80 | (cp & 0x3F));
    }
}

// Decodes one code point at s[i], advancing i. Invalid bytes decode as themselves.
uint32_t next_cp(const std::string& s, size_t& i) {
    const auto b = static_cast<unsigned char>(s[i]);
    int len = b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xE ? 3 : (b >> 3) == 0x1E ? 4 : 1;
    if (i + len > s.size()) len = 1;
    uint32_t cp = len == 1 ? b : b & (0x7F >> len);
    for (int k = 1; k < len; ++k) {
        const auto c = static_cast<unsigned char>(s[i + k]);
        if ((c & 0xC0) != 0x80) {
            len = 1;
            cp = b;
            break;
        }
        cp = (cp << 6) | (c & 0x3F);
    }
    i += len;
    return cp;
}

// GPT-2's bytes_to_unicode: printable bytes map to themselves, the rest to 256+n.
void build_byte_map(uint32_t (&byte_to_cp)[256]) {
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        byte_to_cp[b] = printable ? b : 256 + n++;
    }
}

// Minimal parser for a flat JSON object of string -> integer.
std::unordered_map<std::string, int32_t> parse_vocab(const std::string& js) {
    std::unordered_map<std::string, int32_t> out;
    size_t i = 0;
    auto skip_ws = [&] { while (i < js.size() && std::isspace(static_cast<unsigned char>(js[i]))) ++i; };
    auto expect = [&](char c) {
        skip_ws();
        if (i >= js.size() || js[i] != c) throw std::runtime_error(std::string("vocab.json: expected '") + c + "'");
        ++i;
    };
    auto hex4 = [&]() {
        if (i + 4 > js.size()) throw std::runtime_error("vocab.json: bad \\u escape");
        uint32_t v = std::stoul(js.substr(i, 4), nullptr, 16);
        i += 4;
        return v;
    };
    auto parse_string = [&]() {
        expect('"');
        std::string s;
        while (i < js.size() && js[i] != '"') {
            char c = js[i++];
            if (c != '\\') {
                s += c;
                continue;
            }
            char e = js[i++];
            switch (e) {
                case '"': s += '"'; break;
                case '\\': s += '\\'; break;
                case '/': s += '/'; break;
                case 'b': s += '\b'; break;
                case 'f': s += '\f'; break;
                case 'n': s += '\n'; break;
                case 'r': s += '\r'; break;
                case 't': s += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && i + 6 <= js.size() && js[i] == '\\' && js[i + 1] == 'u') {
                        i += 2;
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (hex4() - 0xDC00);
                    }
                    append_utf8(s, cp);
                    break;
                }
                default: throw std::runtime_error("vocab.json: bad escape");
            }
        }
        ++i;
        return s;
    };

    expect('{');
    skip_ws();
    if (js[i] == '}') return out;
    while (true) {
        std::string key = parse_string();
        expect(':');
        skip_ws();
        size_t end = i;
        while (end < js.size() && (std::isdigit(static_cast<unsigned char>(js[end])) || js[end] == '-')) ++end;
        out[key] = static_cast<int32_t>(std::stol(js.substr(i, end - i)));
        i = end;
        skip_ws();
        if (js[i] == ',') {
            ++i;
            continue;
        }
        expect('}');
        break;
    }
    return out;
}

// Unicode classes used by GPT-2's pre-tokenizer regex. ASCII is exact; beyond
// that this covers common scripts rather than the full Unicode tables.
bool is_space(uint32_t c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F ||
           c == 0x3000;
}

bool is_number(uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 || (c >= 0xBC && c <= 0xBE) ||
           (c >= 0x660 && c <= 0x669) || (c >= 0x6F0 && c <= 0x6F9) || (c >= 0x966 && c <= 0x96F) ||
           (c >= 0xFF10 && c <= 0xFF19);
}

bool is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
    return c == 0xAA || c == 0xB5 || c == 0xBA || (c >= 0xC0 && c <= 0x2C1 && c != 0xD7 && c != 0xF7) ||
           (c >= 0x370 && c <= 0x3FF && c != 0x37E && c != 0x387) || (c >= 0x400 && c <= 0x52F && !(c >= 0x482 && c <= 0x489)) ||
           (c >= 0x531 && c <= 0x587) || (c >= 0x5D0 && c <= 0x5F2) || (c >= 0x620 && c <= 0x64A) ||
           (c >= 0x904 && c <= 0x939) || (c >= 0xE01 && c <= 0xE30) || (c >= 0x10A0 && c <= 0x10FF) ||
           (c >= 0x1E00 && c <= 0x1FBC) || (c >= 0x3041 && c <= 0x3096) || (c >= 0x30A1 && c <= 0x30FA) ||
           (c >= 0x3400 && c <= 0x4DBF) || (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xAC00 && c <= 0xD7A3) ||
           (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A);
}

}  // namespace

Tokenizer Tokenizer::load(const std::string& dir) {
    Tokenizer tk;
    uint32_t byte_to_cp[256];
    build_byte_map(byte_to_cp);
    std::unordered_map<uint32_t, uint8_t> cp_to_byte;
    for (int b = 0; b < 256; ++b) {
        append_utf8(tk.byte_to_sym_[b], byte_to_cp[b]);
        cp_to_byte[byte_to_cp[b]] = static_cast<uint8_t>(b);
    }

    tk.vocab_ = parse_vocab(read_file(dir + "/vocab.json"));
    int32_t max_id = -1;
    for (const auto& kv : tk.vocab_) max_id = std::max(max_id, kv.second);
    tk.id_to_bytes_.assign(max_id + 1, "");
    for (const auto& kv : tk.vocab_) {
        std::string raw;
        for (size_t i = 0; i < kv.first.size();) {
            auto it = cp_to_byte.find(next_cp(kv.first, i));
            if (it == cp_to_byte.end()) throw std::runtime_error("vocab.json: token outside byte alphabet");
            raw += static_cast<char>(it->second);
        }
        tk.id_to_bytes_[kv.second] = raw;
    }
    auto eos = tk.vocab_.find("<|endoftext|>");
    if (eos != tk.vocab_.end()) {
        tk.eos_ = eos->second;
        tk.id_to_bytes_[tk.eos_] = "<|endoftext|>";
    }

    std::istringstream merges(read_file(dir + "/merges.txt"));
    std::string line;
    int32_t rank = 0;
    while (std::getline(merges, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line.rfind("#version", 0) == 0) continue;
        if (line.find(' ') == std::string::npos) throw std::runtime_error("merges.txt: bad line " + line);
        tk.merge_rank_.emplace(line, rank++);
    }
    return tk;
}

std::vector<std::string> Tokenizer::pretokenize(const std::string& text) {
    // Pattern: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    std::vector<uint32_t> cps;
    std::vector<size_t> pos;
    for (size_t i = 0; i < text.size();) {
        pos.push_back(i);
        cps.push_back(next_cp(text, i));
    }
    pos.push_back(text.size());
    const size_t n = cps.size();
    auto is_other = [](uint32_t c) { return !is_space(c) && !is_letter(c) && !is_number(c); };

    std::vector<std::string> out;
    size_t i = 0;
    while (i < n) {
        size_t j = i;
        if (cps[i] == '\'' && i + 1 < n) {
            static const char* kSuffixes[] = {"s", "t", "re", "ve", "m", "ll", "d"};
            for (const char* suf : kSuffixes) {
                size_t len = std::char_traits<char>::length(suf);
                bool ok = i + len < n + 1;
                for (size_t k = 0; ok && k < len; ++k) ok = i + 1 + k < n && cps[i + 1 + k] == static_cast<uint32_t>(suf[k]);
                if (ok) {
                    j = i + 1 + len;
                    break;
                }
            }
        }
        if (j == i) {
            size_t start = (cps[i] == ' ' && i + 1 < n) ? i + 1 : i;
            for (auto cls : {is_letter, is_number}) {
                size_t k = start;
                while (k < n && cls(cps[k])) ++k;
                if (k > start) {
                    j = k;
                    break;
                }
            }
            if (j == i) {
                size_t k = start;
                while (k < n && is_other(cps[k])) ++k;
                if (k > start) j = k;
            }
        }
        if (j == i) {
            size_t k = i;
            while (k < n && is_space(cps[k])) ++k;
            // \s+(?!\S): leave the last space to prefix the next word, unless the
            // run ends the text or is a single character.
            j = (k < n && k - i > 1) ? k - 1 : k;
            if (j == i) j = i + 1;
        }
        out.push_back(text.substr(pos[i], pos[j] - pos[i]));
        i = j;
    }
    return out;
}

void Tokenizer::bpe(const std::string& word, std::vector<int32_t>& out) const {
    auto cached = cache_.find(word);
    if (cached != cache_.end()) {
        out.insert(out.end(), cached->second.begin(), cached->second.end());
        return;
    }

    std::vector<std::string> parts;
    for (unsigned char b : word) parts.push_back(byte_to_sym_[b]);

    while (parts.size() > 1) {
        int32_t best = INT32_MAX;
        size_t best_i = 0;
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            auto it = merge_rank_.find(parts[i] + " " + parts[i + 1]);
            if (it != merge_rank_.end() && it->second < best) {
                best = it->second;
                best_i = i;
            }
        }
        if (best == INT32_MAX) break;
        const std::string left = parts[best_i], right = parts[best_i + 1];
        std::vector<std::string> merged;
        merged.reserve(parts.size());
        for (size_t i = 0; i < parts.size(); ++i) {
            if (i + 1 < parts.size() && parts[i] == left && parts[i + 1] == right) {
                merged.push_back(left + right);
                ++i;
            } else {
                merged.push_back(parts[i]);
            }
        }
        parts = std::move(merged);
    }

    std::vector<int32_t> ids;
    for (const auto& p : parts) {
        auto it = vocab_.find(p);
        if (it == vocab_.end()) throw std::runtime_error("tokenizer: symbol not in vocab");
        ids.push_back(it->second);
    }
    out.insert(out.end(), ids.begin(), ids.end());
    cache_.emplace(word, std::move(ids));
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) const {
    std::vector<int32_t> ids;
    for (const auto& word : pretokenize(text)) bpe(word, ids);
    return ids;
}

const std::string& Tokenizer::token_bytes(int32_t id) const {
    if (id < 0 || static_cast<size_t>(id) >= id_to_bytes_.size()) throw std::out_of_range("token id");
    return id_to_bytes_[id];
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string s;
    for (int32_t id : ids) s += token_bytes(id);
    return s;
}

}  // namespace tinyinfer
