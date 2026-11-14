#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace test {

struct Case {
    std::string name;
    std::function<void()> fn;
};

inline std::vector<Case>& registry() {
    static std::vector<Case> cases;
    return cases;
}

inline int& failures() {
    static int n = 0;
    return n;
}

struct Register {
    Register(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline int run_all() {
    for (auto& c : registry()) {
        int before = failures();
        c.fn();
        std::printf("[%s] %s\n", failures() == before ? "PASS" : "FAIL", c.name.c_str());
    }
    std::printf("%zu tests, %d failures\n", registry().size(), failures());
    return failures() ? EXIT_FAILURE : EXIT_SUCCESS;
}

}  // namespace test

#define TEST_CAT2(a, b) a##b
#define TEST_CAT(a, b) TEST_CAT2(a, b)
#define TEST(name)                                                         \
    static void name();                                                    \
    static test::Register TEST_CAT(reg_, name)(#name, name);               \
    static void name()

#define CHECK(cond)                                                                  \
    do {                                                                             \
        if (!(cond)) {                                                               \
            std::printf("  %s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond);   \
            ++test::failures();                                                      \
            return;                                                                  \
        }                                                                            \
    } while (0)

#define CHECK_NEAR(a, b, tol)                                                                    \
    do {                                                                                         \
        double _a = (a), _b = (b);                                                               \
        if (!(std::fabs(_a - _b) <= (tol))) {                                                    \
            std::printf("  %s:%d: %s=%g vs %s=%g (tol %g)\n", __FILE__, __LINE__, #a, _a, #b, _b, \
                        (double)(tol));                                                          \
            ++test::failures();                                                                  \
            return;                                                                              \
        }                                                                                        \
    } while (0)

#define TEST_MAIN() \
    int main() { return test::run_all(); }
