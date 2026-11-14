#include <cstdio>

#include "version.h"

int main() {
    std::printf("tinyinfer %s\n", tinyinfer::version());
    return 0;
}
