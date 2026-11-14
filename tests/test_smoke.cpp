#include <string>

#include "test.h"
#include "version.h"

TEST(version_is_set) { CHECK(!std::string(tinyinfer::version()).empty()); }

TEST_MAIN()
