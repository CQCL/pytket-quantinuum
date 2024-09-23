// Copyright 2020-2024 Quantinuum
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <gtest/gtest.h>
#include "lib.c"

TEST(TESTS, Test1) {
    reset_pfu();
    int syn = 15;
    set_pfu_value(syn);
    EXPECT_EQ(get_pfu(), 40);
}

TEST(TESTS, Test2) {
    reset_pfu();
    int syn = 4;
    set_pfu_value(syn);
    EXPECT_EQ(get_pfu(), 1);
    update_pfu(15);
    EXPECT_EQ(get_pfu(), 41);
}

TEST(TESTS, Test3) {
    set_pfu_value(5);
    reset_pfu();
    EXPECT_EQ(get_pfu(), 0);
}
