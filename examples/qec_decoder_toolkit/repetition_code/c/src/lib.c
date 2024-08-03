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

void init() {
    return;
}

int syn_old = 0;

int decode3(int syn) {
    int syn_new = syn ^ syn_old;
    int val;
    int pfu = 0;
    if (syn_new == 1) {
        val = 1;
    } else if (syn_new == 3) {
        val = 2;
    } else if (syn_new == 2) {
        val = 4;
    } else {
        return 0;
    }
    syn_old = syn;
    return val ^ pfu;
}

void set_global_syn_old() {
    syn_old = 0;
}