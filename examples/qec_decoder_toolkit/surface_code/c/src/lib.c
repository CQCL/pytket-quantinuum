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

int PFU = 0;
int PFU1 = 0;


void init() {
    // to initialise the Wasm sandbox environment   
}


void set_pfu_values(int syn) {
    if (syn == 1) {
        PFU = 4; // 000000100
    }
    if (syn == 2) {
        PFU = 128; // 010000000
    }
    if (syn == 4) {
        PFU = 1; // 000000001
    }
    if (syn == 8) {
        PFU = 64; // 001000000
    }
    if (syn == 3) {
        PFU = 32; // 000100000
    }
    if (syn == 6) {
        PFU = 16; // 000010000
    }
    if (syn == 12) {
        PFU = 8; // 000001000
    }
    if (syn == 5) {
        PFU = 16; // 000010000
        PFU1 = 32; // 000001000
    }
    if (syn == 10) {
        PFU = 8; // 000001000
        PFU1 = 16; // 000001000
    }
    if (syn == 9) {
        PFU = 4; // 000000100
        PFU1 = 64; // 001000000
    }
    if (syn == 7) {
        PFU = 1; // 000000001
        PFU1 = 32; // 000100000
    }
    if (syn == 11) {
        PFU = 32; // 000100000
        PFU1 = 64; // 001000000
    }
    if (syn == 13) {
        PFU = 8; // 000000100
        PFU1 = 4; // 000000010
    }
    if (syn == 14) {
        PFU = 8; // 000000100
        PFU1 = 128; // 010000000
    }
    if (syn == 15) {
        PFU = 8; // 000000100
        PFU1 = 32; // 000010000
    }
}


int get_pfu() {
    return PFU;
}


int get_pfu1() {
    return PFU1;
}


void reset_pfu() {
    PFU = 0;
    PFU1 = 0;
}
