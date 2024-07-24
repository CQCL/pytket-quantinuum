#include <stdio.h>

void init() {
    return;
}

int syn_old;

int decode3(int syn, int pfu) {
    int syn_new = syn ^ syn_old;
    int val;
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

// int main() {
//     int pfu = decode3(1, 2);
//     printf("%d", pfu);
//     return 0;
// }
