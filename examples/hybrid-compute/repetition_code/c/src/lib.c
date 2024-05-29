
int init() {
    return 0;
}

int decode3(int syn, int pfu) {
    int val;
    if (syn == 1) {
        val = 1;
    } else if (syn == 3) {
        val = 2;
    } else if (syn == 2) {
        val = 4;
    } else {
        return 0;
    }

    int pfu_new = pfu ^ val;
    return pfu_new;
}
