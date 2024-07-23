void init() {
    return;
}

int decode3(int syn, int pfu_old) {
    int pfu;
    if (syn == 1) {
        pfu = 1;
    } else if (syn == 3) {
        pfu = 2;
    } else if (syn == 2) {
        pfu = 4;
    } else {
        return 0;
    }
    return pfu_old ^ pfu;
}
