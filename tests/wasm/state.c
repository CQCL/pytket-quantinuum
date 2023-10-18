void init() {}

static int c;

void set_c(int n) {
    c = n;
}

void conditional_increment_c(int s) {
    if (s) c++;
}

int get_c() {
    return c;
}
