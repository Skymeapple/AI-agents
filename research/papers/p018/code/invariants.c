/* invariants.c -- exact graph invariants for small graphs (n <= 20).
 * Reads graph6 lines on stdin, writes one fixed-width binary record per graph
 * on stdout: NINV uint16 values in the order given by NAMES below.
 * Build: gcc -O2 -o invariants invariants.c
 */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MAXN 20
typedef uint32_t set;
static int n;
static set adj[MAXN];
static set full;

#define POP(x) __builtin_popcount(x)
#define LOW(x) __builtin_ctz(x)

/* NAMES: n m delta Delta alpha omega chi gamma nu diam rad res ann tri */
#define NINV 14

static int parse_g6(const char *s) {
    n = s[0] - 63;
    if (n < 1 || n > MAXN) return 0;
    memset(adj, 0, sizeof adj);
    int k = 0; const char *p = s + 1; int bit = 5; int cur = *p - 63;
    for (int j = 1; j < n; j++)
        for (int i = 0; i < j; i++) {
            if (bit < 0) { p++; cur = *p - 63; bit = 5; }
            if ((cur >> bit) & 1) { adj[i] |= 1u << j; adj[j] |= 1u << i; }
            bit--; k++;
        }
    full = (n == 32) ? 0xffffffffu : ((1u << n) - 1);
    return 1;
}

/* maximum independent set within candidate set P (graph given by g[]) */
static int mis(set P, const set *g) {
    if (!P) return 0;
    int v = LOW(P);
    /* if v has no neighbour in P, take it */
    if (!(g[v] & P)) return 1 + mis(P & ~(1u << v), g);
    int a = 1 + mis(P & ~(1u << v) & ~g[v], g);
    int b = mis(P & ~(1u << v), g);
    return a > b ? a : b;
}

static set cadj[MAXN];
static int clique_number(void) {
    for (int i = 0; i < n; i++) cadj[i] = full & ~adj[i] & ~(1u << i);
    return mis(full, cadj);
}

/* chromatic number by backtracking */
static int col[MAXN], order_[MAXN];
static int try_color(int idx, int k) {
    if (idx == n) return 1;
    int v = order_[idx];
    for (int c = 0; c < k; c++) {
        int ok = 1;
        set nb = adj[v];
        while (nb) { int w = LOW(nb); nb &= nb - 1; if (col[w] == c) { ok = 0; break; } }
        if (!ok) continue;
        col[v] = c;
        if (try_color(idx + 1, k)) return 1;
        col[v] = -1;
        /* symmetry: first use of a new colour need only be tried once */
        int used = 0; for (int t = 0; t < idx; t++) if (col[order_[t]] == c) { used = 1; break; }
        if (!used) break;
    }
    return 0;
}
static int chromatic(int lower) {
    /* order: by degree descending */
    for (int i = 0; i < n; i++) order_[i] = i;
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++)
        if (POP(adj[order_[j]]) > POP(adj[order_[i]])) { int t = order_[i]; order_[i] = order_[j]; order_[j] = t; }
    for (int k = lower; k <= n; k++) {
        for (int i = 0; i < n; i++) col[i] = -1;
        if (try_color(0, k)) return k;
    }
    return n;
}

/* domination number: branch on closed neighbourhood of lowest undominated vertex */
static int best_dom;
static void dom(set D, int size) {
    if (size >= best_dom) return;
    if (D == full) { best_dom = size; return; }
    set U = full & ~D;
    int u = LOW(U);
    set C = adj[u] | (1u << u);
    while (C) { int w = LOW(C); C &= C - 1; dom(D | adj[w] | (1u << w), size + 1); }
}

/* matching number */
static int matching(set R) {
    while (R && !(adj[LOW(R)] & R)) R &= R - 1; /* drop isolated (in R) vertices */
    if (!R) return 0;
    int v = LOW(R);
    set Rv = R & ~(1u << v);
    int best = matching(Rv);
    set nb = adj[v] & Rv;
    while (nb) { int w = LOW(nb); nb &= nb - 1; int t = 1 + matching(Rv & ~(1u << w)); if (t > best) best = t; }
    return best;
}

static int ecc(int s) {
    set seen = 1u << s, fr = 1u << s; int d = 0;
    while (seen != full) {
        set nf = 0; set f = fr;
        while (f) { int v = LOW(f); f &= f - 1; nf |= adj[v]; }
        nf &= ~seen; if (!nf) return 255; /* disconnected */
        seen |= nf; fr = nf; d++;
    }
    return d;
}

static int residue(void) {
    int d[MAXN]; int k = n;
    for (int i = 0; i < n; i++) d[i] = POP(adj[i]);
    while (1) {
        /* sort descending */
        for (int i = 0; i < k; i++) for (int j = i + 1; j < k; j++) if (d[j] > d[i]) { int t = d[i]; d[i] = d[j]; d[j] = t; }
        if (k == 0 || d[0] == 0) break;
        int x = d[0];
        for (int i = 1; i <= x; i++) d[i]--;
        for (int i = 0; i < k - 1; i++) d[i] = d[i + 1];
        k--;
    }
    return k;
}

static int annihilation(int m) {
    int d[MAXN];
    for (int i = 0; i < n; i++) d[i] = POP(adj[i]);
    for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if (d[j] < d[i]) { int t = d[i]; d[i] = d[j]; d[j] = t; }
    int s = 0, k = 0;
    for (int i = 0; i < n; i++) { s += d[i]; if (s <= m) k = i + 1; else break; }
    return k;
}

int main(void) {
    static char line[512];
    uint16_t rec[NINV];
    while (fgets(line, sizeof line, stdin)) {
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
        if (!L || line[0] == '>') continue;
        if (!parse_g6(line)) continue;
        int m = 0, dmin = 99, dmax = 0, tri = 0;
        for (int i = 0; i < n; i++) { int d = POP(adj[i]); m += d; if (d < dmin) dmin = d; if (d > dmax) dmax = d; }
        m /= 2;
        for (int i = 0; i < n; i++) for (int j = i + 1; j < n; j++) if ((adj[i] >> j) & 1) tri += POP(adj[i] & adj[j] & ~((2u << j) - 1));
        int alpha = mis(full, adj);
        int omega = clique_number();
        int chi = chromatic(omega);
        best_dom = n + 1; dom(0, 0);
        int nu = matching(full);
        int diam = 0, rad = 255;
        for (int v = 0; v < n; v++) { int e = ecc(v); if (e > diam) diam = e; if (e < rad) rad = e; }
        rec[0] = n; rec[1] = m; rec[2] = dmin; rec[3] = dmax; rec[4] = alpha; rec[5] = omega;
        rec[6] = chi; rec[7] = best_dom; rec[8] = nu; rec[9] = diam; rec[10] = rad;
        rec[11] = residue(); rec[12] = annihilation(m); rec[13] = tri;
        fwrite(rec, 2, NINV, stdout);
    }
    return 0;
}
