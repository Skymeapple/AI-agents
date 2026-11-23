/* Minimal WalkSAT (SKC variant) used as a stochastic "generator".
 * Usage: walksat <instance_file> <runs> <maxflips> <noise> <seed>
 * Instance file: first line "nvars nclauses", then nclauses lines of 3 ints (DIMACS literals).
 * Output: one line per run: 1 if a satisfying assignment was found within maxflips else 0
 * (printed as a compact string of 0/1 characters followed by newline).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned long long rng_s;
static inline unsigned long long rnd(void) { /* xorshift64* */
    rng_s ^= rng_s >> 12; rng_s ^= rng_s << 25; rng_s ^= rng_s >> 27;
    return rng_s * 2685821657736338717ULL;
}
static inline double urand(void) { return (rnd() >> 11) * (1.0 / 9007199254740992.0); }

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    FILE *f = fopen(argv[1], "r");
    int runs = atoi(argv[2]); long maxflips = atol(argv[3]);
    double noise = atof(argv[4]); rng_s = strtoull(argv[5], 0, 10) * 0x9E3779B97F4A7C15ULL + 1;
    int nv, nc; if (fscanf(f, "%d %d", &nv, &nc) != 2) return 1;
    int *lit = malloc(sizeof(int) * 3 * nc);
    for (int i = 0; i < 3 * nc; i++) if (fscanf(f, "%d", &lit[i]) != 1) return 1;
    fclose(f);
    /* occurrence lists */
    int *occn = calloc(2 * (nv + 1), sizeof(int));
    for (int i = 0; i < 3 * nc; i++) { int l = lit[i]; int idx = l > 0 ? 2 * l : 2 * (-l) + 1; occn[idx]++; }
    int **occ = malloc(sizeof(int *) * 2 * (nv + 1)); int *fill = calloc(2 * (nv + 1), sizeof(int));
    for (int i = 0; i < 2 * (nv + 1); i++) occ[i] = malloc(sizeof(int) * (occn[i] + 1));
    for (int c = 0; c < nc; c++) for (int j = 0; j < 3; j++) {
        int l = lit[3 * c + j]; int idx = l > 0 ? 2 * l : 2 * (-l) + 1; occ[idx][fill[idx]++] = c; }
    char *val = malloc(nv + 1); int *ntrue = malloc(sizeof(int) * nc);
    int *unsat = malloc(sizeof(int) * nc); int *pos = malloc(sizeof(int) * nc); int nun;
    char *out = malloc(runs + 2);
    for (int r = 0; r < runs; r++) {
        for (int v = 1; v <= nv; v++) val[v] = rnd() & 1;
        nun = 0;
        for (int c = 0; c < nc; c++) {
            int t = 0; for (int j = 0; j < 3; j++) { int l = lit[3 * c + j]; t += (l > 0) ? val[l] : !val[-l]; }
            ntrue[c] = t; if (t == 0) { pos[c] = nun; unsat[nun++] = c; } else pos[c] = -1;
        }
        long fl = 0;
        while (nun > 0 && fl < maxflips) {
            int c = unsat[rnd() % nun]; int pick = 0;
            int brk[3]; int best = 1 << 30, nbest = 0, cand[3];
            for (int j = 0; j < 3; j++) {
                int v = abs(lit[3 * c + j]);
                /* break count: clauses where v's current true literal is the only true literal */
                int idx = val[v] ? 2 * v : 2 * v + 1; int b = 0;
                for (int o = 0; o < occn[idx]; o++) if (ntrue[occ[idx][o]] == 1) b++;
                brk[j] = b;
            }
            int zero = -1; for (int j = 0; j < 3; j++) if (brk[j] == 0) { zero = j; break; }
            if (zero >= 0) pick = zero;
            else if (urand() < noise) pick = rnd() % 3;
            else { for (int j = 0; j < 3; j++) { if (brk[j] < best) { best = brk[j]; nbest = 0; } if (brk[j] == best) cand[nbest++] = j; }
                   pick = cand[rnd() % nbest]; }
            int v = abs(lit[3 * c + pick]);
            int idxT = val[v] ? 2 * v : 2 * v + 1;   /* literal currently true -> becomes false */
            int idxF = val[v] ? 2 * v + 1 : 2 * v;   /* literal currently false -> becomes true */
            val[v] = !val[v];
            for (int o = 0; o < occn[idxT]; o++) { int cc = occ[idxT][o]; ntrue[cc]--;
                if (ntrue[cc] == 0) { pos[cc] = nun; unsat[nun++] = cc; } }
            for (int o = 0; o < occn[idxF]; o++) { int cc = occ[idxF][o]; ntrue[cc]++;
                if (ntrue[cc] == 1) { int p = pos[cc]; int last = unsat[--nun]; unsat[p] = last; pos[last] = p; pos[cc] = -1; } }
            fl++;
        }
        out[r] = nun == 0 ? '1' : '0';
    }
    out[runs] = 0; printf("%s\n", out);
    return 0;
}
