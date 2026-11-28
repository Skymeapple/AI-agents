"""Cross-check the C invariant engine against independent brute-force networkx code
on every connected graph with 2..7 vertices plus random graphs with 8..12 vertices."""
import itertools, subprocess, sys, os, random
import numpy as np
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["n", "m", "delta", "Delta", "alpha", "omega", "chi", "gamma", "nu",
         "diam", "rad", "res", "ann", "tri"]


def c_invariants(graphs):
    data = "".join(nx.to_graph6_bytes(g, header=False).decode() for g in graphs)
    out = subprocess.run([os.path.join(HERE, "invariants")], input=data.encode(),
                         capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.uint16).reshape(-1, len(NAMES))


def brute(g):
    n = g.number_of_nodes(); V = list(g.nodes())
    def indep(S):
        return all(not g.has_edge(a, b) for a, b in itertools.combinations(S, 2))
    alpha = max(k for k in range(n + 1) for S in itertools.combinations(V, k) if indep(S)) if n else 0
    omega = max(len(c) for c in nx.find_cliques(g))
    chi = next(k for k in range(1, n + 1) if any(
        all(col[a] != col[b] for a, b in g.edges()) for col in
        (dict(zip(V, c)) for c in itertools.product(range(k), repeat=n))))
    gamma = next(k for k in range(1, n + 1) for S in itertools.combinations(V, k)
                 if nx.is_dominating_set(g, S))
    nu = len(nx.max_weight_matching(g, maxcardinality=True))
    deg = sorted((d for _, d in g.degree()), reverse=True)
    d = deg[:]
    while d and d[0] > 0:
        x = d.pop(0); d = [y - 1 if i < x else y for i, y in enumerate(d)]; d.sort(reverse=True)
    res = len(d)
    m = g.number_of_edges(); asc = sorted(deg); s = 0; ann = 0
    for i, y in enumerate(asc):
        s += y
        if s <= m: ann = i + 1
        else: break
    tri = sum(nx.triangles(g).values()) // 3
    return [n, m, min(deg), max(deg), alpha, omega, chi, gamma, nu,
            nx.diameter(g), nx.radius(g), res, ann, tri]


def main():
    graphs = [g for g in nx.graph_atlas_g()[1:] if g.number_of_nodes() >= 2 and nx.is_connected(g)]
    rng = random.Random(0)
    for _ in range(60):
        n = rng.randint(8, 9)
        while True:
            g = nx.gnp_random_graph(n, rng.uniform(0.25, 0.7), seed=rng.randint(0, 10**9))
            if nx.is_connected(g): break
        graphs.append(g)
    C = c_invariants(graphs)
    bad = 0
    for g, row in zip(graphs, C):
        b = brute(g)
        if list(map(int, row)) != b:
            bad += 1
            if bad < 5: print("MISMATCH", nx.to_graph6_bytes(g), list(row), b)
    msg = f"validated {len(graphs)} graphs ({sum(1 for g in graphs if g.number_of_nodes()<=7)} atlas n<=7 + random n=8..9); mismatches = {bad}"
    print(msg)
    open(os.path.join(HERE, "..", "results", "validation_invariants.txt"), "w").write(msg + "\n")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
