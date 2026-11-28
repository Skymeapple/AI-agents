"""Out-of-universe falsification sample: connected graphs with 11..16 vertices drawn from
random models and from structured families that are frequent extremal graphs.
Writes data/inv_large.bin (14 uint16 per graph) and data/large_meta.txt (source tag per graph)."""
import os, random, subprocess, itertools
import networkx as nx

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
rng = random.Random(20261122)


def families(n):
    out = []
    out += [("path", nx.path_graph(n)), ("cycle", nx.cycle_graph(n)), ("star", nx.star_graph(n - 1)),
            ("complete", nx.complete_graph(n)), ("wheel", nx.wheel_graph(n))]
    for a in range(1, n):
        out.append(("Kab", nx.complete_bipartite_graph(a, n - a)))
    for k in range(1, n):  # complete split graph K_k + independent set
        g = nx.complete_graph(k); g.add_nodes_from(range(k, n))
        g.add_edges_from((i, j) for i in range(k) for j in range(k, n)); out.append(("csplit", g))
    for k in range(2, n - 1):  # lollipop / kite
        out.append(("lollipop", nx.lollipop_graph(k, n - k)))
    if n % 2 == 1:
        out.append(("friendship", nx.windmill_graph((n - 1) // 2, 3)))
    if n % 2 == 0:
        out.append(("barbell", nx.barbell_graph(n // 2, 0)))
        out.append(("ladder", nx.ladder_graph(n // 2)))
        out.append(("prism", nx.circular_ladder_graph(n // 2)))
        out.append(("cocktail", nx.complement(nx.Graph([(2 * i, 2 * i + 1) for i in range(n // 2)]))))
    for jumps in itertools.combinations(range(1, n // 2 + 1), 2):
        out.append(("circulant", nx.circulant_graph(n, list(jumps))))
    for r in (3, 4):
        if n * r % 2 == 0:
            for _ in range(20):
                out.append(("regular", nx.random_regular_graph(r, n, seed=rng.randint(0, 10**9))))
    for t in range(2, 6):  # complete multipartite, balanced
        parts = [n // t + (1 if i < n % t else 0) for i in range(t)]
        out.append(("multipartite", nx.complete_multipartite_graph(*parts)))
    for _ in range(300):
        out.append(("tree", nx.random_labeled_tree(n, seed=rng.randint(0, 10**9))))
    # corona of a path / caterpillars with many leaves
    if n % 2 == 0:
        g = nx.path_graph(n // 2)
        for i in range(n // 2): g.add_edge(i, n // 2 + i)
        out.append(("corona", g))
    return out


def main():
    graphs, tags = [], []
    for n in range(11, 17):
        for tag, g in families(n):
            g = nx.convert_node_labels_to_integers(g)
            if g.number_of_nodes() == n and nx.is_connected(g):
                graphs.append(g); tags.append(f"{tag}:{n}")
        cnt = 0
        per_n = 3000 if n <= 13 else 1500
        while cnt < per_n:
            p = rng.uniform(0.08, 0.92)
            g = nx.gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
            if nx.is_connected(g):
                graphs.append(g); tags.append(f"gnp:{n}"); cnt += 1
    data = "".join(nx.to_graph6_bytes(g, header=False).decode() for g in graphs)
    out = subprocess.run([os.path.join(HERE, "invariants")], input=data.encode(), capture_output=True, check=True).stdout
    open(os.path.join(DATA, "inv_large.bin"), "wb").write(out)
    open(os.path.join(DATA, "large_meta.txt"), "w").write("\n".join(tags) + "\n")
    print("large sample graphs:", len(graphs))


if __name__ == "__main__":
    main()
