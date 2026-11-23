"""Extract intra-package module dependency DAGs from installed Python packages.

Edge orientation: u -> v  means "v imports u" (u must exist before v; a defect
in u can propagate to v).  Strongly connected components (import cycles) are
condensed into single super-modules so that the result is a DAG.
Package `__init__.py` files are excluded: they mostly re-export names and
otherwise create spurious package-level cycles; an import that resolves only
to a package (not a submodule) is dropped.
Test modules (any path component named test/tests/testing or file test_*.py)
are excluded so that the graph reflects production code only.
Only import-time imports are used (imports inside function bodies are skipped).
"""
import ast, os, importlib
import networkx as nx

PACKAGES = ["numpy", "scipy", "matplotlib", "networkx", "sympy", "pandas",
            "setuptools", "pip", "asyncio", "email", "unittest", "multiprocessing",
            "importlib", "xml", "http", "logging"]

SKIP_DIRS = {"tests", "test", "testing", "_vendor", "vendor", "benchmarks"}


def _modname(root, pkg, path):
    rel = os.path.relpath(path, root)[:-3].split(os.sep)
    if rel[-1] == "__init__":
        rel = rel[:-1]
    return ".".join([pkg] + rel)


def _toplevel_nodes(tree):
    """Yield nodes executed at import time: skip function bodies, where deferred
    imports are conventionally placed to break cycles (load-time dependency only)."""
    stack = list(tree.body)
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def package_graph(pkg):
    mod = importlib.import_module(pkg)
    root = os.path.dirname(mod.__file__)
    files = {}
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS and not x.startswith(".")]
        for f in fs:
            if f.endswith(".py") and not f.startswith("test_") and not f.startswith("conftest"):
                if f == "__init__.py":
                    continue  # package initialisers are re-export aggregators (see docstring)
                p = os.path.join(d, f)
                files[_modname(root, pkg, p)] = p
    names = set(files)
    G = nx.DiGraph(); G.add_nodes_from(names)
    for m, path in files.items():
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="ignore").read())
        except SyntaxError:
            continue
        cur_pkg = m.rsplit(".", 1)[0]
        for node in _toplevel_nodes(tree):
            targets = []
            if isinstance(node, ast.Import):
                targets = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = cur_pkg.split(".")
                    if node.level > 1:
                        base = base[: len(base) - (node.level - 1)]
                    base = ".".join(base)
                    full = base + ("." + node.module if node.module else "")
                else:
                    full = node.module or ""
                targets = [full] + [full + "." + a.name for a in node.names]
            for t in targets:
                # resolve to the longest existing module prefix
                parts = t.split(".")
                while parts and ".".join(parts) not in names:
                    parts = parts[:-1]
                if parts:
                    tgt = ".".join(parts)
                    if tgt != m:
                        G.add_edge(tgt, m)
    return G


def to_dag(G):
    C = nx.condensation(G)
    D = nx.DiGraph()
    D.add_nodes_from(range(C.number_of_nodes()))
    D.add_edges_from(C.edges())
    sizes = [len(C.nodes[i]["members"]) for i in C.nodes]
    return D, sizes


def load_all(min_nodes=20):
    out = {}
    for pkg in PACKAGES:
        try:
            G = package_graph(pkg)
        except Exception as e:  # pragma: no cover
            print("skip", pkg, e); continue
        D, sizes = to_dag(G)
        if D.number_of_nodes() >= min_nodes:
            out[pkg] = (G, D, sizes)
    return out


if __name__ == "__main__":
    for k, (G, D, s) in load_all().items():
        print(k, G.number_of_nodes(), G.number_of_edges(), D.number_of_nodes(), D.number_of_edges(), max(s))
