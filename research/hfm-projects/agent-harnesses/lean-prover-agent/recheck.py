"""Re-check accepted proofs in a fresh file, under their original theorem names, and test
whether the hypotheses of selected statements are contradictory (a benchmark defect that
would let any goal be proved).

Usage: python recheck.py <miniF2F-lean4 checkout> <results.json ...>
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lean_floor as lf

CONTRA = "(first | (simp_all; done) | (norm_num at *; done) | (have := h₁ 0 0; simp_all; done) | aesop | grind)"


def main():
    repo = sys.argv[1]
    stmts = lf.statements(os.path.join(repo, "MiniF2F", "Test"))
    proofs = {}
    for f in sys.argv[2:]:
        for r in json.load(open(f)):
            if r.get("proved") and r.get("wins"):
                proofs[r["name"]] = min(r["wins"], key=lambda w: len(w["proof"]))["proof"]
    path = os.path.join(repo, "recheck.lean")
    with open(path, "w") as fo:
        fo.write(lf.HEADER % 400000)
        for name, p in sorted(proofs.items()):
            fo.write(f"{stmts[name].rstrip()}\n{p}\n\n#print axioms {name}\n\n")
            # the same hypotheses with the goal replaced by False
            head = stmts[name][:stmts[name].rfind(":=")]
            cut = head.rfind(") :")
            if cut > 0:
                binders = head[head.find(name) + len(name):cut + 1]
                fo.write(f"theorem {name}_contra{binders} : False := by {CONTRA}\n\n#print axioms {name}_contra\n\n")
    out, status = lf.lean(repo, path, 1800)
    print(open(path).read())
    print("===== lean output =====")
    print(out)
    print("lean status", status)
    for name in sorted(proofs):
        ok = re.search(rf"'{name}' depends on axioms: \[([^\]]*)\]", out)
        contra = re.search(rf"'{name}_contra' depends on axioms: \[([^\]]*)\]", out)
        print(f"RESULT {name}: proof {'accepted' if ok and 'sorryAx' not in ok.group(1) else 'REJECTED'}; "
              f"hypotheses contradictory: {'YES' if contra and 'sorryAx' not in contra.group(1) else 'not shown'}")


if __name__ == "__main__":
    main()
