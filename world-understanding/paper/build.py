"""Assemble paper.md from its parts and render paper.html (pandoc) and paper.pdf (Chromium).
Run from anywhere: python paper/build.py"""
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = ["header.md", "front.md", "methods.md", "results_a.md", "results_b.md", "results_c.md", "results_d.md", "synthesis.md", "appendix.md"]


def main():
    text = "\n\n".join((HERE / p).read_text() for p in PARTS if (HERE / p).exists())
    (HERE / "paper.md").write_text(text)
    subprocess.run(["pandoc", "paper.md", "-o", "paper.html", "--standalone", "--mathml", "--citeproc",
                    "--bibliography", "references.bib", "--css", "paper.css", "--embed-resources",
                    "--metadata", "link-citations=true"],
                   cwd=HERE, check=True)
    subprocess.run(["node", "topdf.js"], cwd=HERE, check=True)
    print("built", HERE / "paper.pdf")


if __name__ == "__main__":
    main()
