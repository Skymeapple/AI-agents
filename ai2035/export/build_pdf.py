"""Merge the 20 frontier papers into one PDF.

pipeline: front matter + 20 paper.md (image paths made absolute, each paper's H1 kept as a
chapter) -> pandoc (standalone HTML, TOC of chapters) -> Chromium (Playwright) print to A4
PDF with page-number footer.

Usage: python ai2035/export/build_pdf.py   -> ai2035/export/ai2035-papers.pdf
"""
from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_MD = HERE / "_combined.md"
OUT_HTML = HERE / "_combined.html"
OUT_PDF = HERE / "ai2035-papers.pdf"

FRONT = """<div class="cover">

<h1 class="cover-title">面向 2035 的人工智能<br>20 个基础科学问题</h1>

<p style="font-size:15pt">前沿研究论文集</p>

<p style="margin-top:3em">ai2035 独立研究项目 ｜ {date}</p>

</div>

# 说明

**内容**：本论文集收录 20 篇前沿研究论文，对应中国科学院学部“中国学科及前沿领域 2035 发展战略丛书”人工智能卷提出的 20 个基础科学问题。每篇论文先综述该方向 2025–2026 年的 arXiv 前沿文献，再选一个当前的前沿争论，做理论推导，在最小可控模型上做实验检验，最后给出结论与 2035 研判。

**题目清单来源**：编写时无法访问原书，20 个题目是依据可检索到的原书主题重建的工作清单，不是原文逐字条目，使用前请与原书核对。

**证据标注**：[实测] 表示本项目实验结果；[文献] 表示已核实的参考文献；[判断] 表示基于证据的推断。

**审计**：
- 文中引用的 781 个 arXiv 编号全部可在 arXiv API 解析；
- 在隔离环境中从零重跑全部 40 个实验脚本，约 4.2 万个数值逐位一致；
- 文中 6,153 个小数中有 99.6% 可在结果文件中直接追溯。

**局限**：
- 全部实验都是 CPU 上的最小模型（numpy / scikit-learn），只提供机理层面的证据，不能直接外推到大模型；各文“讨论”一节写明了外推边界。
- 论文由 AI 研究代理在统一研究规范下完成，未经人类同行评审。

**配套材料**：每篇论文对应的分析报告、实验代码、结果数据与一页结论卡片，均位于仓库 `ai2035/projects/Qxx-*/` 目录。

"""


def paper_md(q: dict) -> str:
    d = ROOT / "projects" / f"{q['id']}-{q['slug']}"
    text = (d / "paper.md").read_text()
    # absolute image paths
    text = re.sub(r"!\[([^\]]*)\]\((?!https?:|/)([^)]+)\)",
                  lambda m: f"![{m.group(1)}]({(d / m.group(2)).resolve()})", text)
    lines = text.splitlines()
    # chapter title: "Qxx  <title>", with the question as a tag line underneath
    assert lines[0].startswith("# "), d
    title = lines[0][2:].strip()
    head = [f"# {q['id']}　{title}", "",
            f'<p class="paper-tag">基础科学问题 {q["id"]}（{q["group"]}）：{q["title"]}</p>', ""]
    body = lines[1:]
    # demote any other H1 inside a paper so the TOC stays one entry per paper
    body = [("#" + ln) if re.match(r"^# ", ln) else ln for ln in body]
    # a list that follows a paragraph line directly needs a blank line, otherwise
    # CommonMark folds lists not starting at "1." into the preceding paragraph
    LIST = re.compile(r"^\s{0,3}(\d+[.)]|[-*+])\s")
    fixed = []
    for ln in body:
        if LIST.match(ln) and fixed and fixed[-1].strip() and not LIST.match(fixed[-1]) \
                and not fixed[-1].startswith(("|", "    ", "\t")):
            fixed.append("")
        fixed.append(ln)
    return "\n".join(head + fixed) + "\n\n"


def main() -> None:
    qs = yaml.safe_load((ROOT / "questions.yaml").read_text())["questions"]
    md = FRONT.format(date=time.strftime("%Y 年 %m 月")) + "".join(paper_md(q) for q in qs)
    OUT_MD.write_text(md)
    subprocess.run(["pandoc", str(OUT_MD), "-f", "gfm",
                    "-t", "html5", "--standalone", "--toc", "--toc-depth=1",
                    "--metadata", "title=面向2035的人工智能20个基础科学问题：前沿研究论文集",
                                        "--css", str(HERE / "style.css"), "--embed-resources",
                    "-o", str(OUT_HTML)], check=True)
    html = OUT_HTML.read_text()
    # pandoc's title block duplicates the cover; drop it. Move TOC after the 说明 page.
    html = re.sub(r'<header id="title-block-header">.*?</header>', "", html, flags=re.S)
    toc = re.search(r'<nav id="TOC".*?</nav>', html, flags=re.S).group(0)
    html = html.replace(toc, "")
    first_paper = re.search(r'<h1 id="[^"]*">Q01', html).start()
    html = html[:first_paper] + toc + html[first_paper:]
    # remove the 说明 entry from the TOC
    html = re.sub(r'<li><a href="#说明"[^<]*</a></li>', "", html)
    # short ASCII anchors: long CJK ids become oversized PDF names that some readers reject
    ids = re.findall(r'\sid="([^"]+)"', html)
    mapping = {old: f"s{i}" for i, old in enumerate(dict.fromkeys(ids)) if old not in ("TOC",)}
    html = re.sub(r'(\sid=")([^"]+)(")', lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)) + m.group(3), html)
    html = re.sub(r'(href="#)([^"]+)(")', lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)) + m.group(3), html)
    OUT_HTML.write_text(html)

    subprocess.run(["node", str(HERE / "print_pdf.js"), str(OUT_HTML), str(OUT_PDF)], check=True)
    print(OUT_PDF, f"{OUT_PDF.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()
