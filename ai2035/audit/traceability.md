# 数字可追溯审计

报告/论文/结论卡中共抽取 6153 个 ≥2 位小数（不含文献计量节与参考文献），其中 6129 个（99.6%）可在本项目 results/*.json 中按引用精度匹配。
未匹配者多为文献引用值、理论常数、假设阈值或由多个结果推导出的比值，需结合上下文复核。

| 文件 | 小数个数 | 可追溯 | 比例 | 未匹配示例 |
|---|---|---|---|---|
| `Q01-cognition-computation/README.md` | 105 | 104 | 99% | 1.71 |
| `Q01-cognition-computation/paper.md` | 124 | 124 | 100% |  |
| `Q01-cognition-computation/CONCLUSIONS.md` | 14 | 14 | 100% |  |
| `Q02-deep-learning-math/README.md` | 81 | 80 | 99% | 3.11 |
| `Q02-deep-learning-math/paper.md` | 92 | 79 | 86% | -0.64, -0.64, -2.04, -0.64, -0.64, -0.64, -2.04, -0.64 |
| `Q02-deep-learning-math/CONCLUSIONS.md` | 9 | 7 | 78% | -0.64, -0.64 |
| `Q03-scaling-emergence/README.md` | 132 | 132 | 100% |  |
| `Q03-scaling-emergence/paper.md` | 84 | 84 | 100% |  |
| `Q03-scaling-emergence/CONCLUSIONS.md` | 12 | 12 | 100% |  |
| `Q04-statistical-foundations/README.md` | 123 | 122 | 99% | 0.087 |
| `Q04-statistical-foundations/paper.md` | 155 | 155 | 100% |  |
| `Q04-statistical-foundations/CONCLUSIONS.md` | 10 | 10 | 100% |  |
| `Q05-rl-theory/README.md` | 54 | 54 | 100% |  |
| `Q05-rl-theory/paper.md` | 139 | 139 | 100% |  |
| `Q05-rl-theory/CONCLUSIONS.md` | 7 | 7 | 100% |  |
| `Q06-few-label-learning/README.md` | 221 | 221 | 100% |  |
| `Q06-few-label-learning/paper.md` | 174 | 174 | 100% |  |
| `Q06-few-label-learning/CONCLUSIONS.md` | 5 | 5 | 100% |  |
| `Q07-continual-learning/README.md` | 183 | 182 | 99% | 0.70 |
| `Q07-continual-learning/paper.md` | 120 | 120 | 100% |  |
| `Q07-continual-learning/CONCLUSIONS.md` | 7 | 7 | 100% |  |
| `Q08-causal-learning/README.md` | 117 | 117 | 100% |  |
| `Q08-causal-learning/paper.md` | 70 | 68 | 97% | 1.16, 1.56 |
| `Q08-causal-learning/CONCLUSIONS.md` | 1 | 1 | 100% |  |
| `Q09-ood-generalization/README.md` | 175 | 175 | 100% |  |
| `Q09-ood-generalization/paper.md` | 137 | 137 | 100% |  |
| `Q09-ood-generalization/CONCLUSIONS.md` | 7 | 7 | 100% |  |
| `Q10-knowledge-data-fusion/README.md` | 237 | 237 | 100% |  |
| `Q10-knowledge-data-fusion/paper.md` | 67 | 67 | 100% |  |
| `Q10-knowledge-data-fusion/CONCLUSIONS.md` | 7 | 7 | 100% |  |
| `Q11-interpretability/README.md` | 207 | 207 | 100% |  |
| `Q11-interpretability/paper.md` | 195 | 195 | 100% |  |
| `Q11-interpretability/CONCLUSIONS.md` | 3 | 3 | 100% |  |
| `Q12-robustness/README.md` | 222 | 222 | 100% |  |
| `Q12-robustness/paper.md` | 204 | 204 | 100% |  |
| `Q12-robustness/CONCLUSIONS.md` | 2 | 2 | 100% |  |
| `Q13-privacy/README.md` | 193 | 193 | 100% |  |
| `Q13-privacy/paper.md` | 121 | 121 | 100% |  |
| `Q13-privacy/CONCLUSIONS.md` | 1 | 1 | 100% |  |
| `Q14-fairness-alignment/README.md` | 236 | 236 | 100% |  |
| `Q14-fairness-alignment/paper.md` | 169 | 169 | 100% |  |
| `Q14-fairness-alignment/CONCLUSIONS.md` | 3 | 3 | 100% |  |
| `Q15-uncertainty/README.md` | 257 | 256 | 100% | 0.956 |
| `Q15-uncertainty/paper.md` | 138 | 138 | 100% |  |
| `Q15-uncertainty/CONCLUSIONS.md` | 7 | 7 | 100% |  |
| `Q16-brain-inspired/README.md` | 141 | 141 | 100% |  |
| `Q16-brain-inspired/paper.md` | 187 | 186 | 99% | 47.49 |
| `Q16-brain-inspired/CONCLUSIONS.md` | 6 | 6 | 100% |  |
| `Q17-world-models/README.md` | 183 | 183 | 100% |  |
| `Q17-world-models/paper.md` | 170 | 170 | 100% |  |
| `Q17-world-models/CONCLUSIONS.md` | 4 | 4 | 100% |  |
| `Q18-multi-agent/README.md` | 144 | 144 | 100% |  |
| `Q18-multi-agent/paper.md` | 126 | 125 | 99% | 0.736 |
| `Q18-multi-agent/CONCLUSIONS.md` | 11 | 11 | 100% |  |
| `Q19-human-ai-hybrid/README.md` | 106 | 106 | 100% |  |
| `Q19-human-ai-hybrid/paper.md` | 187 | 187 | 100% |  |
| `Q19-human-ai-hybrid/CONCLUSIONS.md` | 3 | 3 | 100% |  |
| `Q20-ai-for-science/README.md` | 119 | 119 | 100% |  |
| `Q20-ai-for-science/paper.md` | 139 | 139 | 100% |  |
| `Q20-ai-for-science/CONCLUSIONS.md` | 0 | 0 | 0% |  |
