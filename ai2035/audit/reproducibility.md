# 复现审计

在隔离副本中从零重跑全部 `experiment.py` 与 `frontier_experiment.py`（单线程 BLAS），逐个比较 JSON 中的全部数值叶子（忽略运行时间与元数据）。

| 结果文件 | 数值数 | 逐位一致比例 | 相对差<1e-6 比例 | 最大相对偏差 | 判定 |
|---|---|---|---|---|---|
| `projects/Q01-cognition-computation/results/frontier.json` | 624 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q01-cognition-computation/results/results.json` | 1443 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q02-deep-learning-math/results/frontier.json` | 673 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q02-deep-learning-math/results/results.json` | 546 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q03-scaling-emergence/results/frontier.json` | 299 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q03-scaling-emergence/results/results.json` | 243 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q04-statistical-foundations/results/frontier.json` | 474 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q04-statistical-foundations/results/results.json` | 2322 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q05-rl-theory/results/frontier.json` | 709 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q05-rl-theory/results/results.json` | 988 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q06-few-label-learning/results/frontier.json` | 652 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q06-few-label-learning/results/results.json` | 673 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q07-continual-learning/results/frontier.json` | 509 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q07-continual-learning/results/results.json` | 1112 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q08-causal-learning/results/frontier.json` | 358 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q08-causal-learning/results/results.json` | 571 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q09-ood-generalization/results/frontier.json` | 715 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q09-ood-generalization/results/results.json` | 481 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q10-knowledge-data-fusion/results/frontier.json` | 1931 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q10-knowledge-data-fusion/results/results.json` | 1014 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q11-interpretability/results/frontier.json` | 885 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q11-interpretability/results/results.json` | 3077 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q12-robustness/results/frontier.json` | 276 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q12-robustness/results/results.json` | 398 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q13-privacy/results/frontier.json` | 1203 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q13-privacy/results/results.json` | 364 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q14-fairness-alignment/results/frontier.json` | 2290 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q14-fairness-alignment/results/results.json` | 1613 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q15-uncertainty/results/frontier.json` | 218 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q15-uncertainty/results/results.json` | 747 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q16-brain-inspired/results/frontier.json` | 927 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q16-brain-inspired/results/results.json` | 1487 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q17-world-models/results/frontier.json` | 2100 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q17-world-models/results/results.json` | 451 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q18-multi-agent/results/frontier.json` | 746 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q18-multi-agent/results/results.json` | 859 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q19-human-ai-hybrid/results/frontier.json` | 2894 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q19-human-ai-hybrid/results/results.json` | 2543 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q20-ai-for-science/results/frontier.json` | 288 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |
| `projects/Q20-ai-for-science/results/results.json` | 546 | 1.0000 | 1.0000 | 0.00e+00 | 逐位一致 |

存在差异的文件数：0 / 40
