# Academic write-up — three length tiers

One canonical academic treatment of the Alken meta-labelling metamodel, rendered at three depths. The three documents tell the identical honest-negative story with the same figures and the same citations; they differ only in how much detail they carry. Read whichever matches the time available.

| File | Length | Read it when you want… |
| :--- | :--- | :--- |
| [`methodology-full.md`](methodology-full.md) | ~6,500 words | the complete methodology — every feature block, the validation scheme, the cluster-importance discipline, the full significance and deflation tables, and a reproducibility appendix. |
| [`methodology-report.md`](methodology-report.md) | ~5,000 words | a submission-scale scholarly report: the argument and the load-bearing tables, without the second-order detail. |
| [`methodology-brief.md`](methodology-brief.md) | ~2,100 words | the verdict and the reasoning behind it in a few minutes — the five-lens convergence, the two dissolved traps, and the limitations. |

**The finding, in one line.** On the six-month out-of-sample window the pooled net Sharpe ratio of 1.31 is not statistically distinguishable from zero (*t* = 0.93, n = 127; bootstrap 95% interval [−0.04, +0.19], containing zero), and five independent diagnostics agree: the conclusion is *insufficient evidence of a deployable edge*, not a proven failure.

**Provenance.** These documents supersede the internal `../../docs/methodology.md` and the earlier `../T3_03_Alken_Metamodel_Report.md`, which are retained unchanged. Every quoted figure is labelled by its estimator and evaluation window and is anchored to the seeded regeneration command `uv run --project metamodel-apb python -m alken_metamodel.emit` (which reproduces the deliverable byte-for-byte); the page-exact reference list lives in `../research/nlr-cw-v1.md`.
