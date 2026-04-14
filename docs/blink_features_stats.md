# Statistical testing in blink_features.py

## What is being compared

For each of the six per-trace blink features (duty cycle, blinking rate, mean on-time, mean off-time, CV on-times, CV off-times), the distributions of the two protein classes (e.g. HTHTL vs HTIA) are compared. Each observation is one MCD-filtered trace.

---

## Test: Mann-Whitney U (Wilcoxon rank-sum)

**Why not a t-test?**
A Welch t-test assumes that sample means are approximately normally distributed. While the central limit theorem makes this reasonable for large n, the features here are clearly non-normal:

- Off-times follow an approximately exponential distribution (CV > 1).
- On-times, duty cycle, and blinking rate are right-skewed and bounded at zero.
- CV on/off-times are also positive and skewed.

Fitting a normal model to these distributions would be misleading, so a non-parametric test is preferred.

**Why Mann-Whitney U?**
The Mann-Whitney U test (equivalent to the Wilcoxon rank-sum test) makes no distributional assumptions. It tests whether, for a randomly selected trace from protein A and a randomly selected trace from protein B, it is equally likely that either has the larger value. It is the standard two-sample non-parametric test for independent groups and is appropriate for all six features.

**Implementation:** `scipy.stats.mannwhitneyu` with `alternative='two-sided'`.

---

## Effect size: rank-biserial correlation (r)

With n ~ 200 per class, even biologically negligible differences can reach p < 0.05. p-values alone are therefore insufficient. The rank-biserial correlation r is reported alongside each test:

```
r = 1 - 2U / (n1 * n2)
```

where U is the Mann-Whitney U statistic. r ranges from -1 to +1, with conventional thresholds:

| |r| range | Interpretation |
|-----------|----------------|
| < 0.10    | Negligible     |
| 0.10–0.30 | Small          |
| 0.30–0.50 | Medium         |
| > 0.50    | Large          |

---

## Multiple comparison correction: Benjamini-Hochberg FDR

Six features are tested simultaneously. Without correction, the probability of at least one false positive at α = 0.05 rises to ~ 26% (1 − 0.95^6). Two common correction strategies are:

- **Bonferroni**: multiplies each p-value by m = 6. Controls the family-wise error rate (FWER) — the probability of any false positive. Very conservative when tests are correlated.
- **Benjamini-Hochberg (BH) FDR**: controls the expected fraction of significant results that are false positives. Less conservative than Bonferroni when tests are correlated, which they are here (e.g. duty cycle, blinking rate, mean on-time, and mean off-time are all derived from the same underlying GMM peak calls).

BH FDR is the appropriate choice and is applied at α = 0.05.

**Procedure:**
1. Sort the m raw p-values in ascending order: p_(1) ≤ … ≤ p_(m).
2. Compute adjusted p-values: p_adj_(i) = p_(i) × m / i.
3. Enforce monotonicity by scanning from rank m down to 1: p_adj_(i) = min(p_adj_(i), p_adj_(i+1)).
4. Clip at 1.0.

Significance thresholds displayed on the plot:

| Symbol | BH-adjusted p |
|--------|---------------|
| ns     | ≥ 0.05        |
| *      | < 0.05        |
| **     | < 0.01        |
| ***    | < 0.001       |
| ****   | < 0.0001      |

---

## Data included

Only MCD-filtered traces (Wasserstein distance ≥ configured threshold) are included. No trace is excluded based on its peak properties; all peaks from all passing traces contribute to the per-trace summary statistics.
