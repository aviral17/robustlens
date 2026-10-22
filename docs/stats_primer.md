# Stats Primer

## Percentile Bootstrap CI
A non-parametric method to compute confidence intervals by resampling with replacement. We take the $\alpha / 2$ and $1 - \alpha / 2$ percentiles of the bootstrapped statistic distribution.

## BCa Bootstrap CI
Bias-Corrected and accelerated (BCa) bootstrap improves upon the percentile method by adjusting for bias and skewness in the bootstrap distribution.

## Paired Bootstrap Comparisons
Compares two paired samples by resampling indices. The difference in statistics is calculated for each bootstrap sample to construct a CI and compute a p-value for the difference.

## McNemar's Test
A statistical test used on paired nominal data to determine whether the row and column marginal frequencies are equal. Specifically used for comparing the performance of two classifiers.

## Holm and Benjamini-Hochberg Correction
Methods to adjust p-values for multiple comparisons. Holm controls the Family-Wise Error Rate (FWER) and is uniformly more powerful than Bonferroni. Benjamini-Hochberg controls the False Discovery Rate (FDR).

## ECE (Expected Calibration Error)
Measures the difference between predicted probabilities and empirical accuracy, typically evaluated by binning the probabilities.

## Reliability Diagram
A visual representation of calibration, plotting empirical accuracy against predicted confidence across bins.

## Temperature Scaling
A simple calibration technique that scales logits by a learned parameter $T > 0$ before applying the softmax function, preserving accuracy while improving ECE.

## Split Conformal Prediction
A framework to provide valid predictive sets with guaranteed marginal coverage without making strong distributional assumptions. Uses a calibration set to compute a threshold for test prediction sets.

## Ensembles (Soft Voting, Weighted, Stacking)
Techniques to combine multiple models to improve predictive performance. Soft voting averages probabilities, weighted applies specific weights, and stacking trains a meta-model on base predictions.
