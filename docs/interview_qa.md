# Interview Q&A

**Q: Why use BCa over standard percentile bootstrap?**
A: BCa adjusts for both bias and skewness in the statistic's distribution, making it more accurate for skewed data or when the estimator is biased, though it's computationally more complex.

**Q: How does McNemar's test differ from a standard t-test for comparing models?**
A: McNemar's test is specifically for paired nominal data, making it ideal for comparing two classifiers on the same test set by looking at where they disagree (e.g. one correct, one wrong), whereas a t-test is for continuous outcomes.

**Q: When would you use Benjamini-Hochberg over Holm correction?**
A: Benjamini-Hochberg controls the False Discovery Rate (FDR) and is less conservative, making it suitable when you are willing to accept some false positives in exchange for finding more true effects. Holm controls the Family-Wise Error Rate (FWER) and is better when false positives are highly undesirable.

**Q: What is the main advantage of split conformal prediction?**
A: It provides finite-sample, distribution-free guarantees on the marginal coverage of prediction sets, without needing retraining for every test point (unlike full conformal prediction).

**Q: Does temperature scaling affect a model's accuracy?**
A: No, temperature scaling only scales the logits by a constant scalar before softmax. This changes the predicted probabilities (improving calibration) but does not change the argmax, so accuracy remains identical.
