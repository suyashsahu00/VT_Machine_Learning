# Machine Learning Study Notes

## Supervised vs Unsupervised Learning
In **Supervised Learning**, the model is trained on labeled data, meaning each training example is paired with an output label. Common tasks include Regression (predicting continuous values like house prices) and Classification (predicting discrete classes like spam vs. not spam).
In **Unsupervised Learning**, the model finds hidden patterns or structures in unlabeled data. A classic example is Clustering (e.g., K-Means clustering to group similar customers together) and Dimensionality Reduction (e.g., PCA to simplify complex datasets).

## Overfitting and Underfitting
* **Overfitting** occurs when a model learns the training data *too* well, including the noise and outliers. This results in high accuracy on the training set but poor performance (low generalization) on unseen test data.
* **Underfitting** happens when a model is too simple to capture the underlying pattern of the data. This leads to poor performance on both the training and testing datasets.

## The Bias-Variance Tradeoff
* **Bias** represents the error introduced by approximating a real-world problem (which may be complex) by a much simpler model. High bias leads to underfitting.
* **Variance** is the model's sensitivity to small fluctuations in the training set. High variance leads to overfitting because the model fits the training set's random noise instead of the general trend.
* The goal is to find the sweet spot that minimizes both bias and variance, leading to the lowest overall test error.

## Evaluation Metrics for Classification
* **Accuracy**: The ratio of correctly predicted observations to the total observations. It can be misleading for imbalanced datasets.
* **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. (Formula: TP / (TP + FP))
* **Recall (Sensitivity)**: The ratio of correctly predicted positive observations to all actual positives. (Formula: TP / (TP + FN))
* **F1-Score**: The harmonic mean of Precision and Recall. It provides a balanced measure when classes are imbalanced.
