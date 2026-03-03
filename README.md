# Why Are You Poor? A Kernel PCA Analysis of Billionaire Wealth

Unsupervised machine learning project applying **Kernel PCA** to Forbes billionaire data to uncover nonlinear patterns that distinguish the ultra-wealthy.

## 🎯 Objective

Determine whether **nonlinear dimensionality reduction** (Kernel PCA) reveals structure in billionaire characteristics that linear methods miss, using wealth rank as the target variable.

## 📊 Dataset

- **Source**: IBM Skills Network - Billionaires Dataset
- **Size**: ~2,600 billionaires
- **Features**: Country, industry, wealth source, age, gender, self-made status
- **Target**: Inverse rank (1/rank) — higher values indicate greater wealth

## 🔬 Methodology

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | Data Cleaning | Parse net worth strings, handle mixed B/M units |
| 2 | Feature Engineering | One-hot encode categoricals, standardize |
| 3 | Linear PCA | Baseline dimensionality reduction |
| 4 | **Kernel PCA** | Nonlinear pattern extraction (RBF kernel) |
| 5 | Ridge Regression | Evaluate predictive power of components |
| 6 | Visualization | Compare projections colored by wealth |

## 🚀 Key Findings

- Kernel PCA captures **nonlinear clusters** invisible to linear PCA
- Wealth concentration creates distinct **tiers** in feature space
- Industry and geography are stronger predictors than demographic factors

## 🛠️ Tech Stack

```python
pandas, numpy, scikit-learn, matplotlib, seaborn
