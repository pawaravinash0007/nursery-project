<div align="center">

# 🌿 nursery-project

### Nursery Dataset Classification — Multi-Model Comparison

[![Language](https://img.shields.io/badge/Language-Python-blue?style=flat-square&logo=python)](.)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange?style=flat-square&logo=scikit-learn)](.)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange?style=flat-square&logo=jupyter)](.)
[![Dataset](https://img.shields.io/badge/Dataset-UCI%20Nursery-green?style=flat-square)](.)

</div>

---

## 📖 Overview

A machine learning classification project using the **UCI Nursery Dataset** to evaluate and compare multiple classification algorithms. The nursery dataset was developed to rank applications for nursery schools in Ljubljana, Slovenia, and is a classic multi-class classification benchmark.

---

## 📊 Dataset

| Property | Details |
|----------|---------|
| **Source** | UCI Machine Learning Repository |
| **Samples** | 12,960 instances |
| **Features** | 8 categorical attributes |
| **Classes** | 5 (not_recom, recommend, very_recom, priority, spec_prior) |

**Features:** parents, has_nurs, form, children, housing, finance, social, health.

---

## 🤖 Models Compared

| Model | Accuracy |
|-------|---------|
| Decision Tree | ~99% |
| Random Forest | ~99% |
| Naive Bayes | ~90% |
| KNN | ~97% |
| Logistic Regression | ~92% |
| SVM | ~98% |

---

## ✨ Key Findings

- Tree-based models achieve near-perfect accuracy on this dataset
- The dataset is well-structured with clear decision boundaries
- **Health** and **finance** attributes are the most discriminative features
- Naive Bayes underperforms due to feature dependencies

---

## 🚀 Getting Started

```bash
git clone https://github.com/pawaravinash0007/nursery-project.git
cd nursery-project
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook nursery_classification.ipynb
```

---

## 📁 Repository Structure

```
nursery-project/
├── data/
│   └── nursery.data            # UCI nursery dataset
├── nursery_EDA.ipynb           # Exploratory analysis
├── nursery_models.ipynb        # Model comparison
└── README.md
```

---

## 🛠️ Tech Stack

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `Matplotlib` · `Seaborn`

---

## 👤 Author

**Avinash Pawar** | [@pawaravinash0007](https://github.com/pawaravinash0007)

<div align="center">⭐ Star this repo if you find it useful! ⭐</div>
