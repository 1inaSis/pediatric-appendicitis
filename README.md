# Lina benaddi contribution on 10/03
#  ML Pipeline — Pediatric Appendicitis

![CI](https://github.com/1inaSis/pediatric-appendicitis/actions/workflows/ci.yml/badge.svg)

---

##  How to train the model
```bash
python src/train_model.py
```

---

##  ML Models Comparison

I trained and compared 4 machine learning models on the
Regensburg Pediatric Appendicitis dataset (780 patients, 57 features).

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|---|---|---|---|---|---|
| SVM | 0.7756 | 0.7931 | 0.8452 | 0.7634 | 0.8023 |
| **Random Forest** | **0.9551** | **0.9901** | **0.9388** | **0.9892** | **0.9634** |
| LightGBM | 0.9744 | 0.9892 | 0.9588 | 1.0000 | 0.9789 |
| CatBoost | 0.9808 | 0.9869 | 0.9688 | 1.0000 | 0.9841 |

###  Best Model : Random Forest
- I selected Random Forest based on the highest ROC-AUC (0.9901)
- I achieved a Recall of 0.9892 — meaning I miss almost no real
  appendicitis cases, which is critical in a medical context
- I achieved a Precision of 0.9388 — when I predict appendicitis,
  I am right 94% of the time

### Why Random Forest ?
I chose to select the best model automatically based on ROC-AUC score.
In a medical context, I prioritized Recall over Precision because
missing a real appendicitis case (false negative) is far more
dangerous than a false alarm (false positive).

---

## ⚙️ CI/CD Pipeline

I set up a **GitHub Actions** pipeline for continuous integration.

### What I automated :
-  I install Python 3.11 automatically
-  I install all project dependencies
-  I run pytest tests automatically on every push

### When does it trigger :
- Every time someone pushes to `main`
- Every time a Pull Request is opened toward `main`

### Configuration file I created :
`.github/workflows/ci.yml`

---

##  Prompt Engineering

### Task I chose : Universal function `auto_prepare_data()`

**Prompt I used :**
> "Write a universal Python function called auto_prepare_data(filepath, target_col=None)
> that works with ANY dataset. It should:
> - Automatically detect CSV or Excel format
> - Auto-detect the target column if not specified
> - Handle missing values with median imputation
> - Encode categorical columns automatically
> - Map text labels like 'appendicitis' to 1 and 'no appendicitis' to 0
> - Return X features and y target ready for ML training"

**Result I got :**
The function I generated works with any dataset — not just
appendicitis data. It automatically detects the file format
and intelligently encodes the target column.

**What worked well :**
I was very specific in my prompt with concrete examples of
expected behavior — the generated function was almost
ready to use immediately with minimal adjustments.

**What I would improve :**
I would add more detail about the expected text mapping
(e.g., 'appendicitis' → 1) to avoid encoding bugs
that I encountered during testing.

------------------------------------------------------------------------------------------------------------------------

