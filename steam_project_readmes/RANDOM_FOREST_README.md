# Steam Game Success Prediction - Random Forest

## Project Description

This project predicts Steam game success using Random Forest Classifier.

## Main Files

```text
app.py
steam.csv
random_forest_model.pkl
rf_features.pkl
Untitled1 (1).ipynb
```

## How to Save Model Files

In the Random Forest notebook, run:

```python
import joblib

joblib.dump(model, r"D:\projects\randomForest\random_forest_model.pkl")
joblib.dump(list(X.columns), r"D:\projects\randomForest\rf_features.pkl")
```

## How to Run

```powershell
cd D:\projects\randomForest
..\deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

## Evaluation

The notebook includes:

- accuracy score
- classification report
- confusion matrix
- cross-validation scores
- mean cross-validation accuracy
- standard deviation of cross-validation accuracy

The project removes leakage columns such as `positive`, `negative`, `rating_ratio`, and `ccu`.
The achieved test accuracy was approximately `0.759`.

