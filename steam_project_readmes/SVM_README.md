# Steam Game Success Prediction - SVM

## Project Description

This project predicts whether a Steam game is successful using Support Vector Machine.

## Main Files

```text
app.py
steam.csv
svm_model.pkl
scaler.pkl
svm_features.pkl
Untitled.ipynb
```

## How to Save Model Files

In the SVM notebook, run:

```python
import joblib

joblib.dump(model, r"D:\projects\SVM_Project\svm_model.pkl")
joblib.dump(scaler, r"D:\projects\SVM_Project\scaler.pkl")
joblib.dump(list(X.columns), r"D:\projects\SVM_Project\svm_features.pkl")
```

## How to Run

```powershell
cd D:\projects\SVM_Project
..\deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

## Evaluation

The notebook includes:

- accuracy score
- classification report
- cross-validation scores
- mean cross-validation accuracy
- standard deviation of cross-validation accuracy

Cross-validation is implemented with a `Pipeline` so that `StandardScaler` is fitted separately inside each fold.

