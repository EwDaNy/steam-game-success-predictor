# Steam Game Success Prediction - KNN

## Project Description

This project predicts whether a Steam game is successful using the K-Nearest Neighbors algorithm.

## Main Files

```text
app.py
steam.csv
knn_model.pkl
scaler.pkl
Untitled.ipynb
```

## How to Run

```powershell
cd D:\projects\KNN_Project
..\deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

## Evaluation

The notebook includes:

- accuracy score
- classification report
- cross-validation scores
- mean cross-validation accuracy
- standard deviation of cross-validation accuracy

KNN is distance-based, so scaling is important. Cross-validation is implemented with a `Pipeline` so that `StandardScaler` is fitted separately inside each fold.

