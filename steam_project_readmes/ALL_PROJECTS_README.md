# Steam Game Success Prediction Projects

This workspace contains four related machine learning projects that use the Steam games dataset to predict whether a game is successful.

## Models

1. K-Nearest Neighbors (KNN)
2. Support Vector Machine (SVM)
3. Random Forest
4. Deep Learning neural network

Each project has its own notebook, saved model files, and Streamlit deployment app. There is also one combined Streamlit app in the root folder that compares all four models at the same time.

## Folder Structure

```text
D:\projects
|-- app.py
|-- KNN_Project
|   |-- app.py
|   |-- steam.csv
|   |-- knn_model.pkl
|   |-- scaler.pkl
|   `-- Untitled.ipynb
|-- SVM_Project
|   |-- app.py
|   |-- steam.csv
|   |-- svm_model.pkl
|   |-- scaler.pkl
|   |-- svm_features.pkl
|   `-- Untitled.ipynb
|-- randomForest
|   |-- app.py
|   |-- steam.csv
|   |-- random_forest_model.pkl
|   |-- rf_features.pkl
|   `-- Untitled1 (1).ipynb
|-- deepLearning
|   |-- app.py
|   |-- steam.csv
|   |-- deep_learning_model.keras
|   |-- dl_scaler.pkl
|   |-- dl_features.pkl
|   |-- deep_learn.ipynb
|   `-- .venv
`-- report
    `-- steam_ml_deep_learning_report.tex
```

## Combined Deployment

The root `app.py` is the main deployment version. It loads all four trained models and shows their predictions in one interface.

Run it from the root project folder:

```powershell
cd D:\projects
deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

The combined app includes two preset buttons:

- `Use successful game example`
- `Use unsuccessful game example`

These buttons fill all input fields with example game data, so the app can be demonstrated quickly.

## Presentation

The project presentation is available here:

```text
report\steam_game_success_predictor_presentation.pptx
```

It summarizes the problem, dataset, methodology, models, evaluation, cross-validation and Streamlit deployment. The presentation uses the same diagrams and screenshots as the LaTeX documentation.

## Individual Deployment

Each model can also be launched separately from its own folder.

KNN:

```powershell
cd D:\projects\KNN_Project
..\deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

SVM:

```powershell
cd D:\projects\SVM_Project
..\deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

Random Forest:

```powershell
cd D:\projects\randomForest
..\deepLearning\.venv\Scripts\python.exe -m streamlit run app.py
```

Deep Learning:

```powershell
cd D:\projects\deepLearning
.venv\Scripts\python.exe -m streamlit run app.py
```

## Required Model Files

| Project | Required Files |
|---|---|
| KNN | `knn_model.pkl`, `scaler.pkl` |
| SVM | `svm_model.pkl`, `scaler.pkl`, `svm_features.pkl` |
| Random Forest | `random_forest_model.pkl`, `rf_features.pkl` |
| Deep Learning | `deep_learning_model.keras`, `dl_scaler.pkl`, `dl_features.pkl` |

## Evaluation

The notebooks evaluate models using:

- train/test split accuracy
- classification report
- confusion matrix
- cross-validation
- mean cross-validation accuracy
- standard deviation of cross-validation accuracy

Cross-validation was added to:

- `KNN_Project\Untitled.ipynb`
- `SVM_Project\Untitled.ipynb`
- `randomForest\Untitled1 (1).ipynb`

Deep Learning does not use cross-validation in this version because neural network cross-validation is slower and more complex.

Example cross-validation code:

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")

print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())
```

For KNN and SVM, cross-validation is implemented with a `Pipeline` so that `StandardScaler` is fitted separately inside each fold.

## Data Leakage Note

Some columns were removed from the Random Forest and Deep Learning feature sets to avoid data leakage:

- `positive`
- `negative`
- `rating_ratio`
- `ccu`

These columns are related to the target variable, so leaving them in the input features can make the model look unrealistically accurate.

## Important Notes

- Do not place project files inside `.venv`.
- Keep model files in their project folders.
- If notebooks are rerun, saved `.pkl` or `.keras` files may be overwritten. This is normal.
- If you only need cross-validation results, you do not need to delete any saved model files before running notebooks.
