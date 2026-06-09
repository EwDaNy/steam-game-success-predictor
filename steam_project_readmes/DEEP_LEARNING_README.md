# Steam Game Success Prediction — Deep Learning

## Project Description

This project predicts Steam game success using a neural network built with TensorFlow/Keras.

## Main Files

```text
app.py
steam.csv
deep_learning_model.keras
dl_scaler.pkl
dl_features.pkl
```

## Correct Project Structure

```text
D:\projects\deepLearning
│ app.py
│ steam.csv
│ deep_learning_model.keras
│ dl_scaler.pkl
│ dl_features.pkl
│ notebook.ipynb
│
└── .venv
```

Do not put project files inside `.venv`.

## Activate Environment

```bash
cd D:\projects\deepLearning
.venv\Scripts\activate
```

## Check Python

```bash
python -c "import sys; print(sys.executable)"
```

It should show:

```text
D:\projects\deepLearning\.venv\Scripts\python.exe
```

## Install Libraries

```bash
pip install pandas numpy scikit-learn joblib matplotlib streamlit tensorflow
```

## Save Model Files

In the notebook, run:

```python
model.save(r"D:\projects\deepLearning\deep_learning_model.keras")
joblib.dump(scaler, r"D:\projects\deepLearning\dl_scaler.pkl")
joblib.dump(list(X.columns), r"D:\projects\deepLearning\dl_features.pkl")
```

## Run App

```bash
python -m streamlit run app.py
```

## Notes

The model achieved test accuracy around `0.763`.

Very large unrealistic input values may produce strange predictions because the model only understands values similar to the training data.
