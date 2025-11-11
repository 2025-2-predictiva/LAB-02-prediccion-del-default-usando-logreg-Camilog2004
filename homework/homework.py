# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score, balanced_accuracy_score,
    recall_score, f1_score, confusion_matrix
)


import os


train_path = "files/input/train_data.csv.zip"
test_path = "files/input/test_data.csv.zip"

df_train = pd.read_csv(train_path, compression="zip")
df_test = pd.read_csv(test_path, compression="zip")


def limpiar(df):
    df = df.copy()

    
    df.rename(columns={"default payment next month": "default"}, inplace=True)

    df.drop(columns=["ID"], inplace=True)

    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: 4 if v > 4 else v)

    df["EDUCATION"] = df["EDUCATION"].apply(lambda v: v if v > 0 else np.nan)
    df["MARRIAGE"] = df["MARRIAGE"].apply(lambda v: v if v > 0 else np.nan)

    df.dropna(inplace=True)

    return df


df_train = limpiar(df_train)
df_test = limpiar(df_test)


X_train = df_train.drop(columns=["default"])
y_train = df_train["default"]

X_test = df_test.drop(columns=["default"])
y_test = df_test["default"]


cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

transformador = ColumnTransformer(
    transformers=[
        ("categoricos", OneHotEncoder(), cat_cols),
        ("numericos", MinMaxScaler(), num_cols)
    ],
    remainder="drop"
)

selector = SelectKBest(score_func=f_classif)
modelo_base = LogisticRegression(max_iter=500)

pipeline = Pipeline(steps=[
    ("pre", transformador),
    ("select", selector),
    ("clf", modelo_base)
])

parametros = {
    "select__k": range(1, len(X_train.columns) + 1)
}

busqueda = GridSearchCV(
    pipeline,
    parametros,
    scoring="balanced_accuracy",
    cv=10,
    n_jobs=-1
)

busqueda.fit(X_train, y_train)

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(busqueda, f)

def obtener_metricas(y_true, y_pred, tipo):
    return {
        "type": "metrics",
        "dataset": tipo,
        "precision": float(precision_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred)),
    }


pred_train = busqueda.predict(X_train)
pred_test = busqueda.predict(X_test)

metricas_train = obtener_metricas(y_train, pred_train, "train")
metricas_test = obtener_metricas(y_test, pred_test, "test")

def cm_a_dict(cm, tipo):
    return {
        "type": "cm_matrix",
        "dataset": tipo,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1])
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1])
        }
    }


cm_train = confusion_matrix(y_train, pred_train)
cm_test = confusion_matrix(y_test, pred_test)

cm_train_dict = cm_a_dict(cm_train, "train")
cm_test_dict = cm_a_dict(cm_test, "test")

os.makedirs("files/output", exist_ok=True)
output_file = "files/output/metrics.json"

with open(output_file, "w", encoding="utf-8") as fw:
    fw.write(json.dumps(metricas_train) + "\n")
    fw.write(json.dumps(metricas_test) + "\n")
    fw.write(json.dumps(cm_train_dict) + "\n")
    fw.write(json.dumps(cm_test_dict) + "\n")


