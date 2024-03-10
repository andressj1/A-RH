###librerias
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler 
import plotly.express as px
import matplotlib.pyplot as plt  # gráficos


def plot_attrition_counts(df, colors=['blue', 'orange']):
    attrition_counts = df['Attrition'].value_counts()

    # Crear el gráfico de barras horizontal
    plt.figure(figsize=(7, 3))
    attrition_counts.plot(kind='barh', color=colors)
    plt.xlabel('Cantidad de empleados')
    plt.ylabel('Continuidad')
    plt.title('Comportamiento')

    # Mostrar el gráfico centrado en la pantalla
    plt.tight_layout()
    plt.show()
