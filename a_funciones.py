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


#### Grafico para numericas vs variable objetivo 
def vovsnum(df, x_col, y_col):
    fig = px.box(df, x=x_col, y=y_col, 
                 title=f'Distribución de {y_col} por {x_col}',
                 points='outliers',
                 labels={x_col: x_col, y_col: y_col})
    return fig

##### Grafico para categoricas; encuestas

def vovsenc(df, x_col, y_col, color_col):
    # Calcular el conteo de valores
    counts = df.groupby([x_col, color_col]).size().reset_index(name=y_col)
    
    # Crear el gráfico de barras
    fig = px.bar(counts, x=x_col, y=y_col, color=color_col, 
                 title=f'Distribución de {y_col} por {x_col}',
                 labels={'Category': 'Categoría', 'Count': 'Cantidad de Empleados', 'Attrition': 'Attrition'})
    fig.update_xaxes(tickvals=[1, 2, 3, 4, 5, 6])
    return fig
