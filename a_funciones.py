###librerias
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
import joblib
from sklearn.preprocessing import StandardScaler 
import plotly.express as px


#######Grafico de torta 

def plot_pie_chart(data, values_col, names_col):
    fig = px.pie(data, values=values_col, names=names_col)
    fig.show()

    