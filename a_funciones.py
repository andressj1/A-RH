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
from sklearn.metrics import confusion_matrix #### Matriz de confusion 
import seaborn as sns #####Graficos



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
                 title=f'Distribución de {color_col} por {x_col}',
                 labels={'Category': 'Categoría', 'Count': 'Cantidad de Empleados', 'Attrition': 'Attrition'})
    fig.update_xaxes(tickvals=[1, 2, 3, 4, 5, 6])
    return fig

#### Grafico para categoricas; str

def vovsstr(df, x_col, y_col, color_col):
    # Calcular el conteo de valores
    counts = df.groupby([x_col, color_col]).size().reset_index(name=y_col)
    
    # Ordenar los datos para que la categoría con menos valores aparezca al final
    counts = counts.sort_values(by=y_col)
    
    # Crear el gráfico de barras
    fig = px.bar(counts, x=y_col, y=x_col, color=color_col, 
                 title=f'Distribución de {color_col} por {x_col}',
                 labels={'Category': 'Categoría', 'Count': 'Cantidad de Empleados', 'Attrition': 'Attrition'},
                 orientation='h',
                 color_discrete_map={'rf': 'red', 'rg': 'green', 'rh': 'blue'})  # Cambiar los colores aquí
    return fig

#### Imputar

def imputar_f (df,list_cat):  
        
    
    df_c=df[list_cat]

    df_n=df.loc[:,~df.columns.isin(list_cat)]

    imputer_n=SimpleImputer(strategy='median')
    imputer_c=SimpleImputer( strategy='most_frequent')

    imputer_n.fit(df_n)
    imputer_c.fit(df_c)
    imputer_c.get_params()
    imputer_n.get_params()

    X_n=imputer_n.transform(df_n)
    X_c=imputer_c.transform(df_c)


    df_n=pd.DataFrame(X_n,columns=df_n.columns)
    df_c=pd.DataFrame(X_c,columns=df_c.columns)
    df_c.info()
    df_n.info()

    df =pd.concat([df_n,df_c],axis=1)
    return df

##### Seleccion de variables 

def sel_variables(modelos, X, y, threshold):
    var_names_ac = np.array([])
    for modelo in modelos:
        modelo.fit(X, y)
        sel = SelectFromModel(modelo, threshold=threshold, prefit=True)
        var_names = X.columns[sel.get_support()]
        var_names_ac = np.append(var_names_ac, var_names)
    
    var_names_ac = np.unique(var_names_ac)  # Movido fuera del bucle para conservar todas las variables seleccionadas
    
    return var_names_ac

###### Medir modelos 

def medir_modelos(modelos,scoring,X,y,cv):

    metric_modelos=pd.DataFrame()
    for modelo in modelos:
        scores=cross_val_score(modelo,X,y, scoring=scoring, cv=cv )
        pdscores=pd.DataFrame(scores)
        metric_modelos=pd.concat([metric_modelos,pdscores],axis=1)
    
    metric_modelos.columns=["LogisticRegression","RandomForestClassifier","DecisionTreeClassifier","GradientBoostingClassifier"]
    return metric_modelos


##### Matriz de confusion 

def plot_confusion_matrix(y_true, y_pred):
    # Crear la matriz de confusión
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Crear un mapa de calor con seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

