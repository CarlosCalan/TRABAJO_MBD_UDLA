# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:56:15 2024

@author: ccalan
"""

#%%% IMPORTAMOS LAS LIBRERIAS NECESARIAS
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

#%%% EXPLORATORY DATA ANALYSIS
#CARGAMOS LA BASE DE DATOS DESDE LA RUTA DONDE SE ALMACENA
df=pd.read_csv(r"C:\\OneD\\OneDrive - Farmaenlace\\Escritorio\\healthcare-dataset-stroke-data.csv")
#MOSTRAMOS LA FORMA DEL DATAFRAME
df.head()
#MOSTRAMOS LOS TIPOS DE DATOS 
df.dtypes
print(df.shape)
df.id.nunique() 
#VERIFICAMOS NULOS
df.isnull().sum()
#MOSTRAMOS LAS MÉTRICAS MÁS RELEVANTES
df.describe()
# Conteo de frecuencia por valores únicos.
print(df.stroke.value_counts())
# En porcentaje de acuerdo a la totalidad de registros, ES UN PROBLEMA DE CLASES DESBALANCEADAS
print(df.stroke.value_counts(normalize=True))

#ELIMINAMOS DATOS ATIPICOS, UN BMI SUPERIOR A 70 ES IMPOSIBLE EN HUMANOS
df=df[df['bmi']<=70]
df['bmi'] = df['bmi'].fillna(round (df['bmi'].median(), 2))
#VERIFICAMOS LOS NULOS
df.isnull().sum()
#CAMBIAMOS EL TIPO DE DATO DE CIERTAS COLUMNAS
df['id'] = df['id'].astype(str)
df['hypertension'] = df['hypertension'].astype(str)
df['heart_disease'] = df['heart_disease'].astype(str)
df['stroke'] = df['stroke'].astype(str)
df = df.drop(columns=['id'])
df.describe(include='all')# LOS VALORES DESCRIPTIVOS ANTES Y DESPUES DEL TRATAMIENTO NO AFECTAN EN LAS 
#MÉTRICAS DEL DATAFRAME
# SEPARAMOS VARIABLES CUALITATIVAS Y CUANTITATIVAS
var_cuantitativas = df.select_dtypes('number').columns
var_cualitativas  =df.select_dtypes('object').columns


#REALIZAMOS LOS BOXPLOTS DE LAS VARIABLES CONTINUAS PARA REVISAR COMO ESTÁN SUS VALORES ATIPICOS
#A PESAR QUE EXISTEN VALORES POR ENCIMA DEL BIGOTE SUPERIOR NO LOS ELIMINAMOS AL SER VALORES 
#QUE SI SON POSIBLES EN UN SER HUMANO.

fig, axs = plt.subplots(3, 1, figsize=(15, 10))

# Graficar boxplots y añadir títulos y etiquetas
sns.boxplot(x=df['age'], ax=axs[0])
axs[0].set_title('Distribución de Edad')
axs[0].set_xlabel('Edad')
axs[0].set_ylabel('Valores')

sns.boxplot(x=df['avg_glucose_level'], ax=axs[1])
axs[1].set_title('Distribución del Nivel Promedio de Glucosa')
axs[1].set_xlabel('Nivel Promedio de Glucosa')
axs[1].set_ylabel('Valores')

sns.boxplot(x=df['bmi'], ax=axs[2])
axs[2].set_title('Distribución de BMI')
axs[2].set_xlabel('BMI')
axs[2].set_ylabel('Valores')

# Ajustar diseño para evitar solapamientos
plt.tight_layout()

# Mostrar gráfico
plt.show()

#GRAFICAMOS LAS DISTRIBUCIONES DE LAS VARIABLES CUANTITATIVAS
# Crear subplots
fig, ax = plt.subplots(1, 3, figsize=(10, 5))

# Graficar cada columna cuantitativa en un subplot
for variable, subplot in zip(var_cuantitativas, ax.flatten()):
    sns.kdeplot(df[variable], ax=subplot, shade=True)
    subplot.set_title(f'Distribución de {variable}')

# Ajustar el diseño para evitar solapamientos
plt.tight_layout()
plt.show()

#GRAFICAMOS LA DISTRIBUCION DE LAS VARIABLES CUALITATIVAS
# Crear subplots
fig, ax = plt.subplots(2, 4, figsize=(15, 10))

# Graficar cada columna categórica en un subplot
for variable, subplot in zip(var_cualitativas, ax.flatten()):
    sns.countplot(x=df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
    subplot.set_title(variable)

# Ajustar el diseño para evitar solapamientos
plt.tight_layout()

# Mostrar la figura completa
plt.show()

#VEMOS LA VARIABLE EDAD RESPECTO A LA VARIABLE OBJETIVO
#SE PUEDE DECIR QUE LA EDAD SI AFECTA A LA VARIABLE OBJETIVO
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))

# Primer gráfico: Distribución total de edades
sns.kdeplot(data=df, x='age', fill=True, alpha=1, ax=ax1)
ax1.set_title('Distribución de Edades')

# Segundo gráfico: Distribuciones por grupo de accidente cerebrovascular
sns.kdeplot(data=df[df['stroke'] == '0'], x='age', fill=True, alpha=1, ax=ax2, label='Sin ACV')
sns.kdeplot(data=df[df['stroke'] == '1'], x='age', fill=True, alpha=0.8, ax=ax2, label='Con ACV')
ax2.set_title('Distribución de Edades por ACV')
ax2.legend()

# Ajustes adicionales
plt.tight_layout()
plt.show()

# GRAFICAMOS CADA VARIABLE CUALITATIVA RESPECTO A LA VARIABLE OBJETIVO
fig, ax = plt.subplots(2, 4, figsize=(15, 10))

for variable, subplot in zip(var_cualitativas, ax.flatten()):
    sns.countplot(x=df[variable], ax=subplot)
    for label in subplot.get_xticklabels():
        label.set_rotation(90)
    subplot.set_title(variable)
    
    # Añadir anotaciones a cada barra
    for p in subplot.patches:
        height = int(p.get_height())
        subplot.annotate(f'{height}', 
                         xy=(p.get_x() + p.get_width() / 2., height), 
                         xytext=(0, 5),  # 5 points vertical offset
                         textcoords='offset points', 
                         ha='center', va='center')

# Ajustar el diseño para evitar solapamientos
plt.tight_layout()

# Mostrar la figura completa
plt.show()
#REVISAMOS EL GRADO DE CORRELACION ENTRE LAS VARIABLES
## Gráfico de calor para las correlaciones
sns.set(font_scale=2)
corr_matrix = df[var_cuantitativas].corr()
plt.figure(figsize=(8, 4))
ax = sns.heatmap(corr_matrix,annot=True, fmt=".1f",cmap="YlGnBu") 
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))


#%%% 
#REALIZAMOS LA CODIFICACION DE LAS VARIABLES
labelencoder = LabelEncoder()
df[var_cualitativas]=df[var_cualitativas].apply(LabelEncoder().fit_transform)
#SEPARAMOS FEATURES EN X Y OBJETIVO EN Y
X = df.iloc[:, 0:-1].values
Y = df.iloc[:, -1].values
#SEPARAMOS EN TRAIN Y TEST CON PROPORCION DE 80 Y 20 
X_train, x_test, Y_train, y_test = train_test_split(X, Y, test_size= 0.2, random_state= 0)
#APLICAMOS LA TÉCNICA DE OVERSAMPLING DEBIDO A QUE ES UN PROBLEMA DE CLASES DESBALANCEADAS
print("Antes OverSampling, numero de registros con '1': {}".format(sum(Y_train==1)))
print("Antes OverSampling, numero de registros con '0': {} \n".format(sum(Y_train==0)))
sm = SMOTE(random_state=1)
X_train_res, Y_train_res = sm.fit_resample(X_train, Y_train.ravel())
print('Después del OverSampling, el tamaño de train_X: {}'.format(X_train_res.shape))
print('Después del OverSampling, el tamaño de train_y: {} \n'.format(Y_train_res.shape))
print("Después del OverSampling, numero de registros con '1': {}".format(sum(Y_train_res==1)))
print("Después del OverSampling, numero de registros con '0': {}".format(sum(Y_train_res==0)))
#%%%
#ELEGIMOS LOS DOS MODELOS QUE VAMOS A TRABAJAR
models = [
    ['Logistic Regression', LogisticRegression(random_state=2)],
    ['Random Forest', RandomForestClassifier(random_state=2)],
    ]

results = []

#ENTRENAMOS CADA MODELO CON EL TRAIN Y OBTENEMOS LAS MÉTRICAS CON EL TEST
#LOS RESULTADOS SON DE LOS MODELOS SIN NIGÚN AJUSTE DE HIPERPARÁMETROS
for name, model in models:
    model.fit(X_train_res, Y_train_res)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracies = cross_val_score(estimator=model, X=X_train_res, y=Y_train_res, cv=10)
    roc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append([name, accuracy_score(y_test, y_pred) * 100, roc, precision, recall, f1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {model}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')

results= pd.DataFrame(results, columns=['Model', 'Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1'])
#%%%CREAMOS UN MODELO DE REDES NEURONALES EN SU FORMA MÁS SIMPLE
# Crear el modelo de perceptrón
perceptron = Perceptron()
# Entrenar el perceptrón con los datos balanceados
perceptron.fit(X_train_res, Y_train_res)
# Predecir en el conjunto de prueba
Y_pred = perceptron.predict(x_test)
# Calcular métricas
accuracy = accuracy_score(y_test, Y_pred)
roc_auc = roc_auc_score(y_test, Y_pred)  
precision = precision_score(y_test, Y_pred)
recall = recall_score(y_test, Y_pred)
f1 = f1_score(y_test, Y_pred)
print(f'Exactitud (Accuracy): {accuracy}')
print(f'ROC AUC: {roc_auc}')
print(f'Precisión (Precision): {precision}')
print(f'Recall: {recall}')
print(f'F1-score: {f1}')
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1-score'],
    'Value': [accuracy, roc_auc, precision, recall, f1]
})

conf_matrix = confusion_matrix(y_test, Y_pred)
# Crear DataFrame para la matriz de confusión
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
print("Matriz de Confusión:")
print(conf_matrix_df)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Red Neuronal')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

#%%%
#BUSCAMOS LOS MEJORES HIPERPARAMETROS PARA CADA MODELO
# Definir los modelos y sus respectivos parámetros para Grid Search
# Vamos a buscar los parametros que maximizen la métrica Recall
df = pd.DataFrame(columns=['Model', 'Best Recall'])

# Lista de solvers a probar
solvers = ['liblinear', 'lbfgs', 'sag', 'saga']
grid_models = [
    (LogisticRegression(),
     [{'C': [0.25, 0.5, 0.75, 1],
       'random_state': [2],
       'solver': solvers}]),
    (RandomForestClassifier(),
     [{'n_estimators': [100, 150, 200],
       'criterion': ['gini', 'entropy'],
       'random_state': [2]}]),
]

# Realizar el Grid Search para cada modelo
for model, param_grid in grid_models:
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='recall', cv=10)
    grid.fit(X_train_res, Y_train_res)
    
    # Obtener los resultados
    best_recall = grid.best_score_
    best_params = grid.best_params_
    
    # Imprimir los resultados
    print('{}:\nBest Recall: {:.4f}'.format(model.__class__.__name__, best_recall))
    print('Best Parameters:', best_params)
    print('')

    # Actualizar el DataFrame con los resultados
    if model.__class__.__name__ in df['Model'].values:
        df.loc[df['Model'] == model.__class__.__name__, 'Best Recall'] = best_recall
    else:
        new_row = pd.DataFrame({'Model': [model.__class__.__name__], 'Best Recall': [best_recall]})
        df = pd.concat([df, new_row], ignore_index=True)

    print('----------------')
    print('')

# Mostrar el DataFrame actualizado con los mejores valores de Recall para cada modelo
print('DataFrame actualizado con mejores valores de Recall para cada modelo:')
print(df)

#%%%
# Definir los mejores hiperparámetros encontrados
best_params_logreg = {'C': 0.5, 'random_state': 2, 'solver': 'liblinear'}#MEJOR RECALL
#best_params_logreg = {'C': 0.5, 'random_state': 2, 'solver': 'newton-cg'}
#best_params_logreg = {'C': 0.5, 'random_state': 2, 'solver': 'lbfgs'}#MEJOR PRECISION

best_params_rf = {'criterion': 'entropy', 'n_estimators': 200, 'random_state': 2}
#best_params_rf = {'criterion': 'gini', 'n_estimators': 150, 'random_state': 2}

# Elegir los modelos con los mejores hiperparámetros
models = [
    ['Logistic Regression', LogisticRegression(**best_params_logreg)],
    ['Random Forest', RandomForestClassifier(**best_params_rf)]
]

results = []

# Entrenar cada modelo con el train y obtener las métricas con el test
# Los resultados son de los modelos ajustados con los mejores hiperparámetros
for name, model in models:
    model.fit(X_train_res, Y_train_res)
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(x_test)
    
    cm = confusion_matrix(y_test, y_pred)
    accuracies = cross_val_score(estimator=model, X=X_train_res, y=Y_train_res, cv=10)
    roc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append([name, accuracy_score(y_test, y_pred) * 100, roc, precision, recall, f1])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {name}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')


# Convertir los resultados en un DataFrame
results = pd.DataFrame(results, columns=['Model', 'Accuracy', 'ROC AUC', 'Precision', 'Recall', 'F1'])

# Mostrar el DataFrame con los resultados
print(results)

#%%%RED NEURONAL MÁS SOFISTICADA

# Preprocesamiento de datos: Estandarizar 
scaler = StandardScaler()
X_train_res1 = scaler.fit_transform(X_train_res)
x_test1 = scaler.transform(x_test)

# Construir el modelo de la red neuronal
model = Sequential()
model.add(Dense(32, input_shape=(X_train_res1.shape[1],), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))  # Supone una clasificación binaria

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['recall'])

# Entrenar el modelo
model.fit(X_train_res1, Y_train_res, epochs=50, batch_size=32, validation_data=(x_test1, y_test))

# Evaluar el modelo
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# Predecir etiquetas para los datos de prueba
Y_pred_prob = model.predict(x_test1)
Y_pred = (Y_pred_prob > 0.5).astype(int).ravel()

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(y_test, Y_pred)
print(conf_matrix)

plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Red Neuronal')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()
#%%%
# EL MEJOR MODELO ES LA REGRESION LINEAL, CON HIPERPARAMETROS PROCEDEMOS A SABER CUALES 
# SON LOS FEATURES QUE MÁS APORTAN AL MODELO.
import statsmodels.api as sm
feature_names = df.columns[:-1]  # Excluir la columna objetivo
X_train_res_df = pd.DataFrame(X_train, columns=feature_names)
X_train_res = sm.add_constant(X_train_res)
# Ajustar el modelo de regresión logística con statsmodels
logit_model = sm.Logit(Y_train, X_train_res_df)
result = logit_model.fit()
# Imprimir el resumen del modelo
print(result.summary())
