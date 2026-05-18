import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# =====================================================================
# PREPROCESAMIENTO Y TRATAMIENTO DE DATOS
# =====================================================================

df = pd.read_csv('../data/titanic.csv')

# Eliminación de columnas no representativas para la RNA
columnas_a_eliminar = ['PassengerId', 'Name', 'Ticket', 'Cabin']
df = df.drop(columns=columnas_a_eliminar)

# Tratamiento de valores nulos 
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Codificación de variables categóricas
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Convertir booleanos resultantes a enteros (0 y 1) para la RNA
df = df.astype({col: 'int32' for col in df.select_dtypes(include='bool').columns})

# Separar características (X) y etiqueta objetivo (y)
X = df.drop(columns=['Survived'])
y = df['Survived']

# Escalado de características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Balanceo de datos mediante Over-sampling
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

print(f"Distribución original de clases: {np.bincount(y)}")
print(f"Distribución tras SMOTE: {np.bincount(y_resampled)}")

# =====================================================================
# PROCESAMIENTO 
# =====================================================================
print("\n--- Division el conjunto de datos ---")

# Dividir en Entrenamiento (70%) y una porción Temporal (30%)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled, test_size=0.30, random_state=42
)

# Dividir la porción temporal en Validación (15%) y Prueba (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Registros de Entrenamiento: {X_train.shape[0]}")
print(f"Registros de Validación: {X_val.shape[0]}")
print(f"Registros de Prueba: {X_test.shape[0]}")

# =====================================================================
# DISEÑO Y ENTRENAMIENTO: MODELO 1 
# =====================================================================
print("\n--- Entrenando Modelo 1: Arquitectura Simple ---")

# Definicion de la arquitectura
model_1 = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),  # Capa oculta simple
    Dense(1, activation='sigmoid')                                  # Capa de salida binaria
])

# Configuracion  del optimizador con la Tasa de Aprendizaje (Learning Rate)
learning_rate_1 = 0.01
optimizador_1 = Adam(learning_rate=learning_rate_1)

model_1.compile(
    optimizer=optimizador_1,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Callbacks: Intervalo de iteraciones y Detención Temprana
callbacks_model_1 = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_1.keras', save_best_only=True, monitor='val_loss')
]

# Ejecución del entrenamiento medido en tiempo
inicio_tiempo = time.time()

historia_1 = model_1.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,              # Número máximo de épocas
    batch_size=32,
    callbacks=callbacks_model_1,
    verbose=1                # Muestra el progreso por cada época
)

tiempo_ejecucion_1 = time.time() - inicio_tiempo
print(f"Tiempo de entrenamiento de Modelo 1: {tiempo_ejecucion_1:.2f} segundos")

# =====================================================================
# POSPROCESAMIENTO (EVALUACIÓN DEL MODELO 1)
# =====================================================================
print("\n--- Evaluando Modelo 1 en datos de Prueba (Test) ---")

# Medición del tiempo de inferencia/predicción
inicio_prediccion = time.time()
predicciones_prob = model_1.predict(X_test)
tiempo_inferencia_1 = time.time() - inicio_prediccion

# Convertir probabilidades en clases (0 o 1) usando umbral de 0.5
predicciones = (predicciones_prob > 0.5).astype(int)

# Aplicar métricas de desempeño
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, predicciones))

print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones))
print(f"Tiempo de Inferencia: {tiempo_inferencia_1:.4f} segundos")

# =====================================================================
# DISEÑO Y ENTRENAMIENTO: MODELO 2 (Arquitectura Intermedia)
# =====================================================================
print("\n" + "="*50)
print("--- Entrenando Modelo 2: Arquitectura Intermedia ---")
print("="*50)

# Definir la arquitectura (Más capas y neuronas, tasa de aprendizaje menor)
model_2 = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)), # Capa 1
    Dense(16, activation='relu'),                                  # Capa 2
    Dense(1, activation='sigmoid')                                 # Salida
])

# Configuración (Bajamos la tasa de aprendizaje a 0.001 para ser más precisos)
learning_rate_2 = 0.001
model_2.compile(
    optimizer=Adam(learning_rate=learning_rate_2),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks_model_2 = [
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_2.keras', save_best_only=True, monitor='val_loss')
]

# Entrenamiento
inicio_tiempo = time.time()
historia_2 = model_2.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=120,
    batch_size=32,
    callbacks=callbacks_model_2,
    verbose=1
)
tiempo_ejecucion_2 = time.time() - inicio_tiempo
print(f"Tiempo de entrenamiento de Modelo 2: {tiempo_ejecucion_2:.2f} segundos")

# Evaluación Modelo 2
inicio_prediccion = time.time()
predicciones_prob_2 = model_2.predict(X_test)
tiempo_inferencia_2 = time.time() - inicio_prediccion
predicciones_2 = (predicciones_prob_2 > 0.5).astype(int)

print("\n[MÉTRICAS MODELO 2]")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, predicciones_2))
print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones_2))


# =====================================================================
# DISEÑO Y ENTRENAMIENTO: MODELO 3 (Complejo con Regularización Dropout)
# =====================================================================
print("\n" + "="*50)
print("--- Entrenando Modelo 3: Complejo con Regularización ---")
print("="*50)

# Definir la arquitectura (Red profunda. Añadimos Dropout para apagar neuronas al azar)
model_3 = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),                                                  # Evita que las neuronas co-dependan
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Configuración
learning_rate_3 = 0.001
model_3.compile(
    optimizer=Adam(learning_rate=learning_rate_3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks_model_3 = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('mejor_modelo_3.keras', save_best_only=True, monitor='val_loss')
]

# Entrenamiento
inicio_tiempo = time.time()
historia_3 = model_3.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=32,
    callbacks=callbacks_model_3,
    verbose=1
)
tiempo_ejecucion_3 = time.time() - inicio_tiempo
print(f"Tiempo de entrenamiento de Modelo 3: {tiempo_ejecucion_3:.2f} segundos")

# Evaluación Modelo 3
inicio_prediccion = time.time()
predicciones_prob_3 = model_3.predict(X_test)
tiempo_inferencia_3 = time.time() - inicio_prediccion
predicciones_3 = (predicciones_prob_3 > 0.5).astype(int)

print("\n[MÉTRICAS MODELO 3]")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, predicciones_3))
print("\nReporte de Clasificación:")
print(classification_report(y_test, predicciones_3))