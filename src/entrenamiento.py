import pandas as pd
import time
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from preprocesamiento import X_train, X_test, y_train, y_test

def entrenar_modelo(nombre, hidden_layers, activation='relu', lr=0.001):
    print(f"\n--- Entrenando {nombre} ---")
    
    # Configuramos la RNA según los hiperparámetros de la consigna
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation=activation,
        learning_rate_init=lr,
        max_iter=500, # Épocas
        random_state=42,
        verbose=False # Poner en True si querés ver el error por cada iteración
    )
    
    inicio = time.time()
    mlp.fit(X_train, y_train)
    fin = time.time()
    
    tiempo_ejecucion = fin - inicio
    predicciones = mlp.predict(X_test)
    precision = accuracy_score(y_test, predicciones)
    
    print(f"Tiempo: {tiempo_ejecucion:.4f}s")
    print(f"Precisión (Accuracy): {precision:.4f}")
    return {"Modelo": nombre, "Accuracy": precision, "Tiempo": tiempo_ejecucion}

# ---------------------------------------------------------
# DEFINICIÓN DE LOS 3 MODELOS (Diseño de experimentos)
# ---------------------------------------------------------

# Modelo 1: Básico (1 capa oculta pequeña)
m1 = entrenar_modelo("RNA Simple", (8,), activation='logistic', lr=0.01)

# Modelo 2: Profundo (2 capas ocultas)
m2 = entrenar_modelo("RNA Profunda", (16, 8), activation='relu', lr=0.001)

# Modelo 3: Grande (Más neuronas, menor tasa de aprendizaje)
m3 = entrenar_modelo("RNA Compleja", (32, 16, 8), activation='relu', lr=0.0001)

# Resumen para la tabla del informe
resultados = pd.DataFrame([m1, m2, m3])
print("\n--- TABLA RESUMEN DE RESULTADOS ---")
print(resultados)