# IA_TP5_Titanic_RNA

Trabajo Práctico 5 — IA Conexionista. Clasificación de sobrevivientes del Titanic mediante Redes Neuronales Artificiales (MLP), comparando tres arquitecturas distintas bajo criterios de eficacia y eficiencia.

**Dataset:** [Titanic — Kaggle](https://www.kaggle.com/c/titanic/data)

---

## 📁 Estructura del Proyecto

```
IA_TP5_TITANIC_RNA/
├── data/
│   └── titanic.csv            # Dataset original
├── src/
│   ├── ia_titanic.ipynb
│   └── mejor_modelo_1.keras
|   └── mejor_modelo_2.keras
|   └── mejor_modelo_3.keras
├── .gitignore
└── README.md
```

### ia_titanic.ipynb

### Celda 1: Configuración del Entorno y Librerías
Se importaron las librerías base para la manipulación de datos (Pandas, NumPy), preparación y escalado (Scikit-Learn), balanceo de clases (Imbalanced-Learn) y el motor de aprendizaje profundo (TensorFlow/Keras).
Objetivo: Preparar el entorno de ejecución y configurar filtros para silenciar advertencias del sistema, garantizando una consola limpia.

### Celda 2: Carga, Limpieza y Preprocesamiento de Datos
Se eliminaron columnas irrelevantes para el modelado (PassengerId, Name, Ticket, Cabin).
Se imputaron los valores faltantes usando la mediana para variables numéricas (Age, Fare) y la moda para categóricas (Embarked).Se convirtieron las variables de texto a variables numéricas binarias mediante codificación One-Hot Encoding (pd.get_dummies).
Se aplicó la técnica SMOTE (Synthetic Minority Over-sampling Technique) para equilibrar la proporción de pasajeros supervivientes y fallecidos.
Objetivo: Transformar los datos brutos en una matriz numérica limpia, normalizada y balanceada apta para la red neuronal.

### Celda 3: División del Conjunto de Datos
Se realizó una partición estratificada de los datos en tres conjuntos independientes: Entrenamiento (70%), Validación (15%) y Prueba (15%) utilizando semillas de aleatoriedad fija (random_state=42).
Objetivo: Aislar los datos con los que la red aprenderá (Train), con los que se calibrará en tiempo real (Val) y con los que se medirá su éxito final en un entorno no visto (Test).

### Celda 4: Modelo 1 - Arquitectura Simple
Se diseñó una red neuronal feed-forward elemental con una única capa oculta de 16 neuronas (ReLU) y una capa de salida binaria (Sigmoid). 
Se entrenó con una tasa de aprendizaje alta ($0.01$) y se incorporó Early Stopping para detener el proceso si el error de validación dejaba de mejorar.
Objetivo: Establecer una línea de base (baseline) para medir el rendimiento mínimo de la red neuronal en la clasificación.

### Celda 5: Modelo 2 - Arquitectura Intermedia
Se incrementó la complejidad agregando una segunda capa oculta (configuración de 32 y 16 neuronas) y se redujo la tasa de aprendizaje a una diez veces menor ($0.001$) para un ajuste de pesos más fino y preciso.
Objetivo: Permitir que la red capture relaciones y patrones más complejos entre las variables del Titanic, mejorando la métrica general de F1-Score.

### Celda 6: Modelo 3 - Arquitectura Compleja con Regularización
Se estructuró una red más profunda (64, 32 y 16 neuronas) combinada con capas de Dropout (0.3 y 0.2). El Dropout apaga neuronas al azar durante el entrenamiento.
Objetivo: Forzar a la red a no depender de conexiones específicas, actuando como una técnica de regularización para combatir activamente el sobreajuste (overfitting).

### Celda 7: Inferencia en Lote (Puesta en Producción)
Se extrajo el modelo entrenado y el escalador vivos en la memoria del cuaderno y se inyectaron los perfiles de dos pasajeros completamente ficticios (uno de perfil de alta probabilidad de supervivencia y otro de baja). Los datos se estructuraron en un DataFrame y se transformaron bajo el mismo pipeline.
Objetivo: Demostrar la viabilidad del modelo en un entorno de producción real (Inferencia), obteniendo predicciones y porcentajes de probabilidad en tiempo real.

### Archivos .keras

Son archivos binarios compactos que almacenan tres cosas:

La arquitectura: El mapa estructural de la red (cuántas capas y cuántas neuronas tiene ese modelo específico).

Los pesos y sesgos: Los valores numéricos exactos de las conexiones que la red estuvo calculando y puliendo durante el entrenamiento.

El estado del optimizador: La configuración exacta en la que se quedó el algoritmo por si en el futuro quisieras retomar el entrenamiento desde donde paró.

---

## 🔧 Pipeline del Proyecto (Metodología)

```
Datos Raw → Limpieza → Codificación → Balanceo (SMOTE) → Partición → Normalización → Entrenamiento → Evaluación
```

## 🚀 Configuración del Entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu_usuario/IA_TP5_Titanic_RNA.git
cd IA_TP5_Titanic_RNA
```

### 2. Crear y activar el entorno virtual

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. Instalar dependencias

%pip install scikit-learn imbalanced-learn pandas numpy

%pip install tensorflow

---


## 👥 Equipo - Marder, Teves, Lezcano Airaldi

Trabajo Práctico 5 — Inteligencia Artificial 
Materia: Inteligencia Artificial | Carrera: Licenciatura en Sistemas
