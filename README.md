# IA_TP5_Titanic_RNA

Trabajo Práctico 5 — IA Conexionista. Clasificación de sobrevivientes del Titanic mediante Redes Neuronales Artificiales (MLP), comparando tres arquitecturas distintas bajo criterios de eficacia y eficiencia.

**Dataset:** [Titanic — Kaggle](https://www.kaggle.com/c/titanic/data)

---

## 📁 Estructura del Proyecto

```
IA_TP5_TITANIC_RNA/
├── data/
│   └── titanic.csv            # Dataset original
├── notebooks/                 # Notebooks de exploración
├── src/
│   ├── preprocesamiento.py    # Limpieza, codificación, balanceo y partición
│   └── entrenamiento.py       # Definición, entrenamiento y comparación de modelos
├── .gitignore
└── README.md
```

---

## 🔧 Pipeline del Proyecto (Metodología)

```
Datos Raw → Limpieza → Codificación → Balanceo (SMOTE) → Partición → Normalización → Entrenamiento → Evaluación
```

### `preprocesamiento.py`

Prepara el dataset para que sea consumido por la red neuronal. Ejecuta las siguientes etapas en orden:

**1. Carga**
Lee el archivo `data/titanic.csv` con pandas.

**2. Limpieza**
Elimina columnas que no aportan información útil al modelo: `PassengerId`, `Name`, `Ticket` y `Cabin`. Imputa valores nulos de `Age` con la mediana y de `Embarked` con la moda.

**3. Codificación**
Convierte variables categóricas a numéricas usando `LabelEncoder`:
- `Sex` → `0` (female) / `1` (male)
- `Embarked` → valores numéricos por puerto

**4. Balanceo**
Aplica **SMOTE** (Over-sampling) para equilibrar las clases `Survived = 0` y `Survived = 1`, evitando sesgos en el aprendizaje de la red.

**5. Partición y Normalización**
Divide el dataset en **Train (80%) / Test (20%)** con `train_test_split`. Escala las features con `StandardScaler` ajustado solo sobre el set de entrenamiento y aplicado al de test, para evitar data leakage.

Al finalizar, exporta `X_train`, `X_test`, `y_train` e `y_test` listos para ser importados por `entrenamiento.py`.

---

### `entrenamiento.py`

Importa los datos procesados de `preprocesamiento.py` y entrena tres arquitecturas distintas de `MLPClassifier` (scikit-learn). Para cada modelo registra el tiempo de ejecución y la precisión (Accuracy) sobre el set de test.

Los hiperparámetros variados entre modelos son:

| Hiperparámetro         | Descripción |
|------------------------|-------------|
| `hidden_layer_sizes`   | Número de capas ocultas y neuronas por capa |
| `activation`           | Función de activación (`logistic` / `relu`) |
| `learning_rate_init`   | Tasa de aprendizaje inicial (λ) |
| `max_iter`             | Épocas de entrenamiento (fijado en 500) |

**Modelos entrenados:**

- **RNA Simple** — 1 capa oculta `(8,)`, activación `logistic`, lr `0.01`
- **RNA Profunda** — 2 capas ocultas `(16, 8)`, activación `relu`, lr `0.001`
- **RNA Compleja** — 3 capas ocultas `(32, 16, 8)`, activación `relu`, lr `0.0001`

Al finalizar imprime una tabla resumen comparando Accuracy y Tiempo de ejecución de los tres modelos.

---

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

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---

## ▶️ Ejecución

Los scripts deben ejecutarse desde la raíz del proyecto para que las rutas relativas funcionen correctamente.

```bash
# El entrenamiento importa preprocesamiento automáticamente, basta con correr:
python src/entrenamiento.py
```

---

## 👥 Equipo

Trabajo Práctico 5 — Inteligencia Artificial  
Materia: Inteligencia Artificial | Carrera: Licenciatura en Sistemas
