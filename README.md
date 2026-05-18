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
│   ├── ia_titanic.py    
│   └── mejor_modelo_1.keras
|   └── mejor_modelo_2.keras
|   └── mejor_modelo_3.keras
├── .gitignore
└── README.md
```

### ia_titanic.py

Limpieza Automática: El script elimina lo que no aporta patrones (Name, Ticket, etc.), rellena los vacíos con estadísticas seguras (medianas) y escala los datos numéricos para que características grandes como Fare (tarifa) no eclipsen a Age.

SMOTE Activo: Verás en la consola cómo las clases se igualan perfectamente antes de la separación de los conjuntos.

El checkpoint: El entrenamiento generará un archivo .keras. Ese es el "intervalo de iteración" donde guardamos el estado exacto del modelo cada vez que el error de validación baja.

### Variacion del modelo 2 y 3 con respecto al modelo 1
Variación en Capas y Neuronas: Pasamos de un modelo básico (1 capa / 16 neuronas) a uno intermedio (2 capas / 32 y 16 neuronas) y finalmente a uno complejo (3 capas / 64, 32 y 16 neuronas).

Tasa de Aprendizaje: En el Modelo 1 usamos 0.01 (ajustes más rápidos pero bruscos). En el 2 y el 3 usamos 0.001 para que los pesos se ajusten con mayor delicadeza en el espacio matemático.

Inclusión de Dropout (Modelo 3): Es una técnica de posprocesamiento/procesamiento avanzada. Apaga aleatoriamente un porcentaje de neuronas en cada iteración de entrenamiento, obligando a la red a no depender de una sola combinación de datos y destruyendo cualquier intento de memorización.

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

```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

---


## 👥 Equipo - Marder, Teves, Lezcano Airaldi

Trabajo Práctico 5 — Inteligencia Artificial 
Materia: Inteligencia Artificial | Carrera: Licenciatura en Sistemas
