# IA_TP5_Titanic_RNA
Proyecto de IA Conexionista para la clasificación de sobrevivientes del Titanic utilizando Redes Neuronales Artificiales (RNA).

## 🚀 Configuración del Entorno

### 1. Clonar el repositorio
```bash
git clone [https://github.com/tu_usuario/IA_TP5_Titanic_RNA.git](https://github.com/tu_usuario/IA_TP5_Titanic_RNA.git)
cd IA_TP5_Titanic_RNA

### 2. Crear y activar el entorno virtual
En Linux: Bash

python3 -m venv .venv
source .venv/bin/activate
En Windows:

PowerShell

python -m venv .venv
.\.venv\Scripts\activate

#### 3. Instalar dependencias
Bash

pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn

#### 4. Pipeline del Proyecto (Metodología)
Preprocesamiento: Limpieza de nulos, eliminación de ruidos (Name, Ticket), codificación de variables categóricas y normalización.

Balanceo de Datos: Aplicación de SMOTE (Over-sampling) para equilibrar las clases de sobrevivientes.

Procesamiento: Entrenamiento de 3 arquitecturas distintas de MLP (Multilayer Perceptron).

Posprocesamiento: Evaluación mediante métricas de desempeño (Accuracy, F1, Recall).