import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

# 1. CARGA (Sensores)
df = pd.read_csv('data/titanic.csv')

# 2. LIMPIEZA
# Eliminamos columnas que no aportan al aprendizaje de la RNA
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Tratamiento de nulos (Punto clave de la metodología)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# 3. CODIFICACIÓN (Transformar a lenguaje de neuronas)
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex']) # Male/Female -> 0/1
df['Embarked'] = le.fit_transform(df['Embarked'])

# 4. BALANCEO (Consigna: Over-Sampling)
X = df.drop('Survived', axis=1)
y = df['Survived']

# Aplicamos SMOTE para balancear las clases (Sobrevivió / No sobrevivió)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# 5. PARTICIÓN (Consigna: Training, Test)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Normalización (Vital para que la RNA converja rápido)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Dataset listo. Tamaño del set de entrenamiento: {X_train.shape}")