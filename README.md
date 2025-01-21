# Optimización de la Precisión en la Detección de Noticias Falsas de Política

Este proyecto tiene como objetivo mejorar la precisión en la detección de noticias falsas relacionadas con política mediante la aplicación y comparación de diferentes algoritmos de optimización en la regresión logística.

## Resumen del Proyecto
Se implementaron y optimizaron algoritmos de descenso de gradiente como:
- Gradiente Descendente (GD)
- Gradiente Descendente Estocástico (SGD)
- Mini-Batch Gradiente Descendente (MBGD)
- Adagrad
- Adam
- RMSProp

El conjunto de datos consta de **74,276 registros** divididos en tres subconjuntos:
- **70% (Entrenamiento):** Datos utilizados para entrenar el modelo.
- **15% (Validación):** Datos utilizados para ajustar los hiperparámetros.
- **15% (Prueba):** Datos utilizados para evaluar el desempeño final del modelo.

## Datos y Preprocesamiento
Se trabajó con un conjunto de datos preprocesado con los siguientes pasos principales:
- **Vectorización de texto:** Se utilizó `TfidfVectorizer` para convertir texto en vectores numéricos.
- **Selección de características:** Implementada mediante `SelectKBest` y chi-cuadrado.
- **Estandarización:** Se usó `StandardScaler` para normalizar los datos.

### Bibliotecas Utilizadas
Las bibliotecas principales empleadas fueron:
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from scipy.sparse.linalg import norm
from scipy.stats import zscore
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

## Metodología
1. **División de Datos:**
   - **70% Entrenamiento**: 51,993 registros
   - **15% Validación**: 11,141 registros
   - **15% Prueba**: 11,142 registros

2. **Entrenamiento del Modelo:**
   - Implementación de diferentes algoritmos de optimización.
   - Monitoreo de métricas clave como `accuracy`, `precision`, `f1_score` y análisis de matrices de confusión.

3. **Optimización de Algoritmos:**
   - Comparación entre algoritmos de optimización para determinar el que mejor equilibra precisión y rendimiento.
   - Métricas de evaluación obtenidas para los subconjuntos de validación y prueba.

### Métricas Clave
Las métricas principales evaluadas incluyeron:
- **Exactitud (Accuracy)**
- **Precisión (Precision)**
- **F1 Score**
- **Matriz de Confusión**

### Comparación entre Algoritmos
Se realizó una evaluación comparativa para determinar el desempeño de cada algoritmo de optimización en términos de:
- Convergencia
- Estabilidad
- Precisión final

## Visualización de Resultados
Las gráficas generadas incluyeron:
- **Curvas de aprendizaje:** Para monitorear la convergencia de los algoritmos.
- **Matriz de confusión:** Para analizar errores de clasificación.

## Requisitos
- Python 3.8+
- Bibliotecas necesarias:
  ```bash
  pip install numpy pandas scikit-learn torch matplotlib
  ```

## Ejecución
1. Clona el repositorio:
   ```bash
   git clone <URL del repositorio>
   ```
2. Instala las librerias necesarias.
3. Ejecuta el script con el kernel 3.11:

## Créditos
Proyecto desarrollado como parte de un esfuerzo por mejorar la detección de noticias falsas utilizando técnicas  de machine learning y optimización.

---
**Nota:** Este README es un resumen del trabajo realizado y puede ser actualizado con detalles adicionales sobre los resultados experimentales.
