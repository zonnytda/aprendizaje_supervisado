# 🎓 Proyecto: Predicción de Promedio Final - Egresados UNE

## 📋 Descripción
Aplicación de Machine Learning Supervisado para predecir el promedio final de egresados 
de la Universidad Nacional de Educación Enrique Guzmán y Valle (La Cantuta).

**Algoritmos utilizados:** Regresión Ridge y Lasso

## 🏗️ Estructura del Proyecto
```
proyecto_egresados_une/
├── data/
│   ├── raw/                 # Datos originales
│   ├── processed/           # Datos procesados
│   └── features/            # Features engineered
├── src/
│   ├── data/                # Clases de carga y preprocesamiento
│   ├── features/            # Ingeniería de características
│   ├── models/              # Entrenamiento y evaluación
│   └── utils/               # Utilidades y configuración
├── models/                  # Modelos entrenados (.pkl)
├── results/
│   ├── figures/             # Visualizaciones
│   └── metrics/             # Métricas guardadas
├── static/                  # Archivos estáticos Flask
│   ├── css/
│   ├── js/
│   └── img/
├── templates/               # Templates HTML Flask
├── notebooks/               # Jupyter notebooks
├── tests/                   # Tests unitarios
├── app.py                   # Aplicación Flask
├── main.py                  # Pipeline completo
└── requirements.txt         # Dependencias
```

## 🔧 Instalación

### 1. Clonar/Descargar el proyecto
```bash
cd proyecto_egresados_une
```

### 2. Crear entorno virtual
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Colocar dataset
Copiar archivo `EGRESADOSUNE20202024.csv` en la carpeta `data/raw/`

## 🚀 Uso

### Ejecutar Pipeline Completo (Entrenamiento)
```bash
python main.py
```

### Ejecutar Aplicación Web Flask
```bash
python app.py
```
Luego abrir navegador en: `http://localhost:5000`

## 📊 Clases POO Implementadas

### 1. DataLoader
- Carga datos desde CSV/Excel
- Validación de archivos
- Vista previa de datos

### 2. DataPreprocessor
- Limpieza de datos
- Manejo de valores nulos
- Detección de outliers
- Split train/val/test

### 3. FeatureEngineer
- Creación de features derivadas
- Encoding de categóricas
- Escalado de numéricas
- Persistencia de transformadores

### 4. ModelTrainer
- Entrenamiento Ridge/Lasso
- Hyperparameter tuning (Grid Search)
- Validación cruzada
- Guardar/cargar modelos

### 5. ModelEvaluator
- Cálculo de métricas (RMSE, MAE, R², MAPE)
- Visualizaciones (Predicted vs Actual, Residuales, Learning Curves)
- Reportes de evaluación

### 6. ModelComparator
- Comparación Ridge vs Lasso
- Selección del mejor modelo
- Tablas comparativas

### 7. Predictor
- Predicciones en producción
- Validación de entradas
- Explicabilidad de predicciones

## 🎯 Características del Dataset

- **Registros:** 9,898 egresados (2020-2024)
- **Variables:** 17 columnas
- **Target:** PROMEDIO_FINAL (12.23 - 19.30)
- **Features:** 16 predictoras (categóricas y numéricas)

## 📈 Métricas Esperadas

- **R² Score:** 0.60 - 0.75
- **RMSE:** 0.8 - 1.2 puntos
- **MAE:** 0.6 - 0.9 puntos

## 🌐 Aplicación Web

### Páginas disponibles:
1. **Home** - Presentación del proyecto
2. **EDA** - Análisis exploratorio interactivo
3. **Entrenar** - Entrenamiento de modelos
4. **Resultados** - Comparación Ridge vs Lasso
5. **Predicción** - Predictor interactivo
6. **Documentación** - Guía completa

## 🛠️ Tecnologías

- **Python 3.8+**
- **Flask** - Framework web
- **scikit-learn** - Machine Learning
- **pandas, numpy** - Manipulación de datos
- **matplotlib, seaborn, plotly** - Visualización

## 👨‍💻 Autor

**Estudiante de Ingeniería Estadística e Informática**  
Universidad Nacional de Educación - 7mo Semestre  
Curso: Machine Learning Supervisado

## 📅 Fecha

Octubre 2025

## 📝 Licencia

Proyecto académico - UNE
