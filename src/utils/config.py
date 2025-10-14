"""
Módulo: config.py
Proyecto: Predicción Promedio Final - Egresados UNE
Descripción: Configuración centralizada del proyecto
Autor: Estudiante de Ingeniería Estadística e Informática
Fecha: Octubre 2025
"""

from pathlib import Path

# ============================================
# RUTAS DEL PROYECTO
# ============================================

# Directorio base del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent

# Directorios de datos
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
FEATURES_DATA_DIR = DATA_DIR / 'features'

# Directorios de modelos
MODELS_DIR = PROJECT_ROOT / 'models'

# Directorios de resultados
RESULTS_DIR = PROJECT_ROOT / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
METRICS_DIR = RESULTS_DIR / 'metrics'

# Directorios de la aplicación web
STATIC_DIR = PROJECT_ROOT / 'static'
TEMPLATES_DIR = PROJECT_ROOT / 'templates'

# Crear directorios si no existen
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FEATURES_DATA_DIR, 
                  MODELS_DIR, FIGURES_DIR, METRICS_DIR, STATIC_DIR, TEMPLATES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================
# CONFIGURACIÓN DEL DATASET
# ============================================

# Nombre del archivo de datos
RAW_DATA_FILE = 'EGRESADOSUNE20202024.csv'

# Información del dataset
DATASET_INFO = {
    'nombre': 'Egresados UNE 2020-2024',
    'universidad': 'Universidad Nacional de Educación Enrique Guzmán y Valle',
    'periodo': '2020-2024',
    'descripcion': 'Dataset de egresados con información académica y demográfica'
}

# Columna objetivo
TARGET_COLUMN = 'PROMEDIO_FINAL'

# Columnas a excluir del análisis (IDs, fechas, variables redundantes)
EXCLUDE_COLUMNS = [
    'FECHA_CORTE',
    'UUID', 
    'FECHA_EGRESO',
    'SITUACION_ALUMNO',
    'MODALIDAD'
]

# ============================================
# CONFIGURACIÓN DE CARGA DE DATOS
# ============================================

# Configuración para archivos CSV
CSV_SEPARATOR = ';'          # Separador de columnas
CSV_DECIMAL = ','            # Separador decimal (si aplica)
CSV_ENCODING = 'latin1'      # Encoding del archivo

# Configuración robusta para CSVs problemáticos
CSV_OPTIONS = {
    'sep': CSV_SEPARATOR,
    'encoding': CSV_ENCODING,
    'on_bad_lines': 'skip',      # Omitir líneas problemáticas
    'engine': 'python',           # Motor más flexible
    'encoding_errors': 'ignore',  # Ignorar errores de encoding
    # NO incluir 'low_memory' con engine='python'
}


# ============================================
# CONFIGURACIÓN DE PREPROCESAMIENTO
# ============================================

# Estrategia para valores nulos
MISSING_VALUES_STRATEGY = 'drop'  # 'drop', 'impute', 'smart'
MISSING_THRESHOLD = 0.3           # Máximo % de nulos por columna (30%)

# Detección de outliers
OUTLIER_METHOD = 'iqr'            # 'iqr', 'zscore', 'isolation_forest'
OUTLIER_THRESHOLD = 1.5           # Multiplicador de IQR (o Z-score threshold)

# División de datos
TEST_SIZE = 0.15                  # 15% para test
VALIDATION_SIZE = 0.15            # 15% para validación (del restante 85%)
RANDOM_STATE = 42                 # Semilla para reproducibilidad

# Estratificación
STRATIFY_BY = 'NIVEL_ACADEMICO'   # Columna para estratificar el split


# ============================================
# CONFIGURACIÓN DE FEATURE ENGINEERING
# ============================================

# Umbral para agrupar categorías raras
RARE_CATEGORY_THRESHOLD = 50      # Mínimo de ocurrencias

# Tipo de encoding para categóricas
CATEGORICAL_ENCODING = 'onehot'   # 'onehot', 'label', 'target'

# Tipo de escalado para numéricas
NUMERICAL_SCALING = 'standard'    # 'standard', 'minmax', 'robust'

# Features derivadas a crear
DERIVED_FEATURES = {
    'duracion_estudios': True,    # FECHA_EGRESO - FECHA_INGRESO
    'creditos_por_anio': True,    # CREDITOS_ACUMULADOS / DURACION_ESTUDIOS
    'edad_al_egreso': True,       # Edad al momento del egreso
}


# ============================================
# CONFIGURACIÓN DE MODELOS
# ============================================

# Modelos a entrenar
MODELS_TO_TRAIN = ['ridge', 'lasso']  # 'ridge', 'lasso', 'elasticnet'

# Configuración de Ridge
RIDGE_CONFIG = {
    'alpha': 1.0,
    'fit_intercept': True,
    'max_iter': None,
    'tol': 0.001,
    'solver': 'auto',
    'random_state': RANDOM_STATE
}

# Configuración de Lasso
LASSO_CONFIG = {
    'alpha': 1.0,
    'fit_intercept': True,
    'max_iter': 1000,
    'tol': 0.0001,
    'random_state': RANDOM_STATE,
    'selection': 'cyclic'
}

# Grid Search - Hiperparámetros Ridge
RIDGE_PARAM_GRID = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
}

# Grid Search - Hiperparámetros Lasso
LASSO_PARAM_GRID = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
}

# Validación cruzada
CV_FOLDS = 5                      # Número de folds para CV
CV_SCORING = 'neg_mean_squared_error'  # Métrica para CV


# ============================================
# CONFIGURACIÓN DE EVALUACIÓN
# ============================================

# Métricas a calcular
EVALUATION_METRICS = [
    'rmse',      # Root Mean Squared Error
    'mae',       # Mean Absolute Error  
    'r2',        # R² Score
    'mape',      # Mean Absolute Percentage Error
    'mse'        # Mean Squared Error
]

# Configuración de visualizaciones
FIGURE_DPI = 300                  # DPI para guardar figuras
FIGURE_FORMAT = 'png'             # Formato de figuras
FIGURE_SIZE = (10, 6)             # Tamaño por defecto


# ============================================
# NOMBRES DE ARCHIVOS DE SALIDA
# ============================================

# Modelos entrenados
RIDGE_MODEL_FILE = 'ridge_model.pkl'
LASSO_MODEL_FILE = 'lasso_model.pkl'
SCALER_FILE = 'scaler.pkl'
ENCODER_FILE = 'encoder.pkl'

# Archivos de métricas
RIDGE_METRICS_FILE = 'ridge_metrics.json'
LASSO_METRICS_FILE = 'lasso_metrics.json'
COMPARISON_FILE = 'model_comparison.csv'

# Archivos de datos procesados
PROCESSED_TRAIN_FILE = 'train_processed.csv'
PROCESSED_VAL_FILE = 'val_processed.csv'
PROCESSED_TEST_FILE = 'test_processed.csv'


# ============================================
# CONFIGURACIÓN DE FLASK (APLICACIÓN WEB)
# ============================================

FLASK_CONFIG = {
    'SECRET_KEY': 'una-clave-secreta-muy-segura-2025',
    'DEBUG': True,
    'HOST': '0.0.0.0',
    'PORT': 5000,
    'TEMPLATES_AUTO_RELOAD': True,
    'MAX_CONTENT_LENGTH': 16 * 1024 * 1024,  # 16 MB max upload
}


# ============================================
# CONFIGURACIÓN DE LOGGING
# ============================================

LOG_LEVEL = 'INFO'               # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = PROJECT_ROOT / 'app.log'


# ============================================
# MENSAJES DEL SISTEMA
# ============================================

MESSAGES = {
    'loading_data': '📂 Cargando datos...',
    'preprocessing': '🔧 Preprocesando datos...',
    'feature_engineering': '⚙️  Creando features...',
    'training': '🤖 Entrenando modelo...',
    'evaluating': '📊 Evaluando modelo...',
    'success': '✅ Operación completada exitosamente',
    'error': '❌ Error en la operación',
}


# ============================================
# VALIDACIONES
# ============================================

def validate_config():
    """
    Valida la configuración del proyecto.
    Útil para detectar errores antes de ejecutar el pipeline.
    """
    errors = []
    
    # Validar que exista el archivo de datos
    if not (RAW_DATA_DIR / RAW_DATA_FILE).exists():
        errors.append(f"Archivo de datos no encontrado: {RAW_DATA_DIR / RAW_DATA_FILE}")
    
    # Validar rangos
    if not 0 < TEST_SIZE < 1:
        errors.append(f"TEST_SIZE debe estar entre 0 y 1, valor actual: {TEST_SIZE}")
    
    if not 0 < VALIDATION_SIZE < 1:
        errors.append(f"VALIDATION_SIZE debe estar entre 0 y 1, valor actual: {VALIDATION_SIZE}")
    
    # Validar modelos
    valid_models = ['ridge', 'lasso', 'elasticnet']
    for model in MODELS_TO_TRAIN:
        if model not in valid_models:
            errors.append(f"Modelo '{model}' no válido. Opciones: {valid_models}")
    
    # Mostrar errores si existen
    if errors:
        print("\n⚠️  ERRORES DE CONFIGURACIÓN:")
        for error in errors:
            print(f"   • {error}")
        print()
        return False
    
    return True


# ============================================
# INFORMACIÓN DEL PROYECTO
# ============================================

PROJECT_INFO = {
    'nombre': 'Sistema de Predicción de Promedio Final - Egresados UNE',
    'version': '1.0.0',
    'autor': 'Apaza Bruna Luis Wilson',
    'universidad': 'Universidad Nacional del Altiplano',
    'facultad': 'Facultad de Ingeniería Estadística e Informática',
    'carrera': 'Ingeniería Estadística e Informática',
    'fecha': 'Octubre 2025',
    'descripcion': 'Aplicación de Machine Learning para predecir el promedio final de egresados utilizando Ridge y Lasso Regression',
    'ciudad': 'Puno',
    'pais': 'Perú'
}

# ============================================
# INFORMACIÓN DEL DATASET
# ============================================

DATASET_INFO = {
    'nombre': 'Egresados UNE 2020-2024',
    'universidad': 'Universidad Nacional de Educación Enrique Guzmán y Valle',
    'periodo': '2020-2024',
    'descripcion': 'Dataset de egresados con información académica y demográfica'
}


# ============================================
# EXPORTS
# ============================================

__all__ = [
    # Rutas
    'PROJECT_ROOT', 'DATA_DIR', 'RAW_DATA_DIR', 'PROCESSED_DATA_DIR',
    'FEATURES_DATA_DIR', 'MODELS_DIR', 'RESULTS_DIR', 'FIGURES_DIR', 
    'METRICS_DIR', 'STATIC_DIR', 'TEMPLATES_DIR',
    
    # Dataset
    'RAW_DATA_FILE', 'DATASET_INFO', 'TARGET_COLUMN', 'EXCLUDE_COLUMNS',
    
    # Carga de datos
    'CSV_SEPARATOR', 'CSV_DECIMAL', 'CSV_ENCODING', 'CSV_OPTIONS',
    
    # Preprocesamiento
    'MISSING_VALUES_STRATEGY', 'MISSING_THRESHOLD', 'OUTLIER_METHOD',
    'OUTLIER_THRESHOLD', 'TEST_SIZE', 'VALIDATION_SIZE', 'RANDOM_STATE',
    'STRATIFY_BY',
    
    # Feature Engineering
    'RARE_CATEGORY_THRESHOLD', 'CATEGORICAL_ENCODING', 'NUMERICAL_SCALING',
    'DERIVED_FEATURES',
    
    # Modelos
    'MODELS_TO_TRAIN', 'RIDGE_CONFIG', 'LASSO_CONFIG', 'RIDGE_PARAM_GRID',
    'LASSO_PARAM_GRID', 'CV_FOLDS', 'CV_SCORING',
    
    # Evaluación
    'EVALUATION_METRICS', 'FIGURE_DPI', 'FIGURE_FORMAT', 'FIGURE_SIZE',
    
    # Archivos
    'RIDGE_MODEL_FILE', 'LASSO_MODEL_FILE', 'SCALER_FILE', 'ENCODER_FILE',
    'RIDGE_METRICS_FILE', 'LASSO_METRICS_FILE', 'COMPARISON_FILE',
    
    # Flask
    'FLASK_CONFIG',
    
    # Sistema
    'LOG_LEVEL', 'LOG_FORMAT', 'LOG_FILE', 'MESSAGES', 'PROJECT_INFO',
    
    # Funciones
    'validate_config'
]


# ============================================
# EJECUCIÓN AL IMPORTAR
# ============================================

if __name__ == "__main__":
    """
    Ejecuta validaciones y muestra información de configuración.
    """
    print("=" * 70)
    print(f" {PROJECT_INFO['nombre']} ".center(70, "="))
    print("=" * 70)
    print()
    
    print("📋 INFORMACIÓN DEL PROYECTO:")
    for key, value in PROJECT_INFO.items():
        if key != 'nombre':
            print(f"   • {key.capitalize()}: {value}")
    
    print("\n📁 RUTAS CONFIGURADAS:")
    print(f"   • Datos raw: {RAW_DATA_DIR}")
    print(f"   • Modelos: {MODELS_DIR}")
    print(f"   • Resultados: {RESULTS_DIR}")
    
    print("\n🔧 CONFIGURACIÓN:")
    print(f"   • Target: {TARGET_COLUMN}")
    print(f"   • Modelos: {', '.join(MODELS_TO_TRAIN)}")
    print(f"   • Test size: {TEST_SIZE * 100:.0f}%")
    print(f"   • Validation size: {VALIDATION_SIZE * 100:.0f}%")
    
    print("\n🔍 VALIDANDO CONFIGURACIÓN...")
    if validate_config():
        print("✅ Configuración válida")
    else:
        print("❌ Hay errores en la configuración")
    
    print("\n" + "=" * 70)