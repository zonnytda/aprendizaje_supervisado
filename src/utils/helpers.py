"""
Módulo: helpers.py
Proyecto: Predicción Promedio Final - Egresados UNE
Descripción: Funciones auxiliares y utilidades del proyecto
Autor: Estudiante de Ingeniería Estadística e Informática
Fecha: Octubre 2025
"""

import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Callable
from functools import wraps


# ============================================
# FUNCIONES DE IMPRESIÓN Y FORMATO
# ============================================

def print_section_header(title: str, width: int = 70, char: str = "="):
    """
    Imprime un encabezado de sección visualmente atractivo.
    
    Args:
        title: Título de la sección
        width: Ancho del encabezado
        char: Carácter para el borde
    """
    print("\n" + char * width)
    print(title)
    print(char * width + "\n")


def print_progress(message: str, icon: str = "🔄"):
    """Imprime mensaje de progreso con icono"""
    print(f"{icon} {message}")


def print_success(message: str):
    """Imprime mensaje de éxito"""
    print(f"✅ {message}")


def print_warning(message: str):
    """Imprime mensaje de advertencia"""
    print(f"⚠️  {message}")


def print_error(message: str):
    """Imprime mensaje de error"""
    print(f"❌ {message}")


def print_info(message: str):
    """Imprime mensaje informativo"""
    print(f"ℹ️  {message}")


def print_stats_table(data: Dict[str, Any], title: str = "Estadísticas"):
    """
    Imprime diccionario como tabla formateada.
    
    Args:
        data: Diccionario con datos
        title: Título de la tabla
    """
    print(f"\n{title}:")
    max_key_len = max(len(str(k)) for k in data.keys())
    
    for key, value in data.items():
        if isinstance(value, float):
            print(f"   {str(key):<{max_key_len}} : {value:>10.4f}")
        elif isinstance(value, int):
            print(f"   {str(key):<{max_key_len}} : {value:>10,}")
        else:
            print(f"   {str(key):<{max_key_len}} : {value:>10}")
    print()


def print_comparison_table(data1: Dict, data2: Dict, 
                           label1: str = "Modelo 1", 
                           label2: str = "Modelo 2"):
    """
    Imprime tabla comparativa de dos diccionarios.
    
    Args:
        data1: Primer diccionario
        data2: Segundo diccionario
        label1: Etiqueta del primero
        label2: Etiqueta del segundo
    """
    print(f"\n{'='*70}")
    print(f"📊 COMPARACIÓN: {label1.upper()} vs {label2.upper()}")
    print(f"{'='*70}\n")
    
    # Obtener todas las claves
    all_keys = set(data1.keys()) | set(data2.keys())
    
    # Header
    print(f"{'Métrica':<20} | {label1:>15} | {label2:>15} | {'Diferencia':>15}")
    print("-" * 70)
    
    # Filas
    for key in sorted(all_keys):
        val1 = data1.get(key, 'N/A')
        val2 = data2.get(key, 'N/A')
        
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            diff = val2 - val1
            print(f"{key:<20} | {val1:>15.4f} | {val2:>15.4f} | {diff:>+15.4f}")
        else:
            print(f"{key:<20} | {str(val1):>15} | {str(val2):>15} | {'N/A':>15}")
    
    print(f"\n{'='*70}\n")


def format_metrics_table(metrics_dict: Dict[str, Dict], model_names: list = None) -> str:
    """
    Formatea un diccionario de métricas como tabla.
    
    Args:
        metrics_dict: Diccionario con métricas por modelo
                     Formato: {'modelo1': {'RMSE': 0.5, 'MAE': 0.3}, ...}
        model_names: Lista de nombres de modelos (opcional)
    
    Returns:
        String con la tabla formateada
    """
    if not metrics_dict:
        return "No hay métricas para mostrar"
    
    # Obtener modelos
    if model_names is None:
        model_names = list(metrics_dict.keys())
    
    # Obtener todas las métricas únicas
    all_metrics = set()
    for model_metrics in metrics_dict.values():
        all_metrics.update(model_metrics.keys())
    all_metrics = sorted(all_metrics)
    
    # Construir tabla
    lines = []
    
    # Encabezado
    header = f"{'Métrica':<15}"
    for model in model_names:
        header += f" | {model:>12}"
    lines.append(header)
    lines.append("-" * len(header))
    
    # Filas de métricas
    for metric in all_metrics:
        row = f"{metric:<15}"
        for model in model_names:
            value = metrics_dict.get(model, {}).get(metric, 'N/A')
            if isinstance(value, (int, float)):
                row += f" | {value:>12.4f}"
            else:
                row += f" | {str(value):>12}"
        lines.append(row)
    
    return "\n".join(lines)


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """
    Imprime información detallada de un DataFrame.
    
    Args:
        df: DataFrame a analizar
        name: Nombre descriptivo
    """
    print(f"\n{'='*70}")
    print(f"📊 INFORMACIÓN DE {name.upper()}")
    print(f"{'='*70}\n")
    
    print(f"Dimensiones: {df.shape[0]:,} filas x {df.shape[1]} columnas")
    print(f"Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Valores nulos: {df.isnull().sum().sum():,}")
    print(f"Duplicados: {df.duplicated().sum():,}")
    
    print(f"\n📋 Tipos de datos:")
    type_counts = df.dtypes.value_counts()
    for dtype, count in type_counts.items():
        print(f"   • {dtype}: {count} columnas")
    
    print(f"\n📊 Columnas con valores nulos:")
    null_cols = df.isnull().sum()
    null_cols = null_cols[null_cols > 0].sort_values(ascending=False)
    
    if len(null_cols) > 0:
        for col, count in null_cols.head(10).items():
            pct = (count / len(df)) * 100
            print(f"   • {col}: {count:,} ({pct:.1f}%)")
    else:
        print("   ✅ Sin valores nulos")
    
    print(f"\n{'='*70}\n")


# ============================================
# DECORADORES
# ============================================

def measure_time(func: Callable) -> Callable:
    """
    Decorador que mide el tiempo de ejecución de una función.
    
    Args:
        func: Función a decorar
    
    Returns:
        Función decorada
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        
        print(f"\n⏱️  Tiempo de ejecución de '{func.__name__}': {elapsed_time:.2f}s")
        return result
    
    return wrapper


def log_execution(log_file: str = "execution.log"):
    """
    Decorador que registra la ejecución de una función en un archivo.
    
    Args:
        log_file: Archivo donde guardar el log
    
    Returns:
        Decorador
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            
            with open(log_file, 'a') as f:
                f.write(f"[{timestamp}] Ejecutando: {func.__name__}\n")
            
            try:
                result = func(*args, **kwargs)
                
                with open(log_file, 'a') as f:
                    f.write(f"[{timestamp}] ✅ {func.__name__} completado\n")
                
                return result
                
            except Exception as e:
                with open(log_file, 'a') as f:
                    f.write(f"[{timestamp}] ❌ {func.__name__} falló: {str(e)}\n")
                raise
        
        return wrapper
    return decorator


# ============================================
# FUNCIONES DE ARCHIVOS
# ============================================

def ensure_dir(directory: str) -> Path:
    """
    Asegura que un directorio existe, creándolo si es necesario.
    
    Args:
        directory: Ruta del directorio
    
    Returns:
        Path del directorio
    """
    dir_path = Path(directory)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def save_dict_to_json(data: Dict, filepath: str, indent: int = 2):
    """
    Guarda un diccionario como archivo JSON.
    
    Args:
        data: Diccionario a guardar
        filepath: Ruta del archivo
        indent: Nivel de indentación
    """
    # Crear directorio si no existe
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir valores numpy a tipos nativos de Python
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    data_clean = convert_numpy(data)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_clean, f, indent=indent, ensure_ascii=False)
    
    print_success(f"JSON guardado en: {filepath}")


def load_json(filepath: str) -> Dict:
    """
    Carga un archivo JSON como diccionario.
    
    Args:
        filepath: Ruta del archivo
    
    Returns:
        Diccionario con los datos
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data


def create_directory_structure(base_path: Path):
    """
    Crea la estructura de directorios del proyecto.
    
    Args:
        base_path: Directorio base del proyecto
    """
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'models',
        'results/figures',
        'results/metrics',
        'static/css',
        'static/js',
        'static/img',
        'templates',
        'src/data',
        'src/features',
        'src/models',
        'src/utils',
        'notebooks',
        'tests'
    ]
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print_success(f"Estructura de directorios creada en: {base_path}")


def check_file_exists(filepath: str, raise_error: bool = False) -> bool:
    """
    Verifica si un archivo existe.
    
    Args:
        filepath: Ruta del archivo
        raise_error: Si True, lanza excepción si no existe
    
    Returns:
        True si existe, False en caso contrario
    """
    exists = Path(filepath).exists()
    
    if not exists and raise_error:
        raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
    
    return exists


def get_file_size(filepath: str) -> str:
    """
    Obtiene el tamaño de un archivo en formato legible.
    
    Args:
        filepath: Ruta del archivo
    
    Returns:
        String con tamaño formateado
    """
    size_bytes = Path(filepath).stat().st_size
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.2f} TB"


def clear_directory(directory: str, pattern: str = "*"):
    """
    Limpia archivos de un directorio.
    
    Args:
        directory: Directorio a limpiar
        pattern: Patrón de archivos a eliminar (ej: "*.pkl", "*.csv")
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print_warning(f"Directorio no existe: {directory}")
        return
    
    files = list(dir_path.glob(pattern))
    
    if len(files) == 0:
        print_info(f"No hay archivos que coincidan con '{pattern}' en {directory}")
        return
    
    for file in files:
        file.unlink()
    
    print_success(f"{len(files)} archivo(s) eliminado(s) de {directory}")


# ============================================
# FUNCIONES DE ANÁLISIS DE DATOS
# ============================================

def get_data_quality_report(df: pd.DataFrame) -> Dict:
    """
    Genera un reporte de calidad de datos.
    
    Args:
        df: DataFrame a analizar
    
    Returns:
        Diccionario con métricas de calidad
    """
    report = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicated_rows': df.duplicated().sum(),
        'missing_values': {},
        'data_types': df.dtypes.value_counts().to_dict()
    }
    
    # Analizar valores nulos por columna
    for col in df.columns:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            report['missing_values'][col] = {
                'count': int(missing_count),
                'percentage': float(missing_count / len(df) * 100)
            }
    
    return report


def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando el método IQR.
    
    Args:
        data: Serie de datos numéricos
        multiplier: Multiplicador del IQR (típicamente 1.5)
    
    Returns:
        Serie booleana indicando outliers (True = outlier)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    return (data < lower_bound) | (data > upper_bound)


def detect_outliers_zscore(data: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detecta outliers usando Z-score.
    
    Args:
        data: Serie de datos numéricos
        threshold: Umbral de Z-score (típicamente 3.0)
    
    Returns:
        Serie booleana indicando outliers (True = outlier)
    """
    mean = data.mean()
    std = data.std()
    
    z_scores = np.abs((data - mean) / std)
    
    return z_scores > threshold


def get_categorical_summary(df: pd.DataFrame, column: str, top_n: int = 10) -> Dict:
    """
    Resumen estadístico de una variable categórica.
    
    Args:
        df: DataFrame
        column: Nombre de la columna
        top_n: Número de categorías principales a mostrar
    
    Returns:
        Diccionario con resumen
    """
    value_counts = df[column].value_counts()
    
    summary = {
        'total_unique': len(value_counts),
        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
        'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
        'top_categories': value_counts.head(top_n).to_dict()
    }
    
    return summary


def get_numerical_summary(df: pd.DataFrame, column: str) -> Dict:
    """
    Resumen estadístico de una variable numérica.
    
    Args:
        df: DataFrame
        column: Nombre de la columna
    
    Returns:
        Diccionario con resumen
    """
    summary = {
        'count': int(df[column].count()),
        'mean': float(df[column].mean()),
        'std': float(df[column].std()),
        'min': float(df[column].min()),
        'q25': float(df[column].quantile(0.25)),
        'median': float(df[column].median()),
        'q75': float(df[column].quantile(0.75)),
        'max': float(df[column].max()),
        'skewness': float(df[column].skew()),
        'kurtosis': float(df[column].kurtosis())
    }
    
    return summary


# ============================================
# FUNCIONES DE VALIDACIÓN
# ============================================

def validate_dataframe(df: pd.DataFrame, 
                      required_columns: List[str] = None,
                      min_rows: int = 0) -> bool:
    """
    Valida un DataFrame.
    
    Args:
        df: DataFrame a validar
        required_columns: Columnas requeridas
        min_rows: Mínimo número de filas
    
    Returns:
        True si es válido, False en caso contrario
    """
    # Verificar que no esté vacío
    if df is None or df.empty:
        print_error("DataFrame está vacío")
        return False
    
    # Verificar número mínimo de filas
    if len(df) < min_rows:
        print_error(f"DataFrame tiene {len(df)} filas, se requieren al menos {min_rows}")
        return False
    
    # Verificar columnas requeridas
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            print_error(f"Faltan columnas requeridas: {missing_cols}")
            return False
    
    return True


def validate_model_input(X, y) -> bool:
    """
    Valida entradas para entrenamiento de modelo.
    
    Args:
        X: Features
        y: Target
    
    Returns:
        True si es válido, False en caso contrario
    """
    # Verificar tipos
    if not isinstance(X, (pd.DataFrame, np.ndarray)):
        print_error("X debe ser DataFrame o numpy array")
        return False
    
    if not isinstance(y, (pd.Series, np.ndarray)):
        print_error("y debe ser Series o numpy array")
        return False
    
    # Verificar dimensiones
    if len(X) != len(y):
        print_error(f"X ({len(X)}) y y ({len(y)}) tienen diferentes longitudes")
        return False
    
    # Verificar valores nulos
    if isinstance(X, pd.DataFrame):
        if X.isnull().any().any():
            print_warning("X contiene valores nulos")
            return False
    
    if isinstance(y, pd.Series):
        if y.isnull().any():
            print_warning("y contiene valores nulos")
            return False
    
    return True


def validate_target_range(y, min_val: float = None, max_val: float = None) -> bool:
    """
    Valida que los valores del target estén en un rango válido.
    
    Args:
        y: Variable objetivo (Series o array)
        min_val: Valor mínimo permitido (opcional)
        max_val: Valor máximo permitido (opcional)
    
    Returns:
        True si es válido, False en caso contrario
    
    Ejemplo:
        >>> validate_target_range(y_train, min_val=0, max_val=20)
    """
    if isinstance(y, pd.Series):
        values = y.values
    elif isinstance(y, np.ndarray):
        values = y
    else:
        print_error("y debe ser Series o numpy array")
        return False
    
    # Verificar valores no numéricos
    if not np.issubdtype(values.dtype, np.number):
        print_error("Target debe contener valores numéricos")
        return False
    
    # Verificar rango mínimo
    if min_val is not None:
        if np.any(values < min_val):
            count = np.sum(values < min_val)
            print_warning(f"{count} valores están por debajo del mínimo ({min_val})")
            return False
    
    # Verificar rango máximo
    if max_val is not None:
        if np.any(values > max_val):
            count = np.sum(values > max_val)
            print_warning(f"{count} valores están por encima del máximo ({max_val})")
            return False
    
    return True


# ============================================
# FUNCIONES DE FORMATO
# ============================================

def format_number(num: float, decimals: int = 2) -> str:
    """Formatea número con separadores de miles"""
    if isinstance(num, (int, float)):
        return f"{num:,.{decimals}f}"
    return str(num)


def format_percentage(value: float, decimals: int = 1) -> str:
    """Formatea como porcentaje"""
    if isinstance(value, (int, float)):
        return f"{value:.{decimals}f}%"
    return str(value)


def format_time(seconds: float) -> str:
    """
    Formatea tiempo en formato legible.
    
    Args:
        seconds: Tiempo en segundos
    
    Returns:
        String formateado (ej: "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ============================================
# FUNCIONES DE CONVERSIÓN
# ============================================

def numpy_to_python(obj: Any) -> Any:
    """
    Convierte tipos numpy a tipos nativos de Python.
    Útil para serialización JSON.
    
    Args:
        obj: Objeto a convertir
    
    Returns:
        Objeto convertido
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python(item) for item in obj]
    return obj


def encode_categorical_simple(series: pd.Series, mapping: Dict = None) -> tuple:
    """
    Codifica una serie categórica a numérica.
    
    Args:
        series: Serie categórica
        mapping: Mapeo opcional (si None, se crea automáticamente)
    
    Returns:
        Tupla (serie_codificada, mapping_usado)
    """
    if mapping is None:
        unique_values = series.unique()
        mapping = {val: idx for idx, val in enumerate(unique_values)}
    
    encoded = series.map(mapping)
    return encoded, mapping


def decode_categorical_simple(series: pd.Series, mapping: Dict) -> pd.Series:
    """
    Decodifica una serie numérica a categórica.
    
    Args:
        series: Serie numérica
        mapping: Mapeo de codificación original
    
    Returns:
        Serie categórica decodificada
    """
    reverse_mapping = {v: k for k, v in mapping.items()}
    decoded = series.map(reverse_mapping)
    return decoded


# ============================================
# FUNCIONES MISCELÁNEAS
# ============================================

def create_feature_name(base_name: str, *suffixes) -> str:
    """
    Crea nombre de feature combinando base y sufijos.
    
    Args:
        base_name: Nombre base
        *suffixes: Sufijos a agregar
    
    Returns:
        Nombre de feature
    """
    parts = [base_name] + list(suffixes)
    return '_'.join(str(p).lower() for p in parts)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    División segura que maneja división por cero.
    
    Args:
        numerator: Numerador
        denominator: Denominador
        default: Valor por defecto si denominador es cero
    
    Returns:
        Resultado de la división o default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def chunk_list(lst: List, chunk_size: int) -> List[List]:
    """
    Divide una lista en chunks de tamaño específico.
    
    Args:
        lst: Lista a dividir
        chunk_size: Tamaño de cada chunk
    
    Returns:
        Lista de chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """
    Aplana un diccionario anidado.
    
    Args:
        d: Diccionario a aplanar
        parent_key: Clave padre para recursión
        sep: Separador entre claves
    
    Returns:
        Diccionario aplanado
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# ============================================
# FUNCIONES DE PROGRESO
# ============================================

class ProgressTracker:
    """
    Clase simple para trackear progreso de operaciones.
    """
    
    def __init__(self, total: int, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = time.time()
        print(f"\n{description}: 0/{total} (0.0%)")
    
    def update(self, n: int = 1):
        """Actualiza el progreso"""
        self.current += n
        percentage = (self.current / self.total) * 100
        elapsed = time.time() - self.start_time
        
        # Estimar tiempo restante
        if self.current > 0:
            eta = (elapsed / self.current) * (self.total - self.current)
            eta_str = format_time(eta)
        else:
            eta_str = "?"
        
        # Imprimir progreso (sobrescribir línea anterior)
        print(f"\r{self.description}: {self.current}/{self.total} ({percentage:.1f}%) - ETA: {eta_str}", end='', flush=True)
    
    def close(self):
        """Cierra el tracker"""
        elapsed = time.time() - self.start_time
        print(f"\n✅ {self.description} completado en {format_time(elapsed)}\n")


# ============================================
# CONSTANTES Y CONFIGURACIONES
# ============================================

# Emojis para diferentes tipos de mensajes
ICONS = {
    'success': '✅',
    'error': '❌',
    'warning': '⚠️',
    'info': 'ℹ️',
    'progress': '🔄',
    'data': '📊',
    'model': '🤖',
    'file': '📁',
    'search': '🔍',
    'time': '⏱️',
    'chart': '📈',
    'train': '🚂',
    'test': '🧪',
    'save': '💾',
    'load': '📂',
    'check': '✓',
    'cross': '✗'
}


# ============================================
# EXPORTS
# ============================================

__all__ = [
    # Impresión
    'print_section_header',
    'print_progress',
    'print_success',
    'print_warning',
    'print_error',
    'print_info',
    'print_stats_table',
    'print_dataframe_info',
    'print_comparison_table',
    'format_metrics_table',
    
    # Decoradores
    'measure_time',
    'log_execution',
    
    # Archivos
    'save_dict_to_json',
    'load_json',
    'create_directory_structure',
    'ensure_dir',
    'check_file_exists',
    'get_file_size',
    'clear_directory',
    
    # Análisis de datos
    'get_data_quality_report',
    'detect_outliers_iqr',
    'detect_outliers_zscore',
    'get_categorical_summary',
    'get_numerical_summary',
    
    # Validación
    'validate_dataframe',
    'validate_model_input',
    'validate_target_range',
    
    # Formato
    'format_number',
    'format_percentage',
    'format_time',
    
    # Conversión
    'numpy_to_python',
    'encode_categorical_simple',
    'decode_categorical_simple',
    
    # Utilidades
    'create_feature_name',
    'safe_divide',
    'chunk_list',
    'flatten_dict',
    
    # Clases
    'ProgressTracker',
    
    # Constantes
    'ICONS'
]


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    """
    Tests básicos de las funciones auxiliares
    """
    print_section_header("🧪 TESTING HELPERS MODULE")
    
    # Test 1: Formato de números
    print("Test 1: Formato de números")
    print(f"   {format_number(1234567.89, 2)}")
    print(f"   {format_percentage(45.678, 1)}")
    print(f"   {format_time(3725)}")
    
    # Test 2: División segura
    print("\nTest 2: División segura")
    print(f"   10 / 2 = {safe_divide(10, 2)}")
    print(f"   10 / 0 = {safe_divide(10, 0, default=999)}")
    
    # Test 3: Crear nombre de feature
    print("\nTest 3: Nombre de feature")
    print(f"   {create_feature_name('creditos', 'por', 'anio')}")
    
    # Test 4: Chunk list
    print("\nTest 4: Chunk list")
    print(f"   {chunk_list([1,2,3,4,5,6,7], 3)}")
    
    # Test 5: Flatten dict
    print("\nTest 5: Flatten dict")
    nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    print(f"   Original: {nested}")
    print(f"   Aplanado: {flatten_dict(nested)}")
    
    # Test 6: Progress tracker
    print("\nTest 6: Progress tracker")
    tracker = ProgressTracker(total=50, description="Procesando")
    for i in range(50):
        time.sleep(0.02)  # Simular trabajo
        tracker.update(1)
    tracker.close()
    
    print_success("Todos los tests completados")