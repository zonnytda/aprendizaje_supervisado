"""
Módulo: data_preprocessor.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: DataPreprocessor
Responsabilidad: Preprocesar y limpiar datos para Machine Learning
Principio POO: Single Responsibility
Autor: zonny
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    get_data_quality_report,
    detect_outliers_iqr
)

import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocesa y limpia datos para Machine Learning.
    
    Características:
    - Identificación automática de tipos de variables
    - Manejo inteligente de valores nulos
    - Detección y eliminación de outliers (IQR/Z-score)
    - División estratificada train/val/test
    - Conversión de tipos de datos
    - Generación de reportes de calidad
    
    Atributos:
        data (pd.DataFrame): Dataset original
        target_column (str): Nombre de la variable objetivo
        exclude_columns (List[str]): Columnas a excluir del análisis
        numeric_features (List[str]): Lista de features numéricas
        categorical_features (List[str]): Lista de features categóricas
        preprocessed_data (pd.DataFrame): Datos preprocesados
    
    Ejemplo de uso:
        >>> preprocessor = DataPreprocessor(df, target_column='PROMEDIO_FINAL')
        >>> preprocessor.identify_feature_types()
        >>> df_clean = preprocessor.handle_missing_values(strategy='drop')
        >>> df_clean = preprocessor.remove_outliers(method='iqr')
        >>> X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data()
    """
    
    def __init__(self, data: pd.DataFrame, target_column: str, 
                 exclude_columns: Optional[List[str]] = None):
        """
        Inicializa el preprocesador.
        
        Args:
            data: DataFrame con los datos
            target_column: Nombre de la columna objetivo
            exclude_columns: Columnas a excluir del análisis (IDs, fechas, etc.)
        
        Raises:
            ValueError: Si la columna objetivo no existe
        """
        if target_column not in data.columns:
            raise ValueError(f"❌ Columna objetivo '{target_column}' no encontrada")
        
        self.data = data.copy()
        self.target_column = target_column
        self.exclude_columns = exclude_columns or []
        
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = []
        self.preprocessed_data: Optional[pd.DataFrame] = None
        
        # Estadísticas iniciales
        self.initial_shape = self.data.shape
        self.processing_log = []
        
        print(f"✅ DataPreprocessor inicializado")
        print(f"   Dataset: {len(self.data):,} filas x {len(self.data.columns)} columnas")
        print(f"   Target: {self.target_column}")
        print(f"   Columnas excluidas: {len(self.exclude_columns)}")
        
        # Log inicial
        self._add_log("Inicialización", f"Shape: {self.data.shape}")
    
    def _add_log(self, action: str, details: str):
        """Agrega entrada al log de procesamiento"""
        self.processing_log.append({
            'action': action,
            'details': details,
            'shape': self.data.shape if self.data is not None else None
        })
    
    def identify_feature_types(self, max_unique_categorical: int = 100) -> Tuple[List[str], List[str]]:
        """
        Identifica automáticamente tipos de variables.
        
        Args:
            max_unique_categorical: Máximo de valores únicos para considerar categórica
        
        Returns:
            Tuple con (features numéricas, features categóricas)
        """
        print_progress("Identificando tipos de variables...", "🔍")
        
        # Columnas a analizar (excluir target y columnas especificadas)
        columns_to_analyze = [
            col for col in self.data.columns 
            if col != self.target_column and col not in self.exclude_columns
        ]
        
        # Identificar numéricas
        self.numeric_features = []
        for col in columns_to_analyze:
            if self.data[col].dtype in ['int64', 'float64']:
                # Si tiene muchos valores únicos, es numérica continua
                if self.data[col].nunique() > max_unique_categorical:
                    self.numeric_features.append(col)
                else:
                    # Podría ser categórica ordinal, pero tratarla como numérica
                    self.numeric_features.append(col)
        
        # Identificar categóricas
        self.categorical_features = [
            col for col in columns_to_analyze
            if col not in self.numeric_features
        ]
        
        print(f"\n📊 Análisis de variables:")
        print(f"   {'Variable':<40} {'Tipo':<15} {'Únicos':<10}")
        print("   " + "-"*65)
        
        for col in self.numeric_features:
            print(f"   {col:<40} {'Numérica':<15} {self.data[col].nunique():<10}")
        
        for col in self.categorical_features:
            print(f"   {col:<40} {'Categórica':<15} {self.data[col].nunique():<10}")
        
        print(f"\n✅ Identificación completada:")
        print(f"   • Variables numéricas: {len(self.numeric_features)}")
        print(f"   • Variables categóricas: {len(self.categorical_features)}")
        
        self._add_log("Feature identification", 
                     f"Numeric: {len(self.numeric_features)}, Categorical: {len(self.categorical_features)}")
        
        return self.numeric_features, self.categorical_features
    
    def handle_missing_values(self, strategy: str = 'drop', 
                              threshold: float = 0.5,
                              numeric_strategy: str = 'median',
                              categorical_strategy: str = 'mode') -> pd.DataFrame:
        """
        Maneja valores nulos en el dataset.
        
        Args:
            strategy: Estrategia general ('drop', 'impute', 'smart')
            threshold: % máximo de nulos permitido por columna (0-1)
            numeric_strategy: Para imputación numérica ('mean', 'median', 'mode')
            categorical_strategy: Para imputación categórica ('mode', 'constant')
        
        Returns:
            DataFrame sin valores nulos
        """
        print_progress(f"Manejando valores nulos (estrategia: {strategy})...", "🔧")
        
        # Analizar valores nulos
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data)) * 100
        
        print(f"\n📊 Análisis de valores nulos:")
        has_missing = False
        for col in self.data.columns:
            if missing[col] > 0:
                has_missing = True
                print(f"   • {col}: {missing[col]} ({missing_pct[col]:.2f}%)")
        
        if not has_missing:
            print("   ✅ No hay valores nulos!")
            self.preprocessed_data = self.data.copy()
            return self.preprocessed_data
        
        # Aplicar estrategia
        df_clean = self.data.copy()
        rows_before = len(df_clean)
        
        if strategy == 'drop':
            # 1. Eliminar filas con nulos en target
            if df_clean[self.target_column].isnull().any():
                df_clean = df_clean.dropna(subset=[self.target_column])
                print(f"   • Eliminadas {rows_before - len(df_clean)} filas (nulos en target)")
            
            # 2. Eliminar columnas con muchos nulos
            cols_to_drop = missing_pct[missing_pct > threshold * 100].index.tolist()
            if self.target_column in cols_to_drop:
                cols_to_drop.remove(self.target_column)
            
            if cols_to_drop:
                df_clean = df_clean.drop(columns=cols_to_drop)
                print(f"   • Eliminadas {len(cols_to_drop)} columnas (>{threshold*100}% nulos)")
            
            # 3. Eliminar filas restantes con nulos
            rows_before2 = len(df_clean)
            df_clean = df_clean.dropna()
            if rows_before2 > len(df_clean):
                print(f"   • Eliminadas {rows_before2 - len(df_clean)} filas adicionales")
        
        elif strategy == 'impute':
            # Imputar numéricas
            for col in self.numeric_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    if numeric_strategy == 'mean':
                        value = df_clean[col].mean()
                    elif numeric_strategy == 'median':
                        value = df_clean[col].median()
                    else:  # mode
                        value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else df_clean[col].mean()
                    
                    df_clean[col].fillna(value, inplace=True)
                    print(f"   • {col}: imputado con {numeric_strategy} = {value:.2f}")
            
            # Imputar categóricas
            for col in self.categorical_features:
                if col in df_clean.columns and df_clean[col].isnull().any():
                    if categorical_strategy == 'mode':
                        mode_value = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'DESCONOCIDO'
                        df_clean[col].fillna(mode_value, inplace=True)
                        print(f"   • {col}: imputado con moda = '{mode_value}'")
                    elif categorical_strategy == 'constant':
                        df_clean[col].fillna('DESCONOCIDO', inplace=True)
                        print(f"   • {col}: imputado con constante 'DESCONOCIDO'")
        
        elif strategy == 'smart':
            # Estrategia inteligente: combina ambas
            # 1. Eliminar columnas con muchos nulos
            cols_to_drop = missing_pct[missing_pct > threshold * 100].index.tolist()
            if self.target_column in cols_to_drop:
                cols_to_drop.remove(self.target_column)
            if cols_to_drop:
                df_clean = df_clean.drop(columns=cols_to_drop)
                print(f"   • Eliminadas {len(cols_to_drop)} columnas (>{threshold*100}% nulos)")
            
            # 2. Imputar nulos restantes
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if col in self.numeric_features:
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif col in self.categorical_features:
                        mode = df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'DESCONOCIDO'
                        df_clean[col].fillna(mode, inplace=True)
        
        print(f"\n✅ Datos limpios:")
        print(f"   Antes: {rows_before:,} filas x {len(self.data.columns)} columnas")
        print(f"   Después: {len(df_clean):,} filas x {len(df_clean.columns)} columnas")
        print(f"   Pérdida: {rows_before - len(df_clean):,} filas ({(rows_before - len(df_clean))/rows_before*100:.1f}%)")
        
        self.preprocessed_data = df_clean
        self._add_log("Missing values handled", f"Strategy: {strategy}, Rows lost: {rows_before - len(df_clean)}")
        
        return df_clean
    
    def remove_outliers(self, method: str = 'iqr', 
                       threshold: float = 1.5,
                       columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detecta y elimina outliers en variables numéricas.
        
        Args:
            method: Método de detección ('iqr' o 'zscore')
            threshold: Umbral (1.5 para IQR, 3 para Z-score)
            columns: Columnas específicas a analizar (None = todas las numéricas)
        
        Returns:
            DataFrame sin outliers
        """
        print_progress(f"Detectando outliers (método: {method})...", "🔍")
        
        if self.preprocessed_data is not None:
            df = self.preprocessed_data.copy()
        else:
            df = self.data.copy()
        
        # Determinar columnas a analizar
        cols_to_check = columns if columns else self.numeric_features
        cols_to_check = [col for col in cols_to_check if col in df.columns and col != self.target_column]
        
        if not cols_to_check:
            print("   ⚠️  No hay columnas numéricas para analizar")
            return df
        
        outliers_mask = pd.Series([False] * len(df), index=df.index)
        outliers_by_column = {}
        
        for col in cols_to_check:
            if method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                col_outliers = z_scores > threshold
            
            else:
                raise ValueError(f"Método no válido: {method}")
            
            outliers_count = col_outliers.sum()
            if outliers_count > 0:
                outliers_by_column[col] = outliers_count
                print(f"   • {col}: {outliers_count} outliers ({outliers_count/len(df)*100:.1f}%)")
                outliers_mask = outliers_mask | col_outliers
        
        # Eliminar outliers
        before = len(df)
        df_clean = df[~outliers_mask]
        after = len(df_clean)
        
        print(f"\n✅ Outliers eliminados:")
        print(f"   Total de filas con outliers: {before - after} ({(before - after)/before*100:.1f}%)")
        print(f"   Dataset final: {after:,} filas")
        
        self.preprocessed_data = df_clean
        self._add_log("Outliers removed", f"Method: {method}, Rows removed: {before - after}")
        
        return df_clean
    
    def convert_data_types(self) -> pd.DataFrame:
        """
        Convierte tipos de datos apropiadamente.
        
        Returns:
            DataFrame con tipos corregidos
        """
        print_progress("Convirtiendo tipos de datos...", "🔄")
        
        if self.preprocessed_data is not None:
            df = self.preprocessed_data.copy()
        else:
            df = self.data.copy()
        
        conversions_made = 0
        
        # Convertir categóricas a tipo 'category'
        for col in self.categorical_features:
            if col in df.columns:
                if df[col].dtype != 'category':
                    df[col] = df[col].astype('category')
                    conversions_made += 1
                    print(f"   • {col}: → category ({df[col].nunique()} categorías)")
        
        # Asegurar que numéricas sean del tipo correcto
        for col in self.numeric_features:
            if col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        conversions_made += 1
                        print(f"   • {col}: → numeric")
                    except:
                        print(f"   ⚠️  {col}: no se pudo convertir")
        
        print(f"\n✅ {conversions_made} conversiones realizadas")
        
        self.preprocessed_data = df
        self._add_log("Data types converted", f"Conversions: {conversions_made}")
        
        return df
    
    def split_data(self, test_size: float = 0.15, 
                   val_size: float = 0.15,
                   stratify_column: Optional[str] = None,
                   random_state: int = 42) -> Tuple:
        """
        Divide datos en train, validation y test sets.
        
        Args:
            test_size: Proporción del test set (0-1)
            val_size: Proporción del validation set (0-1)
            stratify_column: Columna para estratificación (None = sin estratificación)
            random_state: Semilla aleatoria para reproducibilidad
        
        Returns:
            Tuple (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print_progress("Dividiendo datos en train/val/test...", "✂️")
        
        if self.preprocessed_data is not None:
            df = self.preprocessed_data.copy()
        else:
            df = self.data.copy()
        
        # Calcular tamaños
        train_size = 1 - test_size - val_size
        print(f"\n📊 División:")
        print(f"   • Train: {train_size*100:.1f}%")
        print(f"   • Validation: {val_size*100:.1f}%")
        print(f"   • Test: {test_size*100:.1f}%")
        
        # Separar features y target
        X = df.drop(columns=[self.target_column] + self.exclude_columns, errors='ignore')
        y = df[self.target_column]
        
        # Verificar estratificación
        stratify = None
        if stratify_column and stratify_column in df.columns:
            stratify = df[stratify_column]
            print(f"   • Estratificación por: {stratify_column}")
        
        # Primera división: (train+val) vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        
        # Segunda división: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        
        # Estratificación para segunda división
        stratify_temp = None
        if stratify_column and stratify_column in X_temp.columns:
            stratify_temp = X_temp[stratify_column]
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=stratify_temp
        )
        
        print(f"\n✅ División completada:")
        print(f"   Train: {len(X_train):,} muestras")
        print(f"   Validation: {len(X_val):,} muestras")
        print(f"   Test: {len(X_test):,} muestras")
        print(f"   Features: {X_train.shape[1]}")
        
        self._add_log("Data split", 
                     f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def get_preprocessed_data(self) -> pd.DataFrame:
        """
        Obtiene los datos preprocesados.
        
        Returns:
            DataFrame preprocesado (o original si no se ha preprocesado)
        """
        if self.preprocessed_data is None:
            return self.data
        return self.preprocessed_data
    
    def generate_report(self) -> dict:
        """
        Genera reporte completo de preprocesamiento.
        
        Returns:
            Diccionario con estadísticas del preprocesamiento
        """
        current_data = self.preprocessed_data if self.preprocessed_data is not None else self.data
        
        report = {
            'initial_shape': self.initial_shape,
            'final_shape': current_data.shape,
            'rows_lost': self.initial_shape[0] - current_data.shape[0],
            'rows_lost_pct': ((self.initial_shape[0] - current_data.shape[0]) / self.initial_shape[0]) * 100,
            'cols_lost': self.initial_shape[1] - current_data.shape[1],
            'numeric_features': len(self.numeric_features),
            'categorical_features': len(self.categorical_features),
            'target_column': self.target_column,
            'processing_steps': len(self.processing_log),
            'processing_log': self.processing_log,
        }
        
        return report
    
    def print_report(self):
        """Imprime reporte de preprocesamiento formateado"""
        report = self.generate_report()
        
        print_section_header("📋 REPORTE DE PREPROCESAMIENTO")
        
        print(f"🔢 Datos iniciales: {report['initial_shape'][0]:,} filas x {report['initial_shape'][1]} columnas")
        print(f"🔢 Datos finales: {report['final_shape'][0]:,} filas x {report['final_shape'][1]} columnas")
        print(f"\n📊 Cambios:")
        print(f"   • Filas perdidas: {report['rows_lost']:,} ({report['rows_lost_pct']:.2f}%)")
        print(f"   • Columnas perdidas: {report['cols_lost']}")
        
        print(f"\n🎯 Features identificadas:")
        print(f"   • Numéricas: {report['numeric_features']}")
        print(f"   • Categóricas: {report['categorical_features']}")
        
        print(f"\n📝 Pasos de procesamiento ({report['processing_steps']}):")
        for i, step in enumerate(report['processing_log'], 1):
            print(f"   {i}. {step['action']}: {step['details']}")
        
        print("\n" + "="*70 + "\n")
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        current_data = self.preprocessed_data if self.preprocessed_data is not None else self.data
        return f"DataPreprocessor(rows={len(current_data):,}, target='{self.target_column}')"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de DataPreprocessor
    """
    from data_loader import DataLoader
    
    print_section_header("🧪 TESTING DATAPREPROCESSOR CLASS")
    
    try:
        # 1. Cargar datos
        print("📂 Paso 1: Cargar datos")
        loader = DataLoader('../../data/raw/EGRESADOSUNE20202024.csv')
        df = loader.load_csv()
        
        # 2. Preprocesar
        print("\n🔧 Paso 2: Inicializar preprocesador")
        exclude_cols = ['FECHA_CORTE', 'UUID', 'FECHA_EGRESO', 'SITUACION_ALUMNO', 'MODALIDAD']
        
        preprocessor = DataPreprocessor(
            data=df,
            target_column='PROMEDIO_FINAL',
            exclude_columns=exclude_cols
        )
        
        # 3. Pipeline de preprocesamiento
        print("\n⚙️  Paso 3: Pipeline de preprocesamiento")
        preprocessor.identify_feature_types()
        df_clean = preprocessor.handle_missing_values(strategy='drop')
        df_clean = preprocessor.remove_outliers(method='iqr', threshold=1.5)
        df_clean = preprocessor.convert_data_types()
        
        # 4. Split
        print("\n✂️  Paso 4: División de datos")
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            test_size=0.15,
            val_size=0.15,
            stratify_column='NIVEL_ACADEMICO',
            random_state=42
        )
        
        # 5. Reporte final
        preprocessor.print_report()
        
        print("✅ EJEMPLO COMPLETADO EXITOSAMENTE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()