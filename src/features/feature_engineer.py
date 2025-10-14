"""
Módulo: feature_engineer.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: FeatureEngineer
Responsabilidad: Ingeniería de características y transformaciones
Principio POO: Single Responsibility + Encapsulamiento
Autor: Estudiante de Ingeniería Estadística e Informática
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import joblib

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    save_dict_to_json
)

import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Realiza ingeniería de características para Machine Learning.
    
    Funciones principales:
    - Crea features derivadas (duración de estudios, créditos por año, etc.)
    - Agrupa categorías raras en 'OTROS'
    - Codifica variables categóricas (One-Hot Encoding)
    - Escala variables numéricas (StandardScaler/MinMaxScaler)
    - Selecciona features importantes
    - Persiste transformadores para producción
    
    Atributos:
        data (pd.DataFrame): Dataset original
        categorical_features (List[str]): Features categóricas
        numerical_features (List[str]): Features numéricas
        encoder (OneHotEncoder): Transformador para categóricas
        scaler (StandardScaler/MinMaxScaler): Transformador para numéricas
        feature_names (List[str]): Nombres finales de features
        is_fitted (bool): Indica si los transformadores están ajustados
    
    Ejemplo de uso:
        >>> engineer = FeatureEngineer(X_train, categorical_cols, numerical_cols)
        >>> engineer.create_derived_features()
        >>> engineer.group_rare_categories(threshold=50)
        >>> X_transformed = engineer.fit_transform(X_train)
        >>> X_val_transformed = engineer.transform(X_val)
        >>> engineer.save_transformers('models/')
    """
    
    def __init__(self, data: pd.DataFrame,
                 categorical_features: List[str],
                 numerical_features: List[str]):
        """
        Inicializa el ingeniero de features.
        
        Args:
            data: DataFrame con los datos
            categorical_features: Lista de columnas categóricas
            numerical_features: Lista de columnas numéricas
        """
        self.data = data.copy()
        self.categorical_features = categorical_features.copy()
        self.numerical_features = numerical_features.copy()
        
        # Transformadores
        self.encoder: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        # Metadata de features derivadas
        self.derived_features_info: Dict = {}
        
        # Mapeo de categorías raras
        self.rare_categories_mapping: Dict = {}
        
        print(f"✅ FeatureEngineer inicializado")
        print(f"   Categóricas: {len(self.categorical_features)}")
        print(f"   Numéricas: {len(self.numerical_features)}")
        print(f"   Total features: {len(self.categorical_features) + len(self.numerical_features)}")
    
    def create_derived_features(self) -> pd.DataFrame:
        """
        Crea features derivadas del dataset.
        
        Features creadas:
        1. DURACION_ESTUDIOS: años entre primera matrícula y egreso
        2. ANNIO_EGRESO_NORM: año normalizado (0-4 para 2020-2024)
        3. CREDITOS_POR_ANNIO: créditos acumulados / duración
        4. ES_POSTGRADO: binaria (1 si es posgrado, 0 si no)
        5. SEXO_BIN: binaria (1 si es femenino, 0 si es masculino)
        6. TIEMPO_TRAMITE: años entre egreso y solicitud de trámite
        7. ES_REGULAR: binaria (1 si tipo de estudio es regular)
        
        Returns:
            DataFrame con features adicionales
        """
        print_progress("Creando features derivadas...", "⚙️")
        
        df = self.data.copy()
        features_created = []
        
        # 1. Duración de estudios (años)
        if 'ANNIO_EGRESO' in df.columns and 'ANNIO_MATRICULA1' in df.columns:
            df['DURACION_ESTUDIOS'] = df['ANNIO_EGRESO'] - df['ANNIO_MATRICULA1']
            # Limpiar valores negativos o muy altos (>15 años es poco común)
            df['DURACION_ESTUDIOS'] = df['DURACION_ESTUDIOS'].clip(0, 15)
            features_created.append('DURACION_ESTUDIOS')
            self.numerical_features.append('DURACION_ESTUDIOS')
            self.derived_features_info['DURACION_ESTUDIOS'] = {
                'tipo': 'numérica',
                'descripción': 'Años entre primera matrícula y egreso',
                'rango': [0, 15]
            }
        
        # 2. Año de egreso normalizado (2020=0, 2024=4)
        if 'ANNIO_EGRESO' in df.columns:
            df['ANNIO_EGRESO_NORM'] = df['ANNIO_EGRESO'] - 2020
            features_created.append('ANNIO_EGRESO_NORM')
            self.numerical_features.append('ANNIO_EGRESO_NORM')
            self.derived_features_info['ANNIO_EGRESO_NORM'] = {
                'tipo': 'numérica',
                'descripción': 'Año de egreso normalizado (2020=0)',
                'rango': [0, 4]
            }
        
        # 3. Créditos por año de estudio
        if 'CREDITOS_ACUMULADOS' in df.columns and 'DURACION_ESTUDIOS' in df.columns:
            # Evitar división por cero sumando 1
            df['CREDITOS_POR_ANNIO'] = df['CREDITOS_ACUMULADOS'] / (df['DURACION_ESTUDIOS'] + 1)
            # Limpiar infinitos
            df['CREDITOS_POR_ANNIO'] = df['CREDITOS_POR_ANNIO'].replace([np.inf, -np.inf], 0)
            features_created.append('CREDITOS_POR_ANNIO')
            self.numerical_features.append('CREDITOS_POR_ANNIO')
            self.derived_features_info['CREDITOS_POR_ANNIO'] = {
                'tipo': 'numérica',
                'descripción': 'Promedio de créditos por año de estudio',
                'cálculo': 'CREDITOS_ACUMULADOS / (DURACION_ESTUDIOS + 1)'
            }
        
        # 4. Feature binaria: es posgrado
        if 'NIVEL_ACADEMICO' in df.columns:
            df['ES_POSTGRADO'] = (df['NIVEL_ACADEMICO'] == 'POSGRADO').astype(int)
            features_created.append('ES_POSTGRADO')
            self.numerical_features.append('ES_POSTGRADO')
            self.derived_features_info['ES_POSTGRADO'] = {
                'tipo': 'binaria',
                'descripción': '1 si es posgrado, 0 si no',
                'valores': [0, 1]
            }
        
        # 5. Feature binaria: sexo (F=1, M=0)
        if 'SEXO' in df.columns:
            df['SEXO_BIN'] = (df['SEXO'] == 'F').astype(int)
            features_created.append('SEXO_BIN')
            self.numerical_features.append('SEXO_BIN')
            self.derived_features_info['SEXO_BIN'] = {
                'tipo': 'binaria',
                'descripción': '1 si es femenino, 0 si es masculino',
                'valores': [0, 1]
            }
        
        # 6. Tiempo entre egreso y solicitud de trámite
        if 'ANNIO_SOLICITUD_TRAMITE' in df.columns and 'ANNIO_EGRESO' in df.columns:
            df['TIEMPO_TRAMITE'] = df['ANNIO_SOLICITUD_TRAMITE'] - df['ANNIO_EGRESO']
            df['TIEMPO_TRAMITE'] = df['TIEMPO_TRAMITE'].clip(0, 10)
            features_created.append('TIEMPO_TRAMITE')
            self.numerical_features.append('TIEMPO_TRAMITE')
            self.derived_features_info['TIEMPO_TRAMITE'] = {
                'tipo': 'numérica',
                'descripción': 'Años entre egreso y solicitud de trámite',
                'rango': [0, 10]
            }
        
        # 7. Feature binaria: es regular
        if 'DETALLE_TIPO_ESTUDIO' in df.columns:
            df['ES_REGULAR'] = (df['DETALLE_TIPO_ESTUDIO'] == 'REGULAR').astype(int)
            features_created.append('ES_REGULAR')
            self.numerical_features.append('ES_REGULAR')
            self.derived_features_info['ES_REGULAR'] = {
                'tipo': 'binaria',
                'descripción': '1 si tipo de estudio es regular, 0 si no',
                'valores': [0, 1]
            }
        
        print(f"\n✅ Creadas {len(features_created)} features derivadas:")
        for i, feat in enumerate(features_created, 1):
            info = self.derived_features_info[feat]
            print(f"   {i}. {feat} ({info['tipo']})")
            print(f"      └─ {info['descripción']}")
        
        self.data = df
        return df
    
    def group_rare_categories(self, threshold: int = 50, 
                             exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Agrupa categorías poco frecuentes en 'OTROS'.
        
        Args:
            threshold: Frecuencia mínima para mantener categoría
            exclude_columns: Columnas a excluir de la agrupación
        
        Returns:
            DataFrame con categorías agrupadas
        """
        print_progress(f"Agrupando categorías raras (threshold={threshold})...", "🔄")
        
        df = self.data.copy()
        exclude_columns = exclude_columns or []
        
        total_grouped = 0
        
        for col in self.categorical_features:
            if col not in df.columns or col in exclude_columns:
                continue
            
            value_counts = df[col].value_counts()
            rare_categories = value_counts[value_counts < threshold].index.tolist()
            
            if len(rare_categories) > 0:
                # Guardar mapeo para uso posterior
                self.rare_categories_mapping[col] = rare_categories
                
                # Reemplazar con 'OTROS'
                df[col] = df[col].replace(rare_categories, 'OTROS')
                
                total_grouped += len(rare_categories)
                print(f"   • {col}: {len(rare_categories)} categorías → OTROS")
                print(f"     Ahora tiene {df[col].nunique()} categorías únicas")
        
        print(f"\n✅ Agrupación completada:")
        print(f"   Total de categorías agrupadas: {total_grouped}")
        print(f"   Columnas afectadas: {len(self.rare_categories_mapping)}")
        
        self.data = df
        return df
    
    def encode_categorical(self, method: str = 'onehot',
                          drop_first: bool = True,
                          handle_unknown: str = 'ignore') -> pd.DataFrame:
        """
        Codifica variables categóricas.
        
        Args:
            method: Método de codificación ('onehot' o 'label')
            drop_first: Si True, elimina primera columna (evita multicolinealidad)
            handle_unknown: Cómo manejar categorías desconocidas ('ignore' o 'error')
        
        Returns:
            DataFrame con categóricas codificadas
        """
        print_progress(f"Codificando variables categóricas ({method})...", "🔤")
        
        df = self.data.copy()
        
        if method == 'onehot':
            # Verificar que las columnas categóricas existen
            cat_cols_present = [col for col in self.categorical_features if col in df.columns]
            
            if not cat_cols_present:
                print("   ⚠️  No hay columnas categóricas para codificar")
                return df
            
            # Configurar encoder
            self.encoder = OneHotEncoder(
                drop='first' if drop_first else None,
                sparse_output=False,
                handle_unknown=handle_unknown,
                dtype=np.float64
            )
            
            # Fit encoder
            encoded = self.encoder.fit_transform(df[cat_cols_present])
            
            # Obtener nombres de features
            feature_names = self.encoder.get_feature_names_out(cat_cols_present)
            
            # Crear DataFrame con features codificadas
            df_encoded = pd.DataFrame(
                encoded,
                columns=feature_names,
                index=df.index
            )
            
            print(f"\n   📊 Codificación One-Hot:")
            print(f"      Columnas originales: {len(cat_cols_present)}")
            print(f"      Columnas generadas: {encoded.shape[1]}")
            
            # Obtener features numéricas presentes
            num_cols_present = [col for col in self.numerical_features if col in df.columns]
            
            # Combinar numéricas con codificadas
            df_final = pd.concat([
                df[num_cols_present],
                df_encoded
            ], axis=1)
            
            print(f"      Total features final: {df_final.shape[1]}")
            
            self.data = df_final
            self.feature_names = list(df_final.columns)
            
        elif method == 'label':
            # Label encoding simple (solo para árboles de decisión)
            from sklearn.preprocessing import LabelEncoder
            
            df_encoded = df.copy()
            for col in self.categorical_features:
                if col in df_encoded.columns:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    print(f"   • {col}: codificado con valores 0-{df_encoded[col].max()}")
            
            self.data = df_encoded
            self.feature_names = list(df_encoded.columns)
        
        else:
            raise ValueError(f"Método no válido: {method}")
        
        return self.data
    
    def scale_numerical(self, method: str = 'standard',
                       exclude_binary: bool = True) -> pd.DataFrame:
        """
        Escala variables numéricas.
        
        Args:
            method: Método de escalado ('standard' o 'minmax')
            exclude_binary: Si True, no escala variables binarias (0/1)
        
        Returns:
            DataFrame con numéricas escaladas
        """
        print_progress(f"Escalando variables numéricas ({method})...", "📏")
        
        df = self.data.copy()
        
        # Identificar columnas numéricas actuales
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            print("   ⚠️  No hay columnas numéricas para escalar")
            return df
        
        # Excluir binarias si se solicita
        if exclude_binary:
            binary_cols = []
            for col in numeric_cols:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    binary_cols.append(col)
            
            cols_to_scale = [col for col in numeric_cols if col not in binary_cols]
            
            if binary_cols:
                print(f"\n   ℹ️  Variables binarias excluidas del escalado: {len(binary_cols)}")
        else:
            cols_to_scale = numeric_cols
        
        if not cols_to_scale:
            print("   ⚠️  No hay columnas para escalar después de excluir binarias")
            return df
        
        # Configurar scaler
        if method == 'standard':
            self.scaler = StandardScaler()
        elif method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Método no válido: {method}")
        
        # Escalar
        scaled_data = self.scaler.fit_transform(df[cols_to_scale])
        
        # Crear DataFrame escalado
        df_scaled = df.copy()
        df_scaled[cols_to_scale] = scaled_data
        
        print(f"\n   ✅ Escalado completado:")
        print(f"      Variables escaladas: {len(cols_to_scale)}")
        if exclude_binary and binary_cols:
            print(f"      Variables binarias (sin escalar): {len(binary_cols)}")
        
        self.data = df_scaled
        self.feature_names = list(df_scaled.columns)
        
        return df_scaled
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Ejecuta pipeline completo de transformación (para train set).
        
        Pipeline:
        1. Crear features derivadas
        2. Agrupar categorías raras
        3. Codificar categóricas (One-Hot)
        4. Escalar numéricas (Standard)
        
        Args:
            X: DataFrame de entrada
        
        Returns:
            Array numpy transformado
        """
        print_section_header("🔧 PIPELINE DE FEATURE ENGINEERING")
        
        # Actualizar data
        self.data = X.copy()
        
        # 1. Features derivadas
        self.create_derived_features()
        
        # 2. Agrupar categorías raras
        self.group_rare_categories(threshold=50)
        
        # 3. Codificar categóricas
        self.encode_categorical(method='onehot', drop_first=True)
        
        # 4. Escalar numéricas
        self.scale_numerical(method='standard', exclude_binary=True)
        
        self.is_fitted = True
        
        print(f"\n{'='*70}")
        print(f"✅ PIPELINE COMPLETADO")
        print(f"{'='*70}")
        print(f"   Shape final: {self.data.shape}")
        print(f"   Total features: {len(self.feature_names)}")
        print(f"   Features derivadas: {len(self.derived_features_info)}")
        print(f"{'='*70}\n")
        
        return self.data.values
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforma datos usando transformadores ya ajustados (para val/test set).
        
        Args:
            X: DataFrame de entrada
        
        Returns:
            Array numpy transformado
        """
        if not self.is_fitted:
            raise ValueError("❌ Debe ejecutar fit_transform() primero en el train set")
        
        print_progress("Transformando nuevos datos...", "🔄")
        
        df = X.copy()
        
        # 1. Crear features derivadas (mismo proceso que en fit_transform)
        if 'ANNIO_EGRESO' in df.columns and 'ANNIO_MATRICULA1' in df.columns:
            df['DURACION_ESTUDIOS'] = (df['ANNIO_EGRESO'] - df['ANNIO_MATRICULA1']).clip(0, 15)
        
        if 'ANNIO_EGRESO' in df.columns:
            df['ANNIO_EGRESO_NORM'] = df['ANNIO_EGRESO'] - 2020
        
        if 'CREDITOS_ACUMULADOS' in df.columns and 'DURACION_ESTUDIOS' in df.columns:
            df['CREDITOS_POR_ANNIO'] = df['CREDITOS_ACUMULADOS'] / (df['DURACION_ESTUDIOS'] + 1)
            df['CREDITOS_POR_ANNIO'] = df['CREDITOS_POR_ANNIO'].replace([np.inf, -np.inf], 0)
        
        if 'NIVEL_ACADEMICO' in df.columns:
            df['ES_POSTGRADO'] = (df['NIVEL_ACADEMICO'] == 'POSGRADO').astype(int)
        
        if 'SEXO' in df.columns:
            df['SEXO_BIN'] = (df['SEXO'] == 'F').astype(int)
        
        if 'ANNIO_SOLICITUD_TRAMITE' in df.columns and 'ANNIO_EGRESO' in df.columns:
            df['TIEMPO_TRAMITE'] = (df['ANNIO_SOLICITUD_TRAMITE'] - df['ANNIO_EGRESO']).clip(0, 10)
        
        if 'DETALLE_TIPO_ESTUDIO' in df.columns:
            df['ES_REGULAR'] = (df['DETALLE_TIPO_ESTUDIO'] == 'REGULAR').astype(int)
        
        # 2. Agrupar categorías raras (usar mapeo guardado)
        for col, rare_cats in self.rare_categories_mapping.items():
            if col in df.columns:
                df[col] = df[col].replace(rare_cats, 'OTROS')
        
        # 3. Codificar categóricas (usando encoder ya fitted)
        if self.encoder is not None:
            cat_cols_present = [col for col in self.categorical_features if col in df.columns]
            if cat_cols_present:
                encoded = self.encoder.transform(df[cat_cols_present])
                feature_names = self.encoder.get_feature_names_out(cat_cols_present)
                df_encoded = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                
                num_cols_present = [col for col in self.numerical_features if col in df.columns]
                df = pd.concat([df[num_cols_present], df_encoded], axis=1)
        
        # 4. Escalar numéricas (usando scaler ya fitted)
        if self.scaler is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Excluir binarias
            binary_cols = []
            for col in numeric_cols:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    binary_cols.append(col)
            
            cols_to_scale = [col for col in numeric_cols if col not in binary_cols]
            
            if cols_to_scale:
                scaled_data = self.scaler.transform(df[cols_to_scale])
                df[cols_to_scale] = scaled_data
        
        print(f"   ✅ Datos transformados: {df.shape}")
        
        return df.values
    
    def get_feature_names(self) -> List[str]:
        """
        Obtiene nombres de features finales.
        
        Returns:
            Lista de nombres de features
        """
        return self.feature_names
    
    def get_derived_features_info(self) -> Dict:
        """
        Obtiene información de features derivadas.
        
        Returns:
            Diccionario con info de cada feature derivada
        """
        return self.derived_features_info
    
    def save_transformers(self, path: str) -> None:
        """
        Guarda encoder, scaler y metadata en archivos.
        
        Args:
            path: Directorio donde guardar
        """
        from pathlib import Path
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        print_progress("Guardando transformadores...", "💾")
        
        # Guardar encoder
        if self.encoder is not None:
            encoder_path = path / 'encoder.pkl'
            joblib.dump(self.encoder, encoder_path)
            print(f"   ✓ Encoder: {encoder_path}")
        
        # Guardar scaler
        if self.scaler is not None:
            scaler_path = path / 'scaler.pkl'
            joblib.dump(self.scaler, scaler_path)
            print(f"   ✓ Scaler: {scaler_path}")
        
        # Guardar feature names
        features_path = path / 'feature_names.pkl'
        joblib.dump(self.feature_names, features_path)
        print(f"   ✓ Feature names: {features_path}")
        
        # Guardar metadata
        metadata = {
            'categorical_features': self.categorical_features,
            'numerical_features': self.numerical_features,
            'feature_names': self.feature_names,
            'derived_features_info': self.derived_features_info,
            'rare_categories_mapping': self.rare_categories_mapping,
            'is_fitted': self.is_fitted,
            'total_features': len(self.feature_names)
        }
        
        metadata_path = path / 'feature_engineering_metadata.json'
        save_dict_to_json(metadata, str(metadata_path))
        print(f"   ✓ Metadata: {metadata_path}")
        
        print(f"\n✅ Transformadores guardados en: {path}")
    
    def load_transformers(self, path: str) -> None:
        """
        Carga encoder, scaler y metadata desde archivos.
        
        Args:
            path: Directorio de donde cargar
        """
        from pathlib import Path
        import json
        
        path = Path(path)
        print_progress("Cargando transformadores...", "📂")
        
        # Cargar encoder
        encoder_path = path / 'encoder.pkl'
        if encoder_path.exists():
            self.encoder = joblib.load(encoder_path)
            print(f"   ✓ Encoder cargado")
        
        # Cargar scaler
        scaler_path = path / 'scaler.pkl'
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"   ✓ Scaler cargado")
        
        # Cargar feature names
        features_path = path / 'feature_names.pkl'
        if features_path.exists():
            self.feature_names = joblib.load(features_path)
            print(f"   ✓ Feature names cargados")
        
        # Cargar metadata
        metadata_path = path / 'feature_engineering_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.categorical_features = metadata.get('categorical_features', [])
            self.numerical_features = metadata.get('numerical_features', [])
            self.derived_features_info = metadata.get('derived_features_info', {})
            self.rare_categories_mapping = metadata.get('rare_categories_mapping', {})
            self.is_fitted = metadata.get('is_fitted', False)
            
            print(f"   ✓ Metadata cargado")
        
        print(f"\n✅ Transformadores cargados desde: {path}")
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        return f"FeatureEngineer(features={len(self.feature_names)}, fitted={self.is_fitted})"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de FeatureEngineer
    """
    import sys
    sys.path.append('../..')
    
    from src.data.data_loader import DataLoader
    from src.data.data_preprocessor import DataPreprocessor
    
    print_section_header("🧪 TESTING FEATUREENGINEER CLASS")
    
    try:
        # 1. Cargar y preprocesar
        print("📂 Paso 1: Cargar y preprocesar datos")
        loader = DataLoader('../../data/raw/EGRESADOSUNE20202024.csv')
        df = loader.load_csv()
        
        exclude_cols = ['FECHA_CORTE', 'UUID', 'FECHA_EGRESO', 'SITUACION_ALUMNO', 'MODALIDAD']
        preprocessor = DataPreprocessor(df, 'PROMEDIO_FINAL', exclude_cols)
        preprocessor.identify_feature_types()
        preprocessor.handle_missing_values()
        
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
            test_size=0.15,
            val_size=0.15,
            stratify_column='NIVEL_ACADEMICO'
        )
        
        # 2. Feature Engineering en Train
        print("\n⚙️  Paso 2: Feature Engineering en Train Set")
        engineer = FeatureEngineer(
            data=X_train,
            categorical_features=preprocessor.categorical_features,
            numerical_features=preprocessor.numerical_features
        )
        
        X_train_transformed = engineer.fit_transform(X_train)
        print(f"\n✅ Train transformado: {X_train_transformed.shape}")
        
        # 3. Transformar Val y Test
        print("\n🔄 Paso 3: Transformar Val y Test")
        X_val_transformed = engineer.transform(X_val)
        X_test_transformed = engineer.transform(X_test)
        print(f"✅ Val transformado: {X_val_transformed.shape}")
        print(f"✅ Test transformado: {X_test_transformed.shape}")
        
        # 4. Guardar transformadores
        print("\n💾 Paso 4: Guardar transformadores")
        engineer.save_transformers('../../models/')
        
        # 5. Mostrar info de features derivadas
        print("\n📋 Features derivadas creadas:")
        for name, info in engineer.get_derived_features_info().items():
            print(f"   • {name}: {info['descripción']}")
        
        print("\n✅ EJEMPLO COMPLETADO EXITOSAMENTE!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()