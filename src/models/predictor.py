"""
Módulo: predictor.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: Predictor
Responsabilidad: Realizar predicciones en producción con modelos entrenados
Principio POO: Composición + Encapsulamiento
Autor: ZONNY
Fecha: Octubre 2025
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import joblib
import json

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    validate_target_range
)

import warnings
warnings.filterwarnings('ignore')


class Predictor:
    """
    Realiza predicciones con modelos entrenados en entorno de producción.
    
    Características:
    - Carga modelos y transformadores persistidos
    - Valida y preprocesa inputs automáticamente
    - Predicción única o por lotes (batch)
    - Intervalos de confianza estimados
    - Explicabilidad: contribución de cada feature
    - Comparación con promedios históricos
    - Manejo robusto de errores
    
    Atributos:
        model: Modelo de ML cargado (Ridge/Lasso)
        feature_engineer: Ingeniero de features cargado
        feature_names (List[str]): Nombres de features esperados
        model_type (str): Tipo de modelo ('ridge' o 'lasso')
        is_ready (bool): Indica si está listo para predecir
        metadata (Dict): Información del modelo cargado
    
    Ejemplo de uso:
        >>> predictor = Predictor()
        >>> predictor.load_model('models/ridge_model.pkl')
        >>> predictor.load_transformers('models/')
        >>> 
        >>> input_data = {
        >>>     'FACULTAD': 'CIENCIAS',
        >>>     'PROGRAMA_ESTUDIOS': 'MATEMÁTICA',
        >>>     'NIVEL_ACADEMICO': 'PREGRADO',
        >>>     'SEXO': 'F',
        >>>     'CREDITOS_ACUMULADOS': 216,
        >>>     'ANNIO_EGRESO': 2024,
        >>>     'ANNIO_MATRICULA1': 2020,
        >>>     # ... más features
        >>> }
        >>> 
        >>> resultado = predictor.predict_single(input_data)
        >>> print(f"Promedio predicho: {resultado['prediction']:.2f}")
        >>> print(f"Intervalo: [{resultado['lower_bound']:.2f}, {resultado['upper_bound']:.2f}]")
    """
    
    def __init__(self):
        """
        Inicializa el predictor.
        """
        self.model = None
        self.feature_engineer = None
        self.feature_names: List[str] = []
        self.model_type: Optional[str] = None
        self.is_ready: bool = False
        self.metadata: Dict = {}
        
        # Valores históricos (para comparación)
        self.historical_stats: Dict = {
            'mean': 15.96,
            'std': 1.15,
            'min': 12.23,
            'max': 19.30
        }
        
        print("✅ Predictor inicializado")
        print("   Listo para cargar modelo y transformadores")
    
    def load_model(self, model_path: str) -> None:
        """
        Carga un modelo entrenado desde archivo.
        
        Args:
            model_path: Ruta del archivo .pkl con el modelo
        
        Raises:
            FileNotFoundError: Si el modelo no existe
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"❌ Modelo no encontrado: {model_path}")
        
        print_progress(f"Cargando modelo desde {model_path.name}...", "📂")
        
        try:
            # Cargar modelo completo
            model_data = joblib.load(model_path)
            
            self.model = model_data['model']
            self.model_type = model_data['model_type']
            self.metadata = model_data.get('metadata', {})
            self.feature_names = model_data.get('feature_names', [])
            
            print(f"✅ Modelo cargado exitosamente")
            print(f"   Tipo: {self.model_type.upper()}")
            print(f"   Alpha: {self.model.alpha}")
            print(f"   Features: {len(self.model.coef_)}")
            
        except Exception as e:
            raise Exception(f"❌ Error al cargar modelo: {str(e)}")
    
    def load_transformers(self, path: str) -> None:
        """
        Carga transformadores de feature engineering.
        
        Args:
            path: Directorio con los transformadores
        """
        from src.features.feature_engineer import FeatureEngineer
        
        path = Path(path)
        
        print_progress("Cargando transformadores...", "📂")
        
        try:
            # Crear instancia temporal de FeatureEngineer
            self.feature_engineer = FeatureEngineer(
                data=pd.DataFrame(),
                categorical_features=[],
                numerical_features=[]
            )
            
            # Cargar transformadores
            self.feature_engineer.load_transformers(str(path))
            
            # Actualizar feature names
            if self.feature_engineer.feature_names:
                self.feature_names = self.feature_engineer.feature_names
            
            print("✅ Transformadores cargados exitosamente")
            
            # Verificar que está listo
            if self.model is not None and self.feature_engineer.is_fitted:
                self.is_ready = True
                print("✅ Predictor listo para realizar predicciones")
            
        except Exception as e:
            print(f"❌ Error al cargar transformadores: {str(e)}")
            raise
    
    def validate_input(self, input_data: Dict) -> Tuple[bool, List[str]]:
        """
        Valida que el input tenga todas las features necesarias.
        
        Args:
            input_data: Diccionario con datos de entrada
        
        Returns:
            Tuple (es_válido, lista_de_errores)
        """
        errors = []
        
        # Features categóricas requeridas
        required_categorical = [
            'FACULTAD', 'PROGRAMA_ESTUDIOS', 'SEDE', 
            'SEXO', 'NIVEL_ACADEMICO', 'DETALLE_TIPO_ESTUDIO'
        ]
        
        # Features numéricas requeridas
        required_numerical = [
            'ANNIO_EGRESO', 'CREDITOS_ACUMULADOS', 
            'ANNIO_MATRICULA1', 'ANNIO_SOLICITUD_TRAMITE', 
            'SEMESTRE_EGRESO'
        ]
        
        # Verificar categóricas
        for feature in required_categorical:
            if feature not in input_data:
                errors.append(f"Feature faltante: {feature}")
            elif input_data[feature] is None or input_data[feature] == '':
                errors.append(f"Feature vacío: {feature}")
        
        # Verificar numéricas
        for feature in required_numerical:
            if feature not in input_data:
                errors.append(f"Feature faltante: {feature}")
            elif input_data[feature] is None:
                errors.append(f"Feature vacío: {feature}")
            elif not isinstance(input_data[feature], (int, float)):
                errors.append(f"Feature {feature} debe ser numérico")
        
        # Validaciones de rango
        if 'ANNIO_EGRESO' in input_data:
            if input_data['ANNIO_EGRESO'] < 2020 or input_data['ANNIO_EGRESO'] > 2030:
                errors.append("ANNIO_EGRESO debe estar entre 2020 y 2030")
        
        if 'CREDITOS_ACUMULADOS' in input_data:
            if input_data['CREDITOS_ACUMULADOS'] < 0 or input_data['CREDITOS_ACUMULADOS'] > 300:
                errors.append("CREDITOS_ACUMULADOS debe estar entre 0 y 300")
        
        if 'ANNIO_MATRICULA1' in input_data and 'ANNIO_EGRESO' in input_data:
            if input_data['ANNIO_MATRICULA1'] > input_data['ANNIO_EGRESO']:
                errors.append("ANNIO_MATRICULA1 no puede ser mayor que ANNIO_EGRESO")
        
        is_valid = len(errors) == 0
        
        return is_valid, errors
    
    def _preprocess_input(self, input_data: Dict) -> np.ndarray:
        """
        Preprocesa input para predicción.
        
        Args:
            input_data: Diccionario con datos de entrada
        
        Returns:
            Array numpy procesado y listo para el modelo
        """
        # Convertir dict a DataFrame
        df = pd.DataFrame([input_data])
        
        # Aplicar transformaciones usando feature_engineer
        X_transformed = self.feature_engineer.transform(df)
        
        return X_transformed
    
    def predict_single(self, input_data: Dict, 
                      return_details: bool = True) -> Dict:
        """
        Realiza predicción para una sola instancia.
        
        Args:
            input_data: Diccionario con features de entrada
            return_details: Si retornar detalles adicionales
        
        Returns:
            Diccionario con predicción y detalles
        
        Raises:
            ValueError: Si el predictor no está listo o input es inválido
        """
        if not self.is_ready:
            raise ValueError(
                "❌ Predictor no está listo. "
                "Ejecuta load_model() y load_transformers() primero."
            )
        
        # Validar input
        is_valid, errors = self.validate_input(input_data)
        if not is_valid:
            raise ValueError(f"❌ Input inválido:\n" + "\n".join(f"  • {e}" for e in errors))
        
        try:
            # Preprocesar
            X = self._preprocess_input(input_data)
            
            # Predecir
            prediction = self.model.predict(X)[0]
            
            # Asegurar que está en rango válido
            prediction = np.clip(prediction, self.historical_stats['min'], 
                               self.historical_stats['max'])
            
            result = {
                'prediction': float(prediction),
                'input': input_data
            }
            
            # Agregar detalles si se solicita
            if return_details:
                # Intervalo de confianza (estimado usando std del modelo)
                std_error = self.historical_stats['std'] * 0.5  # Aproximación
                result['lower_bound'] = float(max(prediction - 1.96 * std_error, 
                                                 self.historical_stats['min']))
                result['upper_bound'] = float(min(prediction + 1.96 * std_error, 
                                                 self.historical_stats['max']))
                result['confidence_interval'] = '95%'
                
                # Comparación con promedio histórico
                diff_from_mean = prediction - self.historical_stats['mean']
                result['comparison'] = {
                    'historical_mean': self.historical_stats['mean'],
                    'difference': float(diff_from_mean),
                    'percentile': float(self._calculate_percentile(prediction)),
                    'interpretation': self._get_interpretation(prediction)
                }
                
                # Metadata del modelo
                result['model_info'] = {
                    'type': self.model_type,
                    'alpha': float(self.model.alpha)
                }
            
            return result
            
        except Exception as e:
            raise Exception(f"❌ Error durante la predicción: {str(e)}")
    
    def predict_batch(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza predicciones para múltiples instancias.
        
        Args:
            input_df: DataFrame con múltiples instancias
        
        Returns:
            DataFrame con predicciones agregadas
        """
        if not self.is_ready:
            raise ValueError(
                "❌ Predictor no está listo. "
                "Ejecuta load_model() y load_transformers() primero."
            )
        
        print_progress(f"Realizando predicciones batch ({len(input_df)} instancias)...", "🔮")
        
        try:
            # Preprocesar todas las instancias
            X = self.feature_engineer.transform(input_df)
            
            # Predecir
            predictions = self.model.predict(X)
            
            # Asegurar rango válido
            predictions = np.clip(predictions, 
                                self.historical_stats['min'], 
                                self.historical_stats['max'])
            
            # Crear DataFrame de resultados
            results_df = input_df.copy()
            results_df['PROMEDIO_PREDICHO'] = predictions
            
            # Agregar intervalos de confianza
            std_error = self.historical_stats['std'] * 0.5
            results_df['LIMITE_INFERIOR'] = np.maximum(
                predictions - 1.96 * std_error, 
                self.historical_stats['min']
            )
            results_df['LIMITE_SUPERIOR'] = np.minimum(
                predictions + 1.96 * std_error, 
                self.historical_stats['max']
            )
            
            print(f"✅ Predicciones completadas")
            print(f"   Promedio predicho: {predictions.mean():.2f}")
            print(f"   Rango: [{predictions.min():.2f}, {predictions.max():.2f}]")
            
            return results_df
            
        except Exception as e:
            raise Exception(f"❌ Error durante predicción batch: {str(e)}")
    
    def get_feature_contribution(self, input_data: Dict, 
                                 top_n: int = 10) -> Dict:
        """
        Calcula la contribución de cada feature a la predicción.
        
        Args:
            input_data: Diccionario con features de entrada
            top_n: Número de features top a retornar
        
        Returns:
            Diccionario con contribuciones de features
        """
        if not self.is_ready:
            raise ValueError("❌ Predictor no está listo")
        
        try:
            # Preprocesar
            X = self._preprocess_input(input_data)
            
            # Obtener coeficientes del modelo
            coefs = self.model.coef_
            
            # Calcular contribuciones (coef * valor)
            contributions = X[0] * coefs
            
            # Crear DataFrame con contribuciones
            contrib_df = pd.DataFrame({
                'feature': self.feature_names[:len(contributions)],
                'value': X[0],
                'coefficient': coefs,
                'contribution': contributions,
                'abs_contribution': np.abs(contributions)
            })
            
            # Ordenar por contribución absoluta
            contrib_df = contrib_df.sort_values('abs_contribution', ascending=False)
            
            # Top N
            top_features = contrib_df.head(top_n)
            
            result = {
                'intercept': float(self.model.intercept_),
                'top_features': top_features.to_dict('records'),
                'total_contribution': float(contributions.sum()),
                'prediction': float(self.model.intercept_ + contributions.sum())
            }
            
            return result
            
        except Exception as e:
            raise Exception(f"❌ Error al calcular contribuciones: {str(e)}")
    
    def predict_with_explanation(self, input_data: Dict) -> Dict:
        """
        Predicción completa con explicación detallada.
        
        Args:
            input_data: Diccionario con features de entrada
        
        Returns:
            Diccionario con predicción y explicación completa
        """
        # Predicción básica
        result = self.predict_single(input_data, return_details=True)
        
        # Agregar contribuciones de features
        contributions = self.get_feature_contribution(input_data, top_n=10)
        result['feature_contributions'] = contributions
        
        # Agregar interpretación detallada
        result['detailed_explanation'] = self._generate_explanation(
            result['prediction'],
            contributions,
            input_data
        )
        
        return result
    
    def _calculate_percentile(self, value: float) -> float:
        """
        Calcula en qué percentil está el valor predicho.
        
        Args:
            value: Valor predicho
        
        Returns:
            Percentil (0-100)
        """
        from scipy import stats
        
        # Asumiendo distribución normal con stats históricos
        percentile = stats.norm.cdf(
            value, 
            loc=self.historical_stats['mean'], 
            scale=self.historical_stats['std']
        ) * 100
        
        return percentile
    
    def _get_interpretation(self, prediction: float) -> str:
        """
        Genera interpretación textual de la predicción.
        
        Args:
            prediction: Valor predicho
        
        Returns:
            String con interpretación
        """
        mean = self.historical_stats['mean']
        std = self.historical_stats['std']
        
        if prediction >= mean + std:
            return "Excelente - Por encima del promedio histórico"
        elif prediction >= mean:
            return "Bueno - Por encima del promedio"
        elif prediction >= mean - std:
            return "Regular - Cerca del promedio"
        else:
            return "Bajo - Por debajo del promedio histórico"
    
    def _generate_explanation(self, prediction: float, 
                             contributions: Dict, 
                             input_data: Dict) -> str:
        """
        Genera explicación en lenguaje natural.
        
        Args:
            prediction: Valor predicho
            contributions: Contribuciones de features
            input_data: Datos de entrada
        
        Returns:
            String con explicación detallada
        """
        explanation = f"Predicción: {prediction:.2f} puntos\n\n"
        
        explanation += "Factores principales que influyeron:\n"
        
        top_3 = contributions['top_features'][:3]
        for i, feat in enumerate(top_3, 1):
            impact = "positivamente" if feat['contribution'] > 0 else "negativamente"
            explanation += f"{i}. {feat['feature']}: influyó {impact} "
            explanation += f"(contribución: {feat['contribution']:.4f})\n"
        
        explanation += f"\nComparación:\n"
        explanation += f"• Promedio histórico UNE: {self.historical_stats['mean']:.2f}\n"
        diff = prediction - self.historical_stats['mean']
        explanation += f"• Tu predicción está {abs(diff):.2f} puntos "
        explanation += "por encima" if diff > 0 else "por debajo"
        explanation += " del promedio\n"
        
        return explanation
    
    def set_historical_stats(self, stats: Dict) -> None:
        """
        Actualiza estadísticas históricas para comparación.
        
        Args:
            stats: Diccionario con estadísticas (mean, std, min, max)
        """
        self.historical_stats.update(stats)
        print(f"✅ Estadísticas históricas actualizadas")
    
    def get_model_info(self) -> Dict:
        """
        Obtiene información completa del predictor.
        
        Returns:
            Diccionario con información del modelo
        """
        info = {
            'is_ready': self.is_ready,
            'model_type': self.model_type,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'metadata': self.metadata,
            'historical_stats': self.historical_stats,
        }
        
        if self.model is not None:
            info['model_params'] = {
                'alpha': float(self.model.alpha),
                'intercept': float(self.model.intercept_)
            }
        
        return info
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        status = "ready" if self.is_ready else "not ready"
        return f"Predictor(model={self.model_type}, {status})"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de Predictor
    """
    print_section_header("🧪 TESTING PREDICTOR CLASS")
    
    # Nota: Este ejemplo requiere modelos ya entrenados
    # Para probarlo realmente, primero entrena y guarda un modelo
    
    print("📝 Ejemplo de uso del Predictor:\n")
    
    print("# 1. Inicializar predictor")
    print("predictor = Predictor()\n")
    
    print("# 2. Cargar modelo y transformadores")
    print("predictor.load_model('models/ridge_model.pkl')")
    print("predictor.load_transformers('models/')\n")
    
    print("# 3. Preparar datos de entrada")
    print("input_data = {")
    print("    'FACULTAD': 'CIENCIAS',")
    print("    'PROGRAMA_ESTUDIOS': 'MATEMÁTICA E INFORMÁTICA',")
    print("    'NIVEL_ACADEMICO': 'PREGRADO',")
    print("    'SEDE': 'LA MOLINA',")
    print("    'SEXO': 'F',")
    print("    'DETALLE_TIPO_ESTUDIO': 'REGULAR',")
    print("    'ANNIO_EGRESO': 2024,")
    print("    'SEMESTRE_EGRESO': 20241,")
    print("    'CREDITOS_ACUMULADOS': 216,")
    print("    'ANNIO_MATRICULA1': 2020,")
    print("    'ANNIO_SOLICITUD_TRAMITE': 2024,")
    print("}\n")
    
    print("# 4. Realizar predicción")
    print("resultado = predictor.predict_single(input_data)\n")
    
    print("# 5. Ver resultados")
    print("print(f\"Promedio predicho: {resultado['prediction']:.2f}\")")
    print("print(f\"Intervalo 95%: [{resultado['lower_bound']:.2f}, {resultado['upper_bound']:.2f}]\")")
    print("print(f\"Interpretación: {resultado['comparison']['interpretation']}\")\n")
    
    print("# 6. Predicción con explicación")
    print("resultado_explicado = predictor.predict_with_explanation(input_data)")
    print("print(resultado_explicado['detailed_explanation'])\n")
    
    print("# 7. Predicción por lotes")
    print("df_batch = pd.DataFrame([input_data1, input_data2, input_data3])")
    print("resultados_batch = predictor.predict_batch(df_batch)")
    print("print(resultados_batch[['FACULTAD', 'PROMEDIO_PREDICHO']])\n")
    
    print("✅ Ejemplo de uso completado")
    print("\n💡 Para usar realmente el Predictor:")
    print("   1. Entrena un modelo con ModelTrainer")
    print("   2. Guarda el modelo con save_model()")
    print("   3. Guarda los transformadores con FeatureEngineer.save_transformers()")
    print("   4. Luego usa Predictor para hacer predicciones en producción")