"""
Módulo: model_trainer.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: ModelTrainer
Responsabilidad: Entrenar modelos Ridge y Lasso con optimización de hiperparámetros
Principio POO: Single Responsibility + Polimorfismo
Autor: zonny
Fecha: Octubre 2025
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
import joblib
import time

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    format_number,
    measure_time
)

import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """
    Entrena modelos de regresión Ridge y Lasso con optimización.
    
    Características:
    - Entrena Ridge y Lasso con parámetros configurables
    - Optimiza hiperparámetros con Grid Search CV
    - Realiza validación cruzada exhaustiva
    - Extrae importancia de features (coeficientes)
    - Guarda y carga modelos con metadata
    - Tracking de tiempo de entrenamiento
    
    Atributos:
        model_type (str): Tipo de modelo ('ridge' o 'lasso')
        model: Instancia del modelo (Ridge o Lasso)
        params (dict): Hiperparámetros del modelo
        best_params (dict): Mejores hiperparámetros encontrados
        cv_scores (List[float]): Scores de validación cruzada
        training_time (float): Tiempo de entrenamiento en segundos
        grid_search_results (pd.DataFrame): Resultados completos del Grid Search
    
    Ejemplo de uso:
        >>> trainer = ModelTrainer(model_type='ridge')
        >>> trainer.train(X_train, y_train)
        >>> best_params = trainer.hyperparameter_tuning(X_train, y_train, param_grid)
        >>> cv_results = trainer.cross_validate(X_train, y_train, cv=5)
        >>> y_pred = trainer.predict(X_test)
        >>> importance = trainer.get_feature_importance(feature_names)
        >>> trainer.save_model('models/ridge_model.pkl')
    """
    
    def __init__(self, model_type: str = 'ridge', **kwargs):
        """
        Inicializa el entrenador de modelos.
        
        Args:
            model_type: Tipo de modelo ('ridge' o 'lasso')
            **kwargs: Parámetros adicionales para el modelo
                - alpha: Parámetro de regularización
                - fit_intercept: Si ajustar intercepto
                - max_iter: Iteraciones máximas
                - random_state: Semilla aleatoria
        
        Raises:
            ValueError: Si model_type no es válido
        """
        if model_type.lower() not in ['ridge', 'lasso']:
            raise ValueError(f"❌ Tipo de modelo no válido: {model_type}. Use 'ridge' o 'lasso'")
        
        self.model_type = model_type.lower()
        self.params = kwargs
        self.best_params: Optional[Dict] = None
        self.cv_scores: List[float] = []
        self.training_time: float = 0.0
        self.feature_names: Optional[List[str]] = None
        self.grid_search_results: Optional[pd.DataFrame] = None
        
        # Metadata del modelo
        self.metadata = {
            'model_type': self.model_type,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'trained': False
        }
        
        # Inicializar modelo
        self._initialize_model()
        
        print(f"✅ ModelTrainer inicializado")
        print(f"   Tipo: {self.model_type.upper()}")
        print(f"   Parámetros: {self.params}")
    
    def _initialize_model(self) -> None:
        """
        Inicializa el modelo según el tipo especificado.
        """
        # Parámetros por defecto
        default_params = {
            'alpha': 1.0,
            'fit_intercept': True,
            'max_iter': 1000,
            'random_state': 42
        }
        
        # Actualizar con parámetros proporcionados
        default_params.update(self.params)
        self.params = default_params
        
        # Crear modelo
        if self.model_type == 'ridge':
            self.model = Ridge(**self.params)
        else:  # lasso
            self.model = Lasso(**self.params)
    
    def set_hyperparameters(self, params: Dict) -> None:
        """
        Actualiza hiperparámetros del modelo.
        
        Args:
            params: Diccionario con nuevos parámetros
        """
        self.params.update(params)
        self._initialize_model()
        print(f"⚙️  Hiperparámetros actualizados:")
        for key, value in params.items():
            print(f"   • {key}: {value}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              verbose: bool = True, feature_names: Optional[List[str]] = None) -> None:
        """
        Entrena el modelo con los datos proporcionados.
        
        Args:
            X_train: Features de entrenamiento (array o DataFrame)
            y_train: Target de entrenamiento
            verbose: Si True, imprime información detallada
            feature_names: Nombres de las features (opcional)
        """
        if verbose:
            print_section_header(f"🤖 ENTRENANDO MODELO {self.model_type.upper()}")
            print(f"📊 Datos de entrenamiento:")
            print(f"   • Muestras: {X_train.shape[0]:,}")
            print(f"   • Features: {X_train.shape[1]}")
            print(f"\n⚙️  Configuración del modelo:")
            for key, value in self.params.items():
                print(f"   • {key}: {value}")
        
        # Guardar feature names
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Medir tiempo
        start_time = time.time()
        
        try:
            # Entrenar modelo
            self.model.fit(X_train, y_train)
            
            # Calcular tiempo
            self.training_time = time.time() - start_time
            
            # Actualizar metadata
            self.metadata['trained'] = True
            self.metadata['train_samples'] = X_train.shape[0]
            self.metadata['n_features'] = X_train.shape[1]
            self.metadata['training_time'] = self.training_time
            
            if verbose:
                print(f"\n{'='*70}")
                print(f"✅ ENTRENAMIENTO COMPLETADO")
                print(f"{'='*70}")
                print(f"   ⏱️  Tiempo: {self.training_time:.2f}s")
                print(f"   📊 Alpha final: {self.model.alpha}")
                print(f"   📈 Intercepto: {self.model.intercept_:.4f}")
                
                # Mostrar número de coeficientes no-cero (importante para Lasso)
                if self.model_type == 'lasso':
                    non_zero = np.sum(np.abs(self.model.coef_) > 1e-10)
                    print(f"   🎯 Coeficientes no-cero: {non_zero}/{len(self.model.coef_)}")
                print(f"{'='*70}\n")
            
        except Exception as e:
            print(f"❌ Error durante el entrenamiento: {str(e)}")
            raise
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                             param_grid: Optional[Dict] = None,
                             cv: int = 5,
                             scoring: str = 'neg_mean_squared_error',
                             n_jobs: int = -1,
                             verbose: int = 1) -> Dict:
        """
        Optimiza hiperparámetros usando Grid Search con validación cruzada.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            param_grid: Diccionario con grid de parámetros a probar
            cv: Número de folds para cross-validation
            scoring: Métrica a optimizar
            n_jobs: Número de cores a usar (-1 = todos)
            verbose: Nivel de verbosidad
        
        Returns:
            Diccionario con mejores parámetros encontrados
        """
        print_section_header(f"🔍 OPTIMIZACIÓN DE HIPERPARÁMETROS - {self.model_type.upper()}")
        
        print(f"⚙️  Configuración:")
        print(f"   • Cross-validation: {cv} folds")
        print(f"   • Métrica: {scoring}")
        print(f"   • Procesamiento paralelo: {'Sí' if n_jobs == -1 else f'{n_jobs} cores'}")
        
        # Param grid por defecto si no se proporciona
        if param_grid is None:
            if self.model_type == 'ridge':
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
                }
            else:  # lasso
                param_grid = {
                    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
                }
        
        print(f"\n📋 Grid de parámetros:")
        for param_name, param_values in param_grid.items():
            print(f"   • {param_name}: {param_values}")
        print(f"   Total de combinaciones: {np.prod([len(v) for v in param_grid.values()])}")
        
        # Grid Search
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=verbose,
            return_train_score=True
        )
        
        print(f"\n⏳ Ejecutando Grid Search...")
        start_time = time.time()
        
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        
        # Guardar mejores parámetros
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.grid_search_results = pd.DataFrame(grid_search.cv_results_)
        
        # Calcular estadísticas
        best_score = -grid_search.best_score_  # Convertir a MSE positivo
        best_rmse = np.sqrt(best_score)
        
        print(f"\n{'='*70}")
        print(f"✅ OPTIMIZACIÓN COMPLETADA")
        print(f"{'='*70}")
        print(f"   ⏱️  Tiempo total: {elapsed_time:.2f}s")
        print(f"   🎯 Mejores parámetros:")
        for param, value in self.best_params.items():
            print(f"      • {param}: {value}")
        print(f"   📊 Mejor MSE (CV): {best_score:.6f}")
        print(f"   📊 Mejor RMSE (CV): {best_rmse:.6f}")
        print(f"{'='*70}\n")
        
        # Actualizar metadata
        self.metadata['best_params'] = self.best_params
        self.metadata['best_cv_score'] = float(best_score)
        self.metadata['tuning_time'] = elapsed_time
        self.metadata['trained'] = True
        
        return self.best_params
    
    def cross_validate(self, X_train: np.ndarray, y_train: np.ndarray,
                       cv: int = 5,
                       scoring: Optional[List[str]] = None,
                       return_train_score: bool = True) -> Dict:
        """
        Realiza validación cruzada exhaustiva del modelo.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            cv: Número de folds
            scoring: Lista de métricas a evaluar
            return_train_score: Si calcular scores en train
        
        Returns:
            Diccionario con resultados de CV
        """
        print_progress(f"Ejecutando validación cruzada ({cv} folds)...", "🔄")
        
        # Métricas por defecto
        if scoring is None:
            scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        
        # Realizar CV
        cv_results = cross_validate(
            self.model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=return_train_score
        )
        
        # Procesar resultados
        results = {}
        
        for metric in scoring:
            # Test scores
            test_scores = -cv_results[f'test_{metric}'] if 'neg_' in metric else cv_results[f'test_{metric}']
            results[f'{metric}_test_mean'] = test_scores.mean()
            results[f'{metric}_test_std'] = test_scores.std()
            
            # Train scores (si están disponibles)
            if return_train_score:
                train_scores = -cv_results[f'train_{metric}'] if 'neg_' in metric else cv_results[f'train_{metric}']
                results[f'{metric}_train_mean'] = train_scores.mean()
                results[f'{metric}_train_std'] = train_scores.std()
        
        # Guardar MSE scores
        mse_scores = -cv_results['test_neg_mean_squared_error']
        self.cv_scores = mse_scores.tolist()
        
        print(f"\n✅ Validación cruzada completada:")
        print(f"   • MSE: {results['neg_mean_squared_error_test_mean']:.6f} ± {results['neg_mean_squared_error_test_std']:.6f}")
        print(f"   • RMSE: {np.sqrt(results['neg_mean_squared_error_test_mean']):.6f}")
        print(f"   • MAE: {results['neg_mean_absolute_error_test_mean']:.6f} ± {results['neg_mean_absolute_error_test_std']:.6f}")
        print(f"   • R²: {results['r2_test_mean']:.6f} ± {results['r2_test_std']:.6f}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predicciones con el modelo entrenado.
        
        Args:
            X: Features para predicción
        
        Returns:
            Array con predicciones
        
        Raises:
            ValueError: Si el modelo no está entrenado
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError(
                "❌ El modelo no está entrenado. "
                "Ejecuta train() o hyperparameter_tuning() primero."
            )
        
        return self.model.predict(X)
    
    def get_coefficients(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Obtiene los coeficientes del modelo.
        
        Args:
            feature_names: Nombres de las features (opcional)
        
        Returns:
            DataFrame con coeficientes ordenados por importancia
        """
        if not hasattr(self.model, 'coef_'):
            raise ValueError("❌ El modelo no está entrenado")
        
        # Usar feature names guardados si no se proporcionan
        if feature_names is None:
            feature_names = self.feature_names
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.coef_))]
        
        # Crear DataFrame
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        # Ordenar por importancia absoluta
        coef_df = coef_df.sort_values('abs_coefficient', ascending=False).reset_index(drop=True)
        
        return coef_df
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None,
                               top_n: int = 20) -> pd.DataFrame:
        """
        Obtiene las features más importantes según coeficientes.
        
        Args:
            feature_names: Nombres de las features
            top_n: Número de features top a retornar
        
        Returns:
            DataFrame con top features
        """
        coef_df = self.get_coefficients(feature_names)
        
        print(f"\n📊 Top {top_n} Features más importantes ({self.model_type.upper()}):")
        print("=" * 80)
        print(f"{'Ranking':<8} {'Feature':<50} {'Coeficiente':>15}")
        print("-" * 80)
        
        top_features = coef_df.head(top_n)
        for idx, row in top_features.iterrows():
            feature_name = row['feature'][:47] + '...' if len(row['feature']) > 50 else row['feature']
            sign = '+' if row['coefficient'] > 0 else ''
            print(f"{idx+1:<8} {feature_name:<50} {sign}{row['coefficient']:>14.6f}")
        
        print("=" * 80 + "\n")
        
        return top_features
    
    def save_model(self, path: str) -> None:
        """
        Guarda el modelo entrenado con toda su metadata.
        
        Args:
            path: Ruta donde guardar el modelo (.pkl)
        """
        if not hasattr(self.model, 'coef_'):
            print("⚠️  Advertencia: Guardando modelo sin entrenar")
        
        # Preparar datos para guardar
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'best_params': self.best_params,
            'training_time': self.training_time,
            'cv_scores': self.cv_scores,
            'feature_names': self.feature_names,
            'metadata': self.metadata,
            'grid_search_results': self.grid_search_results
        }
        
        # Crear directorio si no existe
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar
        joblib.dump(model_data, path)
        
        print(f"💾 Modelo guardado exitosamente")
        print(f"   Ruta: {path}")
        print(f"   Tipo: {self.model_type.upper()}")
        print(f"   Alpha: {self.model.alpha}")
    
    def load_model(self, path: str) -> None:
        """
        Carga un modelo desde un archivo.
        
        Args:
            path: Ruta del modelo a cargar
        """
        print_progress(f"Cargando modelo desde {path}...", "📂")
        
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.params = model_data['params']
        self.best_params = model_data.get('best_params')
        self.training_time = model_data.get('training_time', 0.0)
        self.cv_scores = model_data.get('cv_scores', [])
        self.feature_names = model_data.get('feature_names')
        self.metadata = model_data.get('metadata', {})
        self.grid_search_results = model_data.get('grid_search_results')
        
        print(f"✅ Modelo cargado exitosamente")
        print(f"   Tipo: {self.model_type.upper()}")
        print(f"   Alpha: {self.model.alpha}")
        print(f"   Features: {len(self.model.coef_)}")
    
    def get_model_info(self) -> Dict:
        """
        Obtiene información completa del modelo.
        
        Returns:
            Diccionario con información del modelo
        """
        info = {
            'model_type': self.model_type,
            'alpha': self.model.alpha if hasattr(self.model, 'alpha') else None,
            'n_features': len(self.model.coef_) if hasattr(self.model, 'coef_') else 0,
            'intercept': float(self.model.intercept_) if hasattr(self.model, 'intercept_') else None,
            'training_time': self.training_time,
            'best_params': self.best_params,
            'cv_scores_mean': np.mean(self.cv_scores) if self.cv_scores else None,
            'cv_scores_std': np.std(self.cv_scores) if self.cv_scores else None,
            'is_trained': hasattr(self.model, 'coef_'),
            'metadata': self.metadata
        }
        
        # Para Lasso, agregar info de sparsity
        if self.model_type == 'lasso' and hasattr(self.model, 'coef_'):
            non_zero = np.sum(np.abs(self.model.coef_) > 1e-10)
            info['non_zero_coefs'] = int(non_zero)
            info['sparsity'] = float(non_zero / len(self.model.coef_))
        
        return info
    
    def print_model_summary(self):
        """Imprime resumen formateado del modelo"""
        info = self.get_model_info()
        
        print_section_header(f"📋 RESUMEN DEL MODELO - {self.model_type.upper()}")
        
        print(f"🤖 Tipo: {info['model_type'].upper()}")
        print(f"⚙️  Alpha: {info['alpha']}")
        print(f"📊 Features: {info['n_features']}")
        print(f"📈 Intercepto: {info['intercept']:.6f}" if info['intercept'] else "")
        
        if info['is_trained']:
            print(f"⏱️  Tiempo de entrenamiento: {info['training_time']:.2f}s")
            
            if info['cv_scores_mean']:
                print(f"🎯 MSE (CV): {info['cv_scores_mean']:.6f} ± {info['cv_scores_std']:.6f}")
                print(f"🎯 RMSE (CV): {np.sqrt(info['cv_scores_mean']):.6f}")
            
            if self.model_type == 'lasso':
                print(f"🎯 Coeficientes no-cero: {info['non_zero_coefs']}/{info['n_features']}")
                print(f"🎯 Sparsity: {info['sparsity']*100:.1f}%")
        else:
            print("⚠️  Modelo no entrenado")
        
        print("\n" + "="*70 + "\n")
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        trained = "trained" if hasattr(self.model, 'coef_') else "untrained"
        return f"ModelTrainer(type={self.model_type}, alpha={self.params.get('alpha')}, {trained})"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de ModelTrainer
    """
    from sklearn.datasets import make_regression
    
    print_section_header("🧪 TESTING MODELTRAINER CLASS")
    
    # Generar datos sintéticos
    print("📊 Generando datos sintéticos...")
    X_train, y_train = make_regression(
        n_samples=1000, 
        n_features=50, 
        n_informative=20,
        noise=10, 
        random_state=42
    )
    X_test, y_test = make_regression(
        n_samples=200, 
        n_features=50,
        n_informative=20,
        noise=10, 
        random_state=43
    )
    
    feature_names = [f'Feature_{i}' for i in range(50)]
    
    # ============================================
    # Ejemplo 1: Ridge con tuning
    # ============================================
    print("\n" + "="*70)
    print("EJEMPLO 1: RIDGE REGRESSION CON OPTIMIZACIÓN")
    print("="*70 + "\n")
    
    ridge_trainer = ModelTrainer(model_type='ridge')
    
    # Hyperparameter tuning
    param_grid = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
    best_params = ridge_trainer.hyperparameter_tuning(
        X_train, y_train, 
        param_grid=param_grid, 
        cv=5
    )
    
    # Cross-validation
    cv_results = ridge_trainer.cross_validate(X_train, y_train, cv=5)
    
    # Feature importance
    ridge_trainer.get_feature_importance(feature_names, top_n=10)
    
    # Predicción
    y_pred = ridge_trainer.predict(X_test)
    print(f"📊 Primeras 5 predicciones: {y_pred[:5]}")
    
    # Resumen
    ridge_trainer.print_model_summary()
    
    # Guardar
    ridge_trainer.save_model('../../models/ridge_example.pkl')
    
    # ============================================
    # Ejemplo 2: Lasso simple
    # ============================================
    print("\n" + "="*70)
    print("EJEMPLO 2: LASSO REGRESSION SIMPLE")
    print("="*70 + "\n")
    
    lasso_trainer = ModelTrainer(model_type='lasso', alpha=0.1)
    lasso_trainer.train(X_train, y_train, feature_names=feature_names)
    
    # Feature importance (Lasso hace selección automática)
    lasso_trainer.get_feature_importance(top_n=10)
    
    # Resumen
    lasso_trainer.print_model_summary()
    
    print("\n✅ TODOS LOS EJEMPLOS COMPLETADOS!")