"""
Módulo: model_comparator.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: ModelComparator
Responsabilidad: Comparar múltiples modelos y seleccionar el mejor
Principio POO: Composición - usa múltiples ModelEvaluators
Autor: ZONNY
Fecha: Octubre 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    format_metrics_table,
    save_dict_to_json,
    ensure_dir
)

import warnings
warnings.filterwarnings('ignore')


class ModelComparator:
    """
    Compara el rendimiento de múltiples modelos de regresión.
    
    Características:
    - Compara Ridge vs Lasso (o más modelos)
    - Genera tabla comparativa profesional
    - Visualiza comparaciones de métricas
    - Selecciona automáticamente el mejor modelo
    - Análisis estadístico de diferencias
    - Exporta resultados en múltiples formatos
    
    Atributos:
        models (Dict): Diccionario {nombre: modelo}
        evaluators (Dict): Diccionario {nombre: evaluator}
        predictions (Dict): Diccionario {nombre: y_pred}
        comparison_results (pd.DataFrame): Tabla comparativa
        best_model_name (str): Nombre del mejor modelo
        best_metric (str): Métrica usada para selección
    
    Ejemplo de uso:
        >>> comparator = ModelComparator()
        >>> comparator.add_model('Ridge', ridge_model, y_test, y_pred_ridge)
        >>> comparator.add_model('Lasso', lasso_model, y_test, y_pred_lasso)
        >>> comparison_df = comparator.compare_metrics()
        >>> best = comparator.get_best_model(metric='rmse')
        >>> comparator.plot_comparison(save_path='results/figures/comparison.png')
        >>> comparator.plot_metrics_radar()
        >>> comparator.save_comparison('results/metrics/')
    """
    
    def __init__(self):
        """
        Inicializa el comparador de modelos.
        """
        self.models: Dict = {}
        self.evaluators: Dict = {}
        self.predictions: Dict = {}
        self.y_true: Optional[np.ndarray] = None
        self.comparison_results: Optional[pd.DataFrame] = None
        self.best_model_name: Optional[str] = None
        self.best_metric: str = 'rmse'
        
        print("✅ ModelComparator inicializado")
        print("   Listo para comparar modelos Ridge vs Lasso")
    
    def add_model(self, name: str, model, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Agrega un modelo al comparador.
        
        Args:
            name: Nombre del modelo ('Ridge', 'Lasso', etc.)
            model: Modelo entrenado
            y_true: Valores reales del target
            y_pred: Predicciones del modelo
        """
        from src.models.model_evaluator import ModelEvaluator
        
        # Guardar y_true (debe ser el mismo para todos los modelos)
        if self.y_true is None:
            self.y_true = y_true
        elif not np.array_equal(self.y_true, y_true):
            print("⚠️  Advertencia: y_true difiere entre modelos")
        
        # Crear evaluador para este modelo
        evaluator = ModelEvaluator(model, y_true, y_pred, model_name=name)
        evaluator.calculate_metrics()
        
        # Guardar
        self.models[name] = model
        self.evaluators[name] = evaluator
        self.predictions[name] = y_pred
        
        print(f"✅ Modelo '{name}' agregado al comparador")
        print(f"   RMSE: {evaluator.metrics['RMSE']:.4f}")
    
    def compare_metrics(self) -> pd.DataFrame:
        """
        Genera tabla comparativa de métricas.
        
        Returns:
            DataFrame con comparación de métricas
        """
        if len(self.evaluators) == 0:
            raise ValueError("❌ No hay modelos para comparar. Use add_model() primero.")
        
        print_progress("Generando tabla comparativa de métricas...", "📊")
        
        # Recopilar métricas de todos los modelos
        comparison_data = []
        
        for model_name, evaluator in self.evaluators.items():
            metrics = evaluator.metrics.copy()
            metrics['Model'] = model_name
            comparison_data.append(metrics)
        
        # Crear DataFrame
        self.comparison_results = pd.DataFrame(comparison_data)
        
        # Reordenar columnas
        cols = ['Model', 'RMSE', 'MAE', 'R2', 'MAPE', 'MSE', 'Max_Error']
        self.comparison_results = self.comparison_results[cols]
        
        # Agregar ranking por métrica
        self.comparison_results['RMSE_Rank'] = self.comparison_results['RMSE'].rank()
        self.comparison_results['MAE_Rank'] = self.comparison_results['MAE'].rank()
        self.comparison_results['R2_Rank'] = self.comparison_results['R2'].rank(ascending=False)
        
        print("✅ Tabla comparativa generada")
        
        return self.comparison_results
    
    def print_comparison(self) -> None:
        """
        Imprime tabla comparativa formateada.
        """
        if self.comparison_results is None:
            self.compare_metrics()
        
        print_section_header("🔬 COMPARACIÓN DE MODELOS")
        
        print("📊 Métricas de Evaluación:")
        print(self.comparison_results[['Model', 'RMSE', 'MAE', 'R2', 'MAPE']].to_string(index=False))
        
        print("\n🏆 Rankings:")
        print(self.comparison_results[['Model', 'RMSE_Rank', 'MAE_Rank', 'R2_Rank']].to_string(index=False))
        
        # Determinar ganador por métrica
        print("\n🥇 Mejor modelo por métrica:")
        best_rmse = self.comparison_results.loc[self.comparison_results['RMSE'].idxmin(), 'Model']
        best_mae = self.comparison_results.loc[self.comparison_results['MAE'].idxmin(), 'Model']
        best_r2 = self.comparison_results.loc[self.comparison_results['R2'].idxmax(), 'Model']
        
        print(f"   • Mejor RMSE: {best_rmse} ({self.comparison_results['RMSE'].min():.4f})")
        print(f"   • Mejor MAE: {best_mae} ({self.comparison_results['MAE'].min():.4f})")
        print(f"   • Mejor R²: {best_r2} ({self.comparison_results['R2'].max():.4f})")
        
        print("\n" + "="*70 + "\n")
    
    def get_best_model(self, metric: str = 'rmse') -> str:
        """
        Selecciona el mejor modelo según una métrica.
        
        Args:
            metric: Métrica a usar ('rmse', 'mae', 'r2', 'mape')
        
        Returns:
            Nombre del mejor modelo
        """
        if self.comparison_results is None:
            self.compare_metrics()
        
        metric = metric.upper()
        
        if metric not in ['RMSE', 'MAE', 'R2', 'MAPE', 'MSE']:
            raise ValueError(f"Métrica no válida: {metric}")
        
        # Para R² queremos el máximo, para el resto el mínimo
        if metric == 'R2':
            best_idx = self.comparison_results[metric].idxmax()
        else:
            best_idx = self.comparison_results[metric].idxmin()
        
        self.best_model_name = self.comparison_results.loc[best_idx, 'Model']
        self.best_metric = metric
        
        best_value = self.comparison_results.loc[best_idx, metric]
        
        print(f"🏆 Mejor modelo según {metric}: {self.best_model_name}")
        print(f"   Valor: {best_value:.4f}")
        
        return self.best_model_name
    
    def plot_comparison(self, save_path: Optional[str] = None,
                       figsize: tuple = (12, 6),
                       show_plot: bool = True) -> None:
        """
        Gráfica comparativa de métricas principales.
        
        Args:
            save_path: Ruta para guardar la figura
            figsize: Tamaño de la figura
            show_plot: Si mostrar la gráfica
        """
        if self.comparison_results is None:
            self.compare_metrics()
        
        print_progress("Generando gráfica comparativa...", "📊")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        models = self.comparison_results['Model'].values
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(models)]
        
        # Subplot 1: RMSE
        axes[0].bar(models, self.comparison_results['RMSE'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('RMSE', fontsize=11, fontweight='bold')
        axes[0].set_title('Root Mean Squared Error', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for i, (model, rmse) in enumerate(zip(models, self.comparison_results['RMSE'])):
            axes[0].text(i, rmse, f'{rmse:.4f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 2: MAE
        axes[1].bar(models, self.comparison_results['MAE'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('MAE', fontsize=11, fontweight='bold')
        axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        for i, (model, mae) in enumerate(zip(models, self.comparison_results['MAE'])):
            axes[1].text(i, mae, f'{mae:.4f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Subplot 3: R²
        axes[2].bar(models, self.comparison_results['R2'], 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[2].set_ylabel('R²', fontsize=11, fontweight='bold')
        axes[2].set_title('R² Score', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].set_ylim([0, 1])
        
        for i, (model, r2) in enumerate(zip(models, self.comparison_results['R2'])):
            axes[2].text(i, r2, f'{r2:.4f}', 
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.suptitle('Comparación de Modelos - Ridge vs Lasso', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_metrics_radar(self, save_path: Optional[str] = None,
                          figsize: tuple = (10, 10),
                          show_plot: bool = True) -> None:
        """
        Gráfica de radar para comparación multidimensional.
        
        Args:
            save_path: Ruta para guardar
            figsize: Tamaño de figura
            show_plot: Si mostrar
        """
        if self.comparison_results is None:
            self.compare_metrics()
        
        print_progress("Generando gráfica de radar...", "📊")
        
        # Preparar datos (normalizar métricas)
        df = self.comparison_results.copy()
        
        # Normalizar: para RMSE, MAE, MAPE más bajo es mejor (invertir)
        # Para R² más alto es mejor
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE']
        
        for metric in ['RMSE', 'MAE', 'MAPE']:
            if metric in df.columns:
                # Invertir y normalizar
                max_val = df[metric].max()
                min_val = df[metric].min()
                df[f'{metric}_norm'] = 1 - (df[metric] - min_val) / (max_val - min_val + 1e-10)
        
        # R² ya está normalizado, pero asegurar rango 0-1
        if 'R2' in df.columns:
            df['R2_norm'] = df['R2']
        
        # Crear figura
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Ángulos para cada métrica
        metrics_norm = ['RMSE_norm', 'MAE_norm', 'R2_norm', 'MAPE_norm']
        angles = np.linspace(0, 2 * np.pi, len(metrics_norm), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        # Colores para cada modelo
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        # Plot para cada modelo
        for idx, (model_name, color) in enumerate(zip(df['Model'], colors[:len(df)])):
            values = df.loc[df['Model'] == model_name, metrics_norm].values.flatten().tolist()
            values += values[:1]  # Cerrar el polígono
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        # Configurar ejes
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(['RMSE', 'MAE', 'R²', 'MAPE'], fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
        plt.title('Comparación Multidimensional de Modelos\n(Valores normalizados: 1.0 = mejor)',
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_predictions_comparison(self, save_path: Optional[str] = None,
                                   figsize: tuple = (14, 6),
                                   show_plot: bool = True) -> None:
        """
        Compara las predicciones de todos los modelos side-by-side.
        
        Args:
            save_path: Ruta para guardar
            figsize: Tamaño de figura
            show_plot: Si mostrar
        """
        print_progress("Generando comparación de predicciones...", "📊")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=figsize, sharey=True)
        
        if n_models == 1:
            axes = [axes]
        
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for idx, (model_name, y_pred) in enumerate(self.predictions.items()):
            ax = axes[idx]
            
            # Scatter plot
            ax.scatter(self.y_true, y_pred, alpha=0.5, s=30, 
                      edgecolors='k', linewidth=0.5, c=colors[idx])
            
            # Línea perfecta
            min_val = min(self.y_true.min(), y_pred.min())
            max_val = max(self.y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'r--', lw=2, label='Perfecta')
            
            # Métricas
            evaluator = self.evaluators[model_name]
            textstr = f"RMSE: {evaluator.metrics['RMSE']:.4f}\nR²: {evaluator.metrics['R2']:.4f}"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', bbox=props)
            
            ax.set_xlabel('Real', fontsize=11, fontweight='bold')
            if idx == 0:
                ax.set_ylabel('Predicho', fontsize=11, fontweight='bold')
            ax.set_title(model_name, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right')
        
        plt.suptitle('Comparación de Predicciones: Ridge vs Lasso', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def analyze_differences(self) -> Dict:
        """
        Analiza diferencias estadísticas entre modelos.
        
        Returns:
            Diccionario con análisis de diferencias
        """
        if len(self.models) < 2:
            print("⚠️  Se necesitan al menos 2 modelos para comparar")
            return {}
        
        print_progress("Analizando diferencias entre modelos...", "🔬")
        
        # Comparar predicciones
        model_names = list(self.predictions.keys())
        
        analysis = {
            'n_models': len(self.models),
            'models': model_names,
            'differences': {}
        }
        
        # Comparación por pares
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                pred1 = self.predictions[model1]
                pred2 = self.predictions[model2]
                
                # Calcular diferencias
                diff = pred1 - pred2
                
                comparison_key = f"{model1}_vs_{model2}"
                analysis['differences'][comparison_key] = {
                    'mean_diff': float(np.mean(diff)),
                    'std_diff': float(np.std(diff)),
                    'max_diff': float(np.max(np.abs(diff))),
                    'correlation': float(np.corrcoef(pred1, pred2)[0, 1]),
                    'agreement_within_0_5': int(np.sum(np.abs(diff) <= 0.5)),
                    'agreement_within_1_0': int(np.sum(np.abs(diff) <= 1.0)),
                }
        
        print("✅ Análisis de diferencias completado")
        
        return analysis
    
    def save_comparison(self, path: str) -> None:
        """
        Guarda resultados de comparación en múltiples formatos.
        
        Args:
            path: Directorio donde guardar los resultados
        """
        path = Path(path)
        ensure_dir(path)
        
        print_progress("Guardando resultados de comparación...", "💾")
        
        if self.comparison_results is None:
            self.compare_metrics()
        
        # 1. Guardar tabla comparativa (CSV)
        csv_path = path / 'model_comparison.csv'
        self.comparison_results.to_csv(csv_path, index=False)
        print(f"   ✓ CSV: {csv_path}")
        
        # 2. Guardar métricas detalladas (JSON)
        detailed_metrics = {}
        for model_name, evaluator in self.evaluators.items():
            detailed_metrics[model_name] = evaluator.metrics
        
        json_path = path / 'detailed_metrics.json'
        save_dict_to_json(detailed_metrics, str(json_path))
        print(f"   ✓ JSON: {json_path}")
        
        # 3. Guardar análisis de diferencias
        differences = self.analyze_differences()
        diff_path = path / 'model_differences.json'
        save_dict_to_json(differences, str(diff_path))
        print(f"   ✓ Diferencias: {diff_path}")
        
        # 4. Guardar mejor modelo
        if self.best_model_name:
            best_info = {
                'best_model': self.best_model_name,
                'metric_used': self.best_metric,
                'metrics': self.evaluators[self.best_model_name].metrics
            }
            best_path = path / 'best_model_info.json'
            save_dict_to_json(best_info, str(best_path))
            print(f"   ✓ Mejor modelo: {best_path}")
        
        print(f"\n✅ Resultados guardados en: {path}")
    
    def generate_comparison_report(self) -> Dict:
        """
        Genera reporte completo de comparación.
        
        Returns:
            Diccionario con reporte completo
        """
        if self.comparison_results is None:
            self.compare_metrics()
        
        report = {
            'n_models': len(self.models),
            'models': list(self.models.keys()),
            'comparison_table': self.comparison_results.to_dict('records'),
            'best_model': {
                'name': self.best_model_name,
                'metric': self.best_metric,
            },
            'differences': self.analyze_differences(),
        }
        
        # Agregar ganador por cada métrica
        report['winners'] = {
            'best_rmse': self.comparison_results.loc[self.comparison_results['RMSE'].idxmin(), 'Model'],
            'best_mae': self.comparison_results.loc[self.comparison_results['MAE'].idxmin(), 'Model'],
            'best_r2': self.comparison_results.loc[self.comparison_results['R2'].idxmax(), 'Model'],
        }
        
        return report
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        return f"ModelComparator(models={len(self.models)}, best={self.best_model_name})"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de ModelComparator
    """
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.model_selection import train_test_split
    
    print_section_header("🧪 TESTING MODELCOMPARATOR CLASS")
    
    # Generar datos
    print("📊 Generando datos sintéticos...")
    X, y = make_regression(n_samples=1000, n_features=20, noise=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar Ridge
    print("\n🤖 Entrenando Ridge...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred_ridge = ridge.predict(X_test)
    
    # Entrenar Lasso
    print("🤖 Entrenando Lasso...")
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train, y_train)
    y_pred_lasso = lasso.predict(X_test)
    
    # Comparar
    print("\n🔬 Comparando modelos...")
    comparator = ModelComparator()
    comparator.add_model('Ridge', ridge, y_test, y_pred_ridge)
    comparator.add_model('Lasso', lasso, y_test, y_pred_lasso)
    
    # Tabla comparativa
    comparator.compare_metrics()
    comparator.print_comparison()
    
    # Mejor modelo
    best = comparator.get_best_model(metric='rmse')
    
    # Visualizaciones
    print("\n📊 Generando visualizaciones...")
    comparator.plot_comparison(show_plot=False)
    comparator.plot_metrics_radar(show_plot=False)
    comparator.plot_predictions_comparison(show_plot=False)
    
    # Reporte
    report = comparator.generate_comparison_report()
    print(f"\n📋 Reporte generado con {report['n_models']} modelos")
    
    print("\n✅ EJEMPLO COMPLETADO!")