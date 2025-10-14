"""
Módulo: model_evaluator.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: ModelEvaluator
Responsabilidad: Evaluar rendimiento de modelos con métricas y visualizaciones
Principio POO: Single Responsibility
Autor: ZONNY
Fecha: Octubre 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    format_number,
    ensure_dir
)

import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Evalúa rendimiento de modelos de regresión con métricas y visualizaciones.
    
    Características:
    - Calcula métricas completas: RMSE, MAE, R², MAPE, MSE
    - Genera visualizaciones profesionales:
        * Predicted vs Actual (scatter plot)
        * Residuals plot (distribución y scatter)
        * Learning curves (training/validation)
        * Feature importance (barplot)
        * Error distribution
    - Reportes detallados de evaluación
    - Guardado automático de gráficas
    
    Atributos:
        model: Modelo entrenado
        y_true (np.ndarray): Valores reales
        y_pred (np.ndarray): Valores predichos
        metrics (dict): Métricas calculadas
        model_name (str): Nombre del modelo
        residuals (np.ndarray): Residuales calculados
    
    Ejemplo de uso:
        >>> evaluator = ModelEvaluator(model, y_test, y_pred, model_name='Ridge')
        >>> metrics = evaluator.calculate_metrics()
        >>> evaluator.print_metrics()
        >>> evaluator.plot_predictions_vs_actual(save_path='results/figures/pred_vs_actual.png')
        >>> evaluator.plot_residuals(save_path='results/figures/residuals.png')
        >>> evaluator.plot_learning_curves(X_train, y_train, save_path='results/figures/learning.png')
        >>> report = evaluator.generate_evaluation_report()
    """
    
    def __init__(self, model, y_true: np.ndarray, y_pred: np.ndarray,
                 model_name: str = 'Model'):
        """
        Inicializa el evaluador de modelos.
        
        Args:
            model: Modelo entrenado (Ridge/Lasso)
            y_true: Valores reales del target
            y_pred: Valores predichos por el modelo
            model_name: Nombre descriptivo del modelo
        """
        self.model = model
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.metrics: Dict = {}
        self.residuals = y_true - y_pred
        
        # Configurar estilo de gráficos
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        print(f"✅ ModelEvaluator inicializado")
        print(f"   Modelo: {self.model_name}")
        print(f"   Muestras: {len(y_true):,}")
    
    def _calculate_rmse(self) -> float:
        """Calcula Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(self.y_true, self.y_pred))
    
    def _calculate_mae(self) -> float:
        """Calcula Mean Absolute Error"""
        return mean_absolute_error(self.y_true, self.y_pred)
    
    def _calculate_r2(self) -> float:
        """Calcula R² Score"""
        return r2_score(self.y_true, self.y_pred)
    
    def _calculate_mape(self) -> float:
        """Calcula Mean Absolute Percentage Error"""
        # Evitar división por cero
        mask = self.y_true != 0
        if not np.any(mask):
            return np.nan
        return np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
    
    def _calculate_mse(self) -> float:
        """Calcula Mean Squared Error"""
        return mean_squared_error(self.y_true, self.y_pred)
    
    def calculate_metrics(self) -> Dict:
        """
        Calcula todas las métricas de evaluación.
        
        Returns:
            Diccionario con todas las métricas calculadas
        """
        print_progress("Calculando métricas de evaluación...", "📊")
        
        self.metrics = {
            'RMSE': self._calculate_rmse(),
            'MAE': self._calculate_mae(),
            'R2': self._calculate_r2(),
            'MAPE': self._calculate_mape(),
            'MSE': self._calculate_mse(),
        }
        
        # Métricas adicionales
        self.metrics['Max_Error'] = np.max(np.abs(self.residuals))
        self.metrics['Mean_Residual'] = np.mean(self.residuals)
        self.metrics['Std_Residual'] = np.std(self.residuals)
        
        print("✅ Métricas calculadas")
        
        return self.metrics
    
    def print_metrics(self) -> None:
        """
        Imprime las métricas de evaluación de forma profesional y formateada.
        """
        if not self.metrics:
            self.calculate_metrics()
        
        print_section_header(f"📈 MÉTRICAS DE EVALUACIÓN - {self.model_name.upper()}")
        
        print("🎯 Métricas Principales:")
        print(f"   {'Métrica':<25} {'Valor':>15} {'Interpretación'}")
        print("   " + "-"*70)
        print(f"   {'RMSE':<25} {self.metrics['RMSE']:>15.4f}   Error cuadrático medio")
        print(f"   {'MAE':<25} {self.metrics['MAE']:>15.4f}   Error absoluto medio")
        print(f"   {'R²':<25} {self.metrics['R2']:>15.4f}   Varianza explicada")
        print(f"   {'MAPE':<25} {self.metrics['MAPE']:>14.2f}%   Error porcentual")
        print(f"   {'MSE':<25} {self.metrics['MSE']:>15.4f}   Error cuadrático")
        
        print("\n📊 Análisis de Residuales:")
        print(f"   {'Error Máximo':<25} {self.metrics['Max_Error']:>15.4f}")
        print(f"   {'Media de Residuales':<25} {self.metrics['Mean_Residual']:>15.4f}")
        print(f"   {'Desv. Est. Residuales':<25} {self.metrics['Std_Residual']:>15.4f}")
        
        # Interpretación automática
        print("\n💡 INTERPRETACIÓN:")
        r2 = self.metrics['R2']
        mae = self.metrics['MAE']
        
        if r2 > 0.8:
            print("   ✅ Excelente ajuste (R² > 0.8) - El modelo explica muy bien la variabilidad")
        elif r2 > 0.6:
            print("   ✔️  Buen ajuste (R² > 0.6) - El modelo tiene buen poder predictivo")
        elif r2 > 0.4:
            print("   ⚠️  Ajuste moderado (R² > 0.4) - El modelo captura algunas relaciones")
        else:
            print("   ❌ Ajuste pobre (R² < 0.4) - El modelo tiene bajo poder predictivo")
        
        print(f"   📊 Error promedio: ±{mae:.2f} puntos en el promedio final")
        print(f"   📊 El modelo predice con un error típico de {self.metrics['RMSE']:.2f} puntos")
        
        print("\n" + "="*70 + "\n")
    
    def plot_predictions_vs_actual(self, save_path: Optional[str] = None,
                                   figsize: tuple = (10, 8),
                                   show_plot: bool = True) -> None:
        """
        Gráfica profesional de valores predichos vs valores reales.
        
        Args:
            save_path: Ruta para guardar la figura
            figsize: Tamaño de la figura
            show_plot: Si mostrar la gráfica
        """
        print_progress("Generando gráfica: Predicted vs Actual...", "📊")
        
        if not self.metrics:
            self.calculate_metrics()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Scatter plot con transparencia y borde
        ax.scatter(self.y_true, self.y_pred, 
                  alpha=0.6, 
                  s=50, 
                  edgecolors='navy',
                  linewidth=0.5,
                  c='skyblue',
                  label='Predicciones')
        
        # Línea de perfecta predicción
        min_val = min(self.y_true.min(), self.y_pred.min())
        max_val = max(self.y_true.max(), self.y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', lw=2.5, label='Predicción Perfecta', alpha=0.8)
        
        # Intervalos de confianza (±1 punto)
        ax.fill_between([min_val, max_val], 
                       [min_val-1, max_val-1], 
                       [min_val+1, max_val+1],
                       alpha=0.2, color='green', label='±1 punto')
        
        # Etiquetas y título
        ax.set_xlabel('Promedio Final Real', fontsize=13, fontweight='bold')
        ax.set_ylabel('Promedio Final Predicho', fontsize=13, fontweight='bold')
        ax.set_title(f'Predicciones vs Valores Reales\n{self.model_name}', 
                    fontsize=15, fontweight='bold', pad=20)
        
        # Agregar métricas al gráfico
        textstr = (f"R² = {self.metrics['R2']:.4f}\n"
                  f"RMSE = {self.metrics['RMSE']:.4f}\n"
                  f"MAE = {self.metrics['MAE']:.4f}\n"
                  f"n = {len(self.y_true):,}")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.9, edgecolor='black', linewidth=1.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props, family='monospace')
        
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_residuals(self, save_path: Optional[str] = None,
                      figsize: tuple = (14, 6),
                      show_plot: bool = True) -> None:
        """
        Gráfica completa de análisis de residuales.
        
        Args:
            save_path: Ruta para guardar la figura
            figsize: Tamaño de la figura
            show_plot: Si mostrar la gráfica
        """
        print_progress("Generando gráfica: Análisis de Residuales...", "📊")
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Subplot 1: Residuals vs Predicted
        axes[0].scatter(self.y_pred, self.residuals, 
                       alpha=0.6, s=40, edgecolors='k', linewidth=0.5, c='coral')
        axes[0].axhline(y=0, color='red', linestyle='--', lw=2.5, label='Línea Cero')
        axes[0].axhline(y=1, color='green', linestyle=':', lw=1.5, alpha=0.7, label='±1 punto')
        axes[0].axhline(y=-1, color='green', linestyle=':', lw=1.5, alpha=0.7)
        axes[0].set_xlabel('Valores Predichos', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Residuales', fontsize=11, fontweight='bold')
        axes[0].set_title('Residuales vs Predicciones', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='best', fontsize=9)
        
        # Subplot 2: Histogram of residuals
        axes[1].hist(self.residuals, bins=40, edgecolor='black', alpha=0.7, color='steelblue')
        axes[1].axvline(x=0, color='red', linestyle='--', lw=2.5, label='Media')
        axes[1].axvline(x=np.mean(self.residuals), color='orange', linestyle='-', lw=2, label='Media Real')
        axes[1].set_xlabel('Residuales', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
        axes[1].set_title('Distribución de Residuales', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        axes[1].legend(loc='best', fontsize=9)
        
        # Estadísticas
        textstr = f"Media: {np.mean(self.residuals):.4f}\nStd: {np.std(self.residuals):.4f}\nMin: {np.min(self.residuals):.4f}\nMax: {np.max(self.residuals):.4f}"
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        axes[1].text(0.72, 0.97, textstr, transform=axes[1].transAxes, fontsize=9,
                    verticalalignment='top', bbox=props, family='monospace')
        
        # Subplot 3: Q-Q Plot (manual simple)
        from scipy import stats
        (osm, osr), (slope, intercept, r) = stats.probplot(self.residuals, dist="norm", plot=None)
        axes[2].scatter(osm, osr, alpha=0.6, s=40, edgecolors='k', linewidth=0.5, c='purple')
        axes[2].plot(osm, slope*osm + intercept, 'r--', lw=2, label='Línea Teórica')
        axes[2].set_xlabel('Cuantiles Teóricos', fontsize=11, fontweight='bold')
        axes[2].set_ylabel('Cuantiles Muestrales', fontsize=11, fontweight='bold')
        axes[2].set_title('Q-Q Plot (Normalidad)', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc='best', fontsize=9)
        
        plt.suptitle(f'Análisis de Residuales - {self.model_name}', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_learning_curves(self, X_train: np.ndarray, y_train: np.ndarray,
                            save_path: Optional[str] = None,
                            cv: int = 5,
                            figsize: tuple = (10, 6),
                            show_plot: bool = True) -> None:
        """
        Gráfica de curvas de aprendizaje para detectar overfitting/underfitting.
        
        Args:
            X_train: Features de entrenamiento
            y_train: Target de entrenamiento
            save_path: Ruta para guardar
            cv: Folds para CV
            figsize: Tamaño de figura
            show_plot: Si mostrar la gráfica
        """
        print_progress("Generando curvas de aprendizaje...", "📊")
        
        train_sizes, train_scores, val_scores = learning_curve(
            self.model, X_train, y_train,
            cv=cv,
            n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='neg_mean_squared_error',
            shuffle=True,
            random_state=42
        )
        
        # Convertir a RMSE (positivo)
        train_scores_mean = np.sqrt(-train_scores.mean(axis=1))
        train_scores_std = np.sqrt(train_scores.std(axis=1))
        val_scores_mean = np.sqrt(-val_scores.mean(axis=1))
        val_scores_std = np.sqrt(val_scores.std(axis=1))
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training scores
        ax.plot(train_sizes, train_scores_mean, 'o-', color='#2E86AB',
                label='Training RMSE', linewidth=2.5, markersize=8)
        ax.fill_between(train_sizes,
                        train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std,
                        alpha=0.2, color='#2E86AB')
        
        # Plot validation scores
        ax.plot(train_sizes, val_scores_mean, 'o-', color='#A23B72',
                label='Validation RMSE', linewidth=2.5, markersize=8)
        ax.fill_between(train_sizes,
                        val_scores_mean - val_scores_std,
                        val_scores_mean + val_scores_std,
                        alpha=0.2, color='#A23B72')
        
        ax.set_xlabel('Tamaño del Training Set', fontsize=12, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=12, fontweight='bold')
        ax.set_title(f'Curvas de Aprendizaje - {self.model_name}', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Agregar interpretación
        final_gap = val_scores_mean[-1] - train_scores_mean[-1]
        if final_gap < 0.3:
            diagnosis = "✅ Buen balance (bajo overfitting)"
        elif final_gap < 0.5:
            diagnosis = "⚠️  Leve overfitting"
        else:
            diagnosis = "❌ Overfitting significativo"
        
        textstr = f"Gap final: {final_gap:.4f}\n{diagnosis}"
        props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, feature_names: List[str],
                               save_path: Optional[str] = None,
                               top_n: int = 20,
                               figsize: tuple = (10, 8),
                               show_plot: bool = True) -> None:
        """
        Gráfica de importancia de features basada en coeficientes.
        
        Args:
            feature_names: Nombres de las features
            save_path: Ruta para guardar
            top_n: Número de features top a mostrar
            figsize: Tamaño de figura
            show_plot: Si mostrar la gráfica
        """
        print_progress("Generando gráfica: Feature Importance...", "📊")
        
        if not hasattr(self.model, 'coef_'):
            print("   ⚠️  Modelo no tiene coeficientes")
            return
        
        # Obtener coeficientes
        coef_df = pd.DataFrame({
            'feature': feature_names[:len(self.model.coef_)],
            'coefficient': self.model.coef_,
            'abs_coefficient': np.abs(self.model.coef_)
        })
        
        # Top features
        top_features = coef_df.nlargest(top_n, 'abs_coefficient')
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['#2ECC71' if c > 0 else '#E74C3C' for c in top_features['coefficient']]
        bars = ax.barh(range(len(top_features)), top_features['coefficient'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=9)
        ax.set_xlabel('Coeficiente', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Features más Importantes\n{self.model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Leyenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ECC71', alpha=0.8, label='Impacto Positivo (+)'),
            Patch(facecolor='#E74C3C', alpha=0.8, label='Impacto Negativo (-)')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def plot_error_distribution(self, save_path: Optional[str] = None,
                               figsize: tuple = (10, 6),
                               show_plot: bool = True) -> None:
        """
        Gráfica de distribución de errores por rangos.
        
        Args:
            save_path: Ruta para guardar
            figsize: Tamaño de figura
            show_plot: Si mostrar la gráfica
        """
        print_progress("Generando gráfica: Distribución de Errores...", "📊")
        
        abs_errors = np.abs(self.residuals)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Crear bins de error
        bins = [0, 0.5, 1.0, 1.5, 2.0, np.inf]
        labels = ['< 0.5', '0.5-1.0', '1.0-1.5', '1.5-2.0', '> 2.0']
        
        error_bins = pd.cut(abs_errors, bins=bins, labels=labels)
        error_counts = error_bins.value_counts().sort_index()
        error_pcts = (error_counts / len(abs_errors)) * 100
        
        # Barplot
        colors = ['#2ECC71', '#3498DB', '#F39C12', '#E67E22', '#E74C3C']
        bars = ax.bar(range(len(error_pcts)), error_pcts.values, 
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(range(len(error_pcts)))
        ax.set_xticklabels(labels, fontsize=11)
        ax.set_xlabel('Rango de Error Absoluto (puntos)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Porcentaje de Predicciones (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribución de Errores de Predicción\n{self.model_name}',
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Agregar valores en las barras
        for i, (bar, pct, count) in enumerate(zip(bars, error_pcts.values, error_counts.values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%\n({count})',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir(Path(save_path).parent)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"   💾 Guardado: {save_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()
    
    def generate_evaluation_report(self) -> Dict:
        """
        Genera reporte completo de evaluación.
        
        Returns:
            Diccionario con reporte exhaustivo
        """
        if not self.metrics:
            self.calculate_metrics()
        
        report = {
            'model_name': self.model_name,
            'n_samples': len(self.y_true),
            'metrics': self.metrics,
            'residuals_stats': {
                'mean': float(np.mean(self.residuals)),
                'std': float(np.std(self.residuals)),
                'min': float(np.min(self.residuals)),
                'max': float(np.max(self.residuals)),
                'median': float(np.median(self.residuals)),
                'q25': float(np.percentile(self.residuals, 25)),
                'q75': float(np.percentile(self.residuals, 75)),
            },
            'predictions_stats': {
                'min': float(self.y_pred.min()),
                'max': float(self.y_pred.max()),
                'mean': float(self.y_pred.mean()),
                'std': float(self.y_pred.std()),
            },
            'actual_stats': {
                'min': float(self.y_true.min()),
                'max': float(self.y_true.max()),
                'mean': float(self.y_true.mean()),
                'std': float(self.y_true.std()),
            },
            'error_distribution': {
                'within_0_5': int(np.sum(np.abs(self.residuals) <= 0.5)),
                'within_1_0': int(np.sum(np.abs(self.residuals) <= 1.0)),
                'within_1_5': int(np.sum(np.abs(self.residuals) <= 1.5)),
                'within_2_0': int(np.sum(np.abs(self.residuals) <= 2.0)),
            }
        }
        
        return report
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        return f"ModelEvaluator(model='{self.model_name}', samples={len(self.y_true):,})"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de ModelEvaluator
    """
    from sklearn.datasets import make_regression
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    print_section_header("🧪 TESTING MODELEVALUATOR CLASS")
    
    # Generar datos
    print("📊 Generando datos sintéticos...")
    X, y = make_regression(n_samples=1000, n_features=20, noise=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenar modelo
    print("🤖 Entrenando modelo Ridge...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluar
    evaluator = ModelEvaluator(model, y_test, y_pred, model_name='Ridge Example')
    
    # Métricas
    metrics = evaluator.calculate_metrics()
    evaluator.print_metrics()
    
    # Visualizaciones
    print("\n📊 Generando visualizaciones...")
    evaluator.plot_predictions_vs_actual(show_plot=False)
    evaluator.plot_residuals(show_plot=False)
    evaluator.plot_learning_curves(X_train, y_train, show_plot=False)
    
    feature_names = [f'Feature_{i}' for i in range(20)]
    evaluator.plot_feature_importance(feature_names, show_plot=False)
    evaluator.plot_error_distribution(show_plot=False)
    
    # Reporte
    report = evaluator.generate_evaluation_report()
    print("\n📋 Reporte generado:")
    print(f"   Métricas: {list(report['metrics'].keys())}")
    print(f"   Predicciones dentro de ±1 punto: {report['error_distribution']['within_1_0']}")
    
    print("\n✅ EJEMPLO COMPLETADO!")