"""
Módulo: app.py
Proyecto: Predicción Promedio Final - Egresados UNE
Descripción: Aplicación web Flask para el sistema de Machine Learning
Autor: Estudiante de Ingeniería Estadística e Informática
Fecha: Octubre 2025
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file, send_from_directory
import pandas as pd
import numpy as np
import sys
import json
import os
import base64
import zipfile
import threading
import pickle
from pathlib import Path
from io import BytesIO
from datetime import datetime

# Agregar src al path
sys.path.append(str(Path(__file__).resolve().parent))

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.model_comparator import ModelComparator
from src.models.predictor import Predictor

# Importar configuración
from src.utils.config import (
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RAW_DATA_FILE,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    STATIC_DIR,
    TEMPLATES_DIR,
    TARGET_COLUMN,
    CSV_ENCODING,
    CSV_OPTIONS,
    RIDGE_CONFIG,
    LASSO_CONFIG,
    RIDGE_MODEL_FILE,
    LASSO_MODEL_FILE,
    SCALER_FILE,
    RIDGE_METRICS_FILE,
    LASSO_METRICS_FILE,
    COMPARISON_FILE,
    FLASK_CONFIG,
    PROJECT_INFO,
    DATASET_INFO
)

# Importar helpers
from src.utils.helpers import (
    ensure_dir,
    save_dict_to_json,
    load_json,
    get_file_size,
    print_success,
    print_error,
    print_warning,
    print_info
)

import warnings
warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE FLASK
# ============================================

app = Flask(__name__)
app.config.update(FLASK_CONFIG)

# Variables globales
dataset_info = {}
predictor_ready = False
predictor = None
results_data = {}
models_info = {}
training_status = {
    'running': False,
    'progress': 0,
    'message': 'Esperando...',
    'completed': False,
    'error': None
}


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def crear_directorios():
    """Crea directorios necesarios para la aplicación"""
    try:
        ensure_dir(MODELS_DIR)
        ensure_dir(RESULTS_DIR)
        ensure_dir(FIGURES_DIR)
        ensure_dir(METRICS_DIR)
        ensure_dir(STATIC_DIR)
        ensure_dir(TEMPLATES_DIR)
        ensure_dir(STATIC_DIR / 'figures')
    except Exception as e:
        print_error(f"Error creando directorios: {e}")


def load_dataset_info():
    """Carga información básica del dataset"""
    global dataset_info
    
    try:
        raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
        
        # Buscar archivo limpio primero
        clean_path = RAW_DATA_DIR / RAW_DATA_FILE.replace('.csv', '_CLEAN.csv')
        if clean_path.exists():
            raw_data_path = clean_path
        
        if not raw_data_path.exists():
            print_warning(f"Dataset no encontrado: {raw_data_path}")
            return False
        
        loader = DataLoader(str(raw_data_path), encoding=CSV_ENCODING)
        df = loader.load_csv(**CSV_OPTIONS)
        
        dataset_info = {
            'nombre': DATASET_INFO['nombre'],
            'universidad': DATASET_INFO['universidad'],
            'periodo': DATASET_INFO['periodo'],
            'total_registros': len(df),
            'total_variables': len(df.columns),
            'promedio_mean': float(df[TARGET_COLUMN].mean()),
            'promedio_std': float(df[TARGET_COLUMN].std()),
            'promedio_min': float(df[TARGET_COLUMN].min()),
            'promedio_max': float(df[TARGET_COLUMN].max())
        }
        
        return True
        
    except Exception as e:
        print_error(f"Error cargando dataset info: {e}")
        return False


def load_results():
    """Carga resultados de entrenamiento si existen"""
    global results_data, models_info
    
    try:
        # Cargar métricas de Ridge
        ridge_metrics_path = METRICS_DIR / RIDGE_METRICS_FILE
        if ridge_metrics_path.exists():
            ridge_data = load_json(str(ridge_metrics_path))
            results_data['ridge_metrics'] = ridge_data
        
        # Cargar métricas de Lasso
        lasso_metrics_path = METRICS_DIR / LASSO_METRICS_FILE
        if lasso_metrics_path.exists():
            lasso_data = load_json(str(lasso_metrics_path))
            results_data['lasso_metrics'] = lasso_data
        
        # Cargar comparación
        comparison_path = METRICS_DIR / COMPARISON_FILE
        if comparison_path.exists():
            comp_df = pd.read_csv(comparison_path)
            results_data['comparison'] = comp_df.to_dict('records')
        
        if results_data:
            print_success("Resultados de entrenamiento cargados")
            return True
        else:
            print_info("No hay resultados disponibles")
            return False
            
    except Exception as e:
        print_warning(f"Error cargando resultados: {e}")
        return False


def initialize_predictor():
    """Inicializa el predictor si hay modelos disponibles"""
    global predictor_ready, predictor, models_info
    
    try:
        ridge_model_path = MODELS_DIR / RIDGE_MODEL_FILE
        lasso_model_path = MODELS_DIR / LASSO_MODEL_FILE
        scaler_path = MODELS_DIR / SCALER_FILE
        
        if not ridge_model_path.exists() and not lasso_model_path.exists():
            print_info("No hay modelos entrenados disponibles")
            predictor_ready = False
            return False
        
        # Crear un diccionario simple para almacenar modelos y scaler
        predictor = {
            'models': {},
            'scaler': None,
            'feature_names': None
        }
        
        # Cargar Ridge si existe
        if ridge_model_path.exists():
            with open(ridge_model_path, 'rb') as f:
                predictor['models']['ridge'] = pickle.load(f)
            models_info['ridge_available'] = True
            print_success("✅ Modelo Ridge cargado")
        
        # Cargar Lasso si existe
        if lasso_model_path.exists():
            with open(lasso_model_path, 'rb') as f:
                predictor['models']['lasso'] = pickle.load(f)
            models_info['lasso_available'] = True
            print_success("✅ Modelo Lasso cargado")
        
        # Cargar scaler si existe
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                predictor['scaler'] = pickle.load(f)
            print_success("✅ Scaler cargado")
        
        predictor_ready = True
        print_success("✅ Predictor listo para usar")
        return True
        
    except Exception as e:
        print_warning(f"Error inicializando predictor: {e}")
        predictor_ready = False
        return False
    
def train_models_background(models=['ridge', 'lasso']):
    """Ejecuta el entrenamiento en segundo plano"""
    global training_status, predictor_ready
    
    try:
        training_status['running'] = True
        training_status['progress'] = 0
        training_status['message'] = 'Iniciando entrenamiento...'
        training_status['error'] = None
        
        # 1. Cargar datos
        training_status['message'] = 'Cargando datos...'
        training_status['progress'] = 10
        
        raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
        clean_path = RAW_DATA_DIR / RAW_DATA_FILE.replace('.csv', '_CLEAN.csv')
        
        if clean_path.exists():
            raw_data_path = clean_path
        
        loader = DataLoader(str(raw_data_path), encoding=CSV_ENCODING)
        df = loader.load_csv(**CSV_OPTIONS)
        print_info(f"Datos cargados: {df.shape}")
        
        # 2. Preprocesamiento simple
        training_status['message'] = 'Preprocesando datos...'
        training_status['progress'] = 25
        
        # Limpieza básica
        df_clean = df.copy()
        df_clean = df_clean.drop_duplicates()
        df_clean = df_clean.dropna(subset=[TARGET_COLUMN])
        threshold = len(df_clean) * 0.3
        df_clean = df_clean.dropna(thresh=threshold, axis=1)
        
        print_info(f"Después de limpieza: {df_clean.shape}")
        
        # 3. Preparar datos para entrenamiento
        training_status['message'] = 'Preparando features...'
        training_status['progress'] = 40
        
        y = df_clean[TARGET_COLUMN].copy()
        X = df_clean.drop(columns=[TARGET_COLUMN])
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X_numeric = X[numeric_cols].copy()
        X_numeric = X_numeric.fillna(X_numeric.median())
        y = y.fillna(y.mean())
        
        print_info(f"Features preparadas: X={X_numeric.shape}, y={y.shape}")
        print_info(f"Columnas numéricas: {len(numeric_cols)}")
        
        if len(X_numeric) == 0 or len(y) == 0:
            raise ValueError("No hay datos suficientes después del preprocesamiento")
        
        if X_numeric.shape[1] == 0:
            raise ValueError("No hay features numéricas disponibles")
        
        # 4. División train/test
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric, y, test_size=0.2, random_state=42
        )
        
        print_info(f"Train: {X_train.shape}, Test: {X_test.shape}")
        
        # 5. Lista para almacenar resultados de comparación
        comparison_results = []
        
        # 6. Entrenar Ridge
        if 'ridge' in models:
            training_status['message'] = 'Entrenando Ridge Regression...'
            training_status['progress'] = 50
            
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Escalar datos
            scaler_ridge = StandardScaler()
            X_train_scaled = scaler_ridge.fit_transform(X_train)
            X_test_scaled = scaler_ridge.transform(X_test)
            
            # Entrenar modelo
            model_ridge = Ridge(**RIDGE_CONFIG)
            model_ridge.fit(X_train_scaled, y_train)
            
            # Predecir
            y_pred_train = model_ridge.predict(X_train_scaled)
            y_pred_test = model_ridge.predict(X_test_scaled)
            
            # Calcular métricas
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = float(mean_absolute_error(y_test, y_pred_test))
            r2 = float(r2_score(y_test, y_pred_test))
            mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)
            
            metrics_ridge = {
                'model_name': 'Ridge',
                'metrics': {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                },
                'train_score': float(r2_score(y_train, y_pred_train)),
                'test_score': r2
            }
            
            # Guardar modelo
            with open(MODELS_DIR / RIDGE_MODEL_FILE, 'wb') as f:
                pickle.dump(model_ridge, f)
            
            # Guardar scaler
            with open(MODELS_DIR / SCALER_FILE, 'wb') as f:
                pickle.dump(scaler_ridge, f)
            
            # Guardar métricas
            save_dict_to_json(metrics_ridge, str(METRICS_DIR / RIDGE_METRICS_FILE))
            
            # Agregar a resultados de comparación
            comparison_results.append({
                'Model': 'Ridge',
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            })
            
            print_success(f"✅ Ridge - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # 7. Entrenar Lasso
        if 'lasso' in models:
            training_status['message'] = 'Entrenando Lasso Regression...'
            training_status['progress'] = 70
            
            from sklearn.linear_model import Lasso
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            # Escalar datos
            scaler_lasso = StandardScaler()
            X_train_scaled = scaler_lasso.fit_transform(X_train)
            X_test_scaled = scaler_lasso.transform(X_test)
            
            # Entrenar modelo
            model_lasso = Lasso(**LASSO_CONFIG)
            model_lasso.fit(X_train_scaled, y_train)
            
            # Predecir
            y_pred_train = model_lasso.predict(X_train_scaled)
            y_pred_test = model_lasso.predict(X_test_scaled)
            
            # Calcular métricas
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
            mae = float(mean_absolute_error(y_test, y_pred_test))
            r2 = float(r2_score(y_test, y_pred_test))
            mape = float(np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100)
            
            metrics_lasso = {
                'model_name': 'Lasso',
                'metrics': {
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                },
                'train_score': float(r2_score(y_train, y_pred_train)),
                'test_score': r2
            }
            
            # Guardar modelo
            with open(MODELS_DIR / LASSO_MODEL_FILE, 'wb') as f:
                pickle.dump(model_lasso, f)
            
            # Guardar métricas
            save_dict_to_json(metrics_lasso, str(METRICS_DIR / LASSO_METRICS_FILE))
            
            # Agregar a resultados de comparación
            comparison_results.append({
                'Model': 'Lasso',
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            })
            
            print_success(f"✅ Lasso - R²: {r2:.4f}, RMSE: {rmse:.4f}")
        
        # 8. Guardar comparación
        training_status['message'] = 'Guardando comparación...'
        training_status['progress'] = 90
        
        if comparison_results:
            # Crear DataFrame de comparación
            comparison_df = pd.DataFrame(comparison_results)
            
            # Ordenar por RMSE (menor es mejor)
            comparison_df = comparison_df.sort_values('RMSE')
            
            # Guardar
            comparison_df.to_csv(METRICS_DIR / COMPARISON_FILE, index=False)
            print_success(f"✅ Comparación guardada: {len(comparison_results)} modelo(s)")
        
        # 9. Copiar figuras (si existen)
        training_status['message'] = 'Copiando visualizaciones...'
        training_status['progress'] = 95
        copy_figures_to_static()
        
        # 10. Recargar predictor
        training_status['message'] = 'Inicializando predictor...'
        training_status['progress'] = 98
        
        load_results()
        initialize_predictor()
        
        # 11. Finalizar
        training_status['completed'] = True
        training_status['progress'] = 100
        training_status['message'] = '¡Entrenamiento completado exitosamente!'
        
        print_success("=" * 70)
        print_success("🎉 ¡ENTRENAMIENTO COMPLETADO CON ÉXITO!")
        print_success("=" * 70)
        
        # Mostrar resumen de resultados
        if comparison_results:
            print("\n📊 RESUMEN DE RESULTADOS:")
            for result in comparison_results:
                print(f"   • {result['Model']}: R²={result['R2']:.4f}, RMSE={result['RMSE']:.4f}")
            print()
        
    except Exception as e:
        training_status['error'] = str(e)
        training_status['message'] = f'Error: {str(e)}'
        print_error(f"❌ Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        training_status['running'] = False
# ============================================
# RUTAS DE LA APLICACIÓN
# ============================================

@app.route('/')
def index():
    """Página principal"""
    return render_template(
        'index.html',
        dataset_info=dataset_info,
        models_info=models_info,
        predictor_ready=predictor_ready,
        project_info=PROJECT_INFO
    )


@app.route('/eda')
def eda():
    """Página de análisis exploratorio de datos"""
    if not dataset_info:
        flash('El dataset no está cargado. Verifica que el archivo CSV esté en data/raw/', 'warning')
        return redirect(url_for('index'))
    
    try:
        raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
        clean_path = RAW_DATA_DIR / RAW_DATA_FILE.replace('.csv', '_CLEAN.csv')
        
        if clean_path.exists():
            raw_data_path = clean_path
        
        loader = DataLoader(str(raw_data_path), encoding=CSV_ENCODING)
        df = loader.load_csv(**CSV_OPTIONS)
        
        # Estadísticas descriptivas
        stats = {
            'total_registros': len(df),
            'promedio_stats': df[TARGET_COLUMN].describe().to_dict(),
        }
        
        # Conteos por columnas categóricas
        if 'FACULTAD' in df.columns:
            stats['facultad_counts'] = df['FACULTAD'].value_counts().head(10).to_dict()
        
        if 'NIVEL_ACADEMICO' in df.columns:
            stats['nivel_academico_counts'] = df['NIVEL_ACADEMICO'].value_counts().to_dict()
        
        if 'SEDE' in df.columns:
            stats['sede_counts'] = df['SEDE'].value_counts().to_dict()
        
        if 'SEXO' in df.columns:
            stats['sexo_counts'] = df['SEXO'].value_counts().to_dict()
        
        # Correlaciones
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        for col in numeric_cols:
            if col != TARGET_COLUMN:
                try:
                    corr = df[col].corr(df[TARGET_COLUMN])
                    if not np.isnan(corr):
                        correlations[col] = float(corr)
                except:
                    pass
        
        correlations = dict(sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True))
        
        return render_template(
            'eda.html',
            dataset_info=dataset_info,
            stats=stats,
            correlations=correlations,
            project_info=PROJECT_INFO
        )
        
    except Exception as e:
        flash(f'Error al cargar datos: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/train')
def train():
    """Página de entrenamiento de modelos"""
    return render_template(
        'train.html',
        dataset_info=dataset_info,
        training_status=training_status,
        project_info=PROJECT_INFO
    )


@app.route('/api/train', methods=['POST'])
def api_train():
    """API para entrenar modelos"""
    global training_status
    
    try:
        # Verificar si ya hay entrenamiento en curso
        if training_status['running']:
            return jsonify({
                'success': False,
                'error': 'Ya hay un entrenamiento en curso'
            })
        
        data = request.get_json()
        models_to_train = data.get('models', ['ridge', 'lasso'])
        
        # Validar
        if not models_to_train:
            return jsonify({
                'success': False,
                'error': 'Debes seleccionar al menos un modelo'
            })
        
        # Iniciar entrenamiento en thread separado
        training_thread = threading.Thread(
            target=train_models_background,
            args=(models_to_train,)
        )
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Entrenamiento iniciado en segundo plano',
            'models': models_to_train
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/training-status')
def api_training_status():
    """API para obtener el estado del entrenamiento"""
    return jsonify(training_status)


@app.route('/results')
def results():
    """Página de resultados"""
    if not results_data:
        flash('No hay resultados disponibles. Por favor entrena los modelos primero.', 'warning')
        return redirect(url_for('train'))
    
    try:
        # Verificar figuras disponibles
        figures = {
            'comparison': (FIGURES_DIR / 'model_comparison.png').exists(),
            'ridge_pred_vs_actual': (FIGURES_DIR / 'ridge_predicted_vs_actual.png').exists(),
            'ridge_residuals': (FIGURES_DIR / 'ridge_residuals_plot.png').exists(),
            'ridge_importance': (FIGURES_DIR / 'ridge_feature_importance.png').exists(),
            'lasso_pred_vs_actual': (FIGURES_DIR / 'lasso_predicted_vs_actual.png').exists(),
            'lasso_residuals': (FIGURES_DIR / 'lasso_residuals_plot.png').exists(),
            'lasso_importance': (FIGURES_DIR / 'lasso_feature_importance.png').exists(),
        }
        
        return render_template(
            'results.html',
            results=results_data,
            figures=figures,
            project_info=PROJECT_INFO
        )
        
    except Exception as e:
        flash(f'Error al cargar resultados: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/predict')
def predict():
    """Página de predicción"""
    if not predictor_ready:
        return render_template(
            'predict.html',
            predictor_ready=False,
            project_info=PROJECT_INFO
        )
    
    try:
        # Cargar opciones del dataset
        raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
        clean_path = RAW_DATA_DIR / RAW_DATA_FILE.replace('.csv', '_CLEAN.csv')
        
        if clean_path.exists():
            raw_data_path = clean_path
        
        loader = DataLoader(str(raw_data_path), encoding=CSV_ENCODING)
        df = loader.load_csv(**CSV_OPTIONS)
        
        # Opciones para el formulario
        options = {
            'facultades': sorted(df['FACULTAD'].dropna().unique().tolist()) if 'FACULTAD' in df.columns else [],
            'programas': sorted(df['PROGRAMA_ESTUDIOS'].dropna().unique().tolist()) if 'PROGRAMA_ESTUDIOS' in df.columns else [],
            'sedes': sorted(df['SEDE'].dropna().unique().tolist()) if 'SEDE' in df.columns else [],
            'niveles': sorted(df['NIVEL_ACADEMICO'].dropna().unique().tolist()) if 'NIVEL_ACADEMICO' in df.columns else [],
            'tipos_estudio': sorted(df['DETALLE_TIPO_ESTUDIO'].dropna().unique().tolist()) if 'DETALLE_TIPO_ESTUDIO' in df.columns else [],
        }
        
        return render_template(
            'predict.html',
            predictor_ready=True,
            options=options,
            project_info=PROJECT_INFO
        )
        
    except Exception as e:
        flash(f'Error al cargar opciones: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/api/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API para realizar predicciones"""
    if not predictor_ready or predictor is None:
        return jsonify({
            'success': False,
            'error': 'El predictor no está disponible. Por favor entrena los modelos primero.'
        })
    
    try:
        data = request.get_json()
        
        # Cargar el dataset para obtener las columnas correctas y hacer encoding
        raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
        clean_path = RAW_DATA_DIR / RAW_DATA_FILE.replace('.csv', '_CLEAN.csv')
        
        if clean_path.exists():
            raw_data_path = clean_path
        
        loader = DataLoader(str(raw_data_path), encoding=CSV_ENCODING)
        df_reference = loader.load_csv(**CSV_OPTIONS)
        
        # Obtener solo las columnas numéricas que se usaron en el entrenamiento
        numeric_cols = df_reference.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remover la columna target si está presente
        if TARGET_COLUMN in numeric_cols:
            numeric_cols.remove(TARGET_COLUMN)
        
        # Crear DataFrame con valores del formulario
        input_data = {}
        
        # Mapear los datos del formulario a las columnas del dataset
        for col in numeric_cols:
            if col in data:
                # El dato viene directamente del formulario
                input_data[col] = float(data[col])
            else:
                # Usar la mediana de esa columna del dataset de referencia
                input_data[col] = float(df_reference[col].median())
        
        # Crear DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Asegurar el orden correcto de las columnas
        input_df = input_df[numeric_cols]
        
        print_info(f"Columnas para predicción: {list(input_df.columns)}")
        print_info(f"Valores de entrada: {input_df.iloc[0].to_dict()}")
        
        # Escalar datos
        if predictor['scaler'] is not None:
            input_scaled = predictor['scaler'].transform(input_df)
        else:
            input_scaled = input_df.values
        
        # Realizar predicción con Ridge (modelo por defecto)
        if 'ridge' in predictor['models']:
            model = predictor['models']['ridge']
            prediction = model.predict(input_scaled)[0]
            model_type = 'ridge'
        elif 'lasso' in predictor['models']:
            model = predictor['models']['lasso']
            prediction = model.predict(input_scaled)[0]
            model_type = 'lasso'
        else:
            return jsonify({
                'success': False,
                'error': 'No hay modelos disponibles'
            })
        
        # Asegurar que la predicción está en un rango válido (0-20)
        prediction = float(np.clip(prediction, 0, 20))
        
        # Calcular intervalos de confianza (aproximado)
        std_error = 0.5
        lower_bound = float(max(0, prediction - (1.96 * std_error)))
        upper_bound = float(min(20, prediction + (1.96 * std_error)))
        
        # Calcular percentil (aproximado basado en la distribución del dataset)
        mean_promedio = df_reference[TARGET_COLUMN].mean()
        std_promedio = df_reference[TARGET_COLUMN].std()
        
        # Calcular z-score y convertir a percentil
        from scipy import stats
        z_score = (prediction - mean_promedio) / std_promedio
        percentile = float(stats.norm.cdf(z_score) * 100)
        percentile = max(0, min(100, percentile))
        
        # Interpretación
        if prediction >= 16:
            interpretation = "Excelente rendimiento académico. El estudiante está en el tercio superior."
        elif prediction >= 14:
            interpretation = "Buen rendimiento académico. El estudiante está por encima del promedio."
        elif prediction >= 12:
            interpretation = "Rendimiento académico satisfactorio. El estudiante está cerca del promedio."
        elif prediction >= 10:
            interpretation = "Rendimiento académico aceptable. Hay oportunidades de mejora."
        else:
            interpretation = "Rendimiento académico por debajo del promedio. Se recomienda apoyo adicional."
        
        print_success(f"✅ Predicción exitosa: {prediction:.2f} (Modelo: {model_type})")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'percentile': percentile,
            'interpretation': interpretation,
            'model_type': model_type.upper()
        })
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print_error(f"Error en predicción: {e}")
        print(error_trace)
        
        return jsonify({
            'success': False,
            'error': f'Error al realizar la predicción: {str(e)}'
        })

@app.route('/visualizations')
def visualizations():
    """Página de visualizaciones"""
    try:
        # Verificar si hay figuras disponibles
        figures_available = FIGURES_DIR.exists() and len(list(FIGURES_DIR.glob('*.png'))) > 0
        
        if not figures_available:
            return render_template(
                'visualizations.html',
                figures_available=False,
                project_info=PROJECT_INFO
            )
        
        # Detectar figuras disponibles
        figures = {
            'correlation_matrix': (FIGURES_DIR / 'correlation_matrix.png').exists(),
            'target_distribution': (FIGURES_DIR / 'target_distribution.png').exists(),
            'boxplot_facultad': (FIGURES_DIR / 'boxplot_facultad.png').exists(),
            'ridge_pred_vs_actual': (FIGURES_DIR / 'ridge_predicted_vs_actual.png').exists(),
            'ridge_residuals': (FIGURES_DIR / 'ridge_residuals_plot.png').exists(),
            'ridge_importance': (FIGURES_DIR / 'ridge_feature_importance.png').exists(),
            'ridge_learning': (FIGURES_DIR / 'ridge_learning_curve.png').exists(),
            'lasso_pred_vs_actual': (FIGURES_DIR / 'lasso_predicted_vs_actual.png').exists(),
            'lasso_residuals': (FIGURES_DIR / 'lasso_residuals_plot.png').exists(),
            'lasso_importance': (FIGURES_DIR / 'lasso_feature_importance.png').exists(),
            'lasso_learning': (FIGURES_DIR / 'lasso_learning_curve.png').exists(),
            'comparison': (FIGURES_DIR / 'model_comparison.png').exists(),
        }
        
        # Cargar métricas si existen
        metrics = None
        comparison_path = METRICS_DIR / COMPARISON_FILE
        if comparison_path.exists():
            df_metrics = pd.read_csv(comparison_path)
            metrics = {}
            for _, row in df_metrics.iterrows():
                metrics[row['Model']] = {
                    'RMSE': row['RMSE'],
                    'MAE': row['MAE'],
                    'R2': row['R2'],
                    'best_rmse': row.get('Best_RMSE', False),
                    'best_r2': row.get('Best_R2', False)
                }
        
        return render_template(
            'visualizations.html',
            figures_available=True,
            figures=figures,
            metrics=metrics,
            project_info=PROJECT_INFO
        )
        
    except Exception as e:
        flash(f'Error al cargar visualizaciones: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/download_results')
def download_results():
    """Descargar todas las gráficas en un ZIP"""
    try:
        memory_file = BytesIO()
        
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for fig_path in FIGURES_DIR.glob('*.png'):
                zf.write(fig_path, fig_path.name)
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='visualizaciones_egresados_une.zip'
        )
    except Exception as e:
        flash(f'Error al descargar: {str(e)}', 'danger')
        return redirect(url_for('visualizations'))


# ============================================
# RUTAS ESTÁTICAS
# ============================================

@app.route('/figures/<path:filename>')
def serve_figure(filename):
    """Sirve imágenes de la carpeta results/figures"""
    return send_from_directory(str(FIGURES_DIR), filename)


@app.route('/favicon.ico')
def favicon():
    """Maneja el favicon"""
    return '', 204


# ============================================
# FILTROS DE PLANTILLAS
# ============================================

@app.template_filter('format_decimal')
def format_decimal_filter(value, decimals=2):
    """Formatea números decimales"""
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):.{decimals}f}"
    except:
        return str(value)


@app.template_filter('format_number')
def format_number_filter(value, decimals=2):
    """Formatea números"""
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):,.{decimals}f}"
    except:
        return str(value)


@app.template_filter('format_percent')
def format_percent_filter(value, decimals=1):
    """Formatea porcentajes"""
    if value is None:
        return 'N/A'
    try:
        return f"{float(value):.{decimals}f}%"
    except:
        return str(value)


# ============================================
# MANEJO DE ERRORES
# ============================================

@app.errorhandler(404)
def page_not_found(e):
    """Maneja errores 404"""
    return render_template('404.html', project_info=PROJECT_INFO), 404


@app.errorhandler(500)
def internal_error(e):
    """Maneja errores 500"""
    return render_template('500.html', project_info=PROJECT_INFO, error=str(e)), 500


# ============================================
# FUNCIONES DE INICIALIZACIÓN
# ============================================

def copy_figures_to_static():
    """Copia figuras de results/figures a static/figures"""
    import shutil
    
    source_dir = FIGURES_DIR
    dest_dir = STATIC_DIR / 'figures'
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    if not source_dir.exists():
        print_info("No hay figuras para copiar")
        return
    
    figures = list(source_dir.glob('*.png'))
    
    if figures:
        for fig in figures:
            try:
                shutil.copy2(fig, dest_dir / fig.name)
            except Exception as e:
                print_warning(f"Error copiando {fig.name}: {e}")
        
        print_success(f"{len(figures)} figura(s) copiada(s)")
    else:
        print_info("No hay figuras PNG para copiar")


def initialize_app():
    """Inicializa la aplicación"""
    print("=" * 70)
    print("🚀 INICIALIZANDO APLICACIÓN FLASK - EGRESADOS UNE")
    print("=" * 70)
    print()
    
    crear_directorios()
    
    print("📊 Cargando información del dataset...")
    load_dataset_info()
    
    print("📈 Cargando resultados de entrenamiento...")
    load_results()
    
    print("🤖 Inicializando predictor...")
    initialize_predictor()
    
    print("📸 Copiando figuras a static...")
    copy_figures_to_static()
    
    print()
    print("=" * 70)
    print("✅ APLICACIÓN INICIALIZADA")
    print("=" * 70)
    print()
    print(f"🌐 Servidor: http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    print(f"📊 Dataset: {dataset_info.get('total_registros', 'N/A')} registros")
    print(f"🤖 Predictor: {'Listo ✅' if predictor_ready else 'No disponible ⚠️'}")
    print()
    print("💡 TIPS:")
    if not predictor_ready:
        print("   • Entrena los modelos desde la sección Entrenar en la web")
        print("   • El entrenamiento se ejecutará automáticamente al hacer clic")
    print("   • Navega a http://localhost:5000")
    print("   • Ctrl+C para detener")
    print()


# ============================================
# PUNTO DE ENTRADA
# ============================================

if __name__ == '__main__':
    initialize_app()
    
    app.run(
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT'],
        debug=FLASK_CONFIG['DEBUG']
    )