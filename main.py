"""
main.py
Proyecto: Predicción Promedio Final - Egresados UNE
Descripción: Pipeline completo de Machine Learning ejecutable desde CLI
Autor: Estudiante de Ingeniería Estadística e Informática
Fecha: Octubre 2025

Uso:
    python main.py                    # Ejecuta pipeline completo
    python main.py --quick            # Ejecución rápida (sin tuning)
    python main.py --model ridge      # Solo Ridge
    python main.py --model lasso      # Solo Lasso
"""

import sys
import time
import argparse
from pathlib import Path

# Importar clases del proyecto
from src.data.data_loader import DataLoader
from src.data.data_preprocessor import DataPreprocessor
from src.features.feature_engineer import FeatureEngineer
from src.models.model_trainer import ModelTrainer
from src.models.model_evaluator import ModelEvaluator
from src.models.model_comparator import ModelComparator
from src.utils.config import *
from src.utils.helpers import (
    print_section_header,
    print_progress,
    save_dict_to_json,
    measure_time
)

import warnings
warnings.filterwarnings('ignore')


class MLPipeline:
    """
    Pipeline completo de Machine Learning para predicción de promedios.
    
    Orquesta todas las clases del proyecto en un flujo coherente:
    1. Carga de datos
    2. Preprocesamiento
    3. Feature Engineering
    4. Entrenamiento de modelos (Ridge y Lasso)
    5. Evaluación y comparación
    6. Persistencia de resultados
    """
    
    def __init__(self, quick_mode: bool = False, models_to_train: list = None):
        """
        Inicializa el pipeline.
        
        Args:
            quick_mode: Si True, omite optimización de hiperparámetros
            models_to_train: Lista de modelos a entrenar ['ridge', 'lasso']
        """
        self.quick_mode = quick_mode
        self.models_to_train = models_to_train or ['ridge', 'lasso']
        
        # Componentes del pipeline
        self.loader = None
        self.preprocessor = None
        self.engineer = None
        self.trainers = {}
        self.evaluators = {}
        self.comparator = None
        
        # Datos
        self.df_raw = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_transformed = None
        self.X_val_transformed = None
        self.X_test_transformed = None
        
        # Resultados
        self.results = {
            'execution_time': 0,
            'models_trained': [],
            'best_model': None,
            'metrics': {}
        }
        
        print_section_header("🚀 PIPELINE DE MACHINE LEARNING - EGRESADOS UNE")
        print(f"Configuración:")
        print(f"   • Modo rápido: {'Sí' if quick_mode else 'No'}")
        print(f"   • Modelos a entrenar: {', '.join([m.upper() for m in self.models_to_train])}")
        print()
    
    def step_1_load_data(self):
        """Paso 1: Cargar datos desde archivo CSV"""
        print_section_header("📂 PASO 1: CARGA DE DATOS")
        
        # Buscar primero archivo limpio
        clean_path = RAW_DATA_DIR / RAW_DATA_FILE.replace('.csv', '_CLEAN.csv')
        
        if clean_path.exists():
            print(f"ℹ️  Usando archivo limpio: {clean_path.name}")
            raw_data_path = clean_path
        else:
            raw_data_path = RAW_DATA_DIR / RAW_DATA_FILE
            if not raw_data_path.exists():
                raise FileNotFoundError(
                    f"❌ Archivo de datos no encontrado: {raw_data_path}\n"
                    f"   Por favor, coloca el archivo en: {RAW_DATA_DIR}\n"
                    f"   O ejecuta primero: python quick_fix.py"
                )
        
        self.loader = DataLoader(str(raw_data_path), encoding=CSV_ENCODING)
        
        try:
            # Intentar carga con configuración estándar
            self.df_raw = self.loader.load_csv(**CSV_OPTIONS)
        except Exception as e:
            print(f"⚠️  Error con configuración estándar: {e}")
            print("🔄 Intentando configuración alternativa...")
            
            # Configuración alternativa más permisiva
            self.df_raw = self.loader.load_csv(
                sep=CSV_SEPARATOR,
                encoding=CSV_ENCODING,
                on_bad_lines='skip',
                engine='python'
            )
        
        # Verificar columnas críticas
        if TARGET_COLUMN not in self.df_raw.columns:
            print(f"⚠️  Columna objetivo '{TARGET_COLUMN}' no encontrada")
            print(f"   Columnas disponibles: {list(self.df_raw.columns)[:10]}")
            
            # Buscar columna similar
            similar = [col for col in self.df_raw.columns if 'PROMEDIO' in col.upper()]
            if similar:
                print(f"   💡 Columnas con 'PROMEDIO': {similar}")
                raise ValueError(f"Ajusta TARGET_COLUMN en config.py a una de estas: {similar}")
        
        # Vista previa
        self.loader.preview_data(n_rows=3)
        
        # Guardar datos cargados
        self.loader.save_data(str(PROCESSED_DATA_DIR / 'datos_cargados.csv'))
        
        print(f"✅ Paso 1 completado: {len(self.df_raw):,} registros cargados\n")
    
    def step_2_preprocess_data(self):
        """Paso 2: Preprocesamiento de datos"""
        print_section_header("🔧 PASO 2: PREPROCESAMIENTO")
        
        self.preprocessor = DataPreprocessor(
            data=self.df_raw,
            target_column=TARGET_COLUMN,
            exclude_columns=EXCLUDE_COLUMNS
        )
        
        # Identificar tipos de variables
        self.preprocessor.identify_feature_types()
        
        # Pipeline de limpieza
        df_clean = self.preprocessor.handle_missing_values(
            strategy=MISSING_VALUES_STRATEGY
        )
        
        df_clean = self.preprocessor.remove_outliers(
            method=OUTLIER_METHOD,
            threshold=OUTLIER_THRESHOLD
        )
        
        df_clean = self.preprocessor.convert_data_types()
        
        # Split de datos
        self.X_train, self.X_val, self.X_test, \
        self.y_train, self.y_val, self.y_test = self.preprocessor.split_data(
            test_size=TEST_SIZE,
            val_size=VALIDATION_SIZE,
            stratify_column=STRATIFY_BY,
            random_state=RANDOM_STATE
        )
        
        # Reporte
        self.preprocessor.print_report()
        
        print(f"✅ Paso 2 completado\n")
    
    def step_3_feature_engineering(self):
        """Paso 3: Ingeniería de características"""
        print_section_header("⚙️  PASO 3: FEATURE ENGINEERING")
        
        self.engineer = FeatureEngineer(
            data=self.X_train,
            categorical_features=self.preprocessor.categorical_features,
            numerical_features=self.preprocessor.numeric_features  # CORREGIDO: numeric_features
        )
        
        # Pipeline completo de transformación en train
        self.X_train_transformed = self.engineer.fit_transform(self.X_train)
        
        # Transformar validation y test
        self.X_val_transformed = self.engineer.transform(self.X_val)
        self.X_test_transformed = self.engineer.transform(self.X_test)
        
        # Guardar transformadores
        self.engineer.save_transformers(str(MODELS_DIR))
        
        print(f"\n📊 Features finales: {self.X_train_transformed.shape[1]}")
        print(f"✅ Paso 3 completado\n")
    
    def step_4_train_models(self):
        """Paso 4: Entrenamiento de modelos"""
        print_section_header("🤖 PASO 4: ENTRENAMIENTO DE MODELOS")
        
        feature_names = self.engineer.get_feature_names()
        
        for model_type in self.models_to_train:
            print(f"\n{'='*70}")
            print(f"Entrenando {model_type.upper()}")
            print(f"{'='*70}\n")
            
            # Crear trainer
            trainer = ModelTrainer(model_type=model_type)
            
            if self.quick_mode:
                # Entrenamiento rápido sin tuning
                print("⚡ Modo rápido: entrenamiento sin optimización de hiperparámetros")
                trainer.train(
                    self.X_train_transformed,
                    self.y_train,
                    feature_names=feature_names
                )
            else:
                # Optimización de hiperparámetros
                print("🔍 Optimizando hiperparámetros con Grid Search...")
                param_grid = (RIDGE_PARAM_GRID if model_type == 'ridge' 
                            else LASSO_PARAM_GRID)
                
                best_params = trainer.hyperparameter_tuning(
                    self.X_train_transformed,
                    self.y_train,
                    param_grid=param_grid,
                    cv=CV_FOLDS,
                    scoring=CV_SCORING,
                    verbose=1
                )
                
                print(f"✅ Mejores parámetros encontrados: {best_params}")
                
                # Validación cruzada con mejores parámetros
                print("\n📊 Realizando validación cruzada...")
                cv_results = trainer.cross_validate(
                    self.X_train_transformed,
                    self.y_train,
                    cv=CV_FOLDS
                )
            
            # Feature importance
            trainer.get_feature_importance(feature_names, top_n=15)
            
            # Guardar modelo
            model_filename = RIDGE_MODEL_FILE if model_type == 'ridge' else LASSO_MODEL_FILE
            trainer.save_model(str(MODELS_DIR / model_filename))
            
            # Guardar trainer
            self.trainers[model_type] = trainer
            self.results['models_trained'].append(model_type)
        
        print(f"\n✅ Paso 4 completado: {len(self.trainers)} modelos entrenados\n")
    
    def step_5_evaluate_models(self):
        """Paso 5: Evaluación de modelos"""
        print_section_header("📊 PASO 5: EVALUACIÓN DE MODELOS")
        
        for model_type, trainer in self.trainers.items():
            print(f"\n{'='*70}")
            print(f"Evaluando {model_type.upper()}")
            print(f"{'='*70}\n")
            
            # Predicciones en test set
            y_pred = trainer.predict(self.X_test_transformed)
            
            # Crear evaluador
            evaluator = ModelEvaluator(
                model=trainer.model,
                y_true=self.y_test,
                y_pred=y_pred,
                model_name=model_type.upper()
            )
            
            # Calcular métricas
            metrics = evaluator.calculate_metrics()
            evaluator.print_metrics()
            
            # Guardar métricas
            metrics_file = (RIDGE_METRICS_FILE if model_type == 'ridge' 
                          else LASSO_METRICS_FILE)
            save_dict_to_json(
                evaluator.generate_evaluation_report(),
                str(METRICS_DIR / metrics_file)
            )
            
            # Visualizaciones
            print(f"\n📊 Generando visualizaciones para {model_type.upper()}...")
            
            # 1. Predicted vs Actual
            evaluator.plot_predictions_vs_actual(
                save_path=str(FIGURES_DIR / f'{model_type}_predicted_vs_actual.png'),
                show_plot=False
            )
            
            # 2. Residuales
            evaluator.plot_residuals(
                save_path=str(FIGURES_DIR / f'{model_type}_residuals.png'),
                show_plot=False
            )
            
            # 3. Learning curves
            evaluator.plot_learning_curves(
                self.X_train_transformed,
                self.y_train,
                save_path=str(FIGURES_DIR / f'{model_type}_learning_curves.png'),
                show_plot=False,
                cv=3  # Menos folds para ser más rápido
            )
            
            # 4. Feature importance
            evaluator.plot_feature_importance(
                self.engineer.get_feature_names(),
                save_path=str(FIGURES_DIR / f'{model_type}_feature_importance.png'),
                top_n=20,
                show_plot=False
            )
            
            # 5. Distribución de errores
            evaluator.plot_error_distribution(
                save_path=str(FIGURES_DIR / f'{model_type}_error_distribution.png'),
                show_plot=False
            )
            
            print(f"   ✓ 5 visualizaciones guardadas en {FIGURES_DIR}")
            
            # Guardar evaluador
            self.evaluators[model_type] = evaluator
            self.results['metrics'][model_type] = metrics
        
        print(f"\n✅ Paso 5 completado: {len(self.evaluators)} modelos evaluados\n")
    
    def step_6_compare_models(self):
        """Paso 6: Comparación de modelos"""
        print_section_header("🔬 PASO 6: COMPARACIÓN DE MODELOS")
        
        if len(self.trainers) < 2:
            print("⚠️  Solo hay 1 modelo, omitiendo comparación")
            # Si solo hay un modelo, marcarlo como el mejor
            if self.trainers:
                self.results['best_model'] = list(self.trainers.keys())[0]
            return
        
        # Crear comparador
        self.comparator = ModelComparator()
        
        # Agregar modelos
        for model_type, trainer in self.trainers.items():
            y_pred = trainer.predict(self.X_test_transformed)
            self.comparator.add_model(
                name=model_type.upper(),
                model=trainer.model,
                y_true=self.y_test,
                y_pred=y_pred
            )
        
        # Tabla comparativa
        comparison_df = self.comparator.compare_metrics()
        self.comparator.print_comparison()
        
        # Mejor modelo
        best_model = self.comparator.get_best_model(metric='rmse')
        self.results['best_model'] = best_model
        
        # Visualizaciones comparativas
        print("\n📊 Generando visualizaciones comparativas...")
        
        self.comparator.plot_comparison(
            save_path=str(FIGURES_DIR / 'model_comparison.png'),
            show_plot=False
        )
        
        self.comparator.plot_metrics_radar(
            save_path=str(FIGURES_DIR / 'model_comparison_radar.png'),
            show_plot=False
        )
        
        self.comparator.plot_predictions_comparison(
            save_path=str(FIGURES_DIR / 'predictions_comparison.png'),
            show_plot=False
        )
        
        print(f"   ✓ 3 visualizaciones comparativas guardadas")
        
        # Guardar comparación
        self.comparator.save_comparison(str(METRICS_DIR))
        
        print(f"\n✅ Paso 6 completado\n")
    
    def step_7_generate_report(self):
        """Paso 7: Generar reporte final"""
        print_section_header("📋 PASO 7: REPORTE FINAL")
        
        # Compilar resultados finales
        final_report = {
            'project': PROJECT_INFO['nombre'],
            'version': PROJECT_INFO['version'],
            'execution_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'execution_time_seconds': self.results['execution_time'],
            'quick_mode': self.quick_mode,
            'dataset': {
                'total_samples': len(self.df_raw),
                'train_samples': len(self.X_train),
                'val_samples': len(self.X_val),
                'test_samples': len(self.X_test),
                'n_features_original': self.X_train.shape[1],
                'n_features_engineered': self.X_train_transformed.shape[1],
            },
            'preprocessing': self.preprocessor.generate_report(),
            'feature_engineering': {
                'derived_features': len(self.engineer.get_derived_features_info()),
                'feature_names': self.engineer.get_feature_names()[:20],  # Top 20
            },
            'models': self.results['models_trained'],
            'best_model': self.results['best_model'],
            'metrics': self.results['metrics'],
        }
        
        # Guardar reporte
        report_path = RESULTS_DIR / 'final_report.json'
        save_dict_to_json(final_report, str(report_path))
        
        # Imprimir resumen ejecutivo
        print("\n" + "="*70)
        print("🎯 RESUMEN EJECUTIVO DEL PROYECTO")
        print("="*70 + "\n")
        
        print(f"📊 DATASET:")
        print(f"   • Total registros: {final_report['dataset']['total_samples']:,} egresados")
        print(f"   • Train: {final_report['dataset']['train_samples']:,} ({final_report['dataset']['train_samples']/final_report['dataset']['total_samples']*100:.1f}%)")
        print(f"   • Validation: {final_report['dataset']['val_samples']:,} ({final_report['dataset']['val_samples']/final_report['dataset']['total_samples']*100:.1f}%)")
        print(f"   • Test: {final_report['dataset']['test_samples']:,} ({final_report['dataset']['test_samples']/final_report['dataset']['total_samples']*100:.1f}%)")
        
        print(f"\n⚙️  FEATURE ENGINEERING:")
        print(f"   • Features originales: {final_report['dataset']['n_features_original']}")
        print(f"   • Features engineered: {final_report['dataset']['n_features_engineered']}")
        print(f"   • Features derivadas creadas: {final_report['feature_engineering']['derived_features']}")
        
        print(f"\n🤖 MODELOS:")
        print(f"   • Modelos entrenados: {len(final_report['models'])} ({', '.join([m.upper() for m in final_report['models']])})")
        print(f"   • Mejor modelo: {final_report['best_model']}")
        
        if final_report['best_model']:
            best_key = final_report['best_model'].lower()
            if best_key in final_report['metrics']:
                best_metrics = final_report['metrics'][best_key]
                print(f"\n📈 MÉTRICAS DEL MEJOR MODELO ({final_report['best_model']}):")
                print(f"   • RMSE: {best_metrics['RMSE']:.4f}")
                print(f"   • MAE: {best_metrics['MAE']:.4f}")
                print(f"   • R²: {best_metrics['R2']:.4f}")
                print(f"   • MAPE: {best_metrics.get('MAPE', 'N/A'):.2f}%" if isinstance(best_metrics.get('MAPE'), (int, float)) else "   • MAPE: N/A")
        
        print(f"\n📁 ARCHIVOS GENERADOS:")
        print(f"   • Modelos: {MODELS_DIR}")
        print(f"   • Figuras: {FIGURES_DIR}")
        print(f"   • Métricas: {METRICS_DIR}")
        print(f"   • Reporte: {report_path}")
        
        print(f"\n⏱️  TIEMPO DE EJECUCIÓN:")
        print(f"   • Total: {self.results['execution_time']:.2f} segundos ({self.results['execution_time']/60:.1f} minutos)")
        
        print(f"\n✅ Paso 7 completado\n")
    
    def run(self):
        """Ejecuta el pipeline completo"""
        start_time = time.time()
        
        try:
            # Paso 1: Cargar datos
            self.step_1_load_data()
            
            # Paso 2: Preprocesamiento
            self.step_2_preprocess_data()
            
            # Paso 3: Feature Engineering
            self.step_3_feature_engineering()
            
            # Paso 4: Entrenamiento
            self.step_4_train_models()
            
            # Paso 5: Evaluación
            self.step_5_evaluate_models()
            
            # Paso 6: Comparación
            self.step_6_compare_models()
            
            # Paso 7: Reporte final
            self.results['execution_time'] = time.time() - start_time
            self.step_7_generate_report()
            
            # Mensaje final
            print_section_header("✅ PIPELINE COMPLETADO EXITOSAMENTE")
            print(f"⏱️  Tiempo total: {self.results['execution_time']:.2f}s ({self.results['execution_time']/60:.1f}m)")
            print(f"🏆 Mejor modelo: {self.results['best_model']}")
            print(f"\n💡 SIGUIENTE PASO: Ejecutar la aplicación web")
            print(f"   python app.py")
            print()
            
            return True
            
        except Exception as e:
            print(f"\n❌ ERROR EN EL PIPELINE:")
            print(f"   {str(e)}")
            print("\n📋 STACK TRACE:")
            import traceback
            traceback.print_exc()
            print()
            return False


def parse_arguments():
    """Parsea argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description='Pipeline de ML para Predicción de Promedios - Egresados UNE',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main.py                    # Ejecuta pipeline completo
  python main.py --quick            # Ejecución rápida (sin tuning)
  python main.py --model ridge      # Solo entrena Ridge
  python main.py --model lasso      # Solo entrena Lasso
  python main.py --quick --model ridge  # Rápido, solo Ridge
  
Notas:
  - El modo --quick omite la optimización de hiperparámetros
  - Útil para pruebas rápidas o cuando ya conoces buenos parámetros
  - El pipeline completo puede tomar 5-15 minutos dependiendo del tamaño del dataset
        """
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Modo rápido: omite optimización de hiperparámetros'
    )
    
    parser.add_argument(
        '--model',
        choices=['ridge', 'lasso', 'both'],
        default='both',
        help='Modelo(s) a entrenar (default: both)'
    )
    
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Solo valida la configuración sin ejecutar el pipeline'
    )
    
    return parser.parse_args()


def main():
    """Función principal"""
    # Banner inicial
    print("\n" + "="*70)
    print(f" {PROJECT_INFO['nombre']} ".center(70))
    print(f" v{PROJECT_INFO['version']} ".center(70))
    print("="*70 + "\n")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Si solo quiere validar
    if args.validate:
        print("🔍 VALIDANDO CONFIGURACIÓN...\n")
        if validate_config():
            print("✅ Configuración válida - listo para ejecutar")
            return 0
        else:
            print("❌ Hay errores en la configuración - revisar config.py")
            return 1
    
    # Validar configuración antes de empezar
    print("🔍 Validando configuración...")
    if not validate_config():
        print("❌ Errores en configuración. Abortando.")
        return 1
    print("✅ Configuración válida\n")
    
    # Determinar modelos a entrenar
    if args.model == 'both':
        models = ['ridge', 'lasso']
    else:
        models = [args.model]
    
    # Crear y ejecutar pipeline
    pipeline = MLPipeline(
        quick_mode=args.quick,
        models_to_train=models
    )
    
    # Ejecutar
    success = pipeline.run()
    
    # Código de salida
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)