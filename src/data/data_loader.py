"""
Módulo: data_loader.py
Proyecto: Predicción Promedio Final - Egresados UNE
Clase: DataLoader
Responsabilidad: Cargar datos desde diversos formatos (CSV, Excel)
Principio POO: Single Responsibility
Autor: Estudiante de Ingeniería Estadística e Informática
Fecha: Octubre 2025
"""

import pandas as pd
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.utils.helpers import (
    print_progress,
    print_section_header,
    print_success,
    print_error,
    print_warning,
    get_file_size
)

import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Carga y valida datos desde archivos CSV o Excel.
    
    Características:
    - Carga desde CSV con configuración personalizable
    - Carga desde Excel (múltiples hojas)
    - Validación automática de datos
    - Vista previa de datos
    - Información detallada del dataset
    - Guardado de datos procesados
    
    Atributos:
        filepath (str): Ruta del archivo de datos
        encoding (str): Codificación del archivo
        data (pd.DataFrame): DataFrame cargado
        file_info (Dict): Información del archivo
    
    Ejemplo de uso:
        >>> loader = DataLoader('data/raw/egresados.csv', encoding='latin1')
        >>> df = loader.load_csv(sep=';', on_bad_lines='skip')
        >>> loader.preview_data(n_rows=5)
        >>> loader.save_data('data/processed/egresados_clean.csv')
    """
    
    def __init__(self, filepath: str, encoding: str = 'utf-8'):
        """
        Inicializa el cargador de datos.
        
        Args:
            filepath: Ruta del archivo a cargar
            encoding: Codificación del archivo (default: 'utf-8')
        
        Raises:
            FileNotFoundError: Si el archivo no existe
        """
        self.filepath = Path(filepath)
        self.encoding = encoding
        self.data: Optional[pd.DataFrame] = None
        self.file_info: Dict[str, Any] = {}
        
        # Verificar que el archivo existe
        if not self.filepath.exists():
            raise FileNotFoundError(f"❌ Archivo no encontrado: {self.filepath}")
        
        # Obtener información del archivo
        self._get_file_info()
        
        print(f"✅ DataLoader inicializado")
        print(f"   Archivo: {self.filepath.name}")
        print(f"   Tamaño: {self.file_info['size']}")
        print(f"   Encoding: {self.encoding}")
    
    def _get_file_info(self) -> None:
        """Obtiene información básica del archivo"""
        self.file_info = {
            'name': self.filepath.name,
            'path': str(self.filepath),
            'size': get_file_size(str(self.filepath)),
            'extension': self.filepath.suffix,
            'exists': self.filepath.exists()
        }
    
    def load_csv(self, **kwargs) -> pd.DataFrame:
        """
        Carga datos desde un archivo CSV.
        
        Args:
            **kwargs: Argumentos adicionales para pd.read_csv()
                     (sep, encoding, on_bad_lines, engine, etc.)
        
        Returns:
            DataFrame con los datos cargados
        
        Raises:
            Exception: Si hay error al leer el archivo
        
        Ejemplo:
            >>> df = loader.load_csv(
            ...     sep=';',
            ...     encoding='latin1',
            ...     on_bad_lines='skip',
            ...     engine='python'
            ... )
        """
        print_progress(f"Cargando CSV: {self.filepath.name}...", "📂")
        
        try:
            # Configuración por defecto
            default_config = {
                'encoding': self.encoding,
            }
            
            # Si no se especifica engine o es 'c', agregar low_memory
            # Si es 'python', NO agregar low_memory
            if 'engine' not in kwargs or kwargs['engine'] == 'c':
                default_config['low_memory'] = False
            
            # Combinar configuración por defecto con la proporcionada
            config = {**default_config, **kwargs}
            
            # Cargar datos
            self.data = pd.read_csv(self.filepath, **config)
            
            # Información de carga
            print_success(f"CSV cargado exitosamente")
            print(f"   Registros: {len(self.data):,}")
            print(f"   Columnas: {len(self.data.columns)}")
            print(f"   Memoria: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            return self.data
            
        except Exception as e:
            print_error(f"Error al cargar CSV: {str(e)}")
            raise
    
    def load_excel(self, sheet_name: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """
        Carga datos desde un archivo Excel.
        
        Args:
            sheet_name: Nombre de la hoja a cargar (None = primera hoja)
            **kwargs: Argumentos adicionales para pd.read_excel()
        
        Returns:
            DataFrame con los datos cargados
        
        Raises:
            Exception: Si hay error al leer el archivo
        
        Ejemplo:
            >>> df = loader.load_excel(sheet_name='Egresados')
        """
        print_progress(f"Cargando Excel: {self.filepath.name}...", "📂")
        
        try:
            # Cargar datos
            if sheet_name:
                self.data = pd.read_excel(self.filepath, sheet_name=sheet_name, **kwargs)
            else:
                self.data = pd.read_excel(self.filepath, **kwargs)
            
            # Información de carga
            print_success(f"Excel cargado exitosamente")
            print(f"   Registros: {len(self.data):,}")
            print(f"   Columnas: {len(self.data.columns)}")
            
            if sheet_name:
                print(f"   Hoja: {sheet_name}")
            
            return self.data
            
        except Exception as e:
            print_error(f"Error al cargar Excel: {str(e)}")
            raise
    
    def validate_data(self, required_columns: Optional[list] = None) -> bool:
        """
        Valida que los datos estén correctamente cargados.
        
        Args:
            required_columns: Lista de columnas requeridas (opcional)
        
        Returns:
            True si es válido, False en caso contrario
        
        Ejemplo:
            >>> loader.validate_data(required_columns=['PROMEDIO_FINAL', 'FACULTAD'])
        """
        print_progress("Validando datos...", "🔍")
        
        # Verificar que hay datos
        if self.data is None:
            print_error("No hay datos cargados")
            return False
        
        if self.data.empty:
            print_error("El DataFrame está vacío")
            return False
        
        # Verificar columnas requeridas
        if required_columns:
            missing_cols = set(required_columns) - set(self.data.columns)
            if missing_cols:
                print_error(f"Faltan columnas requeridas: {missing_cols}")
                return False
        
        print_success("Validación exitosa")
        return True
    
    def preview_data(self, n_rows: int = 5) -> None:
        """
        Muestra una vista previa de los datos.
        
        Args:
            n_rows: Número de filas a mostrar
        
        Ejemplo:
            >>> loader.preview_data(n_rows=10)
        """
        if self.data is None:
            print_warning("No hay datos cargados para previsualizar")
            return
        
        print(f"\n{'='*70}")
        print(f"📋 VISTA PREVIA DE DATOS ({n_rows} filas)")
        print(f"{'='*70}\n")
        
        # Mostrar primeras filas
        print(self.data.head(n_rows).to_string())
        print()
    
    def get_info(self) -> Dict[str, Any]:
        """
        Obtiene información detallada del dataset.
        
        Returns:
            Diccionario con información del dataset
        
        Ejemplo:
            >>> info = loader.get_info()
            >>> print(info['total_rows'])
        """
        if self.data is None:
            print_warning("No hay datos cargados")
            return {}
        
        info = {
            'total_rows': len(self.data),
            'total_columns': len(self.data.columns),
            'column_names': list(self.data.columns),
            'dtypes': dict(self.data.dtypes.astype(str)),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': dict(self.data.isnull().sum()),
            'duplicated_rows': int(self.data.duplicated().sum()),
            'file_info': self.file_info
        }
        
        return info
    
    def print_info(self) -> None:
        """
        Imprime información detallada del dataset de forma formateada.
        
        Ejemplo:
            >>> loader.print_info()
        """
        if self.data is None:
            print_warning("No hay datos cargados")
            return
        
        info = self.get_info()
        
        print(f"\n{'='*70}")
        print(f"📊 INFORMACIÓN DEL DATASET")
        print(f"{'='*70}\n")
        
        print(f"📁 Archivo:")
        print(f"   Nombre: {info['file_info']['name']}")
        print(f"   Tamaño: {info['file_info']['size']}")
        print(f"   Ruta: {info['file_info']['path']}")
        
        print(f"\n📊 Dimensiones:")
        print(f"   Filas: {info['total_rows']:,}")
        print(f"   Columnas: {info['total_columns']}")
        print(f"   Memoria: {info['memory_usage_mb']:.2f} MB")
        
        print(f"\n📋 Columnas:")
        for i, col in enumerate(info['column_names'][:10], 1):
            dtype = info['dtypes'][col]
            missing = info['missing_values'][col]
            print(f"   {i:2d}. {col:<30} [{dtype}] - Nulos: {missing}")
        
        if info['total_columns'] > 10:
            print(f"   ... y {info['total_columns'] - 10} columnas más")
        
        print(f"\n⚠️  Calidad:")
        print(f"   Valores nulos: {sum(info['missing_values'].values()):,}")
        print(f"   Filas duplicadas: {info['duplicated_rows']:,}")
        
        print(f"\n{'='*70}\n")
    
    def save_data(self, output_path: str, index: bool = False, **kwargs) -> None:
        """
        Guarda los datos en un archivo CSV.
        
        Args:
            output_path: Ruta donde guardar el archivo
            index: Si True, guarda el índice (default: False)
            **kwargs: Argumentos adicionales para pd.to_csv()
        
        Ejemplo:
            >>> loader.save_data('data/processed/datos_limpios.csv')
        """
        if self.data is None:
            print_warning("No hay datos para guardar")
            return
        
        print_progress(f"Guardando datos en: {output_path}...", "💾")
        
        try:
            # Crear directorio si no existe
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar datos
            self.data.to_csv(output_path, index=index, encoding='utf-8', **kwargs)
            
            # Información del archivo guardado
            saved_size = get_file_size(output_path)
            print_success(f"Datos guardados exitosamente")
            print(f"   Ubicación: {output_path}")
            print(f"   Tamaño: {saved_size}")
            
        except Exception as e:
            print_error(f"Error al guardar datos: {str(e)}")
            raise
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """
        Obtiene el DataFrame cargado.
        
        Returns:
            DataFrame con los datos o None si no hay datos
        
        Ejemplo:
            >>> df = loader.get_data()
        """
        return self.data
    
    def __repr__(self) -> str:
        """Representación string del objeto"""
        if self.data is not None:
            return f"DataLoader(file='{self.filepath.name}', rows={len(self.data):,}, cols={len(self.data.columns)})"
        else:
            return f"DataLoader(file='{self.filepath.name}', loaded=False)"


# ============================================
# EJEMPLO DE USO
# ============================================

if __name__ == "__main__":
    """
    Ejemplo de uso completo de DataLoader
    """
    print_section_header("🧪 TESTING DATALOADER CLASS")
    
    try:
        # 1. Inicializar loader
        print("Paso 1: Inicializar DataLoader")
        loader = DataLoader(
            'data/raw/EGRESADOSUNE20202024.csv',
            encoding='latin1'
        )
        
        # 2. Cargar CSV
        print("\nPaso 2: Cargar CSV")
        df = loader.load_csv(
            sep=';',
            on_bad_lines='skip',
            engine='python'
        )
        
        # 3. Vista previa
        print("\nPaso 3: Vista previa")
        loader.preview_data(n_rows=3)
        
        # 4. Información detallada
        print("\nPaso 4: Información del dataset")
        loader.print_info()
        
        # 5. Validar datos
        print("\nPaso 5: Validar datos")
        is_valid = loader.validate_data(
            required_columns=['PROMEDIO_FINAL']
        )
        
        # 6. Guardar datos
        if is_valid:
            print("\nPaso 6: Guardar datos procesados")
            loader.save_data('data/processed/datos_cargados.csv')
        
        print("\n✅ EJEMPLO COMPLETADO EXITOSAMENTE!")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Asegúrate de que el archivo existe en la ruta especificada")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()