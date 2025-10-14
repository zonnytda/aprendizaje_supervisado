"""
quick_fix.py
Proyecto: Predicción Promedio Final - Egresados UNE
Descripción: Script para limpiar y arreglar archivos CSV problemáticos
Uso: python quick_fix.py

Este script:
1. Lee el CSV con configuración muy permisiva
2. Limpia líneas problemáticas
3. Verifica la estructura
4. Guarda versión limpia
"""

import pandas as pd
import sys
from pathlib import Path


def print_header(text):
    """Imprime encabezado"""
    print("\n" + "="*70)
    print(f" {text} ".center(70, "="))
    print("="*70 + "\n")


def diagnose_csv(filepath):
    """
    Diagnostica problemas en el CSV.
    
    Args:
        filepath: Ruta del archivo CSV
    """
    print("🔍 DIAGNOSTICANDO ARCHIVO CSV...")
    print(f"   Archivo: {filepath}\n")
    
    # Verificar que existe
    if not Path(filepath).exists():
        print(f"❌ Error: Archivo no encontrado")
        return None
    
    # Tamaño del archivo
    size_mb = Path(filepath).stat().st_size / 1024 / 1024
    print(f"📊 Tamaño: {size_mb:.2f} MB")
    
    # Leer primeras líneas para detectar separador
    print("\n🔍 Detectando estructura...")
    with open(filepath, 'r', encoding='latin1', errors='ignore') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    # Detectar separador
    separators = {';': 0, ',': 0, '\t': 0, '|': 0}
    for line in first_lines[:3]:
        for sep in separators:
            separators[sep] += line.count(sep)
    
    detected_sep = max(separators, key=separators.get)
    print(f"   Separador detectado: '{detected_sep}'")
    
    # Mostrar primeras líneas
    print("\n📄 Primeras líneas del archivo:")
    for i, line in enumerate(first_lines[:3], 1):
        print(f"   {i}. {line[:100]}...")
    
    return detected_sep


def clean_csv(input_filepath, output_filepath=None, separator=';'):
    """
    Limpia el archivo CSV.
    
    Args:
        input_filepath: Archivo de entrada
        output_filepath: Archivo de salida (opcional)
        separator: Separador de columnas
    """
    print_header("🧹 LIMPIANDO ARCHIVO CSV")
    
    if output_filepath is None:
        # Crear nombre automático
        input_path = Path(input_filepath)
        output_filepath = input_path.parent / f"{input_path.stem}_CLEAN{input_path.suffix}"
    
    print(f"📂 Entrada: {input_filepath}")
    print(f"💾 Salida:  {output_filepath}\n")
    
    try:
        # Configuración muy permisiva para lectura
        print("📖 Leyendo CSV con configuración permisiva...")
        
        df = pd.read_csv(
            input_filepath,
            encoding='latin1',
            sep=separator,
            on_bad_lines='skip',      # Omitir líneas problemáticas
            engine='python',           # Motor más flexible
            encoding_errors='ignore',  # Ignorar errores de encoding
            low_memory=False           # Evitar warnings de tipos mixtos
        )
        
        print(f"✅ CSV leído exitosamente")
        print(f"   Registros: {len(df):,}")
        print(f"   Columnas: {len(df.columns)}")
        
        # Información de columnas
        print(f"\n📋 Columnas encontradas ({len(df.columns)}):")
        for i, col in enumerate(df.columns[:15], 1):
            print(f"   {i:2d}. {col}")
        if len(df.columns) > 15:
            print(f"   ... y {len(df.columns) - 15} más")
        
        # Análisis de calidad
        print(f"\n📊 Análisis de calidad:")
        
        # Valores nulos
        total_nulls = df.isnull().sum().sum()
        print(f"   • Valores nulos totales: {total_nulls:,}")
        
        # Columnas con nulos
        cols_with_nulls = df.isnull().sum()
        cols_with_nulls = cols_with_nulls[cols_with_nulls > 0]
        if len(cols_with_nulls) > 0:
            print(f"   • Columnas con nulos: {len(cols_with_nulls)}")
            for col in cols_with_nulls.head(5).index:
                count = cols_with_nulls[col]
                pct = (count / len(df)) * 100
                print(f"     - {col}: {count:,} ({pct:.1f}%)")
        
        # Duplicados
        duplicates = df.duplicated().sum()
        print(f"   • Registros duplicados: {duplicates:,}")
        
        # Tipos de datos
        print(f"\n📊 Tipos de datos:")
        dtype_counts = df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   • {dtype}: {count} columnas")
        
        # Guardar versión limpia
        print(f"\n💾 Guardando versión limpia...")
        df.to_csv(output_filepath, index=False, encoding='utf-8')
        
        # Verificar archivo guardado
        size_mb = Path(output_filepath).stat().st_size / 1024 / 1024
        print(f"✅ Archivo guardado exitosamente")
        print(f"   Tamaño: {size_mb:.2f} MB")
        print(f"   Ubicación: {output_filepath}")
        
        return df
        
    except Exception as e:
        print(f"\n❌ Error al limpiar CSV: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_cleaned_csv(filepath):
    """
    Verifica que el CSV limpio se pueda leer correctamente.
    
    Args:
        filepath: Ruta del archivo limpio
    """
    print_header("✅ VERIFICANDO ARCHIVO LIMPIO")
    
    try:
        # Leer con pandas estándar
        df = pd.read_csv(filepath)
        
        print(f"✅ Archivo verificado correctamente")
        print(f"   Registros: {len(df):,}")
        print(f"   Columnas: {len(df.columns)}")
        
        # Mostrar primeras filas
        print(f"\n📋 Primeras 3 filas:")
        print(df.head(3).to_string())
        
        # Estadísticas básicas de columna numérica
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            print(f"\n📊 Estadísticas de '{first_numeric}':")
            stats = df[first_numeric].describe()
            for stat, value in stats.items():
                print(f"   {stat:8s}: {value:>10.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error al verificar: {e}")
        return False


def main():
    """Función principal"""
    print_header("🔧 QUICK FIX - LIMPIADOR DE CSV")
    print("Script para limpiar archivos CSV problemáticos\n")
    
    # Ruta del archivo
    input_file = 'data/raw/EGRESADOSUNE20202024.csv'
    
    # Verificar argumentos de línea de comandos
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    # Verificar que existe
    if not Path(input_file).exists():
        print(f"❌ Error: Archivo no encontrado: {input_file}")
        print(f"\n💡 Uso:")
        print(f"   python quick_fix.py                              # Usa data/raw/EGRESADOSUNE20202024.csv")
        print(f"   python quick_fix.py ruta/al/archivo.csv          # Especifica archivo")
        return 1
    
    # Paso 1: Diagnosticar
    separator = diagnose_csv(input_file)
    
    if separator is None:
        return 1
    
    # Paso 2: Limpiar
    df = clean_csv(input_file, separator=separator)
    
    if df is None:
        return 1
    
    # Paso 3: Verificar
    output_file = Path(input_file).parent / f"{Path(input_file).stem}_CLEAN{Path(input_file).suffix}"
    success = verify_cleaned_csv(output_file)
    
    # Resumen final
    print_header("🎯 RESUMEN")
    
    if success:
        print("✅ ¡Limpieza completada exitosamente!")
        print(f"\n📁 Archivos:")
        print(f"   Original: {input_file}")
        print(f"   Limpio:   {output_file}")
        print(f"\n💡 Siguiente paso:")
        print(f"   El pipeline usará automáticamente el archivo limpio")
        print(f"   Ejecuta: python main.py")
        return 0
    else:
        print("❌ Hubo problemas en la verificación")
        print("\n💡 Revisa los errores arriba e intenta nuevamente")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)