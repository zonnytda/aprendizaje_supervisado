"""
Script de Instalación Automática de Dependencias
Proyecto: Predicción Egresados UNE
Uso: python install_dependencies.py
"""

import subprocess
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f" {text} ".center(70, "="))
    print("="*70 + "\n")

def install_package(package, description=""):
    """Instala un paquete y maneja errores"""
    print(f"📦 Instalando {package}...", end=" ")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--upgrade"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"✅ {description if description else package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Error al instalar {package}")
        return False

def main():
    print_header("🚀 INSTALADOR DE DEPENDENCIAS - EGRESADOS UNE")
    
    print("📌 Actualizando pip...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL
        )
        print("✅ pip actualizado\n")
    except:
        print("⚠️  No se pudo actualizar pip, pero continuaremos\n")
    
    # Lista de paquetes esenciales
    packages = [
        # Web Framework
        ("Flask", "Framework web"),
        ("Flask-CORS", "CORS para Flask"),
        
        # Data Processing
        ("pandas", "Manipulación de datos"),
        ("numpy", "Operaciones numéricas"),
        ("openpyxl", "Leer archivos Excel"),
        
        # Machine Learning
        ("scikit-learn", "Machine Learning"),
        ("scipy", "Computación científica"),
        ("joblib", "Persistencia de modelos"),
        
        # Visualization
        ("matplotlib", "Gráficas estáticas"),
        ("seaborn", "Gráficas estadísticas"),
        ("plotly", "Gráficas interactivas"),
        
        # Utilities
        ("python-dotenv", "Variables de entorno"),
        ("tqdm", "Barras de progreso"),
    ]
    
    print_header("📦 INSTALANDO PAQUETES PRINCIPALES")
    
    successful = 0
    failed = 0
    
    for package, description in packages:
        if install_package(package, description):
            successful += 1
        else:
            failed += 1
    
    print_header("📊 RESUMEN DE INSTALACIÓN")
    print(f"✅ Exitosos: {successful}/{len(packages)}")
    print(f"❌ Fallidos: {failed}/{len(packages)}")
    
    if failed == 0:
        print("\n🎉 ¡Todas las dependencias se instalaron correctamente!")
        print("\n📝 Próximos pasos:")
        print("   1. Crea la estructura de carpetas")
        print("   2. Copia tus archivos Python al proyecto")
        print("   3. Coloca el dataset en data/raw/")
        print("   4. Ejecuta: python main.py")
    else:
        print(f"\n⚠️  {failed} paquete(s) fallaron. Intenta instalarlos manualmente:")
        print("   pip install [nombre_del_paquete]")
    
    # Verificar instalación
    print_header("🔍 VERIFICANDO INSTALACIÓN")
    
    test_imports = [
        ("Flask", "flask"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("matplotlib", "matplotlib"),
    ]
    
    for package_name, import_name in test_imports:
        try:
            __import__(import_name)
            print(f"✅ {package_name} importado correctamente")
        except ImportError:
            print(f"❌ {package_name} NO se puede importar")
    
    print("\n" + "="*70)
    print("✅ INSTALACIÓN COMPLETADA")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()