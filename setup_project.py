"""
setup_project.py
Proyecto: Predicción Promedio Final - Egresados UNE
Descripción: Script de configuración automática del proyecto
Uso: python setup_project.py

Este script:
1. Crea la estructura de directorios
2. Verifica dependencias instaladas
3. Crea archivos __init__.py necesarios
4. Valida la configuración
5. Muestra siguiente pasos
"""

import os
import sys
from pathlib import Path
import subprocess


def print_header(text, char="="):
    """Imprime encabezado"""
    print("\n" + char * 70)
    print(f" {text} ".center(70, char))
    print(char * 70 + "\n")


def print_step(step_num, total_steps, description):
    """Imprime paso actual"""
    print(f"\n{'='*70}")
    print(f"PASO {step_num}/{total_steps}: {description}")
    print('='*70 + "\n")


def create_directory_structure():
    """Crea estructura de directorios del proyecto"""
    print("📁 Creando estructura de directorios...")
    
    directories = [
        'data/raw',
        'data/processed',
        'data/features',
        'models',
        'results/figures',
        'results/metrics',
        'static/css',
        'static/js',
        'static/img',
        'templates',
        'src/data',
        'src/features',
        'src/models',
        'src/utils',
        'notebooks',
        'tests'
    ]
    
    created = 0
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            dir_path.mkdir(parents=True, exist_ok=True)
            created += 1
            print(f"   ✓ Creado: {directory}")
        else:
            print(f"   ○ Ya existe: {directory}")
    
    print(f"\n✅ {created} directorios nuevos creados")
    return True


def create_init_files():
    """Crea archivos __init__.py en todos los paquetes"""
    print("\n📝 Creando archivos __init__.py...")
    
    init_locations = [
        'src',
        'src/data',
        'src/features',
        'src/models',
        'src/utils',
        'tests'
    ]
    
    created = 0
    for location in init_locations:
        init_file = Path(location) / '__init__.py'
        if not init_file.exists():
            init_file.touch()
            created += 1
            print(f"   ✓ Creado: {init_file}")
        else:
            print(f"   ○ Ya existe: {init_file}")
    
    print(f"\n✅ {created} archivos __init__.py nuevos creados")
    return True


def check_python_version():
    """Verifica versión de Python"""
    print("🐍 Verificando versión de Python...")
    
    version = sys.version_info
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("   ⚠️  Se recomienda Python 3.8 o superior")
        return False
    else:
        print("   ✅ Versión compatible")
        return True


def check_dependencies():
    """Verifica si las dependencias están instaladas"""
    print("\n📦 Verificando dependencias instaladas...")
    
    required_packages = {
        'flask': 'Flask',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
    }
    
    missing = []
    installed = []
    
    for import_name, package_name in required_packages.items():
        try:
            __import__(import_name)
            installed.append(package_name)
            print(f"   ✓ {package_name}")
        except ImportError:
            missing.append(package_name)
            print(f"   ✗ {package_name} - NO INSTALADO")
    
    print(f"\n   Instalados: {len(installed)}/{len(required_packages)}")
    
    if missing:
        print(f"\n⚠️  Faltan {len(missing)} paquete(s):")
        for pkg in missing:
            print(f"      • {pkg}")
        print("\n💡 Para instalar:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ Todas las dependencias están instaladas")
        return True


def check_dataset():
    """Verifica si existe el archivo de datos"""
    print("\n📊 Verificando dataset...")
    
    possible_files = [
        'data/raw/EGRESADOSUNE20202024.csv',
        'data/raw/EGRESADOSUNE20202024_CLEAN.csv',
        'data/raw/egresados_une.csv'
    ]
    
    found = False
    for filepath in possible_files:
        if Path(filepath).exists():
            size = Path(filepath).stat().st_size / 1024 / 1024  # MB
            print(f"   ✓ Encontrado: {filepath} ({size:.2f} MB)")
            found = True
            break
    
    if not found:
        print("   ✗ Dataset no encontrado")
        print("\n💡 Coloca tu archivo CSV en:")
        print("   data/raw/EGRESADOSUNE20202024.csv")
        return False
    
    return True


def check_config_file():
    """Verifica si existe el archivo de configuración"""
    print("\n⚙️  Verificando archivo de configuración...")
    
    config_path = Path('src/utils/config.py')
    
    if config_path.exists():
        print(f"   ✓ config.py existe")
        return True
    else:
        print(f"   ✗ config.py NO EXISTE")
        print("\n💡 Crea el archivo src/utils/config.py con la configuración del proyecto")
        return False


def validate_structure():
    """Valida que existan los archivos principales"""
    print("\n🔍 Validando estructura del proyecto...")
    
    required_files = {
        'src/utils/config.py': 'Configuración',
        'src/utils/helpers.py': 'Funciones auxiliares',
        'main.py': 'Pipeline principal',
        'requirements.txt': 'Dependencias'
    }
    
    missing = []
    for filepath, description in required_files.items():
        if Path(filepath).exists():
            print(f"   ✓ {description}: {filepath}")
        else:
            print(f"   ✗ {description}: {filepath} - FALTA")
            missing.append(filepath)
    
    if missing:
        print(f"\n⚠️  Faltan {len(missing)} archivo(s) importantes")
        return False
    else:
        print("\n✅ Estructura del proyecto válida")
        return True


def create_gitignore():
    """Crea archivo .gitignore"""
    print("\n📄 Creando .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints

# PyCharm
.idea/

# VS Code
.vscode/

# Environment
.env
.env.local

# Data
data/raw/*.csv
data/processed/*.csv
data/features/*.csv
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/features/.gitkeep

# Models
models/*.pkl
models/*.joblib

# Results
results/figures/*.png
results/figures/*.jpg
results/metrics/*.json
results/metrics/*.csv
!results/figures/.gitkeep
!results/metrics/.gitkeep

# Logs
*.log
app.log

# OS
.DS_Store
Thumbs.db

# Backup files
*~
*.bak
*.swp
"""
    
    gitignore_path = Path('.gitignore')
    
    if gitignore_path.exists():
        print("   ○ .gitignore ya existe")
    else:
        with open(gitignore_path, 'w') as f:
            f.write(gitignore_content)
        print("   ✓ .gitignore creado")
    
    return True


def create_readme_basic():
    """Crea README básico si no existe"""
    print("\n📖 Verificando README...")
    
    readme_path = Path('README.md')
    
    if readme_path.exists():
        print("   ○ README.md ya existe")
        return True
    
    readme_content = """# 🎓 Proyecto: Predicción de Promedio Final - Egresados UNE

## 📋 Descripción
Sistema de Machine Learning para predecir el promedio final de egresados.

## 🚀 Inicio Rápido

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Colocar dataset
Copiar archivo CSV en: `data/raw/EGRESADOSUNE20202024.csv`

### 3. Ejecutar pipeline
```bash
python main.py
```

### 4. Ejecutar aplicación web
```bash
python app.py
```

## 📁 Estructura del Proyecto
- `data/` - Datos del proyecto
- `src/` - Código fuente
- `models/` - Modelos entrenados
- `results/` - Resultados y visualizaciones
- `static/` y `templates/` - Aplicación web Flask

## 📊 Tecnologías
- Python 3.8+
- scikit-learn (Ridge, Lasso)
- Flask (Web App)
- pandas, numpy
- matplotlib, seaborn

## 👨‍💻 Autor
Estudiante de Ingeniería Estadística e Informática - UNE
"""
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("   ✓ README.md creado")
    return True


def show_next_steps(all_checks_passed):
    """Muestra siguientes pasos"""
    print_header("🎯 SIGUIENTES PASOS", "=")
    
    if all_checks_passed:
        print("✅ El proyecto está configurado correctamente!\n")
        print("📋 Para empezar:")
        print("   1. Asegúrate de tener el CSV en: data/raw/EGRESADOSUNE20202024.csv")
        print("   2. Ejecuta el pipeline: python main.py")
        print("   3. Inicia la web app: python app.py")
        print("   4. Abre el navegador en: http://localhost:5000")
    else:
        print("⚠️  Hay algunas configuraciones pendientes:\n")
        print("📋 Completar:")
        print("   1. Instalar dependencias faltantes: pip install -r requirements.txt")
        print("   2. Colocar el archivo CSV en: data/raw/")
        print("   3. Verificar que existan todos los archivos .py necesarios")
        print("   4. Ejecutar setup_project.py nuevamente para validar")
    
    print()


def main():
    """Función principal"""
    print_header("🚀 CONFIGURACIÓN DEL PROYECTO - EGRESADOS UNE", "=")
    print("Este script configurará automáticamente tu proyecto\n")
    
    total_steps = 9
    checks_passed = []
    
    # Paso 1: Verificar Python
    print_step(1, total_steps, "Verificando Python")
    checks_passed.append(check_python_version())
    
    # Paso 2: Crear directorios
    print_step(2, total_steps, "Creando estructura de directorios")
    checks_passed.append(create_directory_structure())
    
    # Paso 3: Crear __init__.py
    print_step(3, total_steps, "Creando archivos __init__.py")
    checks_passed.append(create_init_files())
    
    # Paso 4: Verificar dependencias
    print_step(4, total_steps, "Verificando dependencias")
    checks_passed.append(check_dependencies())
    
    # Paso 5: Verificar configuración
    print_step(5, total_steps, "Verificando archivo de configuración")
    checks_passed.append(check_config_file())
    
    # Paso 6: Verificar estructura
    print_step(6, total_steps, "Validando estructura del proyecto")
    checks_passed.append(validate_structure())
    
    # Paso 7: Verificar dataset
    print_step(7, total_steps, "Verificando dataset")
    checks_passed.append(check_dataset())
    
    # Paso 8: Crear .gitignore
    print_step(8, total_steps, "Creando .gitignore")
    checks_passed.append(create_gitignore())
    
    # Paso 9: Crear README
    print_step(9, total_steps, "Verificando README")
    checks_passed.append(create_readme_basic())
    
    # Resumen final
    print_header("📊 RESUMEN DE CONFIGURACIÓN", "=")
    
    passed = sum(checks_passed)
    total = len(checks_passed)
    percentage = (passed / total) * 100
    
    print(f"Pasos completados: {passed}/{total} ({percentage:.1f}%)\n")
    
    if passed == total:
        print("🎉 ¡CONFIGURACIÓN COMPLETADA EXITOSAMENTE!")
    elif passed >= total * 0.7:
        print("⚠️  Configuración mayormente completa - revisa los puntos pendientes")
    else:
        print("❌ Hay varios puntos por completar - revisa los errores arriba")
    
    # Mostrar siguientes pasos
    show_next_steps(passed == total)
    
    print_header("", "=")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Configuración interrumpida por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error durante la configuración: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)