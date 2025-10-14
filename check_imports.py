"""
check_imports.py
Script para verificar todas las importaciones faltantes
"""

import ast
import sys
from pathlib import Path
from collections import defaultdict

def extract_imports_from_file(filepath):
    """Extrae todas las importaciones de un archivo Python"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module
                for alias in node.names:
                    imports.append({
                        'module': module,
                        'name': alias.name,
                        'file': filepath.name
                    })
        
        return imports
    except Exception as e:
        print(f"Error procesando {filepath}: {e}")
        return []

def check_helpers_exports():
    """Verifica qué funciones están definidas en helpers.py"""
    helpers_path = Path('src/utils/helpers.py')
    
    if not helpers_path.exists():
        print(f"❌ No se encuentra: {helpers_path}")
        return set()
    
    try:
        with open(helpers_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read())
        
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
        
        return defined
    except Exception as e:
        print(f"❌ Error leyendo helpers.py: {e}")
        return set()

def main():
    print("="*70)
    print("🔍 VERIFICANDO IMPORTACIONES DEL PROYECTO")
    print("="*70 + "\n")
    
    # Archivos a verificar
    files_to_check = [
        'src/data/data_preprocessor.py',
        'src/features/feature_engineer.py',
        'src/models/model_trainer.py',
        'src/models/model_evaluator.py',
        'src/models/model_comparator.py',
    ]
    
    # Obtener funciones definidas en helpers.py
    print("📋 Funciones definidas en helpers.py:")
    helpers_functions = check_helpers_exports()
    for func in sorted(helpers_functions):
        print(f"   ✓ {func}")
    print()
    
    # Verificar cada archivo
    all_imports_from_helpers = defaultdict(list)
    
    print("🔍 Verificando importaciones desde helpers.py...\n")
    
    for filepath in files_to_check:
        path = Path(filepath)
        if not path.exists():
            print(f"⚠️  No existe: {filepath}")
            continue
        
        imports = extract_imports_from_file(path)
        
        # Filtrar solo importaciones desde helpers
        helpers_imports = [
            imp for imp in imports 
            if imp['module'] and 'helpers' in imp['module']
        ]
        
        if helpers_imports:
            print(f"📄 {filepath}:")
            for imp in helpers_imports:
                is_defined = imp['name'] in helpers_functions
                status = "✓" if is_defined else "✗"
                print(f"   {status} {imp['name']}")
                
                if not is_defined:
                    all_imports_from_helpers[imp['name']].append(filepath)
            print()
    
    # Resumen de funciones faltantes
    if all_imports_from_helpers:
        print("="*70)
        print("❌ FUNCIONES FALTANTES EN helpers.py")
        print("="*70 + "\n")
        
        for func_name, files in sorted(all_imports_from_helpers.items()):
            print(f"• {func_name}")
            for f in files:
                print(f"    - Usado en: {f}")
            print()
        
        print("💡 SOLUCIÓN:")
        print("   Agrega estas funciones a src/utils/helpers.py:")
        print()
        for func_name in sorted(all_imports_from_helpers.keys()):
            print(f"   def {func_name}(...):")
            print(f"       pass")
            print()
        
        return False
    else:
        print("="*70)
        print("✅ TODAS LAS IMPORTACIONES ESTÁN CORRECTAS")
        print("="*70)
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)