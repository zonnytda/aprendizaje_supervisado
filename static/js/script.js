// JavaScript del proyecto Egresados UNE
// TODO: Implementar

/* ============================================
   PROYECTO: PREDICCIÓN EGRESADOS UNE
   Archivo: script.js
   Descripción: Scripts JavaScript para interactividad
   Autor: Estudiante de Ingeniería Estadística e Informática
   Fecha: Octubre 2025
   ============================================ */

// ============================================
// VARIABLES GLOBALES
// ============================================
let currentModel = 'ridge';
let predictionResult = null;

// ============================================
// INICIALIZACIÓN
// ============================================
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎓 Aplicación Egresados UNE cargada');
    
    // Inicializar componentes
    initializeTooltips();
    initializeAnimations();
    initializeFormValidation();
    initializeCharts();
    
    // Event listeners
    setupEventListeners();
    
    // Formatear números
    formatNumbers();
});

// ============================================
// TOOLTIPS DE BOOTSTRAP
// ============================================
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(
        document.querySelectorAll('[data-bs-toggle="tooltip"]')
    );
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// ============================================
// ANIMACIONES
// ============================================
function initializeAnimations() {
    // Animación de entrada para cards
    const cards = document.querySelectorAll('.card');
    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        
        setTimeout(() => {
            card.style.transition = 'all 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });
    
    // Animación de números contadores
    animateCounters();
}

function animateCounters() {
    const counters = document.querySelectorAll('[data-count]');
    
    counters.forEach(counter => {
        const target = parseInt(counter.getAttribute('data-count'));
        const duration = 2000; // 2 segundos
        const increment = target / (duration / 16); // 60 FPS
        let current = 0;
        
        const updateCounter = () => {
            current += increment;
            if (current < target) {
                counter.textContent = Math.floor(current).toLocaleString('es-PE');
                requestAnimationFrame(updateCounter);
            } else {
                counter.textContent = target.toLocaleString('es-PE');
            }
        };
        
        updateCounter();
    });
}

// ============================================
// VALIDACIÓN DE FORMULARIOS
// ============================================
function initializeFormValidation() {
    const forms = document.querySelectorAll('.needs-validation');
    
    Array.from(forms).forEach(form => {
        form.addEventListener('submit', event => {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// ============================================
// EVENT LISTENERS
// ============================================
function setupEventListeners() {
    // Botón de predicción
    const predictBtn = document.getElementById('predictBtn');
    if (predictBtn) {
        predictBtn.addEventListener('click', handlePrediction);
    }
    
    // Formulario de predicción
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Seleccionar modelo
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', function(e) {
            currentModel = e.target.value;
            console.log('Modelo seleccionado:', currentModel);
        });
    }
    
    // Actualizar semestre al cambiar año de egreso
    const annioEgreso = document.getElementById('annioEgreso');
    if (annioEgreso) {
        annioEgreso.addEventListener('change', updateSemestreEgreso);
    }
}

// ============================================
// FUNCIONES DE PREDICCIÓN
// ============================================
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    // Validar formulario
    if (!event.target.checkValidity()) {
        event.target.classList.add('was-validated');
        return;
    }
    
    // Mostrar loading
    showLoading();
    
    // Recopilar datos del formulario
    const formData = new FormData(event.target);
    const data = {};
    
    formData.forEach((value, key) => {
        // Convertir a número si es necesario
        if (['CREDITOS_ACUMULADOS', 'ANNIO_MATRICULA1', 'ANNIO_EGRESO', 
             'SEMESTRE_EGRESO', 'ANNIO_SOLICITUD_TRAMITE'].includes(key)) {
            data[key] = parseInt(value);
        } else {
            data[key] = value;
        }
    });
    
    try {
        // Hacer request a la API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayPredictionResult(result);
        } else {
            showError(result.error || 'Error al realizar la predicción');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Error de conexión. Por favor, intenta de nuevo.');
    } finally {
        hideLoading();
    }
}

function displayPredictionResult(result) {
    // Actualizar valores en el DOM
    document.getElementById('prediccionValor').textContent = result.prediction.toFixed(2);
    document.getElementById('intervaloConfianza').textContent = 
        `[${result.lower_bound.toFixed(2)}, ${result.upper_bound.toFixed(2)}]`;
    document.getElementById('percentilValor').textContent = result.percentile.toFixed(1) + '%';
    document.getElementById('interpretacion').textContent = result.interpretation;
    document.getElementById('modeloUsado').textContent = result.model_type.toUpperCase();
    
    // Actualizar barra de percentil
    const percentilBar = document.getElementById('percentilBar');
    percentilBar.style.width = result.percentile + '%';
    percentilBar.textContent = result.percentile.toFixed(1) + '%';
    
    // Cambiar color según el percentil
    if (result.percentile >= 75) {
        percentilBar.className = 'progress-bar bg-success';
    } else if (result.percentile >= 50) {
        percentilBar.className = 'progress-bar bg-info';
    } else if (result.percentile >= 25) {
        percentilBar.className = 'progress-bar bg-warning';
    } else {
        percentilBar.className = 'progress-bar bg-danger';
    }
    
    // Mostrar card de resultados con animación
    const resultCard = document.getElementById('resultadoCard');
    resultCard.style.display = 'block';
    resultCard.classList.add('fade-in');
    
    // Scroll suave a resultados
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    
    // Guardar resultado
    predictionResult = result;
}

// ============================================
// FUNCIONES AUXILIARES
// ============================================
function updateSemestreEgreso() {
    const annio = document.getElementById('annioEgreso').value;
    const semestre = document.getElementById('semestre');
    
    if (annio && semestre) {
        // Sugerir semestre basado en el año
        semestre.value = `${annio}1`;
    }
}

function showLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loadingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

function showError(message) {
    // Crear alerta de error
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger alert-dismissible fade show';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <i class="fas fa-exclamation-circle"></i> <strong>Error:</strong> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Insertar al inicio del container
    const container = document.querySelector('main .container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-remover después de 5 segundos
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    } else {
        alert(message);
    }
}

function showSuccess(message) {
    // Crear alerta de éxito
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-success alert-dismissible fade show';
    alertDiv.role = 'alert';
    alertDiv.innerHTML = `
        <i class="fas fa-check-circle"></i> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('main .container');
    if (container) {
        container.insertBefore(alertDiv, container.firstChild);
        
        setTimeout(() => {
            alertDiv.remove();
        }, 5000);
    }
}

// ============================================
// FORMATEO DE NÚMEROS
// ============================================
function formatNumbers() {
    // Formatear todos los elementos con data-number
    const numberElements = document.querySelectorAll('[data-number]');
    
    numberElements.forEach(element => {
        const number = parseFloat(element.textContent);
        if (!isNaN(number)) {
            element.textContent = number.toLocaleString('es-PE');
        }
    });
    
    // Formatear decimales con data-decimal
    const decimalElements = document.querySelectorAll('[data-decimal]');
    
    decimalElements.forEach(element => {
        const number = parseFloat(element.textContent);
        const decimals = parseInt(element.getAttribute('data-decimal')) || 2;
        if (!isNaN(number)) {
            element.textContent = number.toFixed(decimals);
        }
    });
}

// ============================================
// GRÁFICAS (Chart.js)
// ============================================
function initializeCharts() {
    // Esta función se puede expandir para inicializar gráficas específicas
    console.log('Inicializando gráficas...');
}

// Función auxiliar para crear gráficas
function createChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.warn(`Canvas ${canvasId} no encontrado`);
        return null;
    }
    
    const ctx = canvas.getContext('2d');
    return new Chart(ctx, config);
}

// ============================================
// EXPORTAR RESULTADOS
// ============================================
function exportToPDF() {
    window.print();
}

function exportToCSV(data, filename) {
    // Convertir datos a CSV
    const csv = convertToCSV(data);
    
    // Crear blob y descargar
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    window.URL.revokeObjectURL(url);
}

function convertToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const rows = data.map(row => 
        headers.map(header => JSON.stringify(row[header] || '')).join(',')
    );
    
    return [headers.join(','), ...rows].join('\n');
}

// ============================================
// COPIAR AL PORTAPAPELES
// ============================================
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showSuccess('Copiado al portapapeles');
    }).catch(err => {
        console.error('Error al copiar:', err);
        showError('No se pudo copiar al portapapeles');
    });
}

// ============================================
// VALIDACIONES PERSONALIZADAS
// ============================================
function validateCreditosAcumulados(input) {
    const value = parseInt(input.value);
    
    if (value < 0 || value > 300) {
        input.setCustomValidity('Los créditos deben estar entre 0 y 300');
        return false;
    }
    
    input.setCustomValidity('');
    return true;
}

function validateAnnioMatricula(input) {
    const annioMatricula = parseInt(input.value);
    const annioEgreso = parseInt(document.getElementById('annioEgreso').value);
    
    if (annioMatricula > annioEgreso) {
        input.setCustomValidity('El año de matrícula no puede ser posterior al año de egreso');
        return false;
    }
    
    input.setCustomValidity('');
    return true;
}

// ============================================
// UTILIDADES DE UI
// ============================================
function toggleSection(sectionId) {
    const section = document.getElementById(sectionId);
    if (section) {
        section.classList.toggle('d-none');
    }
}

function smoothScroll(targetId) {
    const target = document.getElementById(targetId);
    if (target) {
        target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// ============================================
// NAVEGACIÓN ENTRE PÁGINAS
// ============================================
function navigateTo(url) {
    window.location.href = url;
}

function navigateBack() {
    window.history.back();
}

// ============================================
// MANEJO DE ERRORES GLOBAL
// ============================================
window.addEventListener('error', function(e) {
    console.error('Error global:', e.error);
    // No mostrar errores al usuario en producción
    // showError('Ha ocurrido un error. Por favor, recarga la página.');
});

// ============================================
// THEME TOGGLE (opcional para dark mode)
// ============================================
function toggleTheme() {
    const body = document.body;
    body.classList.toggle('dark-mode');
    
    // Guardar preferencia
    const isDark = body.classList.contains('dark-mode');
    localStorage.setItem('darkMode', isDark);
}

// Cargar tema guardado
function loadTheme() {
    const isDark = localStorage.getItem('darkMode') === 'true';
    if (isDark) {
        document.body.classList.add('dark-mode');
    }
}

// ============================================
// ESTADÍSTICAS EN TIEMPO REAL
// ============================================
function updateStats() {
    fetch('/api/dataset-info')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                // Actualizar estadísticas en la página
                const stats = data.data;
                
                // Ejemplo de actualización
                const totalElement = document.getElementById('total-registros');
                if (totalElement) {
                    totalElement.textContent = stats.total_registros.toLocaleString('es-PE');
                }
            }
        })
        .catch(error => console.error('Error al obtener estadísticas:', error));
}

// ============================================
// WEBSOCKETS (opcional para actualizaciones en tiempo real)
// ============================================
/*
let socket = null;

function initializeWebSocket() {
    socket = new WebSocket('ws://localhost:5000/ws');
    
    socket.onopen = function(e) {
        console.log('WebSocket conectado');
    };
    
    socket.onmessage = function(event) {
        console.log('Mensaje recibido:', event.data);
        // Manejar mensaje
    };
    
    socket.onclose = function(event) {
        console.log('WebSocket cerrado');
    };
    
    socket.onerror = function(error) {
        console.error('Error WebSocket:', error);
    };
}
*/

// ============================================
// FUNCIONES DE DEBUGGING (solo desarrollo)
// ============================================
if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    // Funciones de debug
    window.debugApp = {
        showPrediction: () => console.log('Prediction result:', predictionResult),
        clearStorage: () => localStorage.clear(),
        testError: () => showError('Error de prueba'),
        testSuccess: () => showSuccess('Éxito de prueba'),
    };
    
    console.log('🔧 Modo debug activado. Usa window.debugApp para funciones de debug.');
}

// ============================================
// INICIALIZACIÓN FINAL
// ============================================
console.log('✅ Scripts cargados correctamente');

// Exportar funciones necesarias globalmente
window.appFunctions = {
    showLoading,
    hideLoading,
    showError,
    showSuccess,
    copyToClipboard,
    exportToPDF,
    exportToCSV,
    toggleTheme,
    navigateTo,
    smoothScroll
};

// ============================================
// FIN DE SCRIPTS
// ============================================