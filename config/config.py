"""
Configuración centralizada del proyecto.
"""
import os

# Reproducibilidad
RANDOM_SEED = 42

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')

RAW_FILE = os.path.join(DATA_RAW_DIR, 'permisos_circulacion_bruto.csv')
CLEAN_FILE = os.path.join(DATA_PROCESSED_DIR, 'permisos_circulacion_limpio.csv')

# Parámetros de generación de datos
NUM_RECORDS = 5000
TIPOS_VEHICULO = ['Coche', 'Moto', 'Camion', 'Furgoneta', 'Bicicleta', 'Monopatin']
ZONAS = ['Zona A', 'Zona B', 'Zona C', 'Zona D']

# Proporciones de errores inyectados
ERROR_RATES = {
    'nulls_duracion': 0.05,
    'nulls_zona': 0.02,
    'formato_upper': 0.03,
    'formato_espacio': 0.02,
    'categoria_invalida': 0.01,
    'duracion_negativa': 0.02,
    'fecha_invalida': 0.01,
    'duplicados': 0.01,
}

# Modelado
TEST_SIZE = 0.2
N_SPLITS_CV = 5
