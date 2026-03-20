"""
Generador de dataset sintético de permisos de circulación.

Genera datos con patrones de negocio reales para que la variable target
('estado') tenga relación con las features, permitiendo que un modelo
de ML pueda aprender patrones significativos.

Incluye inyección intencional de errores para practicar limpieza de datos.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config.config import (
    DATA_RAW_DIR,
    ERROR_RATES,
    NUM_RECORDS,
    RANDOM_SEED,
    RAW_FILE,
    TIPOS_VEHICULO,
    ZONAS,
)


def _calcular_probabilidad_inactivo(row, medianas_monto):
    """
    Calcula la probabilidad de que un permiso sea 'Inactivo' basándose
    en reglas de negocio realistas.
    """
    prob = 0.10  # probabilidad base

    # Duración larga → mayor riesgo de inactividad
    if row['duracion_dias'] >= 25:
        prob += 0.15
    elif row['duracion_dias'] <= 5:
        prob += 0.10

    # Camiones en Zona A (zona restringida) → más inactivos
    if row['tipo_vehiculo'] == 'Camion' and row['zona_circulacion'] == 'Zona A':
        prob += 0.20

    # Monopatines → regulación nueva, más inactivos
    if row['tipo_vehiculo'] == 'Monopatin':
        prob += 0.12

    # Permisos emitidos en fin de semana → mayor irregularidad
    if row['fecha_emision'].weekday() >= 5:
        prob += 0.10

    # Antigüedad alta → más probable que expire
    dias_antiguedad = (datetime(2025, 7, 1) - row['fecha_emision']).days
    if dias_antiguedad > 300:
        prob += 0.15
    elif dias_antiguedad > 200:
        prob += 0.08

    # Infracciones previas → fuerte predictor
    prob += row['infracciones_previas'] * 0.08

    # Monto bajo pagado → posible irregularidad
    mediana = medianas_monto.get(row['tipo_vehiculo'], 50000)
    if row['monto_pagado'] < mediana * 0.5:
        prob += 0.12

    # Renovación → ligeramente más estable
    if row['renovacion']:
        prob -= 0.05

    return np.clip(prob, 0.05, 0.90)


def generate_synthetic_data(num_records=NUM_RECORDS, seed=RANDOM_SEED):
    """
    Genera un dataset sintético de permisos de circulación.

    Args:
        num_records: Número de registros base a generar.
        seed: Semilla para reproducibilidad.

    Returns:
        pd.DataFrame con datos generados incluyendo errores intencionales.
    """
    np.random.seed(seed)

    # --- Features base ---
    tipos = np.random.choice(TIPOS_VEHICULO, num_records)
    zonas = np.random.choice(ZONAS, num_records)
    duracion = np.random.randint(1, 30, num_records)
    fechas = [
        datetime(2025, 7, 1) - timedelta(days=int(np.random.randint(1, 365)))
        for _ in range(num_records)
    ]

    # Monto pagado varía por tipo de vehículo
    monto_base = {
        'Coche': 50000, 'Moto': 25000, 'Camion': 120000,
        'Furgoneta': 80000, 'Bicicleta': 8000, 'Monopatin': 5000
    }
    montos = np.array([
        max(1000, np.random.normal(monto_base[t], monto_base[t] * 0.3))
        for t in tipos
    ])

    # Renovación: 30% son renovaciones
    renovacion = np.random.choice([True, False], num_records, p=[0.3, 0.7])

    # Infracciones previas: distribución sesgada a 0
    infracciones = np.random.choice(
        [0, 1, 2, 3, 4, 5], num_records,
        p=[0.50, 0.25, 0.12, 0.07, 0.04, 0.02]
    )

    df = pd.DataFrame({
        'id_permiso': np.arange(1, num_records + 1),
        'tipo_vehiculo': tipos,
        'fecha_emision': fechas,
        'duracion_dias': duracion,
        'zona_circulacion': zonas,
        'monto_pagado': np.round(montos, 2),
        'renovacion': renovacion,
        'infracciones_previas': infracciones,
    })

    # --- Asignar estado basado en reglas de negocio ---
    probs = df.apply(
        lambda row: _calcular_probabilidad_inactivo(row, monto_base), axis=1
    )
    df['estado'] = np.where(
        np.random.random(len(df)) < probs, 'Inactivo', 'Activo'
    )

    print("Distribución de estado (antes de errores):")
    print(df['estado'].value_counts(normalize=True).round(3))

    # --- Inyectar errores ---
    df = _inject_errors(df)

    return df


def _inject_errors(df):
    """Inyecta errores realistas en el DataFrame para practicar limpieza."""
    # 1. Valores faltantes
    df.loc[df.sample(frac=ERROR_RATES['nulls_duracion']).index, 'duracion_dias'] = np.nan
    df.loc[df.sample(frac=ERROR_RATES['nulls_zona']).index, 'zona_circulacion'] = np.nan

    # 2. Inconsistencias de formato
    idx_upper = df.sample(frac=ERROR_RATES['formato_upper']).index
    df.loc[idx_upper, 'tipo_vehiculo'] = df.loc[idx_upper, 'tipo_vehiculo'].str.upper()

    df.loc[df.sample(frac=ERROR_RATES['formato_espacio']).index, 'tipo_vehiculo'] = 'coche '
    df.loc[df.sample(frac=ERROR_RATES['categoria_invalida']).index, 'tipo_vehiculo'] = 'OTRO'

    # 3. Valores fuera de rango
    df.loc[df.sample(frac=ERROR_RATES['duracion_negativa']).index, 'duracion_dias'] = -10
    df.loc[df.sample(frac=ERROR_RATES['fecha_invalida']).index, 'fecha_emision'] = '2025-25-50'

    # 4. Montos negativos
    df.loc[df.sample(frac=0.01).index, 'monto_pagado'] = -500

    # 5. Duplicados
    df_dup = df.sample(frac=ERROR_RATES['duplicados']).copy()
    df = pd.concat([df, df_dup], ignore_index=True)

    return df


if __name__ == '__main__':
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    df = generate_synthetic_data()
    df.to_csv(RAW_FILE, index=False)

    print(f"\nDataset creado en: {RAW_FILE}")
    print(f"Total de registros: {df.shape[0]}")
    print(f"Columnas: {list(df.columns)}")
    print(f"\nPrimeras filas:\n{df.head()}")
