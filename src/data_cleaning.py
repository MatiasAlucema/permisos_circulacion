"""
data_cleaning.py -- Limpieza y validacion de datos de permisos de circulacion.

Pipeline reproducible que transforma el CSV bruto en un dataset limpio:
1. Elimina duplicados
2. Estandariza tipo_vehiculo (capitalize, mapea 'Otro' a moda)
3. Parsea y valida fechas (elimina filas con fechas invalidas)
4. Corrige duracion_dias negativos/nulos con mediana
5. Corrige monto_pagado negativos con mediana por tipo_vehiculo
6. Imputa nulos categoricos (zona y tipo) con moda
7. Valida constraints finales

Uso:
    python src/data_cleaning.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CLEAN_FILE, DATA_PROCESSED_DIR, RAW_FILE


def clean_data(input_path: str = RAW_FILE, output_path: str = CLEAN_FILE) -> pd.DataFrame:
    """Ejecuta el pipeline completo de limpieza."""
    df = pd.read_csv(input_path)
    n_original = len(df)
    print(f"[INPUT] {n_original} filas x {df.shape[1]} columnas")

    # 1. Eliminar duplicados
    df.drop_duplicates(inplace=True)
    print(f"[Duplicados] {n_original} -> {len(df)} filas (eliminados: {n_original - len(df)})")

    # 2. Estandarizar tipo_vehiculo
    df["tipo_vehiculo"] = df["tipo_vehiculo"].str.strip().str.capitalize()
    df.loc[df["tipo_vehiculo"] == "Otro", "tipo_vehiculo"] = np.nan
    print(f"[Formato] tipo_vehiculo estandarizado. Unicos: {df['tipo_vehiculo'].nunique()}")

    # 3. Parsear fechas (invalidas -> NaT -> eliminar)
    df["fecha_emision"] = pd.to_datetime(df["fecha_emision"], errors="coerce")
    n_fechas_nat = df["fecha_emision"].isnull().sum()
    df.dropna(subset=["fecha_emision"], inplace=True)
    print(f"[Fechas] Invalidas eliminadas: {n_fechas_nat}, restantes: {len(df)}")

    # 4. Corregir duracion_dias negativos/nulos con mediana
    mediana_duracion = df.loc[df["duracion_dias"] > 0, "duracion_dias"].median()
    n_neg = (df["duracion_dias"] < 0).sum()
    n_null = df["duracion_dias"].isnull().sum()
    df.loc[df["duracion_dias"] < 0, "duracion_dias"] = mediana_duracion
    df["duracion_dias"] = df["duracion_dias"].fillna(mediana_duracion)
    print(f"[duracion_dias] Negativos: {n_neg}, Nulos: {n_null} (mediana={mediana_duracion})")

    # 5. Corregir monto_pagado negativos con mediana por tipo_vehiculo
    n_monto_neg = (df["monto_pagado"] < 0).sum()
    medianas_monto = (
        df.loc[df["monto_pagado"] > 0].groupby("tipo_vehiculo")["monto_pagado"].median()
    )
    for tipo, mediana in medianas_monto.items():
        mask = (df["monto_pagado"] < 0) & (df["tipo_vehiculo"] == tipo)
        df.loc[mask, "monto_pagado"] = mediana
    # Negativos restantes (sin tipo asignado)
    df.loc[df["monto_pagado"] < 0, "monto_pagado"] = df.loc[
        df["monto_pagado"] > 0, "monto_pagado"
    ].median()
    print(f"[monto_pagado] Negativos corregidos: {n_monto_neg}")

    # 6. Imputar nulos categoricos con moda
    n_null_zona = df["zona_circulacion"].isnull().sum()
    df["zona_circulacion"] = df["zona_circulacion"].fillna(df["zona_circulacion"].mode()[0])
    print(f"[zona_circulacion] Nulos imputados: {n_null_zona}")

    n_null_tipo = df["tipo_vehiculo"].isnull().sum()
    df["tipo_vehiculo"] = df["tipo_vehiculo"].fillna(df["tipo_vehiculo"].mode()[0])
    print(f"[tipo_vehiculo] Nulos imputados: {n_null_tipo}")

    # 7. Validaciones
    assert df.isnull().sum().sum() == 0, "Quedan valores nulos"
    assert df.duplicated().sum() == 0, "Quedan duplicados"
    assert (df["duracion_dias"] > 0).all(), "Hay duraciones no positivas"
    assert (df["monto_pagado"] > 0).all(), "Hay montos no positivos"
    assert (
        df["tipo_vehiculo"]
        .isin(["Coche", "Moto", "Camion", "Furgoneta", "Bicicleta", "Monopatin"])
        .all()
    ), "Categorias invalidas en tipo_vehiculo"
    assert df["zona_circulacion"].isin(["Zona A", "Zona B", "Zona C", "Zona D"]).all(), (
        "Categorias invalidas en zona"
    )
    print("[OK] Todas las validaciones pasaron")

    # 8. Guardar
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OUTPUT] {len(df)} filas guardadas en {output_path}")

    return df


if __name__ == "__main__":
    clean_data()
