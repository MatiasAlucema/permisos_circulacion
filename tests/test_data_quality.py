"""Tests de calidad de datos.

Valida que los datos generados y procesados cumplan con las reglas de negocio
y constraints esperados. Estos tests actuan como "data contracts".
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CLEAN_FILE, NUM_RECORDS, RAW_FILE, TIPOS_VEHICULO, ZONAS

# --- Fixtures ---


@pytest.fixture(scope="module")
def raw_data():
    """Carga datos brutos."""
    if not os.path.exists(RAW_FILE):
        pytest.skip("Datos brutos no generados. Ejecute data_generation.py primero.")
    return pd.read_csv(RAW_FILE)


@pytest.fixture(scope="module")
def clean_data():
    """Carga datos limpios."""
    if not os.path.exists(CLEAN_FILE):
        pytest.skip("Datos limpios no disponibles. Ejecute notebook 02 primero.")
    return pd.read_csv(CLEAN_FILE)


# --- Tests de datos brutos ---


class TestRawData:
    """Validaciones sobre los datos brutos generados."""

    def test_raw_has_expected_row_count(self, raw_data):
        """El CSV bruto tiene aprox NUM_RECORDS filas (puede tener duplicados)."""
        assert len(raw_data) >= NUM_RECORDS * 0.95

    def test_raw_has_required_columns(self, raw_data):
        """Columnas requeridas estan presentes."""
        required = {
            "id_permiso",
            "tipo_vehiculo",
            "fecha_emision",
            "duracion_dias",
            "zona_circulacion",
            "monto_pagado",
            "renovacion",
            "infracciones_previas",
            "estado",
        }
        assert required.issubset(set(raw_data.columns))

    def test_raw_has_intentional_errors(self, raw_data):
        """Los datos brutos contienen errores inyectados (nulls, etc)."""
        # Debe haber algun null en duracion_dias
        assert raw_data["duracion_dias"].isna().sum() > 0


# --- Tests de datos limpios ---


class TestCleanData:
    """Validaciones sobre los datos limpios (post-cleaning)."""

    def test_no_null_values(self, clean_data):
        """No hay valores nulos en datos limpios."""
        assert clean_data.isna().sum().sum() == 0

    def test_no_duplicates(self, clean_data):
        """No hay filas duplicadas."""
        assert clean_data.duplicated().sum() == 0

    def test_valid_vehicle_types(self, clean_data):
        """Todos los tipos de vehiculo son validos."""
        valid_types = set(TIPOS_VEHICULO)
        actual_types = set(clean_data["tipo_vehiculo"].unique())
        assert actual_types.issubset(valid_types)

    def test_valid_zones(self, clean_data):
        """Todas las zonas son validas."""
        valid_zones = set(ZONAS)
        actual_zones = set(clean_data["zona_circulacion"].unique())
        assert actual_zones.issubset(valid_zones)

    def test_positive_duration(self, clean_data):
        """Todas las duraciones son positivas."""
        assert (clean_data["duracion_dias"] > 0).all()

    def test_positive_amount(self, clean_data):
        """Todos los montos son positivos."""
        assert (clean_data["monto_pagado"] > 0).all()

    def test_valid_infractions_range(self, clean_data):
        """Infracciones estan en rango 0-5."""
        assert (clean_data["infracciones_previas"] >= 0).all()
        assert (clean_data["infracciones_previas"] <= 5).all()

    def test_valid_estado_values(self, clean_data):
        """Estado es 'Activo' o 'Inactivo'."""
        assert set(clean_data["estado"].unique()) == {"Activo", "Inactivo"}

    def test_class_distribution_is_imbalanced(self, clean_data):
        """La distribucion de clases es desbalanceada (~70/30)."""
        ratio = (clean_data["estado"] == "Inactivo").mean()
        assert 0.15 < ratio < 0.50  # No debe ser extremo ni balanceado

    def test_valid_dates(self, clean_data):
        """Todas las fechas son parseables."""
        dates = pd.to_datetime(clean_data["fecha_emision"], errors="coerce")
        assert dates.isna().sum() == 0

    def test_minimum_records(self, clean_data):
        """Hay al menos 4000 registros limpios."""
        assert len(clean_data) >= 4000
