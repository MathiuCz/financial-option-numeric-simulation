import os
from pathlib import Path
import numpy as np
import pytest
from src.extracion_datos import DatosSubyacente

def test_datos_subyacente_obtencion():
    # Usamos un rango de fechas corto y un ticker estándar para que el test sea rápido y descargue pocos datos
    spy = DatosSubyacente("SPY", "2023-01-01", "2023-01-15")
    
    # 1. Probar la obtención de precios
    precios = spy.obtener_precios()
    assert precios is not None
    assert isinstance(precios, np.ndarray)
    assert len(precios) > 0
    
    # 2. Probar la obtención de log-retornos
    retornos = spy.obtener_log_retornos()
    assert retornos is not None
    assert isinstance(retornos, np.ndarray)
    # Los log-retornos deben ser exactamente el número de precios menos 1 (por la pérdida del primer elemento)
    assert len(retornos) == len(precios) - 1

def test_datos_subyacente_guardar_csv(tmp_path):
    # tmp_path es un fixture de pytest que crea un directorio temporal único para esta prueba
    spy = DatosSubyacente("SPY", "2023-01-01", "2023-01-10")
    
    # Definimos una ruta temporal para guardar el archivo
    ruta_temp = tmp_path / "test_spy.csv"
    
    # Ejecutamos el guardado
    spy.guardar_csv(ruta=ruta_temp)
    
    # Verificaciones
    assert os.path.exists(ruta_temp)
    assert ruta_temp.stat().st_size > 0
