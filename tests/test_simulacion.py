import numpy as np
from src.extracion_datos import DatosSubyacente
from src.simulacion import CalibracionMLE, MonteCarloGBM

def test_calibracion_mle():
    # Creamos un activo ficticio con datos conocidos para probar la matemática
    activo = DatosSubyacente("TEST", "2023-01-01", "2023-01-10")
    # Simular precios fijos
    activo.precios = np.array([100.0, 101.0, 102.0, 101.0, 100.0])
    activo.retornos = np.diff(np.log(activo.precios)) # log-retornos
    
    calibrador = CalibracionMLE(activo)
    mu, sigma = calibrador.estimar()
    
    # Verificaciones (asserts)
    assert mu is not None
    assert sigma > 0
    assert len(activo.retornos) == 4

def test_montecarlo_gbm():
    # Creamos y calibramos un activo de prueba
    activo = DatosSubyacente("TEST", "2023-01-01", "2023-01-10")
    activo.precios = np.array([100.0, 101.0, 102.0, 101.0, 100.0])
    activo.retornos = np.diff(np.log(activo.precios))
    
    calibrador = CalibracionMLE(activo)
    calibrador.estimar()
    
    # Probamos el simulador de Monte Carlo
    mc = MonteCarloGBM(calibrador)
    trayectorias = mc.simular(T=0.5, n_pasos=126, n_sim=1000, seed=42)
    
    # Verificaciones
    assert trayectorias is not None
    assert trayectorias.shape == (1000, 127)  # N = 1000, M + 1 = 127
    assert mc.S_T is not None
    assert len(mc.S_T) == 1000
    
    # Todos los trayectos inician en S0
    assert np.all(trayectorias[:, 0] == mc.S0)
    
    # Probabilidad de quedar ITM para un strike igual a S0 (debería rondar el 50%)
    prob = mc.prob_ITM(mc.S0)
    assert 0.4 <= prob <= 0.6
