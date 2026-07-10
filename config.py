"""
config.py — Parámetros centralizados del pipeline de valoración de opciones.
"""

from datetime import date, timedelta


# Activo subyacente y ventana histórica de calibración
TICKER = "NVDA"  # confirmado

_HOY = date(2026, 7, 3)  # Fecha fija 
FECHA_FIN = _HOY.isoformat()
FECHA_INICIO = (_HOY - timedelta(days=365 * 2)).isoformat()  # ventana rodante de 2 años

# Parámetros del contrato
TIPO_OPCION = "call"

# K y T dinámicos: K = ATM sobre S0 (definido en runtime tras descargar precios).
# Fijados al contrato REAL cotizado en Yahoo Finance:
#   NVDA270115C00195000, vence 2027-01-15, Call, strike $195
#   Bid (Oferta) = 25.55, Ask (Precio de compra) = 26.00
K_FIJO = 195.0

_FECHA_VENCIMIENTO_OPCION = date(2027, 1, 15)
# T_FIJO se calcula dinámicamente respecto a hoy, no como fracción fija
T_FIJO = (_FECHA_VENCIMIENTO_OPCION - _HOY).days / 365

T_DEFAULT = 0.5      # valido solo si T_FIJO es None
R = 0.05            # tasa libre de riesgo anual

# Volatilidad implícita — precio de mercado real (Módulo 3)
# Precio de mercado real: midpoint bid-ask de NVDA270115C00195000
# Bid = 25.55, Ask = 26.00 -> midpoint = 25.775
C_MERCADO = 25.775

# Monte Carlo (medida real P)
MC_N_PASOS = 252
MC_N_SIM = 100_000
MC_SEED = 42

# Malla de Crank-Nicolson
S_MAX_FACTOR = 3.0      # S_max = S_MAX_FACTOR * K

BARRIDO_CONVERGENCIA_MN = [
    (25, 25),
    (50, 50),
    (100, 100),
    (200, 200),
    (400, 400),
    (800, 800),
    (1600, 1600),
]

# Newton-Raphson (volatilidad implícita)
NR_TOL = 1e-8
NR_MAX_ITER = 100