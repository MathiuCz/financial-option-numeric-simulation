"""
resultados_io.py — Serialización de resultados intermedios del pipeline.

main.py llama a estas funciones después de cada módulo. Los notebooks
(01, 02, 03) SOLO leen de aquí -- nunca vuelven a instanciar
DatosSubyacente, CrankNicolson, MonteCarloGBM, etc. Así se garantiza que
toda figura del reporte corresponde exactamente a la corrida numérica
que produjo los valores citados en el texto.

Convención de archivos, todos en data/resultados/:
    modulo1_calibracion.npz     -> precios, retornos, mu, sigma
    modulo1_montecarlo.npz      -> S_T, muestra de trayectorias, prob_itm
    modulo2_pde.npz             -> malla_S, malla_V, precio_cn, precio_bs
    modulo2_convergencia.csv    -> tabla de analisis_convergencia
    modulo3_griegas.npz         -> delta/gamma sobre la malla, theta/vega/rho
    modulo3_newton_raphson.csv  -> historial (iter, sigma, error)
    modulo3_volatilidad.json    -> sigma_historica, sigma_implicita, C_mercado
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTADOS_DIR = Path(__file__).resolve().parents[1] / "data" / "resultados"


def _ensure_dir() -> None:
    RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)


# Módulo 1 — calibración MLE y Monte Carlo

def guardar_calibracion(activo, calibrador) -> None:
    _ensure_dir()
    np.savez(
        RESULTADOS_DIR / "modulo1_calibracion.npz",
        precios=activo.precios,
        retornos=activo.retornos,
        mu=calibrador.mu,
        sigma=calibrador.sigma,
    )


def guardar_monte_carlo(mc, K: float, tipo: str, n_paths_muestra: int = 200) -> None:
    """
    Guarda S_T completo (histograma/densidad) y solo una MUESTRA de
    trayectorias completas (spaghetti plot). Guardar las 100,000
    trayectorias completas no aporta nada visualmente y pesa ~100x más.
    """
    _ensure_dir()
    n_sim = mc.trayectorias.shape[0]
    rng = np.random.default_rng(0)
    idx_muestra = rng.choice(n_sim, size=min(n_paths_muestra, n_sim), replace=False)

    np.savez(
        RESULTADOS_DIR / "modulo1_montecarlo.npz",
        S_T=mc.S_T,
        trayectorias_muestra=mc.trayectorias[idx_muestra, :],
        prob_itm=mc.prob_ITM(K, tipo=tipo),
        S0=mc.S0,
        K=K,
        tipo=tipo,
    )

# Módulo 2 — Crank-Nicolson, Black-Scholes, convergencia

def guardar_pde(cn, precio_exacto: float, prob_riesgo_neutral: float) -> None:
    _ensure_dir()
    np.savez(
        RESULTADOS_DIR / "modulo2_pde.npz",
        malla_S=cn.malla_S,
        malla_V=cn.malla_V,
        precio_cn=cn.precio,
        precio_bs=precio_exacto,
        prob_riesgo_neutral=prob_riesgo_neutral,
        S0=cn.S0, K=cn.K, T=cn.T, r=cn.r, sigma=cn.sigma, tipo=cn.tipo,
    )


def guardar_convergencia(tabla_convergencia: "pd.DataFrame") -> None:
    _ensure_dir()
    tabla_convergencia.to_csv(RESULTADOS_DIR / "modulo2_convergencia.csv", index=False)


# Módulo 3 — Griegas, volatilidad implícita

def guardar_griegas(griegas_obj, griegas_dict: dict) -> None:
    _ensure_dir()
    np.savez(
        RESULTADOS_DIR / "modulo3_griegas.npz",
        malla_S=griegas_obj.solver.malla_S,
        delta_malla=griegas_obj.delta,
        gamma_malla=griegas_obj.gamma,
        delta_S0=griegas_dict["delta"],
        gamma_S0=griegas_dict["gamma"],
        theta=griegas_dict["theta"],
        vega=griegas_dict["vega"],
        rho=griegas_dict["rho"],
    )


def guardar_volatilidad_implicita(vi, sigma_historica: float) -> None:
    _ensure_dir()
    historial_df = pd.DataFrame(vi.historial, columns=["iter", "sigma", "error"])
    historial_df.to_csv(RESULTADOS_DIR / "modulo3_newton_raphson.csv", index=False)

    with open(RESULTADOS_DIR / "modulo3_volatilidad.json", "w") as f:
        json.dump({
            "sigma_historica": float(sigma_historica),
            "sigma_implicita": float(vi.sigma_implicita),
            "C_mercado": float(vi.C_mercado),
        }, f, indent=2)


# Funciones de carga

def cargar_calibracion() -> dict:
    d = np.load(RESULTADOS_DIR / "modulo1_calibracion.npz")
    return {
        "precios": d["precios"],
        "retornos": d["retornos"],
        "mu": float(d["mu"]),
        "sigma": float(d["sigma"]),
    }


def cargar_monte_carlo() -> dict:
    d = np.load(RESULTADOS_DIR / "modulo1_montecarlo.npz")
    return {
        "S_T": d["S_T"],
        "trayectorias_muestra": d["trayectorias_muestra"],
        "prob_itm": float(d["prob_itm"]),
        "S0": float(d["S0"]),
        "K": float(d["K"]),
        "tipo": str(d["tipo"]),
    }


def cargar_pde() -> dict:
    d = np.load(RESULTADOS_DIR / "modulo2_pde.npz")
    return {
        "malla_S": d["malla_S"],
        "malla_V": d["malla_V"],
        "precio_cn": float(d["precio_cn"]),
        "precio_bs": float(d["precio_bs"]),
        "prob_riesgo_neutral": float(d["prob_riesgo_neutral"]),
        "S0": float(d["S0"]), "K": float(d["K"]), "T": float(d["T"]),
        "r": float(d["r"]), "sigma": float(d["sigma"]), "tipo": str(d["tipo"]),
    }


def cargar_convergencia() -> "pd.DataFrame":
    return pd.read_csv(RESULTADOS_DIR / "modulo2_convergencia.csv")


def cargar_griegas() -> dict:
    d = np.load(RESULTADOS_DIR / "modulo3_griegas.npz")
    return {
        "malla_S": d["malla_S"],
        "delta_malla": d["delta_malla"],
        "gamma_malla": d["gamma_malla"],
        "delta_S0": float(d["delta_S0"]),
        "gamma_S0": float(d["gamma_S0"]),
        "theta": float(d["theta"]),
        "vega": float(d["vega"]),
        "rho": float(d["rho"]),
    }


def cargar_newton_raphson() -> "pd.DataFrame":
    return pd.read_csv(RESULTADOS_DIR / "modulo3_newton_raphson.csv")


def cargar_volatilidad() -> dict:
    with open(RESULTADOS_DIR / "modulo3_volatilidad.json") as f:
        return json.load(f)