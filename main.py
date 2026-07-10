"""
main.py — Pipeline completo de valoración de opciones financieras.

Cadena de dependencias:
    DatosSubyacente -> CalibracionMLE -> MonteCarloGBM -> CrankNicolson -> Griegas

Ejecuta de punta a punta: descarga de precios, calibración de sigma/mu,
simulación Monte Carlo bajo medida real P, valoración Crank-Nicolson y
Black-Scholes bajo medida neutral al riesgo Q, griegas, y volatilidad
implícita vía Newton-Raphson.

Todos los parámetros viven en config.py — no se redefinen aquí.
"""

from src.extracion_datos import DatosSubyacente
from src.simulacion import CalibracionMLE, MonteCarloGBM
from src.pde_solver import CrankNicolson, BlackScholesAnalitico, analisis_convergencia
from src.griegas import Griegas, VolatilidadImplicita
import src.resultados_io as rio
import config
from src.generar_graficas import regenerar_graficas


def main():

    print("=" * 60)
    print("MÓDULO 0 — Extracción de datos históricos")
    print("=" * 60)

    activo = DatosSubyacente(config.TICKER, config.FECHA_INICIO, config.FECHA_FIN)
    activo.obtener_precios()
    activo.obtener_log_retornos()

    S0 = activo.precios[-1]
    K = config.K_FIJO if config.K_FIJO is not None else S0  # ATM dinámico si no hay strike real
    T = config.T_FIJO if config.T_FIJO is not None else config.T_DEFAULT
    r = config.R
    TIPO = config.TIPO_OPCION

    print(f"Ticker             : {config.TICKER}")
    print(f"Ventana histórica   : {config.FECHA_INICIO} -> {config.FECHA_FIN}")
    print(f"Observaciones       : {len(activo.precios)} precios diarios")
    print(f"S0 (último precio)  : {S0:.2f}")
    print(f"K                   : {K:.2f}  ({'ATM dinámico' if config.K_FIJO is None else 'strike real de mercado'})")
    print(f"T                   : {T:.4f} años  ({'default' if config.T_FIJO is None else 'vencimiento real de mercado'})")

    print("\n" + "=" * 60)
    print("MÓDULO 1 — Calibración MLE y Monte Carlo GBM")
    print("=" * 60)

    calibrador = CalibracionMLE(activo)
    calibrador.estimar()
    calibrador.resume()
    rio.guardar_calibracion(activo, calibrador)

    mc = MonteCarloGBM(calibrador, S0=S0)
    mc.simular(T=T, n_pasos=config.MC_N_PASOS, n_sim=config.MC_N_SIM, seed=config.MC_SEED)
    prob_empirica = mc.prob_ITM(K, tipo=TIPO)
    rio.guardar_monte_carlo(mc, K=K, tipo=TIPO)
    print(f"\nP({'S_T > K' if TIPO == 'call' else 'S_T < K'}) empírica (medida P): {prob_empirica:.4f}")

    print("\n" + "=" * 60)
    print("MÓDULO 2 — Crank-Nicolson vs Black-Scholes analítico")
    print("=" * 60)

    sigma = calibrador.sigma   # sigma viaja al pricing; mu se queda en Monte Carlo

    cn = CrankNicolson(S0, K, T, r, sigma, tipo=TIPO, S_max_factor=config.S_MAX_FACTOR)
    cn.resolver()

    bs = BlackScholesAnalitico(S0, K, T, r, sigma, tipo=TIPO)
    precio_exacto, prob_riesgo_neutral = bs.calcular()
    rio.guardar_pde(cn, precio_exacto, prob_riesgo_neutral)

    error_cn = abs(cn.precio - precio_exacto)

    print(f"Precio Crank-Nicolson : {cn.precio:.4f}")
    print(f"Precio BS analítico   : {precio_exacto:.4f}")
    print(f"Error |CN - BS|       : {error_cn:.6f}")

    prima_riesgo = prob_empirica - prob_riesgo_neutral
    etiqueta_prob = "N(d2)" if TIPO == "call" else "N(-d2)"
    print(f"\n{etiqueta_prob} (medida Q)         : {prob_riesgo_neutral:.4f}")
    print(f"Prima de riesgo (P - Q) : {prima_riesgo:+.4f}")

    print("\n" + "=" * 60)
    print("Análisis de convergencia Crank-Nicolson")
    print("=" * 60)

    # refinamiento espera tuplas (M, N) -- M = pasos temporales, N = nodos
    # espaciales. Sin esto, analisis_convergencia usaba su default interno
    # (que no incluye 1000 y usa 500 en vez de 400).
    tabla_convergencia = analisis_convergencia(
        S0, K, T, r, sigma, tipo=TIPO,
        refinamiento=config.BARRIDO_CONVERGENCIA_MN,
    )
    rio.guardar_convergencia(tabla_convergencia)
    print(tabla_convergencia.to_string(index=False))

    print("\n" + "=" * 60)
    print("MÓDULO 3 — Griegas y Volatilidad Implícita")
    print("=" * 60)

    griegas = Griegas(cn)
    griegas_dict = griegas.calcular_todas()  # una sola vez -- Vega/Rho resuelven 2 CN cada una
    rio.guardar_griegas(griegas, griegas_dict)

    print(f"{'Griega':<10} {'Valor':>12}")
    print("-" * 24)
    for nombre, valor in griegas_dict.items():
        print(f"{nombre:<10} {valor:>12.6f}")

    print("\nVolatilidad implícita:")
    if config.C_MERCADO is not None:
        C_mercado = config.C_MERCADO
        print(f"C_mercado (real, Yahoo Finance): {C_mercado:.4f}")
    else:
        # Placeholder de prueba de humo -- NO USAR EN EL REPORTE FINAL.
        # Es circular: se genera desde el propio precio de Crank-Nicolson.
        C_mercado = cn.precio * 1.05
        print(f"[AVISO] C_MERCADO no está definido en config.py.")
        print(f"Usando placeholder sintético para prueba: {C_mercado:.4f}")
        print("Este valor es circular (deriva de cn.precio) y NO es válido para el reporte.")

    vi = VolatilidadImplicita(C_mercado, S0, K, T, r, tipo=TIPO)
    vi.resolver(sigma_0=sigma, tol=config.NR_TOL, max_iter=config.NR_MAX_ITER)
    vi.resumen()
    rio.guardar_volatilidad_implicita(vi, sigma_historica=sigma)

    print(f"\nsigma histórica (MLE)   : {sigma:.4f}")
    print(f"sigma implícita (NR)    : {vi.sigma_implicita:.4f}")
    print(f"Diferencia              : {vi.sigma_implicita - sigma:+.4f}")

    regenerar_graficas()

    print("\n" + "=" * 60)
    print("Pipeline completo ejecutado.")
    print("=" * 60)


if __name__ == "__main__":
    main()