import os
import matplotlib.pyplot as plt
import config
from src.resultados_io import (
    cargar_calibracion,
    cargar_monte_carlo,
    cargar_pde,
    cargar_convergencia,
    cargar_griegas
)

def regenerar_graficas():
    """Genera las gráficas a partir de los datos almacenados y las guarda en directorios."""
    print("\n" + "=" * 60)
    print("MÓDULO 4 — Generación Automática de Gráficas")
    print("=" * 60)
    
    os.makedirs("imagenes", exist_ok=True)
    os.makedirs(os.path.join("informe", "figuras"), exist_ok=True)
    plt.rcParams["figure.figsize"] = (10, 5)

    print("Cargando datos numéricos para gráficas...")
    calib = cargar_calibracion()
    mc = cargar_monte_carlo()
    tabla_convergencia = cargar_convergencia()

    print("Guardando precios históricos...")
    plt.figure()
    plt.plot(calib["precios"], linewidth=1)
    plt.title(f"Precio de cierre ajustado — {config.TICKER}")
    plt.xlabel("Día de trading")
    plt.ylabel("Precio (USD)")
    plt.grid(alpha=0.3)
    plt.savefig("imagenes/precios_historicos.png", dpi=300, bbox_inches='tight')
    plt.savefig("informe/figuras/precios_historicos.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Guardando log-retornos y distribución...")
    retornos = calib["retornos"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(retornos, linewidth=0.8)
    axes[0].set_title("Log-retornos diarios")
    axes[0].set_xlabel("Día de trading")
    axes[0].set_ylabel("$r_i$")
    axes[0].grid(alpha=0.3)
    axes[1].hist(retornos, bins=40, density=True, alpha=0.7, edgecolor="black")
    axes[1].set_title("Distribución de log-retornos")
    axes[1].set_xlabel("$r_i$")
    axes[1].set_ylabel("Densidad")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("imagenes/log_retornos_distribucion.png", dpi=300, bbox_inches='tight')
    plt.savefig("informe/figuras/log_retornos_distribucion.png", dpi=300, bbox_inches='tight')
    plt.savefig("imagenes/log-retornos y distribucion.png", dpi=300, bbox_inches='tight') # Legacy compatibility
    plt.close()

    print("Guardando trayectorias Monte Carlo...")
    trayectorias_muestra = mc["trayectorias_muestra"]
    plt.figure()
    plt.plot(trayectorias_muestra.T, linewidth=0.4, alpha=0.5)
    plt.axhline(mc["K"], color="red", linestyle="--", label=f"K = {mc['K']:.2f}")
    plt.axhline(mc["S0"], color="black", linestyle=":", label=f"S0 = {mc['S0']:.2f}")
    plt.title(f"Trayectorias GBM simuladas (muestra de {trayectorias_muestra.shape[0]})")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Precio simulado")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("imagenes/trayectorias_montecarlo.png", dpi=300, bbox_inches='tight')
    plt.savefig("informe/figuras/trayectorias_montecarlo.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Guardando distribución S_T...")
    S_T, K = mc["S_T"], mc["K"]
    plt.figure()
    plt.hist(S_T, bins=50, density=True, alpha=0.7, edgecolor="black")
    plt.axvline(K, color="red", linestyle="--", label=f"K = {K:.2f}")
    plt.axvline(S_T.mean(), color="green", linestyle="-", label=f"E[S_T] = {S_T.mean():.2f}")
    plt.title("Distribución de $S_T$ (precio al vencimiento)")
    plt.xlabel("$S_T$")
    plt.ylabel("Densidad")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("imagenes/distribucion_st.png", dpi=300, bbox_inches='tight')
    plt.savefig("informe/figuras/distribucion_st.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Guardando convergencia log-log...")
    fig, ax = plt.subplots()
    ax.loglog(tabla_convergencia["dt"], tabla_convergencia["error"], "o-",
               label="Error numérico", linewidth=1.5, markersize=7)
    dt_ref = tabla_convergencia["dt"].values
    error_ref = tabla_convergencia["error"].iloc[0] * (dt_ref / dt_ref[0])**2
    ax.loglog(dt_ref, error_ref, "--", color="gray", label="Referencia $O(\\Delta t^2)$")
    ax.set_xlabel("$\\Delta t$")
    ax.set_ylabel("Error absoluto")
    ax.set_title("Convergencia de Crank-Nicolson (escala log-log)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    plt.savefig("imagenes/convergencia_loglog.png", dpi=300, bbox_inches='tight')
    plt.savefig("informe/figuras/convergencia_loglog.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Todas las gráficas actualizadas exitosamente en 'imagenes/' e 'informe/figuras/'")

    print("Guardando gráficas de Griegas...")
    pde = cargar_pde()
    griegas = cargar_griegas()
    
    malla_S = griegas["malla_S"]
    delta_malla = griegas["delta_malla"]
    gamma_malla = griegas["gamma_malla"]
    malla_V = pde["malla_V"]
    
    # Recortar la malla a un rango de visualización razonable (ej. +/- 50%) para no ver asintotas extremas
    mask = (malla_S >= pde["S0"] * 0.4) & (malla_S <= pde["S0"] * 1.6)
    S_plot = malla_S[mask]
    V_plot = malla_V[mask]
    D_plot = delta_malla[mask]
    G_plot = gamma_malla[mask]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    axes[0].plot(S_plot, V_plot, 'b-', linewidth=2)
    axes[0].axvline(pde["S0"], color="black", linestyle=":", label="S0")
    axes[0].axvline(pde["K"], color="red", linestyle="--", label="K")
    axes[0].set_title("Precio de la Opción (V)")
    axes[0].set_xlabel("Precio Subyacente (S)")
    axes[0].set_ylabel("V(S)")
    axes[0].grid(alpha=0.3)
    axes[0].legend()
    
    axes[1].plot(S_plot, D_plot, 'g-', linewidth=2)
    axes[1].axvline(pde["S0"], color="black", linestyle=":")
    axes[1].axvline(pde["K"], color="red", linestyle="--")
    axes[1].set_title("Delta ($\Delta$)")
    axes[1].set_xlabel("Precio Subyacente (S)")
    axes[1].set_ylabel("$\partial V / \partial S$")
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(S_plot, G_plot, 'r-', linewidth=2)
    axes[2].axvline(pde["S0"], color="black", linestyle=":")
    axes[2].axvline(pde["K"], color="red", linestyle="--")
    axes[2].set_title("Gamma ($\Gamma$)")
    axes[2].set_xlabel("Precio Subyacente (S)")
    axes[2].set_ylabel("$\partial^2 V / \partial S^2$")
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("imagenes/griegas_perfil.png", dpi=300, bbox_inches='tight')
    plt.savefig("informe/figuras/griegas_perfil.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    regenerar_graficas()
