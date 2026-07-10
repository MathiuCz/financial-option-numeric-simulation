import numpy as np
import matplotlib.pyplot as plt
import sys, os
import config
from src.resultados_io import (
    cargar_calibracion,
    cargar_monte_carlo,
    cargar_pde,
    cargar_convergencia
)

# Crear directorios si no existen
os.makedirs("imagenes", exist_ok=True)
os.makedirs(os.path.join("informe", "figuras"), exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 5)

print("Cargando datos...")
calib = cargar_calibracion()
mc = cargar_monte_carlo()
pde = cargar_pde()
tabla_convergencia = cargar_convergencia()

print("Generando precios historicos...")
plt.figure()
plt.plot(calib["precios"], linewidth=1)
plt.title(f"Precio de cierre ajustado — {config.TICKER}")
plt.xlabel("Día de trading")
plt.ylabel("Precio (USD)")
plt.grid(alpha=0.3)
plt.savefig("imagenes/precios_historicos.png", dpi=300, bbox_inches='tight')
plt.savefig("informe/figuras/precios_historicos.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generando log-retornos...")
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
# Support old filename for legacy
plt.savefig("imagenes/log-retornos y distribucion.png", dpi=300, bbox_inches='tight')
plt.close()

print("Generando trayectorias montecarlo...")
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

print("Generando distribucion S_T...")
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

print("Generando convergencia log-log...")
fig, ax = plt.subplots()
ax.loglog(tabla_convergencia["dt"], tabla_convergencia["error"], "o-",
           label="Error numérico", linewidth=1.5, markersize=7)
dt_ref = tabla_convergencia["dt"].values
error_ref = tabla_convergencia["error"].iloc[0] * (dt_ref / dt_ref[0])**2
ax.loglog(dt_ref, error_ref, "--", color="gray", label="Referencia $O(\Delta t^2)$")
ax.set_xlabel("$\Delta t$")
ax.set_ylabel("Error absoluto")
ax.set_title("Convergencia de Crank-Nicolson (escala log-log)")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.savefig("imagenes/convergencia_loglog.png", dpi=300, bbox_inches='tight')
plt.savefig("informe/figuras/convergencia_loglog.png", dpi=300, bbox_inches='tight')
plt.close()

print("Todas las graficas regeneradas exitosamente.")
