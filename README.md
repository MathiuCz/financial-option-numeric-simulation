# Valoración de Opciones Financieras mediante Métodos Numéricos

**Black-Scholes · Crank-Nicolson · Simulación Monte Carlo**

Proyecto de **Física Computacional** — Universidad Nacional Mayor de San Marcos

---

## Descripción

Este proyecto valora opciones financieras europeas (call y put) explotando la
equivalencia formal entre la PDE de Black-Scholes y la ecuación de calor de
Fourier, lo que permite aplicar métodos numéricos estándar de física
computacional — diferencias finitas (Crank-Nicolson) y simulación estocástica
(Monte Carlo) — a un problema de valoración financiera.

El pipeline completo conecta cinco etapas:

```
DatosSubyacente → CalibracionMLE → MonteCarloGBM → CrankNicolson → Griegas
```

1. **Datos históricos** del activo subyacente (`yfinance`)
2. **Calibración MLE** de `σ` y `μ` a partir de log-retornos diarios
3. **Simulación Monte Carlo GBM** bajo la medida real **P**
4. **Crank-Nicolson + Thomas** para resolver la PDE de Black-Scholes bajo la
   medida neutral al riesgo **Q**, validado contra la solución analítica
5. **Griegas** (Delta, Gamma, Theta, Vega, Rho) y **volatilidad implícita**
   vía Newton-Raphson

---

## Objetivos

**Objetivo 1.** Implementar y validar numéricamente la valoración de opciones
mediante Crank-Nicolson y Monte Carlo, cuantificando la convergencia del error
frente a la solución analítica de Black-Scholes.

**Objetivo 2.** Calcular el perfil de riesgo del contrato (griegas) y la
volatilidad implícita, invirtiendo el modelo de Black-Scholes con
Newton-Raphson según precios observados en el mercado.

---

## Estructura del proyecto

```
/valoracion_opciones
│
├── data/                       # Precios históricos descargados
│   └── historico_precios.csv
│
├── notebooks/                  # Análisis visual y reportes
│   ├── 01_calibracion_mle.ipynb
│   ├── 02_convergencia_crank_nicolson.ipynb
│   └── 03_analisis_griegas_y_volatilidad.ipynb
│
├── src/                        # Motor matemático
│   ├── __init__.py
│   ├── datos.py                # Módulo 0 — DatosSubyacente (yfinance)
│   ├── simulacion.py           # Módulo 1 — CalibracionMLE, MonteCarloGBM
│   ├── pde_solver.py           # Módulo 2 — CrankNicolson, BlackScholesAnalitico
│   └── griegas.py              # Módulo 3 — Griegas, VolatilidadImplicita
│
├── main.py                     # Pipeline completo end-to-end
├── pyproject.toml               # Instalación editable del paquete src
├── requirements.txt            # Dependencias
└── README.md
```

---

## Instalación

```bash
git clone <url-del-repositorio>
cd valoracion_opciones

pip install -r requirements.txt
pip install -e .
```

El segundo comando instala `src/` en modo editable, permitiendo importar
`from src.datos import DatosSubyacente` desde cualquier parte del proyecto
— incluyendo los notebooks — sin necesidad de `sys.path.append`.

**Requisitos:** Python ≥ 3.10.

---

## Uso

### Pipeline completo

```bash
python main.py
```

Ejecuta de punta a punta: descarga de datos, calibración, Monte Carlo,
Crank-Nicolson vs Black-Scholes, análisis de convergencia, griegas y
volatilidad implícita. Los parámetros del contrato (`TICKER`, `K`, `T`, `r`,
`TIPO`) se ajustan al inicio de `main.py`.

### Notebooks individuales

```bash
jupyter notebook notebooks/
```

- **`01_calibracion_mle.ipynb`** — descarga de precios, calibración de `σ`/`μ`,
  simulación Monte Carlo y distribución de `S_T`.
- **`02_convergencia_crank_nicolson.ipynb`** — resolución de la PDE,
  validación contra Black-Scholes, análisis de convergencia.
- **`03_analisis_griegas_y_volatilidad.ipynb`** — perfil de riesgo completo
  (griegas) y volatilidad implícita vía Newton-Raphson.

Cada notebook usa los mismos parámetros de ejemplo (`S0=100, K=105, T=1,
r=0.05`) para que los resultados sean comparables entre ellos. Algunos
valores que en la práctica vienen del notebook anterior (p. ej. `σ` calibrada,
`P(S_T>K)` empírica) están marcados explícitamente como *placeholders* a
reemplazar con el resultado real obtenido.

### Uso programático

```python
from src.datos import DatosSubyacente
from src.simulacion import CalibracionMLE, MonteCarloGBM
from src.pde_solver import CrankNicolson, BlackScholesAnalitico
from src.griegas import Griegas, VolatilidadImplicita

activo = DatosSubyacente("AAPL", "2022-01-01", "2024-01-01")
activo.obtener_precios()
activo.calcular_log_retornos()

calibrador = CalibracionMLE(activo)
calibrador.estimar()

solver = CrankNicolson(S0=activo.precios[-1], K=105, T=1, r=0.05,
                        sigma=calibrador.sigma, tipo="call")
solver.resolver()

griegas = Griegas(solver)
griegas.resumen()
```

---

## Notas de diseño

- **Medidas P vs Q.** `MonteCarloGBM` opera bajo la medida real **P** (usa
  `μ`, calibrado por MLE). `CrankNicolson`/`BlackScholesAnalitico` operan
  bajo la medida neutral al riesgo **Q** (usan `r`, nunca `μ`). La
  diferencia entre `P(S_T>K)` empírica y `N(d2)` cuantifica la prima de
  riesgo del mercado.
- **Matrices densas en Crank-Nicolson.** Se eligieron deliberadamente sobre
  matrices dispersas (`scipy.sparse`) por claridad pedagógica — el código
  refleja directamente la notación matricial `A·v^{n+1} = B·v^n` del marco
  teórico, a costa de mayor uso de memoria en mallas muy finas.
- **Theta vía diferencia hacia atrás.** A diferencia de Delta/Gamma (centradas
  en `S`, `O(ΔS²)`), Theta usa diferencia hacia atrás en el tiempo,
  `O(Δt)` — `t=0` es la frontera del dominio temporal de la malla, y no
  existe un punto `V(t=-Δt)` sin resolver un problema distinto que
  contaminaría la medición con cambios no controlados en `σ` y `S0`.
- **Soporte call/put.** Todas las clases (`CrankNicolson`,
  `BlackScholesAnalitico`, `MonteCarloGBM.prob_ITM`) aceptan
  `tipo="call"` o `tipo="put"`, con condiciones de frontera y payoff
  ajustados según corresponda.

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `numpy` | Álgebra lineal, vectorización |
| `scipy` | `norm.cdf`/`norm.pdf` para Black-Scholes |
| `pandas` | Manejo de series de tiempo, tablas de convergencia |
| `yfinance` | Descarga de precios históricos |
| `matplotlib` | Visualización en los notebooks |

---

## Autor

Mathius — Físico, Universidad Nacional Mayor de San Marcos