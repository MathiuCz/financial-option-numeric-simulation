# Fórmulas Matemáticas del Proyecto de Simulación de Opciones Financieras

Este documento resume las ecuaciones matemáticas implementadas en el proyecto para la descarga, calibración y simulación de precios de activos financieros utilizando el modelo de **Movimiento Browniano Geométrico (GBM)** y la **Estimación por Máxima Verosimilitud (MLE)**.

---

## 1. Cálculo de Log-Retornos

Los rendimientos logarítmicos o continuos se calculan de manera diaria a partir de la serie temporal de precios de cierre del activo.

### Fórmula Matemática
$$R_t = \ln\left( \frac{P_t}{P_{t-1}} \right)$$

### Variables
* **$R_t$**: Log-retorno en el día $t$.
* **$P_t$**: Precio de cierre del activo en el día $t$.
* **$P_{t-1}$**: Precio de cierre del activo en el día anterior $t-1$.
* **$\ln$**: Logaritmo natural.

### Implementación en el Proyecto
Se encuentra en [extracion_datos.py](file:///c:/Users/USUARIO/OneDrive/Documentos/fisica_computacional_projects/proyecto_fisica_computacional/opciones_financieras/src/extracion_datos.py#L31):
```python
self.datos['Log_Returns'] = np.log(self.datos['Close'] / self.datos['Close'].shift(1))
```
* `.shift(1)` desplaza los precios una fila para obtener $P_{t-1}$.
* El primer elemento resultante es `NaN` y se elimina usando `.dropna()`.

---

## 2. Calibración de Parámetros (MLE)

Para proyectar el precio del activo a futuro, calibramos los parámetros históricos del mercado diario a valores **anualizados** utilizando estimadores de Máxima Verosimilitud (MLE).

### A. Volatilidad Anualizada ($\sigma$)
La volatilidad mide el nivel de incertidumbre o riesgo del activo.

#### Fórmula Matemática
$$\sigma = \frac{s}{\sqrt{dt}} = s \sqrt{252}$$

Donde $s$ es la desviación estándar muestral de los log-retornos diarios:
$$s = \sqrt{\frac{1}{N-1} \sum_{t=1}^N (R_t - \bar{R})^2}$$

#### Variables
* **$\sigma$**: Volatilidad anualizada.
* **$s$**: Desviación estándar muestral de los log-retornos.
* **$dt$**: Paso de tiempo correspondiente a un día de trading ($1/252$ años).

#### Deducción Matemática
Bajo el modelo del Movimiento Browniano Geométrico (GBM), el log-retorno diario $R_t$ transcurrido en un paso de tiempo de tamaño $dt$ se modela como:
$$R_t = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma (W_t - W_{t-dt})$$

Dado que los incrementos del movimiento browniano cumplen $W_t - W_{t-dt} = \sqrt{dt} Z_t$ con $Z_t \sim \mathcal{N}(0, 1)$, reescribimos:
$$R_t = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} Z_t$$

Calculamos la varianza matemática a ambos lados de la igualdad (teniendo en cuenta que $\mu$, $\sigma$ y $dt$ son valores constantes y que $\text{Var}(Z_t) = 1$):
$$\text{Var}(R_t) = \text{Var}\left( \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} Z_t \right) = \text{Var}(\sigma \sqrt{dt} Z_t) = \sigma^2 dt \text{Var}(Z_t) = \sigma^2 dt$$

La desviación estándar muestral de los datos históricos ($s$) representa empíricamente la raíz de esta varianza, $\sqrt{\text{Var}(R_t)}$:
$$s = \sigma \sqrt{dt}$$

Despejando la volatilidad anualizada ($\sigma$), obtenemos la fórmula utilizada en el proyecto:
$$\sigma = \frac{s}{\sqrt{dt}}$$

#### Implementación en el Proyecto
Se encuentra en [simulacion.py](file:///c:/Users/USUARIO/OneDrive/Documentos/fisica_computacional_projects/proyecto_fisica_computacional/opciones_financieras/src/simulacion.py#L32):
```python
s = np.std(r, ddof=1) # Desviación estándar insesgada (ddof=1)
self.sigma = s / np.sqrt(self.dt)
```

---

### B. Drift Real Anualizado ($\mu$)
El drift representa la tasa de crecimiento esperada a largo plazo del activo, incluyendo la **corrección de Itô**.

#### Fórmula Matemática
$$\mu = \frac{\bar{R}}{dt} + \frac{\sigma^2}{2} = 252 \bar{R} + \frac{\sigma^2}{2}$$

Donde $\bar{R}$ es la media aritmética de los log-retornos diarios:
$$\bar{R} = \frac{1}{N} \sum_{t=1}^N R_t$$

#### Variables
* **$\mu$**: Tasa de crecimiento (drift) real anualizado del activo.
* **$\bar{R}$**: Media de los log-retornos diarios.
* **$\sigma^2 / 2$**: Corrección de Itô (ajuste matemático requerido debido al comportamiento estocástico no lineal del precio).

#### Deducción Matemática
Tomamos el valor esperado (media) a ambos lados de la ecuación del log-retorno diario:
$$R_t = \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} Z_t$$

Sabiendo que el valor esperado de la variable aleatoria normal estándar es $\mathbb{E}[Z_t] = 0$:
$$\mathbb{E}[R_t] = \mathbb{E}\left[ \left(\mu - \frac{\sigma^2}{2}\right)dt + \sigma \sqrt{dt} Z_t \right] = \left(\mu - \frac{\sigma^2}{2}\right)dt$$

Estimamos la esperanza matemática diaria del rendimiento a través de la media muestral aritmética ($\bar{R}$):
$$\bar{R} \approx \left(\mu - \frac{\sigma^2}{2}\right)dt$$

Despejamos el drift anualizado ($\mu$):
$$\frac{\bar{R}}{dt} = \mu - \frac{\sigma^2}{2} \implies \mu = \frac{\bar{R}}{dt} + \frac{\sigma^2}{2}$$

#### Implementación en el Proyecto
Se encuentra en [simulacion.py](file:///c:/Users/USUARIO/OneDrive/Documentos/fisica_computacional_projects/proyecto_fisica_computacional/opciones_financieras/src/simulacion.py#L33):
```python
r_barra = np.mean(r)
self.mu = r_barra / self.dt + (self.sigma**2) / 2
```

---

## 3. Simulación del Precio (Movimiento Browniano Geométrico)

El precio del activo se modela como una ecuación diferencial estocástica (SDE) de Black-Scholes:
$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Para simularlo en computadora, integramos de forma exacta en pasos discretos de tiempo $dt$:

### Fórmula Matemática (Paso de Simulación)
$$S_{j+1} = S_j \exp\left( \left(\mu - \frac{\sigma^2}{2}\right) dt + \sigma \sqrt{dt} Z_j \right)$$

### Variables
* **$S_{j+1}$**: Precio simulado del activo en el paso siguiente.
* **$S_j$**: Precio del activo en el paso actual.
* **$dt$**: Fracción de año que representa cada paso de la simulación ($dt = T/M$).
* **$Z_j$**: Variable aleatoria extraída de una distribución normal estándar independiente, $Z_j \sim \mathcal{N}(0, 1)$.
* **$\mu$, $\sigma$**: Drift y volatilidad anualizados obtenidos en la calibración.

### Implementación en el Proyecto
Se encuentra en [simulacion.py](file:///c:/Users/USUARIO/OneDrive/Documentos/fisica_computacional_projects/proyecto_fisica_computacional/opciones_financieras/src/simulacion.py#L74):
```python
Z = np.random.standard_normal(N)
trayectorias[:, j+1] = trayectorias[:, j] * np.exp((mu - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)
```

---

## 4. Probabilidad de terminar In-The-Money (ITM)

Calcula la probabilidad empírica de que la opción venza con un valor positivo (que sea rentable para el comprador), dependiendo de si es una opción Call o Put.

### Fórmula Matemática
#### Opción Call (Compra)
$$\text{Prob}_{\text{ITM}} = P(S_T > K) \approx \frac{1}{N} \sum_{i=1}^N \mathbb{I}(S_{T, i} > K)$$

#### Opción Put (Venta)
$$\text{Prob}_{\text{ITM}} = P(S_T < K) \approx \frac{1}{N} \sum_{i=1}^N \mathbb{I}(S_{T, i} < K)$$

### Variables
* **$S_T$**: Precio final simulado del activo al vencimiento (tiempo $T$).
* **$K$**: Precio de ejercicio (strike) fijado en el contrato de opción.
* **$N$**: Número total de trayectorias simuladas de Monte Carlo.
* **$\mathbb{I}$**: Función indicadora (vale $1$ si la condición dentro del paréntesis es verdadera, y $0$ si es falsa).

### Implementación en el Proyecto
Se encuentra en [simulacion.py](file:///c:/Users/USUARIO/OneDrive/Documentos/fisica_computacional_projects/proyecto_fisica_computacional/opciones_financieras/src/simulacion.py#L80):
```python
if tipo == "call":
    prob = np.mean(self.S_T > K)
elif tipo == "put":
    prob = np.mean(self.S_T < K)
```
* La función `np.mean` aplicada sobre una máscara booleana (`self.S_T > K`) calcula de manera vectorizada y eficiente la proporción de trayectorias que cumplen con la condición de éxito.
