# Derivación de Diferencias Finitas para la Ecuación de Black-Scholes

Este documento explica paso a paso, desde cero, cómo discretizar la Ecuación Diferencial Parcial (PDE) de Black-Scholes utilizando el método de diferencias finitas en su espacio original (precios $S$), sin hacer transformaciones a la ecuación de calor.

Esta es la derivación matemática exacta que fundamenta el archivo `src/pde_solver.py` de tu proyecto.

---

## 1. La Ecuación de Black-Scholes

El precio $V(S, t)$ de una opción europea cumple con la siguiente ecuación diferencial:

$$ \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V = 0 $$

Donde:
*   $V$: Precio de la opción.
*   $S$: Precio del activo subyacente.
*   $t$: Tiempo actual.
*   $\sigma$: Volatilidad del activo.
*   $r$: Tasa de interés libre de riesgo.

Reordenando la ecuación para despejar la derivada temporal (y multiplicando por -1), obtenemos:

$$ -\frac{\partial V}{\partial t} = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V $$

Definamos el lado derecho como un "operador espacial" $L(V)$:
$$ L(V) = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + r S \frac{\partial V}{\partial S} - r V $$
Por lo tanto, la ecuación se simplifica a:
$$ -\frac{\partial V}{\partial t} = L(V) $$

---

## 2. Discretización del Dominio (La Malla)

Para resolver esto en una computadora, no podemos usar valores continuos. Creamos una **malla o cuadrícula**:
*   **Eje Espacial (Precio $S$):** Dividimos el rango $[0, S_{max}]$ en $N$ pasos de tamaño $\Delta S$.
    $$ S_j = j \cdot \Delta S \quad \text{para } j = 0, 1, 2, ..., N $$
*   **Eje Temporal (Tiempo $t$):** Dividimos el tiempo hasta el vencimiento $T$ en $M$ pasos de tamaño $\Delta t$.
    $$ t_n = n \cdot \Delta t \quad \text{para } n = 0, 1, 2, ..., M $$

Nuestra incógnita será el valor de la opción en cada nodo de esta malla, al cual denotaremos como:
$$ V_j^n = V(S_j, t_n) $$

> [!NOTE]
> En la valoración de opciones, conocemos el valor al final del contrato ($t=T$) por el payoff. Por lo tanto, el tiempo avanza **hacia atrás**. El paso $n+1$ representa un tiempo futuro (que ya calculamos o conocemos), y el paso $n$ representa el tiempo actual (lo que queremos averiguar).

---

## 3. Aproximación por Diferencias Finitas

El cálculo diferencial nos dice que podemos aproximar las derivadas usando pendientes entre puntos cercanos.

**A. Derivadas Espaciales (Diferencias Centradas)**
Usamos los puntos vecinos en el espacio ($j-1$ y $j+1$) para encontrar la derivada en el punto $j$:

1.  **Primera Derivada:**
    $$ \frac{\partial V}{\partial S} \approx \frac{V_{j+1}^n - V_{j-1}^n}{2 \Delta S} $$
2.  **Segunda Derivada:**
    $$ \frac{\partial^2 V}{\partial S^2} \approx \frac{V_{j+1}^n - 2V_j^n + V_{j-1}^n}{\Delta S^2} $$

**B. Derivada Temporal**
Como vamos hacia atrás en el tiempo (de $n+1$ a $n$), la derivada se aproxima como:
$$ -\frac{\partial V}{\partial t} \approx \frac{V_j^n - V_j^{n+1}}{\Delta t} $$

---

## 4. Evaluando el Operador Espacial $L(V)$

Sustituyamos las diferencias espaciales en nuestro operador $L(V)$ en un tiempo genérico $k$:

$$ L(V_j^k) = \frac{1}{2}\sigma^2 S_j^2 \left( \frac{V_{j+1}^k - 2V_j^k + V_{j-1}^k}{\Delta S^2} \right) + r S_j \left( \frac{V_{j+1}^k - V_{j-1}^k}{2 \Delta S} \right) - r V_j^k $$

> [!IMPORTANT]
> **El Secreto Analítico:** Recuerda que $S_j = j \cdot \Delta S$. 
> - En el término de la segunda derivada: $\frac{S_j^2}{\Delta S^2} = \frac{(j \cdot \Delta S)^2}{\Delta S^2} = j^2$
> - En el término de la primera derivada: $\frac{S_j}{2 \Delta S} = \frac{j \cdot \Delta S}{2 \Delta S} = \frac{j}{2}$

¡Los términos $\Delta S$ se cancelan matemáticamente! Esto es fundamental y explica por qué en tu código no usas `dS` para calcular los coeficientes de las matrices.

Sustituyendo esto, $L(V_j^k)$ queda limpio:
$$ L(V_j^k) = \frac{1}{2}\sigma^2 j^2 (V_{j+1}^k - 2V_j^k + V_{j-1}^k) + \frac{1}{2} r j (V_{j+1}^k - V_{j-1}^k) - r V_j^k $$

Agrupemos todo lo que multiplica a $V_{j-1}^k$, a $V_j^k$, y a $V_{j+1}^k$:

$$ L(V_j^k) = \underbrace{\left( \frac{1}{2}\sigma^2 j^2 - \frac{1}{2} r j \right)}_{\text{Coef. de } j-1} V_{j-1}^k - \underbrace{(\sigma^2 j^2 + r)}_{\text{Coef. de } j} V_j^k + \underbrace{\left( \frac{1}{2}\sigma^2 j^2 + \frac{1}{2} r j \right)}_{\text{Coef. de } j+1} V_{j+1}^k $$

---

## 5. El Esquema de Crank-Nicolson

Si usamos el operador $L(V)$ evaluado en el paso futuro $n+1$, tendríamos un método Explícito (inestable). Si lo evaluamos en el paso actual $n$, sería un método Implícito (estable pero menos preciso).

El método de **Crank-Nicolson** toma el promedio exacto de ambos mundos:
$$ \frac{V_j^n - V_j^{n+1}}{\Delta t} = \frac{1}{2} \left[ L(V_j^n) + L(V_j^{n+1}) \right] $$

Multiplicando todo por $\Delta t$:
$$ V_j^n - V_j^{n+1} = \frac{\Delta t}{2} L(V_j^n) + \frac{\Delta t}{2} L(V_j^{n+1}) $$

### Reorganizando Incógnitas (El paso crucial)
Queremos averiguar $V_j^n$ (tiempo actual), mientras que $V_j^{n+1}$ (tiempo futuro) ya lo conocemos. Pasamos todas las incógnitas (nivel $n$) al lado izquierdo, y todo lo conocido (nivel $n+1$) al lado derecho:

$$ V_j^n - \frac{\Delta t}{2} L(V_j^n) = V_j^{n+1} + \frac{\Delta t}{2} L(V_j^{n+1}) $$

---

## 6. Construcción de los Coeficientes del Código

Multipliquemos nuestro operador agrupado $L(V_j^k)$ por $\frac{\Delta t}{2}$. Al hacer esto, nacen los tres coeficientes de tu código Python:

*   **$\alpha_j$** (Multiplica al nodo inferior $j-1$):
    $$ \alpha_j = \frac{\Delta t}{2} \left( \frac{1}{2}\sigma^2 j^2 - \frac{1}{2} r j \right) = \mathbf{0.25 \cdot \Delta t \cdot (\sigma^2 j^2 - r j)} $$
*   **$\beta_j$** (Multiplica al nodo central $j$):
    $$ \beta_j = \frac{\Delta t}{2} \left( -(\sigma^2 j^2 + r) \right) = \mathbf{-0.5 \cdot \Delta t \cdot (\sigma^2 j^2 + r)} $$
*   **$\gamma_j$** (Multiplica al nodo superior $j+1$):
    $$ \gamma_j = \frac{\Delta t}{2} \left( \frac{1}{2}\sigma^2 j^2 + \frac{1}{2} r j \right) = \mathbf{0.25 \cdot \Delta t \cdot (\sigma^2 j^2 + r j)} $$

*(Si revisas `src/pde_solver.py`, verás que estas tres fórmulas son idénticas a las líneas de código).*

Sustituyendo $\alpha, \beta, \gamma$ en la ecuación de Crank-Nicolson, la expresión $\frac{\Delta t}{2} L(V_j^k)$ se convierte en:
$$ \frac{\Delta t}{2} L(V_j^k) = \alpha_j V_{j-1}^k + \beta_j V_j^k + \gamma_j V_{j+1}^k $$

Y nuestra ecuación global para resolver (Lado izquierdo $n$ = Lado derecho $n+1$) queda como:
$$ V_j^n - \left( \alpha_j V_{j-1}^n + \beta_j V_j^n + \gamma_j V_{j+1}^n \right) = V_j^{n+1} + \left( \alpha_j V_{j-1}^{n+1} + \beta_j V_j^{n+1} + \gamma_j V_{j+1}^{n+1} \right) $$

Agrupando términos semejantes:
$$ -\alpha_j V_{j-1}^n + (1 - \beta_j) V_j^n - \gamma_j V_{j+1}^n = \alpha_j V_{j-1}^{n+1} + (1 + \beta_j) V_j^{n+1} + \gamma_j V_{j+1}^{n+1} $$

---

## 7. El Sistema Matricial

Esta última ecuación se aplica para todos los nodos internos de la malla ($j = 1, 2, ..., N-1$). Esto forma un gigantesco sistema de ecuaciones lineales tridiagonales:

$$ A \cdot V^n = B \cdot V^{n+1} + \text{Condiciones de Frontera} $$

Donde la **Matriz A** (Lado izquierdo, incógnitas) tiene:
*   Diagonal principal: $1 - \beta_j$
*   Diagonal inferior: $-\alpha_j$
*   Diagonal superior: $-\gamma_j$

Y la **Matriz B** (Lado derecho, conocidos) tiene:
*   Diagonal principal: $1 + \beta_j$
*   Diagonal inferior: $\alpha_j$
*   Diagonal superior: $\gamma_j$

Para avanzar un paso temporal, simplemente construimos el lado derecho ($b = B \cdot V^{n+1}$) y resolvemos el sistema lineal $A \cdot V^n = b$ para encontrar los precios del paso actual. ¡Y así iteramos hasta llegar a $t=0$!

---

## 8. Análisis de Estabilidad de Von Neumann

Para justificar la **estabilidad incondicional** del esquema de Crank-Nicolson, se realiza el análisis de estabilidad de Von Neumann. Este análisis asume rigurosamente coeficientes constantes, por lo que aplicamos la técnica de "coeficientes congelados" (asumimos $S$ constante localmente para el análisis del error en un nodo $j$).

Sustituimos un error en forma de onda plana o modo de Fourier:
$$ V_j^n = \xi^n e^{i k j \Delta S} $$
Donde $\xi^n$ es la amplitud del error en el paso de tiempo $n$, $k$ es el número de onda espacial y $i$ es la unidad imaginaria.

Si introducimos esto en nuestra ecuación discreta global y factorizamos $\xi^n$ y $\xi^{n+1}$:
$$ \xi^n \left[ (1 - \beta_j) - \alpha_j e^{-ik\Delta S} - \gamma_j e^{ik\Delta S} \right] = \xi^{n+1} \left[ (1 + \beta_j) + \alpha_j e^{-ik\Delta S} + \gamma_j e^{ik\Delta S} \right] $$

Para simplificar, definamos un número complejo $Z$:
$$ Z = \alpha_j e^{-ik\Delta S} + \beta_j + \gamma_j e^{ik\Delta S} $$

Por lo tanto, la ecuación se reduce a:
$$ \xi^n (1 - Z) = \xi^{n+1} (1 + Z) $$

Como marchamos hacia atrás en el tiempo (de $n+1$ a $n$), el factor de amplificación $G$ que determina cómo crece el error es la proporción entre el paso futuro calculado ($n$) y el paso anterior conocido ($n+1$):
$$ G = \frac{\xi^n}{\xi^{n+1}} = \frac{1 + Z}{1 - Z} $$

Usemos la identidad de Euler ($e^{\pm ix} = \cos x \pm i \sin x$) para separar $Z$ en su parte real e imaginaria:
$$ \text{Re}(Z) = (\alpha_j + \gamma_j)\cos(k\Delta S) + \beta_j $$
$$ \text{Im}(Z) = (\gamma_j - \alpha_j)\sin(k\Delta S) $$

Recordemos la definición de nuestros coeficientes:
* $\alpha_j + \gamma_j = 0.5 \cdot \Delta t \cdot \sigma^2 j^2$
* $\beta_j = -0.5 \cdot \Delta t \cdot (\sigma^2 j^2 + r)$

Sustituyendo esto en la parte real:
$$ \text{Re}(Z) = 0.5 \Delta t \sigma^2 j^2 \cos(k\Delta S) - 0.5 \Delta t (\sigma^2 j^2 + r) $$
Agrupando, tenemos:
$$ \text{Re}(Z) = -0.5 \Delta t \sigma^2 j^2 \left( 1 - \cos(k\Delta S) \right) - 0.5 \Delta t r $$

Dado que la función $\cos$ nunca es mayor a $1$, el término $(1 - \cos)$ es siempre $\ge 0$. Como $\Delta t$, $\sigma^2$ y $r$ también son positivos, es estrictamente cierto que:
$$ \text{Re}(Z) \le 0 $$

Definamos $M = -\text{Re}(Z)$ (donde $M \ge 0$) y $N = \text{Im}(Z)$. Entonces nuestro factor de amplificación es:
$$ G = \frac{1 - M + iN}{1 + M - iN} $$

El tamaño o magnitud al cuadrado del factor de amplificación $|G|^2$ (que debe ser $\le 1$ para que el método no "explote") es:
$$ |G|^2 = \frac{(1 - M)^2 + N^2}{(1 + M)^2 + N^2} $$

Dado que $M \ge 0$, es obvio matemáticamente que $(1 - M)^2 \le (1 + M)^2$. Por lo tanto, el denominador siempre será mayor o igual al numerador, concluyendo que:
$$ |G| \le 1 $$

¡Esta desigualdad se cumple para **cualquier valor** de $\Delta t$ o $\Delta S$! Esto significa que los errores no se amplificarán a medida que iteramos en el tiempo, demostrando que **el esquema de Crank-Nicolson es incondicionalmente estable**.
