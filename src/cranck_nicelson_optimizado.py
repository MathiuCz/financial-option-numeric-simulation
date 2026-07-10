# src/pde_solver.py — versión optimizada de CrankNicolson
#clase a modo didactivo d euna version optimizada, no se usa en ese proyecto por temas de 
#simplicidad conceptual y uso de los metodos numericos adecuados
class CrankNicolson:

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 tipo: str = "call", S_max_factor: float = 3.0,
                 M: int = 1000, N: int = 200):
        if tipo not in ("call", "put"):
            raise ValueError("tipo debe ser 'call' o 'put'")

        self.S0    = S0
        self.K     = K
        self.T     = T
        self.r     = r
        self.sigma = sigma
        self.tipo  = tipo

        self.M = M
        self.N = N
        self.S_max = S_max_factor * K

        self.malla_V = None
        self.malla_S = None
        self.precio  = None

    def _payoff(self, S: np.ndarray) -> np.ndarray:
        if self.tipo == "call":
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def _condiciones_frontera(self, V: np.ndarray, S: np.ndarray, t_actual: float) -> None:
        if self.tipo == "call":
            V[0]  = 0
            V[-1] = self.S_max - self.K * np.exp(-self.r * t_actual)
        else:
            V[0]  = self.K * np.exp(-self.r * t_actual)
            V[-1] = 0

    def resolver(self) -> tuple:
        dS = self.S_max / self.N
        dt = self.T / self.M
        S  = np.linspace(0, self.S_max, self.N + 1)

        V = self._payoff(S)

        # Coeficientes — calculados UNA sola vez, no cambian entre pasos de tiempo
        j     = np.arange(1, self.N)
        alpha = 0.25 * dt * (self.sigma**2 * j**2 - self.r * j)
        beta  = -0.5 * dt * (self.sigma**2 * j**2 + self.r)
        gamma = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

        # Diagonales de A (implícita) y B (explícita) — vectores, no matrices
        a_diag = -alpha[1:]        # subdiagonal de A   (tamaño N-2)
        d_diag = 1 - beta          # diagonal principal de A (tamaño N-1)
        c_diag = -gamma[:-1]       # superdiagonal de A (tamaño N-2)

        b_sub  = alpha[1:]         # subdiagonal de B
        b_diag = 1 + beta          # diagonal principal de B
        b_sup  = gamma[:-1]        # superdiagonal de B

        for paso in range(self.M, 0, -1):
            t_actual = (self.M - paso) * dt
            self._condiciones_frontera(V, S, t_actual)

            v_int = V[1:self.N].copy()

            # B @ v_int sin construir matriz — solo operaciones vectorizadas
            b = b_diag * v_int
            b[1:]  += b_sub * v_int[:-1]
            b[:-1] += b_sup * v_int[1:]

            # Aportes de las condiciones de frontera al lado derecho
            b[0]  += alpha[0]  * V[0]
            b[-1] += gamma[-1] * V[-1]

            V[1:self.N] = self._thomas(a_diag, d_diag.copy(), c_diag, b)

        self.malla_V = V
        self.malla_S = S
        self.precio  = np.interp(self.S0, S, V)
        return self.malla_V, self.malla_S

    @staticmethod
    def _thomas(a: np.ndarray, d: np.ndarray, c: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Resuelve A·x = b donde A es tridiagonal.
        a: subdiagonal (tamaño n-1)
        d: diagonal principal (tamaño n)  — se modifica in-place, por eso .copy() al llamar
        c: superdiagonal (tamaño n-1)
        b: lado derecho (tamaño n) — también se modifica in-place
        """
        n = len(b)
        x = np.zeros(n)

        for i in range(1, n):
            m    = a[i-1] / d[i-1]
            d[i] = d[i] - m * c[i-1]
            b[i] = b[i] - m * b[i-1]

        x[-1] = b[-1] / d[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (b[i] - c[i] * x[i+1]) / d[i]

        return x