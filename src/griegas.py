import numpy as np

try:
    from src.pde_solver import CrankNicolson, BlackScholesAnalitico
except ModuleNotFoundError:
    from pde_solver import CrankNicolson, BlackScholesAnalitico



class Griegas:

    def __init__(self, solver: CrankNicolson):
        if solver.malla_V is None:
            solver.resolver()

        self.solver = solver

        self.delta = None
        self.gamma = None
        self.theta = None
        self.vega  = None
        self.rho   = None

    def calcular_delta_gamma(self) -> tuple:
        """
        Delta y Gamma vía diferencias centrales en la dimensión espacial S.
        """
        V  = self.solver.malla_V
        S  = self.solver.malla_S
        dS = S[1] - S[0]

        delta = np.full_like(V, np.nan)
        gamma = np.full_like(V, np.nan)

        delta[1:-1] = (V[2:] - V[:-2]) / (2 * dS)
        gamma[1:-1] = (V[2:] - 2 * V[1:-1] + V[:-2]) / (dS**2)

        self.delta = delta
        self.gamma = gamma
        return self.delta, self.gamma

    def calcular_theta(self) -> float:
        """
        Theta vía diferencia hacia atrás en el tiempo del calendario.
        """
        s = self.solver

        if s.malla_V_dt is None:
            raise ValueError(
                "El solver no guardó la malla en t=Δt. "
                "Verifica que CrankNicolson.resolver() haya corrido con M >= 1."
            )

        dt = s.T / s.M

        self.theta = (s.precio_dt - s.precio) / dt
        return self.theta

    def calcular_vega(self, d_sigma: float = 1e-4) -> float:
        """
        Vega vía perturbación paramétrica de sigma, diferencia central O(Δσ²).
        """
        s = self.solver
        factor = s.S_max / s.K

        solver_sup = CrankNicolson(s.S0, s.K, s.T, s.r, s.sigma + d_sigma,
                                     tipo=s.tipo, S_max_factor=factor, M=s.M, N=s.N)
        solver_inf = CrankNicolson(s.S0, s.K, s.T, s.r, s.sigma - d_sigma,
                                     tipo=s.tipo, S_max_factor=factor, M=s.M, N=s.N)
        solver_sup.resolver()
        solver_inf.resolver()

        self.vega = (solver_sup.precio - solver_inf.precio) / (2 * d_sigma)
        return self.vega

    def calcular_rho(self, d_r: float = 1e-4) -> float:
        """
        Rho vía perturbación paramétrica de r, diferencia central O(Δr²).
        """
        s = self.solver
        factor = s.S_max / s.K

        solver_sup = CrankNicolson(s.S0, s.K, s.T, s.r + d_r, s.sigma,
                                     tipo=s.tipo, S_max_factor=factor, M=s.M, N=s.N)
        solver_inf = CrankNicolson(s.S0, s.K, s.T, s.r - d_r, s.sigma,
                                     tipo=s.tipo, S_max_factor=factor, M=s.M, N=s.N)
        solver_sup.resolver()
        solver_inf.resolver()

        self.rho = (solver_sup.precio - solver_inf.precio) / (2 * d_r)
        return self.rho

    def calcular_todas(self) -> dict:
        self.calcular_delta_gamma()
        self.calcular_theta()
        self.calcular_vega()
        self.calcular_rho()

        # En lugar de usar np.argmin para el vecino más cercano (que introduce error por discretización),
        # interpolamos linealmente Delta y Gamma en S0 al igual que se hace con el precio de la opción.
        delta_S0 = np.interp(self.solver.S0, self.solver.malla_S[1:-1], self.delta[1:-1])
        gamma_S0 = np.interp(self.solver.S0, self.solver.malla_S[1:-1], self.gamma[1:-1])

        return {
            "delta": delta_S0,
            "gamma": gamma_S0,
            "theta": self.theta,
            "vega":  self.vega,
            "rho":   self.rho,
        }

    def resumen(self) -> None:
        griegas = self.calcular_todas()
        print(f"{'Griega':<10} {'Valor':>12}")
        print("-" * 24)
        for nombre, valor in griegas.items():
            print(f"{nombre:<10} {valor:>12.6f}")

class VolatilidadImplicita:
    """
    Invierte Black-Scholes vía Newton-Raphson: dado un precio de mercado
    observado.
    """

    def __init__(self, C_mercado: float, S0: float, K: float, T: float, r: float,
                 tipo: str = "call"):
        self.C_mercado = C_mercado
        self.S0 = S0
        self.K  = K
        self.T  = T
        self.r  = r
        self.tipo = tipo

        self.sigma_implicita = None
        self.historial = []   # guarda (iter, sigma, error) para inspección

    def resolver(self, sigma_0: float = 0.20, tol: float = 1e-8,
                 max_iter: int = 100) -> float:
        sigma = sigma_0
        self.historial = []

        for it in range(1, max_iter + 1):
            bs = BlackScholesAnalitico(self.S0, self.K, self.T, self.r, sigma,
                                         tipo=self.tipo)
            C_bs, _ = bs.calcular()
            error = C_bs - self.C_mercado

            self.historial.append((it, sigma, error))

            if abs(error) < tol:
                self.sigma_implicita = sigma
                return self.sigma_implicita

            vega = self._vega_analitico(sigma)

            if vega == 0 or np.isnan(vega):
                raise RuntimeError(
                    f"Vega se anuló en la iteración {it} (sigma={sigma:.4f}). "
                    "Newton-Raphson no puede continuar; revisa C_mercado o el punto de partida."
                )

            sigma = sigma - error / vega

            if sigma <= 0:
                raise RuntimeError(
                    f"sigma se volvió no positivo en la iteración {it} "
                    f"(sigma={sigma:.4f}). El precio de mercado puede ser inconsistente "
                    "con los demás parámetros (S0, K, T, r)."
                )

        raise RuntimeError(
            f"Newton-Raphson no convergió en {max_iter} iteraciones "
            f"(último error: {error:.2e})"
        )

    def _vega_analitico(self, sigma: float) -> float:
        """Vega cerrada de BS — evita resolver CN repetidamente, mucho más rápido."""
        from scipy.stats import norm

        d1 = (np.log(self.S0 / self.K) + (self.r + sigma**2 / 2) * self.T) / \
             (sigma * np.sqrt(self.T))
        return self.S0 * norm.pdf(d1) * np.sqrt(self.T)

    def resumen(self) -> None:
        if self.sigma_implicita is None:
            raise ValueError("Ejecuta resolver() antes de mostrar el resumen")

        print(f"{'Iter':<6} {'sigma':>10} {'error':>14}")
        print("-" * 32)
        for it, sigma, error in self.historial:
            print(f"{it:<6} {sigma:>10.6f} {error:>14.2e}")
        print(f"\nsigma implicita convergida: {self.sigma_implicita:.6f}")


