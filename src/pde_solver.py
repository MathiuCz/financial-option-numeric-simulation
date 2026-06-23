import numpy as np 
from scipy.stats import norm

class CrankNicolson:
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 tipo: str = "call", S_max_factor: float = 3.0,
                 M: int = 1000, N: int = 200):

        if tipo not in ("call", "put"):
            raise ValueError("tipo debe ser 'call' o 'put'")
        
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.tipo = tipo

        self.S_max = S_max_factor * K
        self.M = M 
        self.N = N

        self.malla_V = None
        self.Malla_S = None
        self.precio = None

    def _payoff(self, S: np.ndarray) -> np.ndarray: #MAX(0, S-K)
        if self.tipo == "call":
            return np.maximum(S - self.K, 0)
        else:
            return np.maximum(self.K - S, 0)

    def _condiciones_frontera(self, v: np.ndarray, S: np.ndarray, t_actual: float) -> None:
        if self.tipo == "call":
            v[0] = 0 #Precio option cuando S=0 es 0
            #Precio option cuando S es muy grande (s_max)
            v[-1] = self.S_max - self.K * np.exp(-self.r * t_actual)

        else: 
            v[0] = self.K * np.exp(-self.r * t_actual)
            v[-1] = 0

    def resolver(self) -> tuple:
        dS = self.S_max / (self.N) #tamaño del paso en el espacio
        dt = self.T / self.M       #tamaño del paso en el tiempo
        S = np.linspace(0, self.S_max, self.N + 1) #malla en el espacio

        V = self._pay_off(S)  #Vector de precios de la opcion en t = 0 (condicion inicial)

        j = np.arange(1, self.N)
        alpha = 0.25*dt*(self.sigma**2 * j**2 - self.r *j)
        beta = -0.5*dt*(self.sigma**2 * j**2 + self.r)
        gamma = 0.25 * dt * (self.sigma**2 * j**2 + self.r * j)

        A = np.zeros((self.N - 1,self.N - 1))
        B = np.zeros((self.N - 1, self.N - 1))

        for i in range(self.N -1):
            A[i, i] = 1 - beta[i]
            B[i, i] = 1 + beta[i]

            if i > 0:
                A[i, i-1] = -alpha[i]
                B[i, i-1] =  alpha[i]
            
            if i < self.N - 2:
                A[i, i+1] = -gamma[i]
                B[i, i+1] =  gamma[i]
        
        for paso in range(self.M, 0, -1):
            t_actual = (self.M - paso) * dt
            self._condiciones_frontera(V, S, t_actual)

            b = B @ V[1:self.N]
            b[0]  += alpha[0] * V[0]
            b[-1] += gamma[-1] * V[-1]

            V[1:self.N] = self._thomas(A, b)

        self.malla_V = V
        self.malla_S = S
        self.precio  = np.interp(self.S0, S, V)
        return self.malla_V, self.malla_S
        
@staticmethod
def _thomas(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    c = np.diag(A, k = 1).copy()
    d = np.diag(A).copy()
    a = np.diag(A, k=-1).copy()
    x = np.zeros(n)

    for i in range(1, n):
        m = a[i-1]/d[i-1]
        d[i] -= m*c[i-1]
        b[i] -= m*b[i-1]
    
    x[-1] = b[-1]/d[-1]
    
    for i in range(n-2, -1, -1):
        x[i] = (b[i] - c[i]*x[i+1])/d[i]
    return x


class BlackScholesAnalitico:

    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float,
                 tipo: str = "call"):
        if tipo not in ("call", "put"):
            raise ValueError("tipo debe ser 'call' o 'put'")

        self.S0    = S0
        self.K     = K
        self.T     = T
        self.r     = r
        self.sigma = sigma
        self.tipo  = tipo

        self.d1     = None
        self.d2     = None
        self.precio = None

    def calcular(self) -> tuple:
        self.d1 = (np.log(self.S0 / self.K) + (self.r + self.sigma**2 / 2) * self.T) / \
                  (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

        if self.tipo == "call":
            self.precio = self.S0 * norm.cdf(self.d1) - \
                           self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            prob_riesgo_neutral = norm.cdf(self.d2)
        else:
            self.precio = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - \
                           self.S0 * norm.cdf(-self.d1)
            prob_riesgo_neutral = norm.cdf(-self.d2)

        return self.precio, prob_riesgo_neutral
    
        
        
        



        