import numpy as np 
import matplotlib.pyplot as plt

try:
    from src.extracion_datos import DatosSubyacente

except ModuleNotFoundError:
    from extracion_datos import DatosSubyacente



class CalibracionMLE:
    dt = 1/252 #dias de tradings
    
    def __init__(self, activo: DatosSubyacente): #Recibe los objetos de DatosSubyacente
        self.activo = activo
        self.mu = None
        self.sigma = None

    def estimar(self) -> tuple:
        if self.activo.retornos is None:
            self.activo.obtener_log_retornos()
        
        r = self.activo.retornos
        if r is None:
            raise ValueError("No se pudieron obtener los retornos del activo.")

        r_barra = np.mean(r)
        s = np.std(r, ddof=1)#delta degree of freedom (es para que sea insesgado)
        
        # Estimadores MLE + corrección de Itô
        self.sigma = s / np.sqrt(self.dt) #volatilidad anualizada
        self.mu = r_barra / self.dt + (self.sigma**2) / 2 #drift real + corrección de Itô
        return self.mu, self.sigma

    def resume(self) -> None:
        if self.mu is None or self.sigma is None:
            self.estimar()

        if self.mu is None or self.sigma is None:
            raise ValueError("No se pudieron calcular los parámetros mu y sigma.")

        print(f"{'Parametro':<15} {'Valor':>10}  {'Interpretacion'}")
        print("-" * 55)
        print(f"{'sigma':<15} {self.sigma:>9.4f}   Volatilidad anualizada")
        print(f"{'mu':<15} {self.mu:>9.4f}   Drift real anualizado")
        print(f"{'sigma^2/2 (Ito)':<15} {self.sigma**2/2:>9.4f}   Correccion de Ito")

class MonteCarloGBM: 
    def __init__(self, calibrador: CalibracionMLE, S0: float = None):
        self.calibrador = calibrador
        self.S0 = S0 if S0 is not None else calibrador.activo.precios[-1] #si no se pasa S0 usa el ultimo precio observado del activo

        self.trayectorias = None
        self.S_T = None
        self.prob_itm = None

    def simular(self, T: float, n_pasos: int, n_sim: int, seed: int = None) -> np.ndarray:
        if self.calibrador.mu is None or self.calibrador.sigma is None:
            self.calibrador.estimar()
        
        if seed is not None:
            np.random.seed(seed)

        mu = self.calibrador.mu
        sigma = self.calibrador.sigma
        dt = T / n_pasos  # n_pasos de tiempo
        
        trayectorias = np.zeros((n_sim, n_pasos + 1))
        trayectorias[:, 0] = self.S0

        for j in range(n_pasos):
            Z = np.random.standard_normal(n_sim)  # Vectorizado sobre las n_sim trayectorias
            trayectorias[:, j+1] = trayectorias[:, j] * np.exp((mu - sigma**2/2)*dt + sigma*np.sqrt(dt)*Z)

        self.trayectorias = trayectorias
        self.S_T = trayectorias[:, n_pasos]
        return self.trayectorias

    def prob_ITM(self, K: float, tipo: str = "call") -> float:
        if self.S_T is None:
            raise ValueError("Ejecuta simular antes de calcular la probabilidad ITM")
        
        tipo = tipo.lower()
        if tipo == "call":
            prob = np.mean(self.S_T > K)
        elif tipo == "put":
            prob = np.mean(self.S_T < K)
        else:
            raise ValueError("El tipo de opcion debe ser 'call' o 'put'")
            
        self.prob_itm = prob
        return self.prob_itm

    def resumen(self, K: float, tipo: str = "call") -> None:
        if self.S_T is None:
            raise ValueError("Ejecuta simular() antes de mostrar el resumen")

        prob = self.prob_ITM(K, tipo=tipo)
        print(f"S0                 : {self.S0:.2f} (Precio subyacente)")
        print(f"K (strike)         : {K:.2f} (Precio strike)")
        print(f"Tipo de Opcion     : {tipo.upper()}")
        print(f"E[S_T]             : {self.S_T.mean():.2f} (Precio esperado)")
        print(f"std[S_T]           : {self.S_T.std():.2f} (Desviacion estandar)")
        if tipo.lower() == "call":
            print(f"P(S_T > K) empir.  : {prob:.4f} (Probabilidad de terminar ITM)")
        else:
            print(f"P(S_T < K) empir.  : {prob:.4f} (Probabilidad de terminar ITM)")

if __name__ == "__main__":
    activo = DatosSubyacente("SPY", "2022-05-01", "2026-05-30")
    activo.obtener_precios()
    activo.obtener_log_retornos()

    calibrador = CalibracionMLE(activo)
    calibrador.estimar()
    calibrador.resume()

    print()

    mc = MonteCarloGBM(calibrador)
    mc.simular(T=0.5, n_pasos=252, n_sim=10_000, seed=42)
    mc.resumen(K=calibrador.activo.precios[-1] * 1.05)  # K 5% sobre el precio actual
    