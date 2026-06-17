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

        print(f"{'Parámetro':<15} {'Valor':>10}  {'Interpretación'}")
        print("-" * 55)
        print(f"{'σ (sigma)':<15} {self.sigma:>9.4f}   Volatilidad anualizada")
        print(f"{'μ (mu)':<15} {self.mu:>9.4f}   Drift real anualizado")
        print(f"{'σ²/2 (Itô)':<15} {self.sigma**2/2:>9.4f}   Corrección de Itô")


if __name__ == "__main__":
    spy = DatosSubyacente("SPY", "2022-05-01", "2026-05-30")
    calibracion = CalibracionMLE(spy)
    calibracion.resume()
        
        


        
        
        
        