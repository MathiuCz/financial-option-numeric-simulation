import yfinance as yf 
import pandas as pd
import numpy as np
from pathlib import Path

class DatosSubyacente:

    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.datos: pd.DataFrame | None = None
        self.precios: np.ndarray | None =  None
        self.retornos: np.ndarray | None = None

    def obtener_precios(self) -> np.ndarray:
        self.datos = yf.download(self.ticker, start=self.start, end=self.end)
        self.precios = self.datos['Close'].values.flatten() #Obtenemos un vector con los precios de cierre
        return self.precios
        
    def obtener_log_retornos(self) -> np.ndarray:
        if self.datos is None:
            self.obtener_precios()
        
        # Calculamos los log-retornos y los agregamos como una columna en el DataFrame completo
        self.datos['Log_Returns'] = np.log(self.datos['Close'] / self.datos['Close'].shift(1))
        
        # self.retornos almacena los valores sin el NaN inicial para compatibilidad
        self.retornos = self.datos['Log_Returns'].dropna().values
        return self.retornos

    def guardar_csv(self, ruta: str = None ) -> None:
        if self.datos is None:
            self.obtener_precios()
        
        if 'Log_Returns' not in self.datos.columns:
            self.obtener_log_retornos()

        if ruta is None:
            ruta = Path(__file__).resolve().parents[1] / "data" / f"{self.ticker}_{self.start}_{self.end}.csv"

        # Aseguramos que la carpeta de destino exista
        ruta.parent.mkdir(parents=True, exist_ok=True)

        # Guardamos todo el DataFrame (Open, High, Low, Close, Volume, Log_Returns) con el índice de fechas
        self.datos.to_csv(ruta)
        print(f"Archivo guardado exitosamente en: {ruta}")

    
if __name__ == "__main__":
    spy = DatosSubyacente("SPY", "2022-05-01", "2026-05-30")
    precios = spy.obtener_precios()
    retornos = spy.obtener_log_retornos()
    spy.guardar_csv()

    print(f"numero de dias: {len(precios)}")
    print(f"numero de retornos: {len(retornos)}")



        

