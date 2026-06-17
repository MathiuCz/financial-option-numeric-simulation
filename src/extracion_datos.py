import yfinance as yf 
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any

class DatosSubyacente:

    def __init__(self, ticker: str, start: str, end: str):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.datos: pd.DataFrame | None = None
        self.precios: Any =  None
        self.retornos: Any = None

    def obtener_precios(self) -> Any:
        datos = yf.download(self.ticker, start=self.start, end=self.end)
        self.datos = datos
        self.precios = datos['Close'].to_numpy().flatten() #Obtenemos un vector con los precios de cierre
        return self.precios
        
    def obtener_log_retornos(self) -> Any:
        if self.datos is None:
            self.obtener_precios()
        
        if self.datos is None:
            raise ValueError("No se pudieron descargar los datos de precios.")
        
        # Calculamos los log-retornos y los agregamos como una columna en el DataFrame completo
        self.datos['Log_Returns'] = np.log(self.datos['Close'] / self.datos['Close'].shift(1))
        
        # self.retornos almacena los valores sin el NaN inicial para compatibilidad
        retornos = self.datos['Log_Returns'].dropna().to_numpy()
        self.retornos = retornos
        return retornos

    def guardar_csv(self, ruta: str | Path | None = None ) -> None:
        if self.datos is None:
            self.obtener_precios()
        
        if self.datos is None:
            raise ValueError("No hay datos disponibles para guardar.")
        
        if 'Log_Returns' not in self.datos.columns:
            self.obtener_log_retornos()

        if ruta is None:
            ruta_path = Path(__file__).resolve().parents[1] / "data" / f"{self.ticker}_{self.start}_{self.end}.csv"
        else:
            ruta_path = Path(ruta)

        # Aseguramos que la carpeta de destino exista
        ruta_path.parent.mkdir(parents=True, exist_ok=True)

        # Guardamos todo el DataFrame (Open, High, Low, Close, Volume, Log_Returns) con el índice de fechas
        self.datos.to_csv(ruta_path)
        print(f"Archivo guardado exitosamente en: {ruta_path}")

    
if __name__ == "__main__":
    spy = DatosSubyacente("SPY", "2022-05-01", "2026-05-30")
    precios = spy.obtener_precios()
    retornos = spy.obtener_log_retornos()
    spy.guardar_csv()

    print(f"numero de dias: {len(precios)}")
    print(f"numero de retornos: {len(retornos)}")



        

