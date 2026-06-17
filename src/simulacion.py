import numpy as np 
import matplotlib.pyplot as plt

try:
    from src.data import DatosSubyacente

except ModuleNotFoundError:
    from data import DatosSubyacente



class CalibracionMLE:
    