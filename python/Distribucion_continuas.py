from abc import ABC, abstractmethod
import numpy as np
import random
import math
from scipy.integrate import quad
from scipy.special import gammainc

# --- Clase Base ---
class Distribution(ABC):
    @abstractmethod
    def getProbability(self, x: float, acc: bool) -> float:
        pass

    @abstractmethod
    def getSample(self, n: int):
        pass

# --- Distribución Gamma
class GammaDistribution(Distribution):
    def __init__(self, k: float, theta: float):
        self.k = float(k)     
        self.theta = float(theta) 

    def getProbability(self, x: float, acc: bool = False) -> float:
        if x <= 0: return 0.0
        
        if not acc:
            return (1 / (math.gamma(self.k) * (self.theta**self.k))) * \
                   (x**(self.k - 1)) * math.exp(-x / self.theta)
        else:
            return gammainc(self.k, x / self.theta)

    def getSample(self, n: int = 1):
        samples = np.random.gamma(self.k, self.theta, n).tolist()
        pdf_values = [self.getProbability(s, acc=False) for s in samples]
        return samples, pdf_values

# Distribución Exponencial 
class ExponentialDistribution(Distribution):
    def __init__(self, l: float):
        self.l = float(l) 

    def getProbability(self, x: float, acc: bool = False) -> float:
        if x < 0: return 0.0
        
        if not acc:
            return self.l * math.exp(-self.l * x)
        else:

            return 1 - math.exp(-self.l * x)

    def getSample(self, n: int = 1):
        samples = []
        pdf_values = []
        for _ in range(n):
            u = random.random()
            # Transformada inversa
            x = -math.log(1 - u) / self.l
            samples.append(float(x))
            pdf_values.append(self.getProbability(x, acc=False))
        return samples, pdf_values

# Distribución Normal
class NormalDistribution(Distribution):
    def __init__(self, mean: float, sd: float):
        self.mean = float(mean)
        self.sd = float(sd)

    def _pdf(self, x: float) -> float:
        return (1 / (self.sd * math.sqrt(2 * math.pi))) * \
               math.exp(-0.5 * ((x - self.mean) / self.sd)**2)

    def getProbability(self, x: float, acc: bool = False) -> float:
        if not acc:
            return self._pdf(x)
        else:
            p, _ = quad(self._pdf, -np.inf, x)
            return p

    def getSample(self, n: int = 1):
        samples = np.random.normal(self.mean, self.sd, n).tolist()
        pdf_values = [self._pdf(s) for s in samples]
        return samples, pdf_values

# Distribuciones Continuas 
class ContinuousDistributionFactory:
    @staticmethod
    def create(distribution_type: str, **kwargs):
        dtype = distribution_type.lower().strip()
        
        if dtype == "gamma":
            ContinuousDistributionFactory._check_params(["k", "theta"], kwargs)
            return GammaDistribution(k=kwargs["k"], theta=kwargs["theta"])
            
        elif dtype == "exponential":
            l_val = kwargs.get("l") or kwargs.get("lambda_val")
            if l_val is None: raise ValueError("Falta parámetro: l")
            return ExponentialDistribution(l=l_val)
            
        elif dtype == "normal":
            ContinuousDistributionFactory._check_params(["mean", "sd"], kwargs)
            return NormalDistribution(mean=kwargs["mean"], sd=kwargs["sd"])
        
        else:
            raise ValueError(f"Distribución '{distribution_type}' no soportada en este módulo.")

    @staticmethod
    def _check_params(required, provided):
        for p in required:
            if p not in provided:
                raise ValueError(f"Error: El parámetro '{p}' es obligatorio.")

if __name__ == "__main__":
    factory = ContinuousDistributionFactory()
    
    # Probar Gamma
    g = factory.create("gamma", k=2.0, theta=2.0)
    print(f"Gamma P(X<=1): {g.getProbability(1, acc=True):.4f}")
    
    # Probar Exponencial
    e = factory.create("exponential", l=0.5)
    muestras_e, _ = e.getSample(5)
    print(f"Muestras Exponencial: {muestras_e}")
    
    # Probar Normal
    n = factory.create("normal", mean=100, sd=15)
    print(f"Normal P(X=100) [PDF]: {n.getProbability(100, acc=False):.4f}")