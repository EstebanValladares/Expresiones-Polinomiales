from abc import ABC, abstractmethod
from scipy.integrate import quad
import numpy as np
import random
import math

# Clase Base Abstracta
class Distribution(ABC):
    @abstractmethod
    def getProbability(self, value, acc=False): 
        pass

    @abstractmethod
    def getSample(self, n=1):
        pass

# Distribucion Normal
class NormalDistribution(Distribution):
    def __init__(self, mu, sd, a=10):
        self.mu = mu
        self.sd = sd
        self.a = a 
    
    def getFunctionValue(self, z):
        coef = 1 / (math.sqrt(2 * math.pi))
        return coef * np.exp(-0.5 * (z**2))

    def getProbability(self, x, acc=True):
        z_limit = (x - self.mu) / self.sd
        p, err = quad(self.getFunctionValue, -np.inf, z_limit) 
        return p
    
    def getSample(self, n=1):
        sample = []
        pdf_values = [] 
        max_pdf = self.getFunctionValue(0)
        
        for _ in range(n):
            while True:
                u = random.uniform(self.mu - self.a * self.sd, self.mu + self.a * self.sd)
                fu = random.uniform(0, max_pdf)
                z = (u - self.mu) / self.sd
                fz = self.getFunctionValue(z)
                if fu < fz:
                    sample.append(float(u))
                    pdf_values.append(fz / self.sd)
                    break
        return sample, pdf_values

# Distribucion Poisson 
class PoissonDistribution(Distribution):
    def __init__(self, l=1):
        self.l = l
    
    def getProbability(self, k, acc=False):
        if k < 0: return 0.0
        
        def pmf(val):
            return (self.l**val) * np.exp(-self.l) / math.factorial(int(val))
        
        if not acc:
            return pmf(k)
        return sum(pmf(i) for i in range(int(k) + 1))
    
    def getSample(self, n=1):
        sample, pmf_values = [], []
        for _ in range(n):
            u = random.random()
            k, p, f = 0, np.exp(-self.l), np.exp(-self.l)
            while u > f:
                k += 1
                p *= (self.l / k)
                f += p
            sample.append(float(k))
            pmf_values.append(self.getProbability(k)) 
        return sample, pmf_values

# Distribució¿on Binomial Negativa
class NegativeBinomialDistribution(Distribution):
    def __init__(self, r, p):
        self.r = int(r) 
        self.p = float(p)

    def getProbability(self, x, acc=False):
        if x < 0: return 0.0
        
        def pmf(k):
            return math.comb(int(k + self.r - 1), int(k)) * (self.p**self.r) * ((1 - self.p)**k)

        if not acc:
            return pmf(x)
        return sum(pmf(i) for i in range(int(x) + 1))

    def getSample(self, n=1):
        sample, pmf_values = [], []
        for _ in range(n):
            u = random.random()
            k = 0
            curr_p = self.getProbability(0)
            f = curr_p
            while u > f:
                k += 1
                curr_p = self.getProbability(k)
                f += curr_p
            sample.append(float(k))
            pmf_values.append(curr_p)
        return sample, pmf_values

# Distribuciones: 
class DistributionFactory:
    @staticmethod
    def create(distribution_type, **kwargs):
        dtype = distribution_type.lower().replace(" ", "_")
        
        if dtype == 'normal':
            DistributionFactory._check_params(["mu", "sd"], kwargs)
            return NormalDistribution(kwargs["mu"], kwargs["sd"]) 
        
        elif dtype == "poisson":
            DistributionFactory._check_params(["l"], kwargs)
            return PoissonDistribution(kwargs["l"])
        
        elif dtype == "binomial_negativa":
            DistributionFactory._check_params(["r", "p"], kwargs)
            return NegativeBinomialDistribution(kwargs["r"], kwargs["p"])
        
        else:
            raise ValueError(f"Distribucion '{distribution_type}' no soportada")

    @staticmethod
    def _check_params(required, provided):
        for p in required:
            if p not in provided:
                raise ValueError(f"Error: El parametro '{p}' es obligatorio")

# --- PRUEBAS DE EJECUCION ---
if __name__ == "__main__":
    # 1. Probar Binomial Negativa
    #print("Binomial Negativa (r=5, p=0.5): ")
    dist_bn = DistributionFactory.create("binomial_negativa", r=5, p=0.5)
    #print(f"Prob. puntual 3 fracasos P(X=3): {dist_bn.getProbability(3):.4f}")
    #print(f"Prob. acumulada hasta 3 fracasos P(X<=3): {dist_bn.getProbability(3, acc=True):.4f}")
    muestras_bn, _ = dist_bn.getSample(5)
    print(f"Muestra (float[]): {muestras_bn}\n")

    # 2. Probar Normal
    #print("Normal (mu=0, sd=1): ")
    dist_n = DistributionFactory.create("normal", mu=0, sd=1)
    #print(f"Prob. acumulada P(X<=0): {dist_n.getProbability(0):.4f}")
    muestras_n, _ = dist_n.getSample(3)
    print(f"Muestra: {muestras_n}\n")

    # 3. Probar Poisson
    #print("Poisson (l=3): ")
    dist_p = DistributionFactory.create("poisson", l=3)
    #print(f"Prob. puntual P(X=2): {dist_p.getProbability(2):.4f}")
    muestras_p, _ = dist_p.getSample(3)
    #print(f"Muestra: {muestras_p}")