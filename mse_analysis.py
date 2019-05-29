import matplotlib.pyplot as plt
import numpy as np
import math


def uniform_midtread_quantizer(x, Q):
    xQ = np.round(x / Q) * Q
    return xQ

def GaussianClippingAnalysis(Alpha, sigma,bitWidth):
    Analysis = []
    for alpha in Alpha:
        clipping_mse = (sigma**2 + (alpha ** 2)) * (1 - math.erf(alpha / (sigma*np.sqrt(2.0)))) - np.sqrt(2.0/np.pi) * alpha * sigma*(np.e ** ((-1)*(0.5* (alpha ** 2))/sigma**2))
        quant_mse = (alpha ** 2) / (3 * (2 ** (2 * bitWidth)))
        mse = clipping_mse + quant_mse
        Analysis.append(mse)
    return Analysis

def GaussianClippingSimulation(Alpha, sigma,bitWidth):
    highPrecision = np.random.normal(0, sigma, size=100000)
    simulations = []
    for alpha in Alpha:
        s = np.copy(highPrecision)
        Q = (2*alpha)/(2**bitWidth)
        # clipping
        s[s > alpha] = alpha
        s[s < -alpha] = -alpha
        # quabtization
        s = uniform_midtread_quantizer(s, Q)

        mse = ((s - highPrecision) ** 2).mean()
        simulations.append(mse)
    return simulations



def LaplacianClippingAnalysis(Alpha, b,bitWidth):
    Analysis = []
    for alpha in Alpha:
        mse = 2 * (b ** 2) * ((np.e) ** (-alpha / b)) + ((alpha ** 2) / (3 * (2 ** (2 * bitWidth))))
        Analysis.append(mse)
    return Analysis

def LaplacianClippingSimulation(Alpha, b, bitWidth):
    simulations = []
    highPrecision = np.random.laplace(scale=b, size=100000, loc = 0)
    for alpha in Alpha:
        s = np.copy(highPrecision)
        Q = (2*alpha)/(2**bitWidth)

        #clipping
        s[s > alpha ] = alpha
        s[s < -alpha] = -alpha
        # quantization
        s = uniform_midtread_quantizer(s, Q)

        mse = ((s - highPrecision) ** 2).mean()
        simulations.append(mse)
    return simulations


if __name__ == "__main__":
    Alpha = np.arange(5, 20, 0.1)

    #Experiment parameters
    bitWidth = 4
    sigma = 2  # standard deviation

    #Gauss
    # simulation  = GaussianClippingSimulation(Alpha,sigma,bitWidth)
    # analysis = GaussianClippingAnalysis(Alpha, sigma, bitWidth)

    #Laplace
    simulation = LaplacianClippingSimulation(Alpha, sigma, bitWidth)
    analysis = LaplacianClippingAnalysis(Alpha, sigma, bitWidth)


    plt.plot(Alpha,simulation,'b', linewidth=5)
    plt.plot(Alpha,analysis,'r', linewidth=2)
    plt.legend(('simulation', 'analysis')); plt.ylabel('Mean Square Error', size=20) ; plt.xlabel('Clipping Value', size=20)
    plt.title('Bit Width='+ str(bitWidth), size=20)
    plt.show()
