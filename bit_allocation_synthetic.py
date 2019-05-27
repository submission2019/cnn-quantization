import numpy as np
import matplotlib.pyplot as plt

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump
def uniform_midtread_quantizer(x, Q):
    xQ = np.round(x / Q) * Q
    return xQ
def simulator(Val, Q):
    s = np.copy(Val)
    s = uniform_midtread_quantizer(s, Q)
    mse_1D = ((s - Val) ** 2).mean()
    return [mse_1D,Q]
def simulator3(X,Y,Q, Range):
    simulations = []
    Ratio = []
    MSE = []

    for p in Range:

        Delta_X=  (max(X) - min(X))/(p*Q)
        Delta_Y = (max(Y) - min(Y))/((1 - p)*Q)

        mse_X, Q_X = simulator(X, Delta_X)
        mse_Y, Q_Y = simulator(Y, Delta_Y)
        # simulations.append((i,NumOfBins-i,mse_X+mse_Y))
        simulations.append( mse_X + mse_Y)
        # Ratio.append(Q_Y / Q_X)
        MSE.append(mse_X + mse_Y)
    return [simulations, MSE]

if __name__== "__main__":
    Range = list(frange(0.15,0.85,0.01))
    Num_of_elements =  10000

    sigma1a = 2.82845653294  #  sigma2**(0.66666) = 2
    sigma1b = 1              # sigma1**(0.66666) = 1

    sigma2a = 1              # sigma1**(0.66666) = 1
    sigma2b = 2.82845653294  #  sigma2**(0.66666) = 2

    sigma3a = 1              # sigma1**(0.66666) = 1
    sigma3b = 1              #  sigma2**(0.66666) = 1



    X1a = np.random.normal(0, sigma1a,Num_of_elements)
    x1b = np.random.normal(0, sigma1b,Num_of_elements)

    X2a = np.random.normal(0, sigma2a,Num_of_elements)
    x2b = np.random.normal(0, sigma2b,Num_of_elements)

    X3a = np.random.normal(0, sigma3a,Num_of_elements)
    x3b = np.random.normal(0, sigma3b,Num_of_elements)

    [simulations_a, MSE_a] = simulator3(X1a,x1b,Q = 32.0, Range = Range)
    [simulations_b, MSE_b] = simulator3(X2a,x2b,Q = 32.0, Range = Range)
    [simulations_c, MSE_c] = simulator3(X3a,x3b,Q = 32.0, Range = Range)

    plt.plot(Range,MSE_a,'b', linewidth=3, label=r'$\alpha_i^\frac{2}{3}=2,\alpha_j^{\frac{2}{3}}=1$')
    plt.plot(Range,MSE_b,'r', linewidth=3,label = r'$\alpha_i^\frac{2}{3}=1,\alpha_j^\frac{2}{3}=2$')
    plt.plot(Range,MSE_c,'g', linewidth=3,label = r'$\alpha_i^\frac{2}{3}=1,\alpha_j^\frac{2}{3}=1$')

    plt.xlabel('percenatge of bins allocated for channel $i$', size=20)
    plt.legend(loc='best', prop={'size': 15, 'weight': 'bold'})
    plt.tick_params(axis='both', which='major', labelsize=20)
    plt.ylabel('Mean Square Error', size=20)
    plt.tight_layout()
    plt.yticks([])

    plt.ylim(0.000,0.22)
    plt.show()
