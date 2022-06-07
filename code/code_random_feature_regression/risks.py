#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

def Theoretical_risk(norm_beta = 1, norm_delta = 1, beta_delta_dot = 1,
                     tau = 1, psi_1 = 2, psi_2 = 1,
                     mu_1 = 0.5, mu_star = np.sqrt(0.5 - (1/(2 * np.pi)) ** 2 - 0.5 ** 2),\
                    pi = 0.75):

    xi = mu_1/mu_star
    psi = np.min([psi_1, psi_2])
    chi_1 = (psi -1) * xi ** 2 - 1
    chi = - (np.sqrt(chi_1 ** 2 + 4 * (xi ** 2) * psi) + chi_1)/(2 * xi ** 2)
    eps_0 = - chi **5 * xi ** 6    + 3 * chi ** 4 * xi ** 4    + (psi_1 * psi_2 - psi_1 - psi_2 + 1) * chi ** 3 * xi ** 6    - 2 * chi ** 3 * xi ** 4            - 3 * chi ** 3 * xi ** 2            + (psi_1 + psi_2 - 3 * psi_1 * psi_2 + 1) * chi ** 2 * xi ** 4                     + 2 * chi ** 2 * xi ** 2                    + chi **2                         + 3 * psi_1 * psi_2 * chi * xi **2 - psi_1 * psi_2
    eps_1 = psi_2 * chi ** 2 * xi ** 2 * (chi * xi ** 2 - 1 )  + psi_1 * psi_2 * (chi * xi ** 2 - 1 )
    eps_2 =  chi **5 * xi ** 6    - 3 * chi ** 4 * xi ** 4            + (psi_1 - 1) * chi ** 3 * xi ** 6            + 2 * chi ** 3 * xi ** 4            + 3 * chi ** 3 * xi ** 2                    - (psi_1 + 1) * chi ** 2 * xi ** 4                     - 2 * chi ** 2 * xi ** 2                    - chi **2 
    B = eps_1/eps_0
    V = eps_2/eps_0
    m0 = chi / (mu_star ** 2)
    Psi_2 = B - 1 + 2 * (chi + psi)
    B_2 = pi * (1 - pi) * V + pi ** 2 * Psi_2
    C = (B -1 + Psi_2) * pi
    Risk = norm_beta ** 2 * B + norm_delta ** 2 * B_2 + beta_delta_dot * norm_beta * norm_delta * C\
                + tau ** 2 * V + tau ** 2
    
    return Risk


class EmpiricalRisk():

    def __init__(self, beta = 5, delta = 1, beta_delta = 1,
                 sigma = 0.1, n = 100, N = 200,
                 d = 30, p = 0.9, ITER = 100):
        super(EmpiricalRisk, self).__init__()
        self.beta = beta
        self.delta = delta
        self.sigma = sigma
        self.n = n
        self.N = N
        self.d = d
        self.p = p
        self.gamma = self.N / self.n
        self.eta = self.N / self.d
        self.Theta = np.random.normal(size = (self.d , self.N))
        self.Theta = self.Theta / np.linalg.norm(self.Theta, axis = 1).reshape((-1, 1))
        self.Theta *= np.sqrt(d)
        self.ITER = ITER
        self.beta_delta = beta_delta
        self.beta_0, self.beta_1 = self.coeff()

    def coeff(self):
        x = np.random.normal(size = (self.d, 1))
        u = x/np.linalg.norm(x)
        x = np.random.normal(size = (self.d, 1))
        x = x/np.linalg.norm(x)
        v = x - np.sum(x * u) * u
        v = v / np.linalg.norm(v)
        beta_0 = u * self.beta
        delta_0 = self.delta * (self.beta_delta * u + np.sqrt(1 - self.beta_delta ** 2) * v)

        return beta_0, beta_0 + delta_0


    def generate_data(self, beta, n):
        d, _ = beta.shape
        x = np.random.normal(size = (n, d))
        x = x / np.linalg.norm(x, axis = 1).reshape((-1, 1))
        x *= np.sqrt(d)
        y = (x @ beta) + self.sigma * np.random.normal(size = (n,1))
        # y = (1 * x[:, :1]) + self.sigma * np.random.normal(size = (n,1))
        return x, y

    def activation(self, x):
        _, d = x.shape
        z = x @ self.Theta/np.sqrt(d)
        z[z<0] = 0
        return z

    def risk(self):
        
        n0, n1 = int(self.n * (1-self.p)), int(self.n * self.p)

        x0, y0 = self.generate_data(self.beta_0, n0)
        x1, y1 = self.generate_data(self.beta_1, n1)
        x, y = np.vstack((x0, x1)), np.vstack((y0, y1))
        z = self.activation(x)

        beta_hat = np.linalg.lstsq(z, y, rcond = None)[0]
        #print(np.linalg.norm(beta_hat) ** 2)


        
        

        x_minor, y_minor = self.generate_data(self.beta_0, 500)
        z_minor = self.activation(x_minor)
        error_minor = y_minor - z_minor @ beta_hat
        return np.mean(error_minor ** 2)



def get_risks(p = 0.9, gamma = 2, eta = 2, sigma = 0.2, beta = 5, delta = 1, d = 50):
    
    """
    p: majority group prop
    gamma: N/n
    eta: N/d
    sigma: error std
    beta: signal norm (norm of minority group coeff)
    delta: shift norm
    d: data dimension
    N: random feature dimension
    n: total sample size
    """

    N = int(d *  eta)
    n = int(N/gamma)
    P = np.array([n, d, N])
    P = P * 200 / np.min(P)
    n, d, N = int(P[0]), int(P[1]), int(P[2])
    

    re = EmpiricalRisk(sigma=sigma, beta=beta, delta=delta,         p = p, n=n, d = d, N = N)
    erisk = re.risk()

    trisk = Theoretical_risk(norm_beta = beta, norm_delta = delta, tau = sigma, psi_1 = N/d, psi_2 = n/d,
       mu_1 = 0.5, mu_star = np.sqrt(0.5 - (1/(2 * np.pi))  - 0.5 ** 2),\
            pi = p)

    return  erisk, trisk


def get_trisks(p = 0.9, gamma = 2, eta = 2, sigma = 0.2, beta = 5, delta = 1, d = 50, beta_delta = 1):
    
    """
    p: majority group prop
    gamma: N/n
    eta: N/d
    sigma: error std
    beta: signal norm (norm of minority group coeff)
    delta: shift norm
    d: data dimension
    N: random feature dimension
    n: total sample size
    """

    # N = int(d *  eta)
    # n = int(N/gamma)
    # P = np.array([n, d, N])
    # P = P * 200 / np.min(P)
    # n, d, N = int(P[0]), int(P[1]), int(P[2])
    psi_1 = eta
    psi_2 = eta/gamma
    

    trisk = Theoretical_risk(norm_beta = beta, norm_delta = delta, beta_delta_dot=beta_delta, tau = sigma, psi_1 = psi_1, psi_2 = psi_2,\
                            mu_1 = 0.5, mu_star = np.sqrt(0.5 - (1/(2 * np.pi))  - 0.5 ** 2),\
                                pi = p)

    return   trisk


def get_erisks(p = 0.9, gamma = 2, eta = 2, sigma = 0.2,
               beta = 5, delta = 1, beta_delta = 1, d = 50):
    
    """
    p: majority group prop
    gamma: N/n
    eta: N/d
    sigma: error std
    beta: signal norm (norm of minority group coeff)
    delta: shift norm
    d: data dimension
    N: random feature dimension
    n: total sample size
    """

    N = int(d *  eta)
    n = int(N/gamma)
    P = np.array([n, d, N])
    P = P * 200 / np.min(P)
    n, d, N = int(P[0]), int(P[1]), int(P[2])
    

    re = EmpiricalRisk(sigma=sigma, beta=beta, delta=delta,
                       beta_delta = beta_delta, 
                       p = p, n=n, d = d, N = N)
    erisk = re.risk()

   
    return  erisk
