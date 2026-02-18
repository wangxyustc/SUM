import numpy as np
import math
from scipy.optimize import brentq

MAX_ITERATIONS = 6000
LARGE_B = 1000.0
EPSILON = 1e-4

TYPE_CONVEX = 'convex'
TYPE_CONCAVE = 'concave'
TYPE_SIGMOID = 'sigmoid'

class Demand:
    def __init__(self, d_id, source, dest, link_requirements, u_type, a, b):
        self.id = d_id
        self.source = source
        self.dest = dest
        self.link_requirements = link_requirements
        self.path = list(link_requirements.keys())
        
        self.u_type = u_type
        self.a = a
        self.b = b
        
        self.Mi = 0
        self.ri = 0.0
        self.ri_final = 0
        
        self.r0 = 5.0
        self.r_hat = 0
        self.Lambda_hat = 0
        if self.u_type == TYPE_SIGMOID:
            self._precalculate_sigmoid_params()

    def _precalculate_sigmoid_params(self):
        def target_func(r):
            if r <= 0: return -0.01
            u = self.utility(r)
            u_prime = self.marginal_utility(r)
            return u - r * u_prime
        try:
            self.r_hat = brentq(target_func, 0.1, self.r0 * 2)
            self.Lambda_hat = self.marginal_utility(self.r_hat)
        except:
            self.r_hat = self.r0
            self.Lambda_hat = self.marginal_utility(self.r0)

    def utility(self, r):
        if r < 0: return 0
        if self.u_type == TYPE_CONVEX:
            return self.a * (r**2) + self.b * r
        elif self.u_type == TYPE_CONCAVE:
            return self.a * np.log(1 + self.b * r)
        elif self.u_type == TYPE_SIGMOID:
            val = self.a / (1 + np.exp(-self.b * (r - self.r0)))
            val_0 = self.a / (1 + np.exp(-self.b * (0 - self.r0)))
            return max(0, val - val_0)
        return 0

    def marginal_utility(self, r):
        r = max(0, r)
        if self.u_type == TYPE_CONVEX:
            return 2 * self.a * r + self.b
        elif self.u_type == TYPE_CONCAVE:
            return (self.a * self.b) / (1 + self.b * r)
        elif self.u_type == TYPE_SIGMOID:
            exp_term = np.exp(-self.b * (r - self.r0))
            return (self.a * self.b * exp_term) / ((1 + exp_term)**2)
        return 0

    def second_derivative(self, r):
        r = max(0, r)
        if self.u_type == TYPE_CONVEX:
            return 2 * self.a
        elif self.u_type == TYPE_CONCAVE:
            return - (self.a * (self.b**2)) / ((1 + self.b * r)**2)
        elif self.u_type == TYPE_SIGMOID:
            exp_term = np.exp(-self.b * (r - self.r0))
            term1 = (1 + exp_term)**(-3)
            term2 = self.a * (self.b**2) * exp_term * (exp_term - 1)
            return term2 * term1
        return 0

    def solve_optimal_r(self, Lambda_i):
        if self.u_type == TYPE_CONVEX:
            val_Mi = self.utility(self.Mi) - Lambda_i * self.Mi
            return float(self.Mi) if val_Mi >= 0 else 0.0
        elif self.u_type == TYPE_CONCAVE:
            u_prime_0 = self.marginal_utility(0)
            u_prime_Mi = self.marginal_utility(self.Mi)
            if u_prime_0 < Lambda_i: return 0.0
            elif u_prime_Mi > Lambda_i: return float(self.Mi)
            else:
                if Lambda_i <= 1e-6: return float(self.Mi)
                opt = (self.a * self.b / Lambda_i - 1) / self.b
                return max(0.0, min(float(self.Mi), opt))
        elif self.u_type == TYPE_SIGMOID:
            candidates = [0.0, float(self.Mi)]
            max_slope = self.marginal_utility(self.r0)
            if max_slope > Lambda_i:
                try:
                    res_r = brentq(lambda r: self.marginal_utility(r) - Lambda_i, 
                                   self.r0, self.Mi + 5) 
                    if 0 <= res_r <= self.Mi:
                        candidates.append(res_r)
                except:
                    pass
            best_r = 0
            best_val = -float('inf')
            for r in candidates:
                val = self.utility(r) - Lambda_i * r
                if val > best_val:
                    best_val = val
                    best_r = r
            return best_r
        return 0.0

    def calculate_derivative_wrt_lambda(self, link_idx, Lambda_i):
        aji = self.link_requirements[link_idx]
        
        if self.u_type == TYPE_CONVEX:
            diff = abs(self.utility(self.Mi) - Lambda_i * self.Mi)
            return -aji * self.Mi if diff < EPSILON else -aji * self.Mi / LARGE_B

        elif self.u_type == TYPE_CONCAVE:
            if 0 < self.ri < self.Mi:
                u_pp = self.second_derivative(self.ri)
                return -LARGE_B if abs(u_pp) < 1e-10 else aji / u_pp
            return -1e-5 

        elif self.u_type == TYPE_SIGMOID:
            if Lambda_i - self.Lambda_hat > EPSILON:
                 if 0 < self.ri < self.Mi:
                     u_pp = self.second_derivative(self.ri)
                     return -LARGE_B if abs(u_pp) < 1e-10 else aji / u_pp
                 else:
                     return -1e-5
            elif abs(Lambda_i - self.Lambda_hat) <= EPSILON:
                return -aji * self.r_hat
            else:
                return -aji * self.r_hat / LARGE_B
        return 0

def run_sum_algorithm(demands, links, capacities):
    L = len(links)
    lambdas = np.ones(L) * 0.1 
    zetas = np.zeros(L)        
    
    history = {
        'utility': [],
        'lambda_norm': [],
        'zeta_norm': [],
        'violation_count': []
    }
    
    for t in range(1, MAX_ITERATIONS + 1):
        eta1 = 0.2 / (1 + 0.005 * t)
        eta2 = 0.05 / (1 + 0.001 * t)

        current_iter_utility = 0
        for d in demands:
            Lambda_i = sum(d.link_requirements[l] * lambdas[l] for l in d.path)
            d.ri = d.solve_optimal_r(Lambda_i)
            current_iter_utility += d.utility(d.ri)
            
        lambda_gradients = np.zeros(L)
        for d in demands:
            sum_zeta_weighted = sum(d.link_requirements[l] * zetas[l] for l in d.path)
            Zi = 1.0 - sum_zeta_weighted
            Lambda_i = sum(d.link_requirements[l] * lambdas[l] for l in d.path)
            
            for j in d.path:
                derivative = d.calculate_derivative_wrt_lambda(j, Lambda_i)
                lambda_gradients[j] += Zi * derivative
        
        lambdas = lambdas + eta1 * lambda_gradients
        lambdas = np.maximum(0, lambdas)
        
        link_loads = np.zeros(L)
        for d in demands:
            for l_idx, aji in d.link_requirements.items():
                link_loads[l_idx] += aji * d.ri
        
        constraint_val = capacities - link_loads
        zetas = zetas - eta2 * constraint_val
        zetas = np.maximum(0, zetas)
        
        history['utility'].append(current_iter_utility)
        history['lambda_norm'].append(np.linalg.norm(lambdas))
        history['zeta_norm'].append(np.linalg.norm(zetas))
        
        violations = np.sum(constraint_val < -0.1)
        history['violation_count'].append(violations)

    for d in demands:
        d.ri_final = int(math.floor(d.ri))
    
    final_u = sum(d.utility(d.ri_final) for d in demands)
    return history, final_u