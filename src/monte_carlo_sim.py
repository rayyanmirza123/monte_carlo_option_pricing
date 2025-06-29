import numpy as np

def simulate_gbm(S0, mu, sigma, T, N, M):
    """Simulate M paths of Geometric Brownian Motion"""
    dt = T / N
    paths = np.zeros((M, N + 1))
    paths[:, 0] = S0
    for t in range(1, N + 1):
        Z = np.random.normal(size=M)
        paths[:, t] = paths[:, t-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return paths

def price_option(S0, K, T, r, sigma, N, M, option_type='call'):
    paths = simulate_gbm(S0, r, sigma, T, N, M)
    ST = paths[:, -1]
    if option_type == 'call':
        payoffs = np.maximum(ST - K, 0)
    elif option_type == 'put':
        payoffs = np.maximum(K - ST, 0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    discounted = np.exp(-r * T) * payoffs
    return paths, np.mean(discounted)

