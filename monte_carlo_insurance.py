
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---- User-tunable assumptions ----
YEARS = 100_000          # number of simulated years
LAMBDA = 1.2             # Poisson mean (expected claim count per year)
SEV_MEAN = 3000.0        # average claim size (dollars)
SEV_CV = 1.3             # severity coefficient of variation (std/mean)
CAPITAL_K = 10_000.0     # capital buffer for ruin probability

# ---- Convert mean & CV to lognormal parameters ----
sigma2 = np.log(1 + SEV_CV**2)
sigma = np.sqrt(sigma2)
mu = np.log(SEV_MEAN) - 0.5 * sigma2

# ---- Simulate ----
N = rng.poisson(LAMBDA, size=YEARS)             # yearly claim counts
total_claims = int(N.sum())
severities = rng.lognormal(mu, sigma, size=total_claims)

# Aggregate to yearly totals
idx = np.cumsum(N)[:-1]
S = np.add.reduceat(severities, np.r_[0, idx]) if len(idx) > 0 else np.zeros(YEARS)

# ---- Metrics ----
def metrics(series, K=CAPITAL_K):
    series = np.asarray(series)
    VaR95 = np.quantile(series, 0.95)
    VaR99 = np.quantile(series, 0.99)
    return {
        "Expected Loss": series.mean(),
        "Std Dev": series.std(ddof=1),
        "P(Zero Loss)": float(np.mean(series == 0)),
        "VaR 95%": VaR95,
        "VaR 99%": VaR99,
        "TVaR 95%": float(series[series >= VaR95].mean()),
        "TVaR 99%": float(series[series >= VaR99].mean()),
        "Ruin Prob (K=$%dk)" % (K/1000): float(np.mean(series > K)),
    }

base_metrics = metrics(S)

# ---- Simple scenarios ----
def run_scenario(freq_mult=1.0, sev_mult=1.0):
    N_s = rng.poisson(LAMBDA*freq_mult, size=YEARS)
    total_s = int(N_s.sum())
    # scale severity mean by multiplying by sev_mult (add to mu in log space)
    sigma2 = np.log(1 + SEV_CV**2)
    sigma = np.sqrt(sigma2)
    mu = np.log(SEV_MEAN) - 0.5 * sigma2 + np.log(sev_mult)
    sev_all_s = rng.lognormal(mu, sigma, size=total_s)
    idx_s = np.cumsum(N_s)[:-1]
    S_s = np.add.reduceat(sev_all_s, np.r_[0, idx_s]) if len(idx_s) > 0 else np.zeros(YEARS)
    return metrics(S_s)

scenarios = pd.DataFrame.from_dict({
    "Base": base_metrics,
    "Higher Freq (x1.3)": run_scenario(freq_mult=1.3),
    "Higher Sev (x1.3)": run_scenario(sev_mult=1.3),
    "Cat Year (freq x1.1, sev x1.5)": run_scenario(freq_mult=1.1, sev_mult=1.5),
}, orient="index")

pd.DataFrame(base_metrics, index=["Base"]).to_csv("metrics.csv", index=True)
scenarios.to_csv("scenarios.csv", index=True)

print("Saved metrics.csv and scenarios.csv")
print(scenarios.round(2))

# ---- Visualizations ----
import matplotlib.pyplot as plt

# Histogram of annual losses
plt.figure(figsize=(8, 5))
plt.hist(S, bins=100, edgecolor='black')
plt.title("Annual Aggregate Loss Distribution")
plt.xlabel("Loss ($)")
plt.ylabel("Frequency")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Empirical CDF (shows probability of being below a given loss)
S_sorted = np.sort(S)
cdf = np.linspace(0, 1, len(S_sorted))

plt.figure(figsize=(8, 5))
plt.plot(S_sorted, cdf)
plt.title("Cumulative Distribution of Annual Loss")
plt.xlabel("Loss ($)")
plt.ylabel("Cumulative Probability")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

