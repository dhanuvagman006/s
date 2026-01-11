#6. Program to display Normal, Binomial Poisson, Bernoulli
#disrtibutions for a given frequency distribution and analyze the
#results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, binom, poisson, bernoulli
# -----------------------------------------------------------
# Create sample frequency distribution data (NO CSV REQUIRED)
# -----------------------------------------------------------
data = pd.DataFrame({
 'Values': [12, 15, 14, 16, 18, 20, 22, 19, 17, 21,
 23, 25, 24, 26, 28, 30, 29, 27, 31, 33]
})
# -----------------------------------------------------------
# Normal Distribution
# -----------------------------------------------------------
# Calculate mean and standard deviation of the data
mean = np.mean(data['Values'])
std_dev = np.std(data['Values'])
print("Normal Distribution")
print("------------------")
# Create normal distribution
normal_dist = norm(loc=mean, scale=std_dev)
# Generate samples
normal_samples = normal_dist.rvs(size=len(data))
# Plot
plt.hist(normal_samples, bins=30, density=True)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
# -----------------------------------------------------------
# Binomial Distribution
# -----------------------------------------------------------
print("\nBinomial Distribution")
print("---------------------")
n = 10
p = 0.5
binom_dist = binom(n=n, p=p)
binom_samples = binom_dist.rvs(size=len(data))
plt.hist(binom_samples, bins=30, density=True)
plt.title("Binomial Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
# -----------------------------------------------------------
# Poisson Distribution
# -----------------------------------------------------------
print("\nPoisson Distribution")
print("---------------------")
lam = 5
poisson_dist = poisson(mu=lam)
poisson_samples = poisson_dist.rvs(size=len(data))
plt.hist(poisson_samples, bins=30, density=True)
plt.title("Poisson Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
# -----------------------------------------------------------
# Bernoulli Distribution
# -----------------------------------------------------------
print("\nBernoulli Distribution")
print("----------------------")
p = 0.5
bernoulli_dist = bernoulli(p=p)
bernoulli_samples = bernoulli_dist.rvs(size=len(data))
plt.hist(bernoulli_samples, bins=30, density=True)
plt.title("Bernoulli Distribution")
plt.xlabel("Value")
plt.ylabel("Probability")
plt.show()
# -----------------------------------------------------------
# Analysis of Results
# -----------------------------------------------------------
print("\nAnalysis of Results")
print("-------------------")
normal_mean = np.mean(normal_samples)
normal_std_dev = np.std(normal_samples)
binom_mean = np.mean(binom_samples)
binom_std_dev = np.std(binom_samples)
poisson_mean = np.mean(poisson_samples)
poisson_std_dev = np.std(poisson_samples)
bernoulli_mean = np.mean(bernoulli_samples)
bernoulli_std_dev = np.std(bernoulli_samples)
print("Normal Distribution: Mean =", normal_mean, ", Standard Deviation =", normal_std_dev)
print("Binomial Distribution: Mean =", binom_mean, ", Standard Deviation =", binom_std_dev)
print("Poisson Distribution: Mean =", poisson_mean, ", Standard Deviation =", poisson_std_dev)
print("Bernoulli Distribution: Mean =", bernoulli_mean, ", Standard Deviation =",
bernoulli_std_dev)
