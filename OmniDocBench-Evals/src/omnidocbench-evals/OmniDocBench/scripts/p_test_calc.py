from statsmodels.stats.proportion import proportions_ztest

# successes and total observations
count = [round(0.8156 * 1000), round(0.8423 * 1000)]
nobs = [1355, 1355]

# two-tailed test: is Model B different?
z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

print("z =", z_stat, "p =", p_value)

