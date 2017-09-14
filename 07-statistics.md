# Statistics

# Table of Contents

[1. Introduction](#section-a)  
[2. Why We Are Using Think Stats](#section-b)  
[3. Instructions for Cloning the Repo](#section-c)  
[4. Required Exercises](#section-d)  
[5. Optional Exercises](#section-e)  
[6. Recommended Reading](#section-f)  
[7. Resources](#section-g)

## <a name="section-a"></a>1.  Introduction

[<img src="img/think_stats.jpg" title="Think Stats"/>](http://greenteapress.com/thinkstats2/)

Use Allen Downey's [Think Stats (second edition)](http://greenteapress.com/thinkstats2/) book for getting up to speed with core ideas in statistics and how to approach them programmatically. This book is available online, or you can buy a paper copy if you would like.

Use this book as a reference when answering the 6 required statistics questions below.  The Think Stats book is approximately 200 pages in length.  **It is recommended that you read the entire book, particularly if you are less familiar with introductory statistical concepts.**

Complete the following exercises along with the questions in this file. Some can be solved using code provided with the book. The preface of Think Stats [explains](http://greenteapress.com/thinkstats2/html/thinkstats2001.html#toc2) how to use the code.  

Communicate the problem, how you solved it, and the solution, within each of the following [markdown](https://guides.github.com/features/mastering-markdown/) files. (You can include code blocks and images within markdown.)

## <a name="section-b"></a>2.  Why We Are Using Think Stats 

The stats exercises have been chosen to introduce/solidify some relevant statistical concepts related to data science.  The solutions for these exercises are available in the [ThinkStats repository on GitHub](https://github.com/AllenDowney/ThinkStats2).  You should focus on understanding the statistical concepts, python programming and interpreting the results.  If you are stuck, review the solutions and recode the python in a way that is more understandable to you. 

For example, in the first exercise, the author has already written a function to compute Cohen's D.  **You could import it, or you could write your own code to practice python and develop a deeper understanding of the concept.** 

Think Stats uses a higher degree of python complexity from the python tutorials and introductions to python concepts, and that is intentional to prepare you for the bootcamp.  

**One of the skills to learn here is to understand other people’s code.  And this author is quite experienced, so it’s good to learn how functions and imports work.**

---

## <a name="section-c"></a>3.  Instructions for Cloning the Repo 
Using the [code referenced in the book](https://github.com/AllenDowney/ThinkStats2), follow the step-by-step instructions below.  

**Step 1. Create a directory on your computer where you will do the prework.  Below is an example:**

```
(Mac):      /Users/yourname/ds/metis/metisgh/prework  
(Windows):  C:/ds/metis/metisgh/prework
```

**Step 2. cd into the prework directory.  Use GitHub to pull this repo to your computer.**

```
$ git clone https://github.com/AllenDowney/ThinkStats2.git
```

**Step 3.  Put your ipython notebook or python code files in this directory (that way, it can pull the needed dependencies):**

```
(Mac):     /Users/yourname/ds/metis/metisgh/prework/ThinkStats2/code  
(Windows):  C:/ds/metis/metisgh/prework/ThinkStats2/code
```

---


## <a name="section-d"></a>4.  Required Exercises

*Include your Python code, results and explanation (where applicable).*

### Q1. [Think Stats Chapter 2 Exercise 4](statistics/2-4-cohens_d.md) (effect size of Cohen's d)  
Cohen's D is an example of effect size.  Other examples of effect size are:  correlation between two variables, mean difference, regression coefficients and standardized test statistics such as: t, Z, F, etc. In this example, you will compute Cohen's D to quantify (or measure) the difference between two groups of data.   

You will see effect size again and again in results of algorithms that are run in data science.  For instance, in the bootcamp, when you run a regression analysis, you will recognize the t-statistic as an example of effect size.

```
import nsfg

def cohens_d(group1, group2):
	"""This function calculates Cohen's d from two input Series."""
	mean1 = group1.mean()
	mean2 = group2.mean()
	
	n1 = len(group1)
	n2 = len(group2)
	
	var1 = group1.var()
	var2 = group2.var()
	pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)

	d = (mean1 - mean2) / (pooled_var ** (1/2))
	return d

df = nsfg.ReadFemPreg()
live = df[df.outcome == 1]
first_babies = live[live.birthord == 1]
other_babies = live[live.birthord != 1]

mean_firsts = first_babies.totalwgt_lb.mean()
mean_others = other_babies.totalwgt_lb.mean()	
cohens_d_babies = cohens_d(first_babies.totalwgt_lb, other_babies.totalwgt_lb)

print("First babies weigh {0!s} lbs on average.".format(mean_firsts))
print("Other babies weigh {0!s} lbs on average.".format(mean_others))
print("The Cohen's d effect size is {0:.3f} standard deviations.".format(cohens_d_babies))
```

*Both the Cohen's d effect size and the difference between mean pregnancy length for firsts babies and other babies are very small.  This suggests that there is not a strong difference between the expected weight of a first baby, versus that of a subsequent baby.  The Cohen's d measure tells us that the mean birthweight for first babies is only a small fraction of a standard deviation away from the mean pregnancy length for other babies.*



### Q2. [Think Stats Chapter 3 Exercise 1](statistics/3-1-actual_biased.md) (actual vs. biased)
This problem presents a robust example of actual vs biased data.  As a data scientist, it will be important to examine not only the data that is available, but also the data that may be missing but highly relevant.  You will see how the absence of this relevant data will bias a dataset, its distribution, and ultimately, its statistical interpretation.

```
import nsfg

def pmf_get_mean(pmf):
	"""This function takes a PMF (frequency dictionary) and returns the mean."""
	mean = 0
	total_count = sum(pmf.values())
	for value, frequency in pmf.items():
		mean += (frequency / total_count) * value
	return mean

# Read the data into a histogram dictionary.
df = nsfg.ReadFemResp()
kid_hist = {}
for response in df.numkdhh:
	kid_hist[response] = kid_hist.get(response, 0) + 1

# Create PMF dictionary of the data.
true_kid_pmf = {}
n = sum(kid_hist.values())
for number_kids, freq in kid_hist.items():
	true_kid_pmf[number_kids] = freq / n

# Bias the PMF:
## What responses would we get if we got family size by interviewing the kids?
biased_kid_pmf = {}
for k, v in kid_hist.items():
	biased_kid_pmf[k] = v
for number_kids, freq in biased_kid_pmf.items():
	biased_kid_pmf[number_kids] *= (number_kids / n)

# Calculate means of PMFs and print findings.
true_kid_mean = pmf_get_mean(true_kid_pmf)
biased_kid_mean = pmf_get_mean(biased_kid_pmf)

print("From the sample, the mean number of kids in a household is {0:.3f}.".format(true_kid_mean))
print("If we surveyed the kids, they would report a mean of {0:.3f}.".format(biased_kid_mean))
```

*From the sample, the mean number of kids in a household is 1.024.
But if we surveyed the kids, they would report a mean of 2.404.*


### Q3. [Think Stats Chapter 4 Exercise 2](statistics/4-2-random_dist.md) (random distribution)  
This questions asks you to examine the function that produces random numbers.  Is it really random?  A good way to test that is to examine the pmf and cdf of the list of random numbers and visualize the distribution.  If you're not sure what pmf is, read more about it in Chapter 3.  

```
import random
import thinkstats2
import thinkplot

nums = []
for i in range(1000):
	nums.append(random.random())

random_pmf = thinkstats2.Pmf(nums)
random_cdf = thinkstats2.Cdf(nums)

thinkplot.PrePlot(2, 2)
thinkplot.Pmf(random_pmf)
thinkplot.PrePlot(2)
thinkplot.SubPlot(2)
thinkplot.Cdf(random_cdf)
thinkplot.Show()
```

*The distribution appears to be uniform.  The PMF of the distribution looks to be a "flat" bar with no peaks or troughs, meaning that all values between 0 and 1 are equally likely to be sampled.  Furthermore, the plot of the CDF is an almost straight diagonal line from the bottom of the distribution to the top.  Because no area of the CDF plot is steeper or flatter than any other area, that means that the probability of sampling any given value is about the same as the probability of sampling any other value; hence, the distribution is uniform, and the numbers are likely to be randomly chosen.*


### Q4. [Think Stats Chapter 5 Exercise 1](statistics/5-1-blue_men.md) (normal distribution of blue men)
This is a classic example of hypothesis testing using the normal distribution.  The effect size used here is the Z-statistic. 

```
from scipy.stats import norm

mean = 178
std = 7.7

males = norm(loc=mean, scale=std)

# The upper and lower bounds are the conversions from ft & in to cm.
lower_bound = 177.8
upper_bound = 185.4

perc_in_range = (males.cdf(upper_bound) - males.cdf(lower_bound)) * 100

print("{0:.1f}% of the US male population is between 5'10\" and 6'1\",\
 and therefore eligible to join the\
 Blue Man Group.".format(perc_in_range))
```
*34.2% of the US male population is between 5'10" and 6'1", and therefore eligible to join the Blue Man Group.*

### Q5. Bayesian (Elvis Presley twin) 

Bayes' Theorem is an important tool in understanding what we really know, given evidence of other information we have, in a quantitative way.  It helps incorporate conditional probabilities into our conclusions.

Elvis Presley had a twin brother who died at birth.  What is the probability that Elvis was an identical twin? Assume we observe the following probabilities in the population: fraternal twin is 1/125 and identical twin is 1/300.  

>> REPLACE THIS TEXT WITH YOUR RESPONSE

---

### Q6. Bayesian &amp; Frequentist Comparison  
How do frequentist and Bayesian statistics compare?

>> REPLACE THIS TEXT WITH YOUR RESPONSE

---

## <a name="section-e"></a>5.  Optional Exercises

The following exercises are optional, but we highly encourage you to complete them if you have the time.

### Q7. [Think Stats Chapter 7 Exercise 1](statistics/7-1-weight_vs_age.md) (correlation of weight vs. age)
In this exercise, you will compute the effect size of correlation.  Correlation measures the relationship of two variables, and data science is about exploring relationships in data.    

```
import matplotlib.pyplot as plt
import nsfg
import numpy as np
from thinkstats2 import Cdf

df = nsfg.ReadFemPreg()
live = df[df.outcome == 1]


# Scatterplot of mother's age vs. birthweight.
plt.scatter(live.totalwgt_lb, live.agepreg, alpha=0.1)
plt.title("Birth Weight vs. Mother's Age at Birth")
plt.xlabel("Birth Weight")
plt.ylabel("Mother's Age at Birth")
plt.show()

# Plot percentiles of birth weight vs mother's age.
bins = np.arange(16, 41, 1)
indices = np.digitize(live.agepreg, bins)
groups = live.groupby(indices)
ages = [group.agepreg.mean() for i, group in groups]
cdfs = [Cdf(group.totalwgt_lb) for i, group in groups]
for percent in [75, 50, 25]:
	birthwgts = [cdf.Percentile(percent) for cdf in cdfs]
	label = '{0:d}th'.format(percent)
	plt.plot(ages, birthwgts, label=label)
plt.xlabel("Mother's age")
plt.ylabel("Birth weight")
plt.title("Percentile ranks for mother's age vs. birth weight")
plt.show()


# Compute both correlation coefficients.
def pearsons_corr(x, y):
	"""This function computes Pearson's correlation coefficient."""
	x_dev = x - x.mean()
	y_dev = y - y.mean()
	x_std = x.std()
	y_std = y.std()
	cov_xy = (x_dev * y_dev).sum() / len(x_dev)
	return cov_xy / (x_std * y_std)

def spearmans_corr(x, y):
	"""This function computes Spearman's correlation coefficient."""
	x_ranks = x.rank()
	y_ranks = y.rank()
	x_dev = x_ranks - x_ranks.mean()
	y_dev = y_ranks - y_ranks.mean()
	x_std = x_ranks.std()
	y_std = y_ranks.std()
	cov_xy = (x_dev * y_dev).sum() / len(x_dev)
	return cov_xy / (x_std * y_std)

pearsons_corr = pearsons_corr(live.totalwgt_lb, live.agepreg)
print("Pearson's Correlation Coefficient: {0:.3f}".format(pearsons_corr))

spearmans_corr = spearmans_corr(live.totalwgt_lb, live.agepreg)
print("Spearman's Correlation Coefficient: {0:.3f}".format(spearmans_corr))
```

*Both correlation coefficients are very low.  This is evidence that the mother's age at birth and birthweight are not meaningfully related.  In other words, knowing information about one variable does not give much, if any, predictive power on the other variable.  Because both correlation coefficients are below 0.1, it is safe to conclude that each variable explains less than a tenth of the variation in the other variable.*

### Q8. [Think Stats Chapter 8 Exercise 2](statistics/8-2-sampling_dist.md) (sampling distribution)
In the theoretical world, all data related to an experiment or a scientific problem would be available.  In the real world, some subset of that data is available.  This exercise asks you to take samples from an exponential distribution and examine how the standard error and confidence intervals vary with the sample size.
```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def simulate_sample(lam=2, n=10, m=1000):
	"""Simulate sampling from an exponential distribution."""
	lam_estimates = []

	# Estimate lambda.
	for i in range(m):
		sample = np.random.exponential(1 / lam, n)
		x_bar = np.mean(sample)
		lam_estimate = 1 / x_bar
		lam_estimates.append(lam_estimate)

	#Compute and print output values.
	std_error = stats.sem(lam_estimates)
	ci_lower = sum(lam_estimates) / len(lam_estimates) - std_error*1.645
	ci_upper = sum(lam_estimates) / len(lam_estimates) + std_error*1.645
	ci = ci_lower.round(3), ci_upper.round(3)

	print("Standard Error: {0:.3f}".format(std_error))
	print("Confidence Interval: {0!s}".format(ci))

	# Plot a histogram of the estimates.
	plt.hist(lam_estimates, bins=50)
	plt.title("Histogram of lambda estimates (n=10)")
	plt.xlabel("Estimate of lambda")
	plt.ylabel("Frequency")
	plt.show()

simulate_sample()
```

### Q9. [Think Stats Chapter 6 Exercise 1](statistics/6-1-household_income.md) (skewness of household income)

```
import hinc
import numpy as np
from scipy.stats import norm

def interpolate_sample(df, log_upper=6.0):
	"""Generate a sample of income dataframe, and take the log of income."""
	
	df['log_upper'] = np.log10(df.income)
	df['log_lower'] = df.log_upper.shift(1)
	df.loc[0, 'log_lower'] = 3.0
	df.loc[41, 'log_upper'] = log_upper
	
	arrays = []
	for item, row in df.iterrows():
		values = np.linspace(row.log_lower, row.log_upper, row.freq)
		arrays.append(values)
		
	log_sample = np.concatenate(arrays)
	return log_sample

income_df = hinc.ReadData()
df = interpolate_sample(income_df)	
sample = interpolate_sample(income_df)

def kth_central_moment(data, k):
	"""Returns the kth central moment of the given data set."""
	x_bar = data.mean()
	n = len(data)
	x_sqd_dev = [(x - x_bar) ** k for x in data]
	return sum(x_sqd_dev) / n

def kth_standardized_moment(data, k):
	"""Returns the kth standardized moment of the given data set."""
	central_moment = kth_central_moment(data, k)
	x_std = data.std()
	return central_moment / x_std ** k
	
def sample_skewness(data):
	"""Return the g1 value of the given data set."""
	return kth_standardized_moment(data, 3)
	
def pearsons_skewness(data):
	"""Return the gp value of the given data set."""
	x_bar = data.mean()
	median = np.median(data)
	x_std = data.std()
	return 3*(x_bar - median)/x_std

def cdf(data, loc):
	"""Return the CDF value for given x."""
	counter = 0
	for x in data:
		if x <= loc:
			counter += 1
	return counter / len(data)

print("Mean log income: {0:.3f}.".format(sample.mean()))
print("Median log income: {0:.3f}.".format(np.median(sample)))
print("Skewness of sample: {0:.3f}.".format(sample_skewness(sample)))
print("Pearson's skewness of sample: {0:.3f}.".format(pearsons_skewness(sample)))	
print("Proportion of sample less than the mean: {0:.3f}.".format(cdf(sample, sample.mean())))
```

*Mean log income: 4.658.
Median log income: 4.709.
Skewness of sample: -0.641.
Pearson's skewness of sample: -0.338.
Proportion of sample less than the mean: 0.451.*

### Q10. [Think Stats Chapter 8 Exercise 3](statistics/8-3-scoring.md) (scoring)

```
import numpy as np
import matplotlib.pyplot as plt

def simulate_game(goals_per_game):
	"""Simulate a game with given parameters.  Return number of goals."""
	lam = goals_per_game
	time = 0
	goals = 0
	while time < 1:
		waiting_time = np.random.exponential(1 / lam, 1)
		time += waiting_time
		if time > 1:
			break
		goals += 1
	return goals

def raw_moment(xs, k):
	return sum([x**k for x in xs]) / len(xs)
	
def central_moment(xs, k):
	mean = raw_moment(xs, 1)
	return sum([(x - mean) ** k for x in xs]) / len(xs)

def simulate_many_games(goals_per_game, n=10000):
	"""
	Simulate many games.
	Record each estimate of lam, and return mean, mean errror, and RMSE.
	"""
	lam_estimates = []
	for i in range(n):
		lam_estimates.append(simulate_game(goals_per_game))
	lam_estimates = np.asarray(lam_estimates)
	L_bar = sum(lam_estimates) / n
	rmse = central_moment(lam_estimates, 2) ** (1/2)
	mean_error = central_moment(lam_estimates, 1)
	return lam_estimates, rmse, mean_error
	
estimates, rmse, mean_error = simulate_many_games(5)
plt.hist(estimates, width=0.9, bins=15)
plt.title("Estimates of gpg")
plt.xlabel("Estimate of lambda")
plt.ylabel("Frequency")
plt.show()
```	

*This seems to be a biased way to estimate lambda, beause the Root Mean Squared Error does not seem to approach zero as the  number of estimates increases.  The standard error remains at about 2.24, even when the number of estimates is increased by a power of ten.*

### Q11. [Think Stats Chapter 9 Exercise 2](statistics/9-2-resampling.md) (resampling)

```
import nsfg
import thinkstats2
import numpy as np

class HypothesisTest(object):
	def __init__(self, data):
		self.data = data
		self.make_model()
		self.actual = self.test_statistic(data)
		
	def p_value(self, iters=1000):
		self.test_stats = [self.test_statistic(self.run_model())
							for _ in range(iters)]
		count = sum([1 for x in self.test_stats if x >= self.actual])
		return count / iters
		
	def test_statistic(self, data):
		raise UnimplementedMethodException()
	
	def make_model(self):
		pass
		
	def run_model(self):
		raise UnimplementedMethodExcpetion()
		
		
class DiffMeansPermute(HypothesisTest):
	def test_statistic(self, data):
		group1, group2 = data
		test_stat = abs(group1.mean() - group2.mean())
		return test_stat
		
	def make_model(self):
		group1, group2 = self.data
		self.n, self.m = len(group1), len(group2)
		self.pool = np.hstack((group1, group2))
		
	def run_model(self):
		np.random.shuffle(self.pool)
		data = self.pool[:self.n], self.pool[self.n:]
		return data
		
		
class DiffMeansResample(DiffMeansPermute):
	def run_model(self):
		group1 = np.random.choice(self.pool, self.n, replace=True)
		group2 = np.random.choice(self.pool, self.m, replace=True)
		return group1, group2


df = nsfg.ReadFemPreg()
live = df[df.outcome == 1]
firsts = live[live.birthord == 1]
others = live[live.birthord != 1]

# Perform hypothesis test for birth weight.
data = firsts.totalwgt_lb, others.totalwgt_lb
ht = DiffMeansResample(data)
print("p-Value for birth weight hyopthesis test: {0:.3f}".format(ht.p_value()))

# Perform hypothesis test for pregnancy length.
data = firsts.prglngth, others.prglngth
ht = DiffMeansResample(data)
print("p-Value for preg length hyopthesis test: {0:.3f}".format(ht.p_value()))
```

*These results do not differ much from the tests performed previously in Chapter 9.  This is probably because the test of statistical significant depends on the underlying asssumptions of the null hypothesis, and what test statistic is used.  Compared with the previous model, this model simply has a different but almost equivalent implementation of computing the test statistic of the total population.  Therefore it is not surprising that this model yields similar results to the previous model; there have been no material changes to the assumptions made in the null hypothesis.*



---

## <a name="section-f"></a>6.  Recommended Reading

Read Allen Downey's [Think Bayes](http://greenteapress.com/thinkbayes/) book.  It is available online for free, or you can buy a paper copy if you would like.

[<img src="img/think_bayes.png" title="Think Bayes"/>](http://greenteapress.com/thinkbayes/) 

---

## <a name="section-g"></a>7.  More Resources

Some people enjoy video content such as Khan Academy's [Probability and Statistics](https://www.khanacademy.org/math/probability) or the much longer and more in-depth Harvard [Statistics 110](https://www.youtube.com/playlist?list=PL2SOU6wwxB0uwwH80KTQ6ht66KWxbzTIo). You might also be interested in the book [Statistics Done Wrong](http://www.statisticsdonewrong.com/) or a very short [overview](http://schoolofdata.org/handbook/courses/the-math-you-need-to-start/) from School of Data.
