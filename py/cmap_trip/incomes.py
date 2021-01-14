import numpy as np
import pandas as pd
from scipy.stats import lognorm

from .random_states import check_random_state

income_levels_1 = {
	1	:  12_000, # Less than $15,000
	2	:  20_000, # $15,000 to $24,999
	3	:  27_500, # $25,000 to $29,999
	4	:  32_500, # $30,000 to $34,999
	5	:  42_500, # $35,000 to $49,999
	6	:  55_000, # $50,000 to $59,999
	7	:  65_000, # $60,000 to $74,999
	8	:  85_000, # $75,000 to $99,999
	9	: 120_000, # $100,000 to $149,999
	10	: 170_000, # $150,000 or more
}

income_levels_2 = {
	1	:  25_001, # Less than $30,000
	2	:  40_001, # $30,000 to $59,999
	3	:  70_001, # $60,000 to $99,999
	4	: 125_001, # $100,000 to $149,999
	5	: 175_001, # $150,000 or more
}



def random_incomes(median_income, replications, random_state=None, sigma=0.42):
	"""
	Draw random income levels from a log-normal distribution

	Parameters
	----------
	median_income : numeric
	replications : int
		Size of resulting array
	random_state : RandomState
	sigma : float, default 0.42
		The standard deviation of the underlying normal distribution.
		The default of 0.42 is retained from prior versions of the CMAP
		trip-based model, although a bug in that model resulted in a
		downward bias in the random draws.

	Returns
	-------
	ndarray
	"""
	random_state = check_random_state(random_state)
	return lognorm(sigma, scale=median_income).rvs(
		replications,
		random_state=random_state,
	)