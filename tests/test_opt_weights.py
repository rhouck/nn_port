"""
tests:
weights sum to 1, are all positive and <= 1
equal weight == highly regularized
optimal weight == barely regularized
test results are same for both norm types when alpha is zero
raises exception if df contains nan
"""

"""
tests:
for each date, never looks back in time
first date in first iteration is same as frist date in input data frame
look exits when fewer additional rows than look ahead input 
(i.e. final date in return df is look_ahead_per less than len of input df)
applied weights with minimum look ahead outperforms equal weights
"""