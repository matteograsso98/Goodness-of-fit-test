## Goodness-of-fit test 

The chi-squared statistics relies on some assumptions that are often not valid in most cosmological scenarios. 
Therefore, a "goodness-of-fit" test is often needed. The code takes the best fit theory vector and create mocks sampling from a multivariate Gaussian distribution. 
Then, the chi-squared and the p-value of the resulting distribution are calculated.

The code GOF.py takes the observed data, its covariance matrix, and the best fit theory vector as an input. 
The best-fit theory (i.e., the mean of the Multivariate Gaussian from which we will sample) is created/formatted in the script mock_data.py. <pre> ``` def XXX(): print("Hello, World!") ``` </pre>
