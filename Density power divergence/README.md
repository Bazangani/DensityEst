Density Power Divergence
===============
One of the primary methods for the estimation of an unknown density is a parametric estimation. In parametric analysis basically, we compare the unknown distribution with a parametric family of distribution.
In order to make this comparison, we need a metric to measure the distance between two distributions. In this study, we focus on Density power divergence as a distance measurement between the unknown distribution and the parametric family of distributions. 


**Reference** : 
1. Basak, Sancharee, Ayanendranath Basu, and M. C. Jones. "On the ‘optimal’density power divergence tuning parameter." Journal of Applied Statistics 48.3 (2021): 536-556.

2.Akrami, Haleh, et al. "A robust variational autoencoder using beta divergence." Knowledge-Based Systems 238 (2022): 107886.


Introduction 
===============
In order to measure the distance between two densities, we need to take into account and balance two factors, efficiency and robustness, in a good manner. Finding an optimal tuning parameter is an important element to keep this trade-off.
Finding an optimal tuning parameter is an important element to keep this trade-off. The Density Power Divegence (DPD) gives us this posiblity to find the optimal place between these two factors in a density estimation challenge.
This family of divergences provide a tuning parameter to tune the importance of efficiency and robustness. 

Density power divegence 
===============
Density power divergence is a family of divergence that masure the distance between two distributions by:

$d_{α}(g,f) = \int [{f^{1+α}(x) - (1+ \frac{1}{α})f^α(x) g(x) + \frac{1}{α} g^{1+α}(x)}] dx$
