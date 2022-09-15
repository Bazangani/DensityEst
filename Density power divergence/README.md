Density Power Divergence (β-divergence)
===============
One of the primary methods for the estimation of an unknown density is a parametric estimation. In parametric analysis basically, we compare the unknown distribution with a parametric family of distribution.
In order to make this comparison, we need a metric to measure the distance between two distributions. In this study, we focus on Density power divergence as a distance measurement between the unknown distribution and the parametric family of distributions. 


**Reference** : 
1. Basak, Sancharee, Ayanendranath Basu, and M. C. Jones. "On the ‘optimal’density power divergence tuning parameter." Journal of Applied Statistics 48.3 (2021): 536-556.

2. Akrami, Haleh, et al. "A robust variational autoencoder using beta divergence." Knowledge-Based Systems 238 (2022): 107886.


Introduction 
===============
In order to measure the distance between two densities, we need to take into account and balance two factors, efficiency and robustness, in a good manner. Finding an optimal tuning parameter is an important element to keep this trade-off.
Finding an optimal tuning parameter is an important element to keep this trade-off. The Density Power Divegence (DPD) gives us this posiblity to find the optimal place between these two factors in a density estimation challenge.
This family of divergences provide a tuning parameter to tune the importance of efficiency and robustness. 

Density power divegence 
===============
Density power divergence or β-divergence is a family of divergence that measures the distance between two distributions $f_{θ}(x)$, the parametric familly of distributions and $g(x)$ the unknwon distribution of the data by:

$d_{α}(g,f) = \int [{f^{1+β}(x) - (1+ \frac{1}{β})f^β(x) g(x) + \frac{1}{β} g^{1+β}(x)}] dx$

Therefore we find the parameters of the model $θ$ by minimizing the PDE and the estimator calls minimum power divergence estimator (MDPDE)
for the sake of simplicity we represent $f_{θ}(x)$ by $f(x)$ 

$β$ is a non-negative tuning parameter in practice the value of $β$ restricted to [0,1]. Although A large value for $β$ leads to low efficiency, it provides high robustness and stability agaist outliers. Therfore we wish to choose a data-driven value for $β$ to balance the robustness and efficency. 

If parameter $β->0$ the divergence convege to :


$d_{0}(g,f) = \int [g(x) + log(\frac{g(x)}{f(x)})] dx$


Which is a version of Kullback-Leibler divergence or equivalent to maximization of the log-likelihood function.

**Review** :
$D_{KL}[f(x|0)||g(x)] = Ε_{x\equivf(x|0)} [log \frac{f(x|0)}{g(x)}] = Ε_{x\equivf(x|0)} [log{f(x|0)}] - Ε_{x\equivf(x|0)} [log{g(x)}]$

The left term is the entropy of $f(x|0)$ and does not depend on parameter $0$. Suppose we drawn $N$ samples from $f(x|0)$ $N->\inf$ the right term  the negative log-likelihood :

$\frac{-1}{N}\sum_{i=1}^{N} logf(x|0) =  -Ε_{x\equivf(x|0)}[log g(x)]$

Therfore KL-divergence and eventually liklihood estimation (MLE) can be considered as a specialcase of the MDPDE when $β = 0$



