
Wasserstein Variational Autoencoder
===============
Wasserstein Variational Autoencoder (W-VAE) is a generative model based on optimal transport (OT) and variational inference. The OT cost is a key to measure the distance between two probability distributions.

**Refrences**
1. Tolstikhin, Ilya, et al. "Wasserstein auto-encoders." arXiv preprint arXiv:1711.01558 (2017).
2. Wasserman, L. "Optimal transport and wasserstein distance." (2017).


Wasserstein distance
---------------
Suppose we have two distributions $p(x)$ and $q(x)$ there are many ways to compute the distance between these two distributions such as KL divergence but there is some drawback with these kinds of distances such as These distances ignore the underlying geometry of the space. Wasserstein distance is useful in practice mainly when the data is supported on low dimensional manifolds in the input space.  Usually, in machine learning, we assume that one's observed data lie on a low-dimensional manifold embedded in a higher-dimensional space. Therefore a distance such as f-divergence which measures the density ratio between distributions often max out and provides no useful gradients for training. 
The $W_2$ distance between two probability measures $p(x)$ and $q(y)$ is defined as follows:
$W_2(p(x),q(x)) = inf E(||x-y||_{2}^{2})^{\frac{1}{2}}$

