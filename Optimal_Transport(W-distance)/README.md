
Wasserstein Variational Autoencoder
===============
Wasserstein Variational Autoencoder (W-VAE) is a generative model based on optimal transport (OT) and variational inference. The OT cost is a key to measure the distance between two probability distributions.

**Refrences**
1. Tolstikhin, Ilya, et al. "Wasserstein auto-encoders." arXiv preprint arXiv:1711.01558 (2017).
2. Wasserman, L. "Optimal transport and wasserstein distance." (2017).
3. Chen, Zichuan, and Peng Liu. "Towards Better Data Augmentation using Wasserstein Distance in Variational Auto-encoder." arXiv preprint arXiv:2109.14795 (2021).


Wasserstein distance
---------------
Suppose we have two distributions there are many ways to compute the distance between these two distributions such as KL divergence but there is some drawback with these kinds of distances such as These distances ignore the underlying geometry of the space. Wasserstein distance is useful in practice mainly when the data is supported on low dimensional manifolds in the input space.  Usually, in machine learning, we assume that one's observed data lie on a low-dimensional manifold embedded in a higher-dimensional space. Therefore a distance such as f-divergence which measures the density ratio between distributions often max out and provides no useful gradients for training. 
The $W_2$ distance between two probabilities $p(x)$ and $q(x)$ is defined as follows:
$p(x)\sim N(m_{1},Σ_{1})$
$q(x)\sim N(m_{2},Σ_{2})$

$d = W_{2}(N(m_{1},Σ_{1}),N(m_{2},Σ_{2}))$


$d_{2}=||m_{1}-m{2}||_{2}^{2}+ +Tr(Σ_1+Σ_2-2(Σ_1^{\frac {1}{2}} Σ_2 Σ_1^{\frac{1}{2}})^{\frac{1}{2}})$


Wasserstein Variational Autoencoder
------------------------------------------
In W-VAE we compress the observed data $x$ into low dimentional latent variable $z$ by an aproximate posterior distribution $Q(z|x)$ by the encoder and the decoder reconstruct $p(x)$. The ELBOW lost fucntion in VAE coonsists of mariginal liklihood of the recounstructed data and the KL divergence of the aproximate and true posterior distribution on $z$:
$ELBO_{kl} = E_{z \sim Q(z|x)} [log(p(x|z))] = \int Q(z|x)log(p(x|z))dz -KL(Q(z|x)||p(z))$

For W-VAE we replaced the KL divergence with wasserstein distance :
$ELBO_{W} = E_{z \sim Q(z|x)} [log(p(x|z))] = \int Q(z|x)log(p(x|z))dz -W_p(Q(z|x)||p(z))$

by replacing the $p(z) \sim 






