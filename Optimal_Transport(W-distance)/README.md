
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

By replacing the $p(z) \sim N(0,1)$ and $Q(z|x) \sim N((μ_1, μ_2, ...,μ_m),{𝜎_12,𝜎_22,…,𝜎_𝑚2})$ with  the unknown mean vector ${𝜇_1,𝜇_2,…,𝜇)𝑚}$ and diagonal variance vector ${𝜎_12,𝜎_22,…,𝜎_𝑚2}$. Since $p(z)$ is multivariate normal we can compute the KL divergence as follows:

$KL(Q(z|x)||p(z)) = \frac {1}{2} [Π_{i=1} ^{m} − log𝜎_𝑖 ^2 + Σ_{i=1} ^{m} (𝜇_𝑖 ^2 + 𝜎_𝑖 ^2)−𝑚]$

So to compare the Kl diveregnce with W-distance we have:

$W_2(𝑄(𝑧|𝑥) || 𝑃(𝑧))=||𝝁−𝟎||_2 ^2+ 𝑇𝑟(Σ+ 𝐈 −2(𝐈^{\frac{1}{2}}Σ𝐈^{\frac{1}{2}})^{\frac{1}{2}})$

Therefore we can define the difrence disstance between KL divergence and W-distance with $T$ as follows:

$T=W_2(𝑄(𝑧|𝑥) ||𝑃(𝑧))− KL(𝑄(𝑧|𝑥) ||𝑃(𝑧))= 𝑙𝑜𝑔 Π_{i=0} ^m 𝜎_𝑖 ^2 +Σ_{i=1} ^m (𝜎_𝑚 − 2)^2 + Σ_{i=1} ^m 𝜇_𝑖^2−𝑚$

The value of $T$ depends on the variances $𝜎_𝑖$

1. When $𝜎_1,𝜎_2,…,𝜎_𝑚=1$, $T=0$, suggesting that ELBOW and ELBOKL are identical at this point.
2. When $𝜎_1,𝜎_2,…,𝜎_𝑚≤1$, it’s easy to show $\frac{𝜕T}{𝜎_1},\frac{𝜕T}{𝜎_2},…,\frac{𝜕T}{𝜎_m}≥0$, suggesting that $T$ increases monotonically across ${𝜎_1,𝜎_2,…,𝜎_𝑚}$, jointly resulting in $T≤0$.

Therefore, when $𝜎_1,𝜎_2,…,𝜎_𝑚≤1$, ELBOW is closer to $log(𝑃(𝑥))$ than ELBOKL in its approximation.
ELBOW may not be a consistent estimator when the model overfits, and the training process may conclude with a higher ELBOW than $log(𝑃(𝑥))$. To enforce the consistency property of ELBOW, it is necessary to place an inductive bias such that $T=0$ upon model convergence. This is achieved by introducing an additional hyperparameter $𝜆$ that controls the weight of the Wasserstein distance term, forming a new objective function $ELBOW_λ$ as follows:

$ELBOW_λ= ∫𝑄(𝑧|𝑥)log(𝑃(𝑥|𝑧))𝑑𝑧 − λ W_2(𝑄(𝑧|𝑥) || 𝑃(𝑧))$









