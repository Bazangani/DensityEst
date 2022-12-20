Gaussian Random field Density parameters Estimation 
=============================



A (Real) random field $F(x_{t})$ is a random function over an arbitrary domain (usually a multi-dimensional space such as ${\displaystyle \mathbb {R} ^{n}})$ consists of a set of correlated random variables $\{x_t,t \in T\}$ where the index set $T$ is some subset of the d-dimensional Euclidean space $R^d$. The set of possible values of $X_t$ is called the state space of the random field. Therefore the random field can classify into four classes based on the state space and index set to be continuous or discrete. 


We can also classify the random fields based on other properties like the distributional properties  for instant, a Gaussian random field $F(x)$ is a random field with 
Gaussian probability distribution function on the random variables $\{x_{t_{1}}, x_{t_{2}}, ..., x_{t_{N}}\}$.


$X =^{def} (x_{t_1}, x_{t_2}, ..., x_{t_N})^T  \sim N ( \mu , \sigma)$


For the covariance matrix $\sigma$ and mean vector $\mu$.

As any linear combination of the random variable in the case of Gaussian field is Gaussian therefore a Gaussian field $F(x_t)$ is entirely determined by its expectation function and its auto-covariance function or equivalently by its mean and its covariance matrix. We can define the mean and the covariance of the random field as follows:


$ m(x) : = E[F(x)] $




$Cov[F(x),F(x_{0})]∶= E[(F(x)-m(x))(F(x_{0})-m(x_0))]$



In this study we focus on a Gaussian random field with discrete index set and continuous state space. For instant by defining the arbitrary domain $R^2$ a 2D image with $N$ pixels with a discrete number of pixels is assigned a continuous gray scale could be a Gaussian random field with discrete index set and continuous state space with $N$ random variables.


\subsection{Stationary Gaussian random field}

A Gaussian process $F(x)$ is said to be first order stationary if the expectation function, $E[F(x)]$ , is constant
and the covariance function, is invariant under translations so for the first order stationary Gaussian random field:


	$\exists \mu \in	R,  m(x) = \mu $


For a second order stationary Gaussian random field the first order condition for stationary field is valid as well as $ Cov[F (x); F (x_0)]$ is a function of only the difference $h = x - x_0$

$Cov[F (x); F (x_0)] = C(h)$



\subsection{Covariance matrix of the stationary Gaussian field}

For a stationary Gaussian process (N random variables) with zero mean and $(N^2 \times N^2)$ covariance matrix $\Sigma$, the covariance matrix is a Toeplitz matrix if the sample points $x_i$ are uniformly spaced \cite{Vinogradova_2015}



\textbf{Toeplitz and Circulant Covariance matrix}:

A Toeplitz matrix is a  diagonal-constant matrix. For an $N^2 \times N^2$ Toeplitz matrix $T_N = [t_{k,j}; k,j =  0,1,... ,N^2-1]$ where $t_{k,j} = t_{(k \text{-} j)}$ , i.e., a matrix of the form \cite{Gray_2006}:


$T_N =
  
      t_0       & ... &  ...   & t_{-(N^2-1)}\\
      t_1       & t_0 & ...     & ... \\
      ...       & ... & ...     & ...\\
      t_{N^2-2}   & ... & ...     & ... \\
      t_{N^2-1}   & ... & t_1     & t_0 \\$
Whereas an $(N^2\times N^2)$ matrix could contain $N^4$ different elements, the Toeplitz matrix contains only $N^2$ elements that are different from each other.


A common special case of Toeplitz matrices results when every row of the matrix is a right cyclic shift of the row above it so that $t_{k} = t_{−(M\text{-} k)} = t_{k\text{-} M} for k = 0,1, 2,... , M-1 $. In this case the picture becomes:

$C_M =
  
      t_0            & t_{-1}    & ...      & t_{-(M-1)}\\
      t_ {-(M-1)}  & t_1       & t_{0}    & ... \\
      ...            & ...       & ...      & ...\\
      ...            & ...       & ...      & ... \\
      t_{-1}         & t_{-2}    & ...      & t_0 \\
$

A matrix of this form is called a Circulant matrix. Symmetric Toeplitz matrices of size $(N^2 \times N^2)$ can always be extended to give symmetric circulant matrices of size $(M \times M)$ by padding them with extra rows and columns if we choose the $(M-N^2)$ big enough. Practically the size of the circulate matrix could be $M = 2N^2$ to keep the matrix symmetric.

%Considering the domain of the field $R^2$ we have block Toeplitz matrix instead of Toeplitz matrix therefore we can extend it to a block circulant matrix.

The Covariance matrix has to be symmetric positive-definite means for a real matrix $\Sigma$ For all non-zero $w \in R^n$:

  $ w^T \Sigma w \geq  0$



 $  \Sigma^T = \Sigma$


For instant for matrix $C_M$ as it's a circulate matrix therefore it's symmetric if $c_{n-i} = c_i$ and it can be defined by $[\frac{M}{2}] + 1$ entries. Any real matrix $\sigma$ is positive-definite if all the eigenvalues of $\Sigma$ are all non-negative.

\section{Estimation of the Covariance matrix}
In this section we propose a method in order to estimate the covariance matrix of a stationary Gaussian random field with variational autoencoder. In order to do so the training dataset is samples from  a stationary Gaussian random filed. As we have in VAE we go through the details in 3 sections, Encoder, sampling and Decoder. But before going through the details we need to know what is the variational Auto-encoder.




\section{Variational Auto-Encoder}

\subsection{Variational inference}
Usually we are interested in computing the distribution over a variable $z$ conditioned on the observed variable $x$ like $p(z|x)$. For estimation of this distribution called $\hat{p_{\theta}}(z|x)$  We define a model with parameter $\theta$ and optimize the parameters to make the estimated distribution as close as possible to the underlying distribution. 


$p(z|x) = \hat{p_{\theta}}(z|x)$
Estimation of a conditional distribution is a hard task when the data distribution is high dimension like image or video. Variational inference is a method to approximate this probability \cite{Blei_2018}.
They are many ways to find the parameters such as maximize the likelihood or  maximize the log posterior w.r.t $\theta$ with Bayesian leaning. The main problem with likelihood estimation is the marginal probability $p_{\theta}(z|x)$ of the data is not tractable and it cause not having an analytical solution. An intractable marginal likelihood means the log of the posterior is intractable as well.
Variational inference is a solution for this estimation so we can define the model $q_{\phi}(z|x)$ with parameters $\phi$ which are variational parameters such that:

 $p(z|x) = \hat{q_{\phi}} (z|x)$

If set of $ X_{i}$,  $i \in \{0,N\}$ the variational inference is to compute the density of the latent variable $p(z|x)$ given the observation. In variational inference we can approximate this density with a defined distribution like $Q(z)$ for the latent space and the main aim is to find the density of $Q$ as close as possible to the posterior density through an optimization process. The complexity of the family of the distribution we choose for the latent space  define the complexity of the optimization \cite{Blei_2018}. This optimization lead us to minimization the distance between two distributions with KL divergence as follows:

$Q(z) = \argmin_Q(z) KL (Q(z)||P(z|x))$


However still we have to compute the evidence for P(z|x) in the closed form of $KL$ we can re-write the $KL$ as follows:



$KL (Q(z)||P(z|x)) = \mathbb{E}[log(Q(z))] - \mathbb{E}[log(P(z|x))]$


$KL (Q(z)||P(z|x)) = \int_{}^{z} Q(z) \frac{Q(z)}{P(z|x)} = - \int_{}^{z} Q(z) \frac{P(z|x)}{Q(z)}$

$KL (Q(z)||P(z|x)) = - \int_{}^{z} Q(z) \frac{P(z,x)}{Q(z)P(x)} = - \int_{}^{z} Q(z) \frac{P(z,x)}{Q(z)} + log P(x) \int_{}^{z} Q(z)$

In (2.7) the $\int_{}^{z} Q(z) = 1$ and the $log P(x)$ is constant:


$log p(x) = KL ( Q(z) || P(z|x) ) +  \int_{}^{z} Q(z) \frac{P(z,x)}{Q(z)}$

So $L =\int_{}^{z} Q(z) \frac{P(z,x)}{Q(z)}$ is the variational lower bound and by minimizing this quantity we can reach the main aim which is minimizing the $KL (Q(z) || P(z|x))$ as the $KL\geq 0$. Minimization of the variational lower band lead us to the $Q(z)$ perfectly as closed as possible to the posterior density \cite{Yang_2017}

\subsection{Variational autoencoder (VAE)}

One of the unsupervised models in generative model is variational auto encoder. Variational autoencoder is a family of probabilistic graphical models which improves the learning process with approximate variational inference. The graphical model is represented in Fig.\ref{fig:VAE_graphical}.

 
 The aim of VAE is to minimize the variational lower band with respect to variational parameters $\phi$ and generative parameters $\theta$.\cite{Kingma_2014}. Calculating the gradient of the lower band with respect to the $\phi$ is tricky so to overcome this issue Kingma et all. proposed a practical estimation of the lower band. The authors re-parameterized the the random variable $\widehat{z} = Q_{\phi}(z|x)$ under certain conditions. To do so a differentiable transformation $\widehat{z} =g_{\phi}(\epsilon,x)$ of a noise variable has been used. 
 

  $\widehat{z} =g_{\phi}(\epsilon,x),                     \epsilon \sim p(\epsilon)$


Based on (2.8) we can have the same formula for VAE as follows:

$L(\theta,\phi;x^{(i)}) = - KL(q_{\phi}(z|x_{i})||p_{\theta}(z)) + \mathbb{E}_{q\theta(z|x_{i})}[log p_{\theta}(x_{i}|z)]$

The first term is the $KL$ divergence (between approximate posterior and the prior) and the second term could be reconstruction error. The process of sampling from the latent space must be differentiable in order to  back-propagate the gradient through the network. In \cite{Kingma_2014} they proposed the re-parameterization trick to solve this issue. It is possible to express the random variable $z = g_{\phi}(\epsilon,x)$.
  

Variational autoencoder in terms of the architecture is composed of 2 parts: an encoder and a decoder to simulate the approximation of the posterior of the generative model represent by $p_{\theta}(x,z)$ and $q_{\phi}(x|z)$ respectively.



The encoder encode the high dimension information into low dimensional representation $z$ and the decoder reconstruct $z$ into the input space. The model learns to the distribution of the latent space with two parameters mean and standard deviation. Optimization of the network has to be done with the minimization of the lower band. 
The prior distribution in VAE is a standard Gaussian distribution.

In the next section, we explain the idea of estimating the covariance matrix of a second order stationary Gaussian field with VAE. To do so we need to adapt the architecture and the sampling method in VAE. We have 3 sections to explain the adaptions in details: Encoder, Sampling and Decoder.




\subsection{Encoder}
Considering an stationary Gaussian random field we have all the specification of the field by knowing the mean and covariance matrix of the field. So the main aim of the encoder is estimating the covariance matrix by having the assumption of the mean is zero.


Estimating the covariance matrix with a neural network is quite tricky because the estimated matrix has to be symmetric and semi-positive definite. To simplify this challenge we use one of the main characteristics of the circulant matrix. Basically, symmetric circulant matrices are factorized by their discrete Fourier transform. So for a circulant matrix $C$ we have :

$C = P^*diag(\Lambda)P$


$P$ is the n-dimensional Fourier transform, $P^*$ is the complex conjugate transpose of the Fourier transform and $diag(\Lambda)$ is the diagnosed matrix of size $(M \times M)$ contains $M$ eigenvalues  $\{s_i, i=1,2, ..., M\}$ of matrix $C$. 

The Fourier transform of matrix $C$ of size $(M \times M)$ in 1-dimensional domain is $P$ and defined as:


$P = (\beta_{kj}) 0 \leq j;k \leq (M); \beta = e^{\frac{-2\pi i}{2M}}= e^{\frac{-\pi i}{M}}$



$P = 
   
      1   & 1             & 1          &...                & 1\\
      1   & \beta         & \beta^2    & ...               & \beta^{(M-1)} \\
      ... &               & ...        & ...               & ... \\
      1   & \beta^{(M-1)}&            & ...      & \beta^{(M-1)^{2}} \\
$

For two-dimensional domain the Fourier transform $P$ is the Kronecker product of two discrete Fourier transform matrices; that is, $P = f \otimes f$ where $f_{jk} = {\frac{1}{\sqrt M}}e^{\frac{-2\pi ijk}{M}}, k=0,1, ..., M-1$ is the 1-dimension Fourier of matrix $C$. 

With using this property of the circulant matrix, the decoder estimate the vector $\{s_i, i=1,2, ..., M\}$ of eigenvalues and then compute the circulant matrix $C$ which is the extended of matrix $T$ or covariance matrix of the stationary Gaussian random field.


We need to check if the computed covariance matrix has the properties of a valid covariance matrix. The first property as mentioned is being a positive-semi definite matrix. The eigenvalues of a positive-definite matrix are non-negative real numbers. 


$s_i \geq 0,  i =\{ 1, 2, ..., M\}$


This aim can be achieved by using ReLu activation function as the last layer of the encoder.
The second properly is being symmetric. The factorization of the circulant matrix is based on having a symmetric matrix therefore using  the equation \ref{Equ:Circulant_factor} gives us a symmetric matrix.


So we can compute matrix $C$ and subsequently $T$ by dropping half of the entries of the $C$ we can have the estimation of the covariance matrix of field $T$. In the next section we explain the process of sampling from the random field by using the estimated of the covariance matrix.



\subsection{Sampling from the field}
we follow the main structure of the VAE so we need to sample from the field and the sampling process needs to be differentiable as it is for VAE.
In order to sample from an stationary Gaussian random field and the special property of the circulant matrix we use circulant embedding method.




\textbf{Circulant embedding method}:
Circulant embedding first proposed by \cite{Dietrich_1997} for covariance matrix decomposition.  Let's $\Sigma_{(M \times M)}$ be covariance matrix the key to efficient simulation of a sample from the field $F(x)$ is that $\Sigma$ is a symmetric block-circulant matrix with circulant blocks. 


The matrix $\Sigma$ is thus completely specified by its first row, which we gather in an $M \times M$ matrix $G$. The vector of eigenvalues $S_i = (s_1, . . . , S_{M} )$ ordered as an $M\times M$ matrix $Γ$ satisfies $\Gamma = N F^* G F$. Since $\Sigma$ is a covariance matrix, the component-wise square root $\sqrt{\Gamma}$ is well-defined and real-valued.

The matrix $B = P^*diag(\sqrt{\Gamma})$ is a complex square root of $\Sigma$, so that sample $Y$ can be generated by drawing $\epsilon = \epsilon +i\epsilon_2$, where $\epsilon_1$ and $\epsilon_2$ are independent standard normal random vectors, and returning the real part of $B\epsilon$. It will be convenient to gather $Y$ into an $M\times M$ matrix $Y$.


In circulant embedding by having the covariance matrix $C$ we can sample from the original field by drooping some entries of sample $Y$. So the original sample from the stationary Gaussian random field $Y_{T}$ is the $N^2 \times N^2$ with $N^2 =  \frac{M}{2}$ firsts entries of the sample $Y$

With this method as we have in VAE we have an stochastic node $\epsilon$ in order to sample out of the gradient flow in the network so the gradient can flow in the network. 


\subsection{Decoder}
After estimating the covariance matrix of the field we can reconstruct the original realization with the covariance matrix of the field. By having the covariance matrix we can define the field and sample from the field as it mentioned in the sampling section. each sample has been used as the input of the decoder in order to reconstruct the realization of the field.



