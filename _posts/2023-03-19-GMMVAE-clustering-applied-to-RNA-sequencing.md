---
layout: post
title:  "GMMVAE clustering applied to RNA sequencing"
date:   2023-03-19 15:00:00 +0100
categories: bayes
header-includes:
   - \usepackage{amsmath,amssymb}
---
## Introduction
One of many classical tasks of machine learining is clustering, based on data one would like to distinguish few clusters of data based on it's internal properties.
There are many classical aproaches to clustering most notable ones including K-means, GMM (Gaussian Mixture Model) trained via EM algorithm and DBSCAN.
While each of those algorithms is widly used in the research and industry all of them try to cluster data using it's orginal representation and mostly fail in the case of hidden similarity between observations.\\
In this post I present GMM+VAE deep learning architecture that can be used to merge learned latent representation which will be then used to cluster the data using GMM model.
Clustering model is based on [paper](https://arxiv.org/abs/1611.02648) while application to RNA sequencing data is based on the work that I performed during my studies at Warsaw University. Orginal project with solution can be found in related [github repository](https://github.com/Wesenheit/SAD2-22W/tree/main/Project1).

# About VAE
Normal VAE architecture can be understood as quiet simple Bayesian Graphical model that tries to describe data $$x\in \mathbb{R}^N$$ as function of some hidden latent varible $$z\in \mathbb{R} ^n $$. $$N$$ represents dimensionality of original data while $$n$$ stands for dimensionality of latent space.
Most basic VAE model can be written as 
\begin{align}
    z\sim \mathcal{N}(0,\mathbf{I})
\end{align}
\begin{equation}
    x\sim P(\theta(z))
\end{equation}
where $$P(\theta(z))$$ where $$P(\theta(z))$$ stands arbitrary probability distribution used to model data while $$\theta(z)$$ denotes parameters of afromentioned distribution which are functions of latent varible $$z$$ modeled by nerual network.
There are few usefull probability distributions that can be used to model data but 
ultimatly distribution should be choosen with true nature of observation in mind. For example most popular distribution that can be used to model images is 
independent Bernouli distributions 
\begin{equation}
    P(x_i)\sim Bernouli(\theta_i(z))
\end{equation} where each pixel is modeled independly from others. This particular choice leads to binary cross entropy loss as one can verify.
In order to perform variational inference one usually take multivariate gaussian posterior with diagonal covariance matrix modeled by neural network 
which we will denote as $$\phi_z$$. We will assume it outputs set of means $$\mu_i$$ and variances $$\sigma_i^2$$
Strictly speaking our varational family is
\begin{equation}
    q(z_i|x)=\mathcal{N}(\mu_i(x),\sigma_i^2(x))
\end{equation}.
After calculations one can derive lower bound of our data (ELBO) which can be used as loss function
\begin{equation}
    \mathcal{L}=-ELBO=-E_q\log{P(x|z)}+\mathcal{D_{KL}}(q(z|x)||p(z))
\end{equation}
where $$\mathcal{D_{KL}}$$ is Kullback-Leibner divergence between posterior $$q(z|x)$$ and prior $$p(z)$$. Due to the fact that both prior and posterior 
are assumed to be gaussian it's very easy to compute both quantities analytically.
Exact method used to infer parameters of distributions is very similar to process of training normal autoencoder with small difference in loss 
and with addition of reparametrization tric.

Why one should use VAE to describe data? Similarly to normal autoencoders one can expect that if two observations are closly related they will be also
closly related in latent space allowing to perform some sort of dimensionality reduction. This observation leads to assumption, that latent space should be good
place to look for similarity in our data but the problem is it's not always true. In many cases latent space is entangled and it's hard to perform any type of clustering.

## GMM VAE 

# How do we model?
In order to overcome afromentioned difficulties with disentanglment of latent space we can write down slightly modified model. Previously we had prior that 
was inherently compact as the gaussian with zero mean tend to cluster everything together. Let's use instead GMM as prior distribution. How we can write it down?
Let's change a little notation as we will need many latent varibles to write down model. First of all we want to model our observations which we will denote as $$y$$.
We want to model data using latent varibles $$w\in \mathbb{R}^{n_w}$$, $$z\in [0,1]^K$$ and $$x\in \mathbb{R}^{n_z}$$ as
\begin{equation}
    w \sim \mathcal{N}(0,\mathbf{I})
\end{equation}
\begin{equation}
    z\sim Mult \bigg(\frac{1}{K},\ldots,\frac{1}{K}\bigg)
\end{equation}
\begin{equation}
    x|w,z\sim \Pi_{k=1}^{K} z_k \mathcal{N}(\mu_{z_k}(w,\beta),diag(\sigma_{z_k}(w,\beta)))
\end{equation}
\begin{equation}
    y\sim P(\theta(x))
\end{equation}
How to understand this model? First of all parameters of our gaussian distributions are generated using latent varible $w$ which is passed through 
neural network with parameters $\beta$. There are is also latent varible $z$ which chooses which cluster is selected like in traditional GMM model. 
Those two parameters are used to construct prior distribution for our final latent varible $x$ which is used together by nerual network $\theta$ to 
parametrize final propability distribution. There are few pros of this approach were most important one can be seen when we write down our variational family for inference.
Following orginal approach we use
\begin{equation}
q(x,w,z|y)=q_{\phi_w}(w|y)q_{\phi_x}(x|y)p_\beta(z|x,w)
\end{equation}
where $$\phi_x$$ and  $$\phi_x$$ denote neural networks used in inference process. In order to obtain posterior on $z$ which is denoted as $$p_\beta(z|x,w)$$
we can write 
\begin{equation}
    p_\beta(z_i=1|x,w)=\frac{p(z_i=1)p(x|z_i=1,w)}{\sum_{j=1}^{K}p(z_j=1)p(x|z_j=1,w)}
\end{equation}
We can see, that in our variational family we have no explicit inference process with regard to $$z$$! This is quiet important as
it's much harder to sample from categorical distributions in the way that will allow to propagate gradient. There are few approaches, one of them standing out from other is Gumbel-softmax reparametrization trick introduced in [this paper](https://arxiv.org/abs/1611.01144).
# Loss function
As usual we write down our objection function which is -ELBO
\begin{equation}
    \mathcal{L}=-E_q \frac{p_{\beta,\theta}(x,y,w,z)}{q(x,w,z|y)}=-E_q\log{p(y|x)}+\mathcal{D}_{KL}(q(w|y)||p(w))+
\end{equation}

\begin{equation}
    E_{q(x|y)q(w|y)} \mathcal{D}_{KL}(p(z|x,w) || p(z) )+
\end{equation}

\begin{equation}
    E_{q(w|y)p(z|x,y)}+\mathcal{D}_{KL}(q(x|y)||p(x|w,z))+
\end{equation}
First two terms represent something we had earlier, KL between prior for $$w$$ and posterior and reconstruction loss. Third term represents 
KL between posterior and prior for $$z$$ but here we have categorical distributions rather then gaussian ones. At the end we have something one can write down as 
conditional prior term, it represents how our posterior on $$x$$ is different from our GMM model. We can write down this term as 
\begin{equation}
    \sum_{i=1}^{K}p(z_i=1|x,w)\mathcal{D}_{KL}(q(x|y)||p(x|w,z_i=1))
\end{equation}