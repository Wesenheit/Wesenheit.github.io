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
where $$\phi_x$$ and  $$\phi_x$$ denote neural networks used in inference process (as previously they parametrize means and variations of gaussian posterior).
In order to obtain posterior on $z$ which is denoted as $$p_\beta(z|x,w)$$
we can write 
\begin{equation}
    p_\beta(z_i=1|x,w)=\frac{p(z_i=1)p(x|z_i=1,w)}{\sum_{j=1}^{K}p(z_j=1)p(x|z_j=1,w)}
\end{equation}
We can see, that in our variational family we have no explicit inference process with regard to $$z$$! This is quiet important as
it's much harder to sample from categorical distributions in the way that will allow to propagate gradient. There are few approaches, one of them is Gumbel-softmax reparametrization trick introduced in [this paper](https://arxiv.org/abs/1611.01144).
# Loss function
As usual we write down our objection function which is -ELBO
\begin{equation}
    \mathcal{L}=-E_q \frac{p_{\beta,\theta}(x,y,w,z)}{q(x,w,z|y)}=-E_q\log{p(y|x)}+\mathcal{D}_{KL}(q(w|y)||p(w))+
\end{equation}

\begin{equation}
    E_{q(x|y)q(w|y)} \mathcal{D}_{KL}(p(z|x,w) || p(z) )+
\end{equation}

\begin{equation}
    E_{q(w|y)p(z|x,y)}+\mathcal{D}_{KL}(q(x|y)||p(x|w,z))
\end{equation}

First two terms represent something we had earlier, KL between prior for $$w$$ and posterior and reconstruction loss. Third term represents 
KL between posterior and prior for $$z$$ but here we have categorical distributions rather then gaussian ones. At the end we have something one can write down as 
conditional prior term, it represents how our posterior on $$x$$ is different from our GMM model. We can write down this term as 
\begin{equation}
    \sum_{i=1}^{K}p(z_i=1|x,w)\mathcal{D}_{KL}(q(x|y)||p(x|w,z_i=1))
\end{equation}
What is easy to compute as again we have KL between two gaussian distributions.
When we see how our loss function looks like we can look at our data.

## Gene expression data
Data used in this study is modified version of data used in this competition. It can be found in [github repository](https://github.com/Wesenheit/SAD2-22W/tree/main/Project1/data). 
Data consist of expression measurments of $$5000$$ genes. There are $$72208$$ observations in training set and $$18052$$ in test set. Data was collected from
$$10$$ donors at $$4$$ lab sites for $$45$$ different cell types. Data is heavily zero inflated as $$91\%$$ of data is equal zero. Totally we have
$$12$$ unique combinations of donors and lab sites which will be important. Our measurments are expressed in counts so it's only discrete, mean value of test set is $$0.44$$ while standard deviation is $$34$$. 

## Implementation of model
In order to implement GMMVAE PyTorch library was used. Here decided dimensionality of data was $$n_x=200$$, $$n_z=200$$ while number of clusters was set to $$K=20$$. As data we are dealing with discret data we can model our data using Negative Binomial distribution with parameters $$p\in [0,1]$$, $$r\in \mathcal{R}^+$$. Total likelihood can be described by formula: 
\begin{equation}
    P(x=k|r_i,p_i)=\left({k+r-1\choose k}(1-p)^k p^r\right)
\end{equation}
As there are two parameters to specify we need to have two heads with dimensionality $$5000$$ each in our decder. 

As previously mentioned there are $$12$$ 
distinct ID of batch, there is great probability that instead of clustering based on purly biological casues we would obtain clustering based on those batches.
In order to overcom this instead of modeling $$p(y|x)$$ we model $$p(y|x,b)$$ where $$b$$ is one-hot encoded vector of batch ID. This hopefully will help to 
clear latent space $$x$$ from any external effects alowing to catch only biological activity.


Decoder and encoder of our VAE were $$2$$ layers deep with parameters:
* Encoder: $$5000-300-250$$ followed by 2 heads with dimensionality $$2n_x$$ and $$2n_w$$ used to describe posterior of $$x$$ and $$w$$
* Decoder: $$n_x+12-250-300$$ followed by 2 heads with dimensionality $$5000$$
Each layer was followed by gelu activation, layer normalization layer and dropout with $$p=0.05$$. $$\beta$$ neural network was one layer deep with dimensions
$$n_w-500-4Kn_x$$. As in data we have few observations with high value we need to scale down our data to zero mean and variation of $1$. Here we can implement relevant class to do so.

```python
import torch
from torch import nn
from torch.nn import functional as F
from typing import List,Optional,Tuple

class norm_layer(nn.Module):
    """
    Normalizing layer to preprocess data using known mean and std
    """
    def __init__(self,mean,std):
        super().__init__()
        self.mean=mean
        self.std=std
    def forward(self,x):
        return (x-self.mean)/self.std
```
Now let's write class of our encoder.

```python
class Encoder(nn.Module):
    """
    General class for encoder
    """
    def __init__(self,sizes:List,
                 dimx:int,
                 dimw:int,
                 preproces=None,
                 dropout:Optional[float]=0.05) -> None:
        """
        sizes
        """
        super().__init__()
        self.latent_size = sizes[-1]
        self.network = []
        if preproces is not None:
            self.network.append(preproces)
        for i in range(len(sizes)-1):
            self.network.append(nn.Linear(sizes[i],sizes[i+1]))
            self.network.append(nn.LayerNorm(sizes[i+1]))
            self.network.append(nn.GELU())
            self.network.append(nn.Dropout(dropout))
        self.infer_x = nn.Linear(sizes[-1],dimx*2)
        self.infer_w = nn.Linear(sizes[-1],dimw*2)
        self.network = nn.Sequential(*self.network)

    def forward(self,inp:torch.Tensor)-> Tuple[Tuple[torch.Tensor]]:
        inp = self.network(inp)
        return (torch.chunk(self.infer_w(inp),2,dim=-1),
                torch.chunk(self.infer_x(inp),2,dim=-1))
```