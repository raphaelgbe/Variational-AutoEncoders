# Variational-AutoEncoders
Implementing and exploring VAEs for various problems and with different approaches using PyTorch.

# Background and resources:

Variational AutoEncoders are models at the intersection of deep learning and variational Bayesian methods, allowing to generate data from low-dimension vectors that ideally encode useful and high-level information about the data.
This work is first a re-implementation, from TensorFlow to PyTorch, the classical VAE and the [infoVAE described and implemented by S. Zhao on his Git page](https://github.com/ShengjiaZhao/InfoVAE/), following his [paper](https://arxiv.org/abs/1706.02262). As a proof of concept, his work consists in showing that more mutual information can be encoded from using a different objective from the ELBO one usually optimized (introduced in [the original VAE by Kingma et al.](https://arxiv.org/abs/1312.6114)) and replacing it with the MMD objective (which intuitively is meant to let more 'freedom' to the latent code at the bottleneck of the VAE). This approach is notably justified as an attempt to mitigate *posterior collapse*, by which we refer to what happens when the VAE ignores the latent code and simply tries to replicate the probability distribution of the data.

The problem of the posterior collapse has been particularly significant when trying to build VAEs for sequential data such as text, as explained by [Bowman et al. in their 2015 paper](https://arxiv.org/abs/1511.06349). Thus this project is also an attempt at implementing an infoVAE for text data, as it doesn't seem to be done as of now in the original repo linked above.
This work is hence a PyTorch implementation of *Bowman et al*'s paper, with a twist on the loss that can be chosen to be optimized with the ELBO or the MMD objective. [Tim Baumgärtner's implementation](https://github.com/timbmg/Sentence-VAE) has been particularly helpful for designing some parts.

# To do:
As of now, some simple additions must be made to the scripts already uploaded (*eg* interpolation functions) and it could be interesting to run experiments on mutual information. Improving the results of the ELBO-based VAE for text should also be feasible.
In addition to that, studying [*delta-VAEs*](https://openreview.net/forum?id=BJe0Gn0cY7) could be extremely valuable for this project.
