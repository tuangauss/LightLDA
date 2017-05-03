# LightLDA
Latent Dirichlet Allocation - inference in topic models (optimized by Metropolis - Hastings - Walker)

## Description:

Implementation of an unsupervised text-clustering machine learning algorithm. Latent Dirichlet Allocation (LDA) is a probabilistic generative model that extracts thematic structure in a documentation of texts (<b>corpus</b>). The algorithm assumes that every topic is a collection of words with certain probabilities and every <b>article</b> (text) in the corpus is a distribution of these topics. 

A good introduction to Latent Dirichlet Allocation can be found on Edwin Chen's data sciene [blog](http://blog.echen.me/2011/08/22/introduction-to-latent-dirichlet-allocation/)

### Reducing sampling complexity:

The usual implementation of LDA -the collapsed Gibbs sampling approach-  can be expensive if we try to capture hundreds, thousands, or hundred of thousands of topic. [Researches](https://pdfs.semanticscholar.org/137a/ec8c56102cea1ac7c083989036bb51331fdc.pdf) has shown that another decomposition of the collapsed conditional probability , using Walker's Alias table and Metropolis-Hasting, can yield an order of magnitude speed up.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Project uses Python and many relevant statistical, graphical libaries, notably:
* panda
* numpy
* matplotlib

Information on how to install these packages can be found only. I highly recommend using Anaconda platform.
```
conda install pandas
```

### Dataset:

The example uses the dataset from the Associated Press (AP) - which contains 2,246 articles and 10,473 words

## Authors

* **Tuan Nguyen Doan** - *Initial work* - [tuangauss](https://github.com/tuangauss)

This is a self-learning project and I hope to learn from the expertise of the community. Please reach out to me if you have any suggestion or ideas.
