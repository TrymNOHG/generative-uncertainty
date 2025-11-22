# Generative Uncertainty

This repository contains my project for the TDT15 theory module "Probabilistic Foundations for Deep Generative Models." The goal of the project was just to generally explore an interesting aspect of generative models, with potential connection to your Master's thesis.

Since my Master's thesis concerns uncertainty quantification, I came up with an idea that combines these two aspects. More specifically, I wanted to determine the uncertainty of a discriminative model by having it predict on data sampled from the generative model close to an original test input. It would work similar to the deep ensembles approach by Lakshminarayanan et. al 2017, but instead of looking at the variance from differently trained models, it looks at variance from semantically similar but different data points.


My initial thought was just to perform Gaussian sampling in the latent space. However, the teacher for the course suggested reading up on the geometry of the manifold produced by the generative models as these may not be Euclidean. This advice lead me down the rabbit-hole of differential geometry and how the images of latent spaces forms Riemmanian manifolds that are only Euclidean in infinitesimally local areas. However, it also gave me a chance to perform my sampling but from a different perspective.

## Project Set-up
The project is split into various experiments and scripts to assist the experiments. All the models that were tested can be trained by going through the training notebook. Moreover, simple examples with these models can be found in the experiments notebook. 

The exploration of shortest distances on the Riemannian manifolds (geodesics) can be found in the geodesics notebook. Here, the Stochman library was used to integrate embedded and stochastic manifolds into the models.

Finally, the uncertainty notebook contains the experiments surrounding quantifying uncertainty using both the properties of the Riemannian manifolds as well as the sampling methods created.



## Interesting Papers for the Curious Mind

For sampling using differential geometry:
- Mario Plays on a Manifold: Generating Functional Content in Latent Space through Differential Geometry - Gonzalez-Duque et. al 2022
- Data generation in low sample size setting using manifold sampling and a geometry-aware VAE - Chadabec et. al 2021

For learning about differential geometry with generative models:
- LATENT SPACE ODDITY: ON THE CURVATURE OF DEEP GENERATIVE MODELS - Arvanitidis et. al 2017
- Only Bayes should learn a manifold  - Hauberg 2018
- Metrics for Deep Generative Models - Chen et. al 2018
- Fast and Robust Shortest Paths on Manifolds Learned from Data - Arvanitidis et. al 2019 
- Geometrically Enriched Latent Spaces - Arvanitidis et. al 2020
- Identifying latent distances with Finslerian geometry - Pouplin et. al 2023
- Decoder ensembling for learned latent geometries - Syrota et. al 2024

Other inspiration for the models used in this repository:
- REGULARIZED AUTOENCODERS FOR ISOMETRIC REPRESENTATION LEARNING by Lee et. al 2022
- Back to the Future: Radial Basis Function Network Revisited - Que et. al 2020
