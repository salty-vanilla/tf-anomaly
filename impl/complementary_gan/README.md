# Complementary GAN
One-Class Adversarial Nets for Fraud Detection [[paper]](https://arxiv.org/abs/1803.01798)  
I implemented Complementary GAN with conv layers.  

## Environments
* python 3.6
* tensorflow 2.1.0

## Experiments
* Inlier: '0' in MNIST
* Outlier: 'Not 0'

### Generated samples
<img src="assets/gen.png">

### Discrimination results
| <img src="assets/normality_histogram.png"> | <img src="assets/roc.png"> |
|-|-|
