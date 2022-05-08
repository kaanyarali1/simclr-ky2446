# Representation Learning 
Kaan Yarali (ky2446)
## A Simple Framework for Contrastive Learning of Visual Representations 
(https://arxiv.org/pdf/2002.05709.pdf) <br>
## Unsupervised Feature Learning via Non-Parametric Instance Discrimination 
(https://arxiv.org/pdf/1805.01978v1.pdf) <br>


In this project, "A Simple Framework for Contrastive Learning of Visual Representations" and "Unsupervised Feature Learning via Non-Parametric Instance Discrimination" are implemented. Labeling data is a very challenging process since it is expensive, time-consuming and can make privacy issues. Recently, self-supervised learning has been heavily used to model the representation of the unlabeled data, without any human annotation. The goal of this project is to train the models using a self-supervised learning approach to learn good feature representations with unlabeled data and compare their performance with the models trained in fully supervised learning fashion. <br>

In the first paper (SimClr), model will learn its representation by contrastive learning by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. We initially define the set consisting of different data augmentation operators such as {random crop (with flip and resize), color distortion, and Gaussian blur.} Two different data augmentation methods will be sampled from the set for any given data example to generate two correlated views of the same example, denoted x_i and x_ j , which we consider as a positive pair. A neural network base encoder h will extract feature embeddings from augmented data examples. A small network (projection head) g will be added on top of this encoder since it is shown that a nonlinear projection head improves the representation quality of the layer before it. NT-Xent (the normalized temperature-scaled cross entropy loss) will be used for contrastive learning. After completing self-supervised learning, projection head will be dropped from the network and a linear classifier will be added to train the network with a small portion of the labeled data to train in a supervised learning fashion. 

![image](https://user-images.githubusercontent.com/77569866/167318402-9a28ed4f-73c6-40cc-b59f-72c2c62509de.png)


In the second paper, it takes the class-wise supervision to the extreme and learns a feature representation that discriminates among individual instances.
This problem is formulated as a non-parametric classification problem at the instance-level, and noisecontrastive estimation is used to tackle the computational challenges imposed by the large number of instance classes.  A memory bank is also implemented to store the feature embeddings of each instance and k nearest neighbor is performed in downstream classification task.

![image](https://user-images.githubusercontent.com/77569866/167318417-b3db88fb-312f-4f0e-81da-46dfa6aedb51.png)

### CIFAR 10
The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research.[1][2] The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. 

![image](https://user-images.githubusercontent.com/77569866/167318519-126a0a3b-899a-4f93-b80b-377a8b861f21.png)

