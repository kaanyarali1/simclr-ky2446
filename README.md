# Representation Learning 
Kaan Yarali (ky2446)
## A Simple Framework for Contrastive Learning of Visual Representations 
(https://arxiv.org/pdf/2002.05709.pdf) <br>
## Unsupervised Feature Learning via Non-Parametric Instance Discrimination 
(https://arxiv.org/pdf/1805.01978v1.pdf) <br>

### Framework used: PyTorch 
### GPUs used: A100, V100, and P100 
### Platform: GCP and Google Colab

### GCP image configuration
Operating System : Deep Learning on Linux
Version: Debian 10 based Deep Learning VM for PyTorch CPU/GPU with CUDA 11.0 M90

All the model weights trained on this project can be found in this Google Drive Link. 
(https://drive.google.com/drive/folders/1LSGz-WAi87WbF3gI0tIVFgv8V136nhv3?usp=sharing)

All the training logs on this project can be found in this Google Drive Link.
(https://drive.google.com/drive/folders/1bR-6sMhYgHrqFsHZCSm5rjxC75F7j0MX?usp=sharing)

For example commands for this project, check the notebooks under the example-notebooks section. I also pushed the notebooks used for training.


In this project, "A Simple Framework for Contrastive Learning of Visual Representations" and "Unsupervised Feature Learning via Non-Parametric Instance Discrimination" are implemented. Labeling data is a very challenging process since it is expensive, time-consuming and can make privacy issues. Recently, self-supervised learning has been heavily used to model the representation of the unlabeled data, without any human annotation. The goal of this project is to train the models using a self-supervised learning approach to learn good feature representations with unlabeled data and compare their performance with the models trained in fully supervised learning fashion. <br>

In the first paper (SimClr), model will learn its representation by contrastive learning by maximizing agreement between differently augmented views of the same data example via a contrastive loss in the latent space. We initially define the set consisting of different data augmentation operators such as {random crop (with flip and resize), color distortion, and Gaussian blur.} Two different data augmentation methods will be sampled from the set for any given data example to generate two correlated views of the same example, denoted x_i and x_ j , which we consider as a positive pair. A neural network base encoder h will extract feature embeddings from augmented data examples. A small network (projection head) g will be added on top of this encoder since it is shown that a nonlinear projection head improves the representation quality of the layer before it. NT-Xent (the normalized temperature-scaled cross entropy loss) will be used for contrastive learning. After completing self-supervised learning, projection head will be dropped from the network and a linear classifier will be added to train the network with a small portion of the labeled data to train in a supervised learning fashion. 

![image](https://user-images.githubusercontent.com/77569866/167318402-9a28ed4f-73c6-40cc-b59f-72c2c62509de.png)


In the second paper, it takes the class-wise supervision to the extreme and learns a feature representation that discriminates among individual instances.
This problem is formulated as a non-parametric classification problem at the instance-level, and noisecontrastive estimation is used to tackle the computational challenges imposed by the large number of instance classes.  A memory bank is also implemented to store the feature embeddings of each instance and k nearest neighbor is performed in downstream classification task.

![image](https://user-images.githubusercontent.com/77569866/167318417-b3db88fb-312f-4f0e-81da-46dfa6aedb51.png)

Finally, four different models are also trained by supervised learning fashion to compare the performance of the results of paper 1 and paper 2.

### CIFAR 10 (Dataset Used)
The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. (https://www.cs.toronto.edu/~kriz/cifar.html)

![image](https://user-images.githubusercontent.com/77569866/167318519-126a0a3b-899a-4f93-b80b-377a8b861f21.png)

## Results

![image](https://user-images.githubusercontent.com/77569866/167318769-f02b182b-cd24-4a0d-b9c7-9313fcc72e83.png)

![image](https://user-images.githubusercontent.com/77569866/167318788-9ab724cc-5f88-4034-a86f-c7bb8e497a81.png)

![image](https://user-images.githubusercontent.com/77569866/167318798-00a52a53-5a80-4ab6-a151-521b36ad99e5.png)

![image](https://user-images.githubusercontent.com/77569866/167318808-d1d1fcba-222f-4b5e-9675-54c1befd3cc6.png)

![image](https://user-images.githubusercontent.com/77569866/167318817-a7422ba8-4e62-4a4a-8f25-867845d64973.png)

![image](https://user-images.githubusercontent.com/77569866/167318828-3114b741-5473-44ea-a11f-cb2dc8567bdb.png)

![image](https://user-images.githubusercontent.com/77569866/167320793-dc6f8a85-273d-46ed-9a7f-71f2644b4f9e.png)

![image](https://user-images.githubusercontent.com/77569866/167320826-4c6516d2-8a86-4e10-80c5-aaeb13ac0fe0.png)

![image](https://user-images.githubusercontent.com/77569866/167320846-b0779251-4f5d-42cb-b80f-d9a57643f6ae.png)

![image](https://user-images.githubusercontent.com/77569866/167320853-91eeda09-71dd-4ca7-a2c3-0ddfe12b4f84.png)

![image](https://user-images.githubusercontent.com/77569866/167320856-0204f3c6-e34d-47cc-aad2-f7ec06a6fe16.png)

# Observations
For linear evaluation, SimClr outperformed both supervised approach and the instance level discrimination approach with a significant gap. Contrastive learning without labels enable model to learn discriminative features. However, the second paper approach (instance level discrimination) could get only % 20 percent accuracy on the test set so we understand that augmentations and selections of negative samples affect the model performance significantly. From supervised learning approaches, ResNet50 TrainAllLayers (imagenet weight init.) got the highest test accuracy. For fine-tuning comparison, I am currently training a model using the weights from this SimClr model (BS:256,PD:64,LARS) and it should be well noted that this time I am training all the layers not only the last linear layer. 

# Organization of this directory
```
.
????????? README.md
????????? requirements.txt
????????? simclr
    ????????? dataloader
        ????????? Cifar10Instance.py
        ????????? dataloader.py
    ????????? example-notebooks
        ????????? Example_InstanceLevelClassification.ipynb
        ????????? Example_SimClr.ipynb
        ????????? Example_SimClr_Downstream.ipynb
    ????????? instance-discrim
        ????????? dloader
            ????????? dloadertest.py
        ????????? notebooks
            ????????? Unsupervised_k1024.ipynb
            ????????? Unsupervised_k256.ipynb
            ????????? Unsupervised_k512.ipynb
        ????????? dloader
        ????????? NCECriterion.py
        ????????? alias_multinomial.py
        ????????? utilsInstance.py
    ????????? layers
        ????????? Normalization.py
        ????????? identity.py
        ????????? linear.py
        ????????? projection.py  
    ????????? loss
        ????????? nxtent.py
        ????????? nxtentgit.py
    ????????? models
        ????????? ResNetCifar.py
        ????????? downstream.py
        ????????? downstreamnew.py
        ????????? resnet.py
        ????????? simclr.py
    ????????? notebooks
        ????????? dstream
            ?????????CIFAR10-RES50-DSTREAM-BS-256-PD128-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS256-PD64-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD128-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD128-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD64-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD64-LARS.ipynb
            ?????????CIFAR10_RES50_SIMCLR_BS128_PD128_ADAM.ipynb
            ?????????CIFAR10_RES50_SIMCLR_BS128_PD128_LARS.ipynb
            ?????????CIFAR10_RES50_SIMCLR_BS128_PD64_ADAM.ipynb
            ?????????CIFAR10_RES50_SIMCLR_BS128_PD64_LARS.ipynb
            ?????????CIFAR10_RES50_SIMCLR_BS256_PD128_ADAM.ipynb
            ?????????CIFAR10_RES50_SIMCLR_BS256_PD64_ADAM.ipynb
        ????????? simclr
            ?????????CIFAR10-RES50-SIMCLR-BS128-PD128-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS128-PD128-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS128-PD64-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS128-PD64-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS256-PD128-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS256-PD128-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS256-PD64-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS256-PD64-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD128-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD128-LARS.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD64-ADAM.ipynb
            ?????????CIFAR10-RES50-SIMCLR-BS64-PD64-LARS.ipynb
    ????????? optim
        ????????? LARS.py
    ????????? plain-models
        ????????? notebooks
            ????????? plainResNet50TrainAll.ipynb
            ????????? plainResNet50TrainAllImagenet.ipynb
            ????????? plainResNet50TrainLinear.ipynb
            ????????? plainResNet50TrainLinearImagenet.ipynb
    ????????? augment.py
    ????????? utils.py
