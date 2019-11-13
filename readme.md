# CNN AutoEncoder-Kmeans on MNIST and CelebA

A simple autoencoder to recover MNIST or celebA data using convolutional and de-convolutional layers and a KMeans_cluster
to divide all images into n-clusters

# Usage

Train an AutoEncoder, generate recoverd Mnist images, and do t-sne on embeddings.

```shell
    python CNN_AE_Mnist.py
```

Train an AutoEncoder, generate recoverd celebA images, and do Kmeans.
```shell
    python CNN_AE_celebA.py
```

# Recovered Image


The dimension of embedding is `10`.

Fig.1 and Fig3 in each row are `real images`, Fig.2 and Fig.4 in each row are `recovered images`.

*Epoch 0* | *Epoch 5* | *Epoch 9*
:---: | :---: | :---: 
<img src="figs/AE/sample_epoch_0.png" width=280px> | <img src="figs/AE/sample_epoch_5.png" width=280px> | <img src="figs/AE/sample_epoch_9.png" width=280px> | 

# t-sne visualization of embeddings

Use t-sne to reduce embeddings' dimension `10` down to `2`, so as to scatter in a coordinate system.

<center>
<img src="figs/AE/tsne.png" width=600px>

