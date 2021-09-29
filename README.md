# Image Generation with VAE
 Reconstruct Images with a Variational Auto Encoder

## Instructions
Implement a Variational Autoencoder (VAE) to reconstruct images with the datasets provided.

1. Show the learning curves and some samples of the reconstructed images.
2. Sample the prior p(z) and use the latent codes z to synthesize some examples when your model is well-trained.
3. Show the synthesized images based on the interpolation of two latent codes z between two real samples.
4. Multiply the Kullback-Leiblier (KL) term with a scale λ and tune λ (e.g. λ = 0 and λ = 100) then show the results based on steps 1, 2, 3.

Hints:
* Convert grayscale images to binary first.
* Use Binary Cross Entropy as the loss function with binary data, and Mean Square Error with real value data.
## Requirements
MNIST dataset: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz  
Anime Faces Dataset: https://www.kaggle.com/soumikrakshit/anime-faces  

The root folder should be structured as follows:
```
📁 root/
  ├─ 📁 dataset/
  |  ├─ 📄 mnist.npz
  |  └─ 📚 archive.zip
  ├─ 📄 train_anime.py
  └─ 📄 train_mnist.py
```

Original Anime Dataset Source: https://github.com/bchao1/Anime-Face-Dataset

### Dependencies  
```
numpy==1.19.2
zipp==3.5.0
matplotlib==3.3.4
tqdm==4.62.2
torch==1.8.0
torchvision==0.9.0
Pillow==8.3.2
```

## Train
Run the following code to train with MNIST dataset  
```bash
python train_mnist.py
```

Run the following code to train with the anime dataset  
```bash
python train_anime.py
```

By default, the scripts should output training results and synthesized images in a `results` folder.