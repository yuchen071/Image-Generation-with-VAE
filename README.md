# Image Generation with VAE
This project is trained with two datasets, the MNIST dataset, and the Anime faces dataset. The VAE for MNIST is first converted from grayscale to binary, then trained with Binary Cross Entropy as its loss function. The Anime faces dataset is normalized to 0~1, and trained with Mean Square Error as its loss function.

## Results
| Dataset | Fake Images | Interpolation between 4 latent codes |
|:--:|:--:|:--:|
| MNIST | ![mni_fake](https://github.com/yuchen071/Image-Generation-with-VAE/blob/main/results/mnist/lambda_1_fake.png) | ![mni_int](https://github.com/yuchen071/Image-Generation-with-VAE/blob/main/results/mnist/lambda_1_interp.png) |
| Anime | ![ani_fake](https://github.com/yuchen071/Image-Generation-with-VAE/blob/main/results/anime/lambda_1_fake.png) | ![ani_int](https://github.com/yuchen071/Image-Generation-with-VAE/blob/main/results/anime/lambda_1_interp.png) |

## Requirements
MNIST dataset: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz  
Anime Faces Dataset: https://www.kaggle.com/soumikrakshit/anime-faces  

The root folder should be structured as follows:
```
📁 root/
  ├─ 📁 dataset/
  |  ├─ 📚 mnist.npz
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

## How to use
### Train
Run the following code to train with MNIST dataset  
```bash
python train_mnist.py
```

Run the following code to train with the anime dataset  
```bash
python train_anime.py
```

By default, the scripts should output training results and synthesized images in a `results` folder.

### Parameters
Global parameters can be tinkered in the script:
```python
PATH_ZIP = "path/to/dataset.zip"
DIR_OUT = "output/image/directory"

EPOCHS          # epochs
LR              # learning rate
BATCH_SIZE      # batch size
SPLIT_PERCENT   # Percantage of the dataset (0~1) to be split for training and testing
LOG_INT         # Interval for outputting testing images

LAMBDA          # Kullback-Leiblier (KL) multiplier λ
LAT_DIM         # Latent space dimension size
```
