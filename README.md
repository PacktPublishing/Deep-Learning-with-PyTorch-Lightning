# Deep Learning with PyTorch Lightning

<a href="https://www.packtpub.com/product/deep-learning-with-pytorch-lightning/9781800561618"><img src="https://static.packt-cdn.com/products/9781800561618/cover/smaller" alt="Deep Learning with PyTorch Lightning" height="256px" align="right"></a>

This is the code repository for [Deep Learning with PyTorch Lightning](https://www.packtpub.com/product/deep-learning-with-pytorch-lightning/9781800561618), published by Packt.

**Swiftly build high-performance Artificial Intelligence (AI) models using Python**

## What is this book about?
PyTorch Lightning lets researchers build their own deep learning (DL) models without having to worry about the boilerplate. With the help of this book, you'll be able to maximize productivity for DL projects while ensuring full flexibility from model formulation through to implementation. You'll take a hands-on approach to implementing PyTorch Lightning models to get up to speed in no time.

This book covers the following exciting features: 
* Customize models that are built for different datasets, model architectures, and optimizers
* Understand how a variety of deep learning models from image recognition and time series to GANs, semi-supervised and self-supervised models can be built
* Use out-of-the-box model architectures and pre-trained models using transfer learning
* Run and tune DL models in a multi-GPU environment using mixed-mode precisions
* Explore techniques for model scoring on massive workloads

If you feel this book is for you, get your [copy](https://www.amazon.com/dp/180056161X) today!

<a href="https://www.packtpub.com/?utm_source=github&utm_medium=banner&utm_campaign=GitHubBanner"><img src="https://raw.githubusercontent.com/PacktPublishing/GitHub/master/GitHub.png" 
alt="https://www.packtpub.com/" border="5" /></a>


## Instructions and Navigations
All of the code is organized into folders. For example, Chapter02.

The code will look like the following:
```
import pytorch_lightning as pl
...
# use only 10% of the training data for each epoch
trainer = pl.Trainer(limit_train_batches=0.1)
# use only 10 batches per epoch
trainer = pl.Trainer(limit_train_batches=10)
```

**Following is what you need for this book:**
This deep learning book is for citizen data scientists and expert data scientists transitioning from other frameworks to PyTorch Lightning. This book will also be useful for deep learning researchers who are just getting started with coding for deep learning models using PyTorch Lightning. Working knowledge of Python programming and an intermediate-level understanding of statistics and deep learning fundamentals is expected.

With the following software and hardware list you can run all code files present in the book (Chapter 1-10).

### Software and Hardware List

| Chapter  | Software required                   | OS required                        |
| -------- | ------------------------------------| -----------------------------------|
| 1 - 10       | PyTorch Lightning                  | Cloud, Anaconda (Mac, Windows) |
| 1 - 10      | Torch            | Cloud, Anaconda (Mac, Windows) |
| 1 - 10     | TensorBoard            | Cloud, Anaconda (Mac, Windows) |


We also provide a PDF file that has color images of the screenshots/diagrams used in this book. [Click here to download it](https://static.packt-cdn.com/downloads/9781800561618_ColorImages.pdf).


### Related products 
* Deep Learning with fastai Cookbook [[Packt]](https://www.packtpub.com/product/deep-learning-with-fastai-cookbook/9781800208100) [[Amazon]](https://www.amazon.com/dp/1800208103)

* Machine Learning Engineering with MLfl ow [[Packt]](https://www.packtpub.com/product/machine-learning-engineering-with-mlflow/9781800560796) [[Amazon]](https://www.amazon.com/dp/1800560796)

## Get to Know the Author
**Kunal Sawarkar**
is a chief data scientist and AI thought leader. He leads the worldwide partner ecosystem in building innovative AI products. He also serves as an advisory board member and an angel investor. He holds a master’s degree from Harvard University with major coursework in applied statistics. He has been applying machine learning to solve previously unsolved problems in industry and society, with a special focus on deep learning and self-supervised learning. Kunal has led various AI product R&D labs and has 20+ patents and papers published in this field. When not diving into data, he loves doing rock climbing and learning to fly aircraft, in addition to an insatiable curiosity for astronomy and wildlife.

