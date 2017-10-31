---
layout: post
title:  "Deep Neural Networks In Cough Detection"
date:   2015-06-15 10:18:00
description: Using Deep Nets for Cough Detection
tags: [dnn, cough]
output:
  html_document:
    keep_md: true
comments: true
published: true
---

# DNN - Application Notes

## Ideas For Deep Learning in Speech/Cough Recognition
- Learn sparse codes for environmental sounds, then use sparse codes as filters for cough detection
- Long-Short Term Memory Neural Networks for end-to-end cough recognition
- Hybrid Systems: use DNNs for modelling state emission probability distribution in HMMs instead of the conventional GMMs
- Tandem Systems: use DNNs to extract bottleneck features as inputs for HMM-GMM training
	- Stacked Denoising Autoencoders: the expectation here is that sdAs will capture the most intrinsic parts of spectral patterns in audio data and lose some of the redundancy, thereby increasing the sensitivity (and maybe specificity) of the system
	- Sparse Autoencoders: motivation here is that sparsity will help us learn an overcomplete basis for feature representation such that, these basis can be more discriminative between coughs and speech. this could boost up the specificity of the system.


## Softwares & Packages
For implementing the DNNs (specifically the autoencoders), I explored both [Theano](http://deeplearning.net/software/theano/) and [Torch](http://torch.ch/) software packages. I decided to go with **Theano** eventually because there it had a more active developer community and support  than **Torch**. 

Theano is a python library and so I installed [Anaconda](https://store.continuum.io/cshop/anaconda/) as an IDE for dev.  This comes with all the necessary python packages like **numpy** and **scipy**. I had to install **pydot** manually though. To install the actual Theano library, I just used pip: `$pip install Theano`.


## System Overview
The structure I have in mind for the proposed system:
*AudioIn -> Feature Extraction(PLP,STFT) -> Autoencoder(sdAs/sAs) -> HMM-GMM* 

## Database
For the audio training data, I start off learning on the TIMIT database. A brief summary of the database:
- 6300 sentences: 10 sentences by 630 speakers
- 8 dialects in the US:
	+ New England, Northern, New York, Southern, etc

### Sentence Distribution
|Sentence Type | #Sentences | #Speakers | Total | #Sentences/Speaker
|------------- | ----------:| ---------:| ----- | :----------------:
|Dialect (SA)  |       2    |    630    |  1260 |         2
|Compact (SX)  |     450    |      7    |  3150 |         5
|Diverse (SI)  |    1890    |      1    |  1890 |         3
|**Total**     |  **2342**  |           | **6300** |    **10**

### Test Data 
Data has been partitioned into training and testing parts. Test **core** test data distribution looks like:


Corpus file structure:
`/<CORPUS>/<USAGE>/<DIALECT>/<SEX><SPEAKER_ID>/<SENTENCE_ID>.<FILE_TYPE>`


## Procedure
### Classical Feature Extraction/Data Augmentation
The first step for me was to extract spectral features from the audio examples. I am considering using Perceptual Linear Prediction, particularly the [RASTA-PLP api](http://labrosa.ee.columbia.edu/matlab/rastamat/). I want to compare that with the typical MFCC features, particularly [this](https://engineering.purdue.edu/~malcolm/interval/1998-010/) 'de-facto' implementation.


I attempted to just concatenate the individual frames together into a long table where each example was a 1x129 vector (129 frequency bins from spectra). **Here the patterns I was hoping to observe were very prominent**. Although these patterns are not as pronounced for the advanced feautres such as PLP or Mfcc. See images below:

![Image of Stft](/assets/imgs/allexamples_spec.png "Spectrum")
![Image of PLPSpeC](/assets/imgs/allexamples_pspec.png "PLP Spectrum")
![Image of PLPCeps](/assets/imgs/allexamples_pceps.png "Cepstrum")
![Image of MFCC](/assets/imgs/allexamples_mfcc.png "MFCCs")
 

>My only concern with this approach is that just by putting the examples(frames) together in their original order, I will have instances of abrupt transitions from one event to the other. I'm thinking though that this may not be as huge a problem for this application since in practice, cough events could occur in-succession.

Next, I attempted to grab the features of equal sizes from all my data(cough and speech). I attempted to use 13x10(13 PLP cepstra, 10 frames) as my 'mini-batch'. Mini-batches at edges which were less than 10 frames were zero-padded. Also, I had a 2-sample overlap on each edge(left and right) of the frames, leaving 6 samples of true intrinsic value in each sample.

![Image of Stacked Mini-Batches](/assets/imgs/spectra_stacked_mini_batch.png "MiniBatches") 

>If I use these exact mini-batches batches in my training, I avoid the abrupt transitions between examples that I mention above.


I am considering using the raw Spectra henceforth since it looks more representative than the PLP or MFCCs. Perhaps I will come back to change this once I understand what features my DNN actually learns.

### Stacked Autoencoder Results
I trained a stacked denoising autoencoder on my ~300 cough dataset. The architecture was as follows;
- input layer   : 128x5
- hidden layer 1: 340
- hidden layer 2: 100
- output layer  : 12
First, the features learned in the first layer didn't look very interesting. 
![SDA H1 Features](/assets/imgs/da_cough_weights.png "sDA Features")

I proceeded to encode my training data then just train a simple SVM on top of the second layer to see whether the encoding was looking good at all. To my surprise, it wasn't performing well at all. The classification performance looked like:
- 61.9% sensitivity, 57.2% specificity for encoded 100 features.
- 80.6% sensitivity, 85.2% specificity for raw 640 STFT features.  

Granted, the results were not optimized for the classification task through backprop. Next steps to try on this are:
- backprop with labels to optimize for task
- try different autoencoder types
- try different activation functions (tanh or relu)
- check regularization params

After the poor autoencoder preliminary results, I decided to change the architecture to convnets to see if I can get any better performance. I found a seemingly good theano based library, [Lasagne](https://github.com/Lasagne/Lasagne) which I plan to use for building my convnet. Currently, the architecture I have in mind looks like this:
- input spectra chunk: 128x20
- filter size/shape: 128x3
- 3 convolutional layers:
	- neurons: 128x20, 256x10 then 512x5
	- stride: 1
	- filter size/shape: 3 (1D convolution - just temporal)
	- max pooling : 2x
- 2 fully connected layers:
	- input layer: 1280 = 2x maxpool(512*5)
	- layer 4: 2000 units
	- layer 5: 2000 units
- 1 Softmax layer: two output classes (cough or speech)  

### Convolutional Neural Network
Concerning the network architecture itself, I previously considered using a stack of autoencoders. But now, given the way I augmented my input data, I am thinking a *1D Convolutional* first layer will be very effective for capturing the temporal properties in the input. 


In my first attempt, I started a 2D convolutional architecture that looks like this:
> Architecture:
- 9x3 conv, 32 filters
- ReLU
- 2x1 maxpool
- 5x3 conv, 32 filters
- ReLU
- 2x1 maxpool
- fully connected layer - 512 units
- 50% dropout
- fully connected layer - 2 units
- softmax

Notable highlights of the architecture are:
- Maintained temporal width of convolutional filters(3) for all layers
- No temporal pooling, just spectral pooling.

With just the above setup, I ran two test:
#### ConvNet Test 1
- learning rate: 0.01
- batch size: 1
- num of epochs: 5
- validation accuracy: 43%

#### ConvNet Test 2
- learning rate: 0.001
- batch size: 1
- num of epochs: 5
- validation accuracy: **92.7%**

#### Baseline Test - SVM on STFT
- Sensitivity: 80.54%
- Specificity: 82.16%
- Validation accuracy: **81.46%**

#### Baseline Test - SVM on MFCC
- Sensitivity: 78.85%
- Specificity: 79.88%
- Validation accuracy: 79.40%

The results, as seen above, is intriguing and informing; the convnet is doing about 10% better than SVM ontop of the raw STFT.

### HyperParameters
#### Learning Rate
From the above two tests, I observe that a **learning rate of ~0.001** is appropriate for this task. And 10% bigger learning rate, significantly decreases performance. 

#### Batch Size
I increased the batch-size from 1 to 5 and realized that the accuracy decreased in the first few epochs, but then it quickly got better and eventually outperformed the single batch case going up to 93% (up to 94% with 20 epochs). I think if optimally increase the batch size, then increase the number of number of epochs, performance can improve.

#### Number of Filters
Increasing the number of filters from 32 to 64 also gave some improvement in performance (94% -> 95%).
 


## Quick References
### Training Error vs Testing Error
![Error Plots](/assets/imgs/training_testing_error.png "Training vs testing Errors")


### Theano Tweaks & Flags
To run with multiple cores:
`OMP_NUM_THREADS=1`
check speed on multiple(2) cores:
`OMP_NUM_THREADS=2 python theano/misc/check_blas.py -q`

BLAS flags and other flags:
set in .theanorc file. See template below
```bash
  [global]
  floatX = float32
  device = cpu
  openmp = True
  base_compiledir = /path/to/base/dir

  [nvcc]
  fastmath = True

  [blas]
  ldflags = -L/path/to/blas/libs 

  [cuda]
  root = /path/to/cuda/
```


## References & Resources
- To be able to read matlab *.mat* files, I had to use the **HDF5** command-line tools and the **h5py** python extension. Apparently, *.mat v7.3* files are of the special [HDF5](http://docs.h5py.org/en/latest/build.html) file format.
- Conversely, I could have converted my *.mat* files to scipy compatible using `save('myfile.mat','-v7')` then I can open in python as: 
```python
import scipy.io
mat = scipy.io.loadmat('filemat')
```
from [StackOverflow](http://stackoverflow.com/questions/874461/read-mat-files-in-python). 
- I saw an interesting [DNN application](http://benanne.github.io/2014/08/05/spotify-cnns.html) on audio(music) from a Spotify intern. It captures some of the things I'm planning to do.
- I found a somewhat code for [loading audio files in theano/python](http://deeplearning.net/software/pylearn/api/pylearn.io.audio-pysrc.html)
- A personal [blog](http://benanne.github.io/research/) from the Spotify intern with the CNN on music applications
- A theano based autoencoder [trainer](https://github.com/caglar/autoencoders). Three different types of autoencoders:
	+ denoising
	+ contractive
	+ sparse 
- Deep learning [lectures](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/) from Oxford. 	
- ConvNets tutorial by Andrej Karpathy [here](http://cs231n.github.io/) is phenomenal, especially for information on hyperparameter tunning.
- The Lasagne theano wrapper has a good [documentation](http://lasagne.readthedocs.org/en/latest/)
- Speedups for convnets in theano:
	- from [beanne](http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html)
