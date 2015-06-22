---
title:  "A Coarse Cough Detection"
date:   2015-01-04 10:18:00
description: A First-Pass Cough Detection Block
published: false
---

## Approaches 
\- *Statistical Tests*: the idea here is to capture the salient feature of the cough waveform by looking at the ratios of novel peaks and dips in the signal. We could have fixed divisions which correspond to distinct portions of the cough waveform. Then a hypothesis test can be performed on these ratios (maybe F-test?) to obtain some p-value (probability of the data being cough).

\- *Optimization*: the idea here is to model the data by some function, then use the training examples/data to optimize the parameters of this function. This will yield a minimum or best-fit curve (template) for all the training cough waveforms. With this template, we can do a cross-correlation with incoming signals to determine if there are similar at all.

\- *SVDD* : a much simpler but probably more efficient way is to employ a one-class classifier such as the Support Vector Data Description. Here, we would use both the time stamps and amplitude of the cough signal as features for training.

For capturing the temporal features / waveform of the cough, I am considering using a one class support vector algorithm called the Support Vector Data Description (SVDD). 

I have been following this URL to get a good understanding of one class SVM in general:

[link](http://rvlasveld.github.io/blog/2013/07/12/introduction-to-one-class-support-vector-machines/)

## Separability In Data
Before moving forward with the SVDD, I have to answer these question: how best can I transform my data(envelope-derivative) into an appropriate training data? What are the features to use? I had two options:

1\. Two Features - Time fraction & Envelope sample point: Here, I would take samples from the envelope of all training data, together with their time stamp(fraction of event duration), and use as two features for my training. This basically makes a 2D representation of the data (time, amplitude) just as our eyes see the waveform in reality. A concern here is that we might be losing some information on the sequential aspects of the data...the transitions between samples are not captured. 

To test this method, I went ahead and generated the suggested 2D data. Since it was just 2D, I can easily visualize without doing any dimensionality reduction like pca. 
<en-media type="image/png" hash="fd3fe612488b1e1d3c30d30a2366e81f"/>

The plot looked promising to me. You can see the generic envelope template I have been looking for in here. Hence, it should be feasible for a classifier, say SVDD, to model this template well and inform when an incoming event deviates significantly from this.

In-fact, I went ahead to generate an average/interpolated waveform from the plot above to better capture the 'template':
<en-media type="image/png" hash="bbed7c0709549a2fc1f2780ade0a7bbb"/>

This plot looks good and can be made even cleaner by low pass filtering to get a smooth line template.

2\. Multiple Features - Each sample of envelope as a feature of one example: In this approach on the other hand, one training example will correspond to an entire waveform/envelope. The samples of the envelope are considered the independent features of the single training example.

To test this approach, I performed pca on a table with row entries as envelope derivatives of coughs, and column entries as the sequential samples. Since the waveforms were of different lengths, I zero-padded shorter ones to make all lengths the same. With PCA, i could visualize the separability in this feature space. 
<en-media type="image/png" hash="3f8c90dfe484a9649c2b61b3f0d3c950"/>

I didn't like the plot very much. I was expecting to see some recognizable pattern (a somewhat generic template of a cough), such that the SVDD algorithm could easily model such a pattern. However, the plots were not very indicative of any such patterns. 

I'm thinking that in this approach, since we don't use a time vector, it is very essential to align all training examples as accurately as possible before sampling/feature extraction. Perhaps if I readjust the time-scale of every waveform(by plotting against fraction of duration before sampling), it might look better? At this point, I am not too excited about this approach because it will require more computation for the alignment or time-adjustments. Also because the data points are small in this approach, it will be tough to get a concrete model. 
n = 5;
fc = 60;
wn = 2*pi*fc; 
[B,A] = butter(5,wn,'low','s');
[C,D] = butter(5,wn*0.5,'low','s');
lpf = tf(B,A);
lpf2 = tf(C,D);

## SVDD Results

I went ahead and trained a few Support Vector Data Descriptions for the two feature case. For some reason, it was very difficult to get the svdd boundaries to capture the 'template'. It kept generating an ellipsoid decision boundary plane irrespective of whatever kernel i used be it RBF, polynomial or even exponential.
<en-media type="image/png" hash="17e8ce33fe0ee264b0dc03edeb32ef28"/>

My best guess is that there is much more data on the zero line, which overwhelms the algorithm. I have two ideas to address this:

\- Use just the first third of the waveforms, that captures the first peak and dip.
\- Remove all the data points on the zero line.

I considered trying other classifiers:

\- Gaussian Distributions didnt work very well either. It seemed to be tripping on the center points yet again.
<en-media type="image/png" hash="dde0e0eb4c739d114c713b7c683b5ef3"/>

\- then i looked at Mixture of Gaussians. Which did look promising with a rejection fraction of 0.05%:
<en-media type="image/png" hash="51ffdc8ef17b6110d3f27e594d93e2b6"/>

From the results of the GMM, I went back to tweak my SVDD parameters to see if I could get better results. I tightened the the param of the RBF kernel (0.02) and reduced the rejection fraction to 0.05 (admit fewer outliers into classifier). I got much improved results:
<en-media type="image/png" hash="eb6695533a19faef7a39dd6b0a757187"/>
<en-media type="image/png" hash="fe69394e60f65a495bedc0c90455e250"/>

I played with the matlab GMMs further to see what results I could get with those too:

With 5 components:
<en-media type="image/png" hash="dc1835b2d1355eaed8bc7e1b3e8f6ff1"/>

With 10 components:
<en-media type="image/png" hash="46cc1ccbe6c868069b2376c811e7075e"/>

**Testing & Evaluation**

## Methods - GMM

I noticed that it wasnt trivial to test my single class gmm models. I could evaluate a test example using one of two commands:

~ *Posterior* which would calculate the posterior probability for fitting a gaussian to each of the components of the gaussians. It's difficult to use this for classification though because the sum of all posteriors across components for each example will always be 1. However, if I use just three Gaussian components, and only consider the posterior from the most probably component for each sample in a test example, that might be revealing. Consider the following plots on the posterior for the three components.
<en-media type="image/png" hash="8d0f6d757a2f91cc97b950c6daeb34c1"/>
<en-media type="image/png" hash="92e5927c4159303e45ae8ddf22b05949"/>
<en-media type="image/png" hash="a96bb84bc63257d25030b632c54cf23d"/>

I tried to use a summation of the first and third component(peak models) posteriors. I observed however, that this doesn't translate well for a single example. I attempted to try on the handel sound track and it failed. It would seem the time information is not being included at all. 
<en-media type="image/png" hash="ddbd08ce112a6e70b0dc2c27bd841609"/>

I then shifted the test example around to verify if the time factor really makes no difference. 
<en-media type="image/png" hash="6fefd7339167496494b58c9268c92468"/>

However, it seems the time factor does make a difference. I think it failed in the handel test case because I was adding the posterior two components which accounted for the peaks and a brief range above & below the noise floor. Since there is almost always a noise floor or the otherwise in the test signal, the sum of posterior identified everything but the noise floor. 

In short, i think we should only use the posterior of one component; the peak component. Taking this further, we should probably just train on the peak data. 

~ *PDF* which gives an indication of the probability of example fitting the model, except the output does not seem bounded so it's hard to set a threshold.

## Methods - SVDD

The SVDD Performance looks fairly good. I have to run a full test and get actual values but on a single event, the confusion matrix looked like:


Labels	| outlier | target | Totals
--------|:-------:|:------:|-------
target  |   27    |   365  | 392  


And the boundaries on the plot looks like: 

  

<en-media type="image/png" hash="0f2c7283efe5d0e9e1cf0c6ee39ea582"/>
