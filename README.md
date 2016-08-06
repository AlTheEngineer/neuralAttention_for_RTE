This is an attempted implementation of the paper titled "Reasoning about Entailment using Neural Attention". 
This package will likely include the data sets needed to carry out validation tests, but incase it is not:
Download the SNLI data sets from: http://nlp.stanford.edu/projects/snli/
Unzip the folder and move it inside this folder
    $cp -r [PATH_TO_SNLI}/snli_1.0 .

Carry out data preprocessing using
    $python data_preprocess.py

This requires the following Python modules to be installed:
    gensim
    nltk
    pandas
    pickle
    numpy

Then, carry out training and testing using
    $THEANO_FLAGS='mode=FAST_RUN,floatX=float32,device=cpu' python neuralAttention.py word_by_word


This requires the bleeding edge versions of Theano and lasagne, which can be installed using:
    $pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
    $pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

The main layers of the RNN are present in the layer.py file. They are essentially derivative classes 
from lasagne.layers.Layer and lasagne.layers.LSTMLayer


At the time of writing this, I have not yet ran the full experiment and thus do not know
the accuracy that is achieved (or honestly if it doesn't give an error)
