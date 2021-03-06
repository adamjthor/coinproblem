# The Coin Problem

In North America, mechanical coin-sorting machines take advantage of the fact that coins of the same monetary value are manufactured to have identical shapes, (i.e. coins can be sorted according to their diameter & thickness). However this isn't the case all over the world. In India, different manufacturers create coins with slightly different dimensions, so traditional mechanical coin-sorting techniques won't work. 

One thing consistent across Indian coin manufacturers to denote monetary value is the symbol stamped into the coin. A machine that leverages computer vision to recognize symbols on coins could effectively sort coins automatically. This project creates that computer vision model, using transfer learning to train a CNN on over 100,000 images of coins.


### Guide to files:

1. _convert_to_recordio.py_ -  Preprocessing the image data to be in the right file format
2. _model_training.ipynb_ - An AWS sagemaker notebook that uses transfer learning to train a CNN for image classification
3. _entry_point.py_ - A blank file needed for the MXNET model in the next file to be deployed
4. _test_images.py_ - Deploying the trained MXNET model to classify the test images for later model evaluation. Confusion matrices and a file containing the soft classification probabilities are created and saved.
5. _results_analysis.py_ - Hard and soft classifications are analyzed for this CNN coin image classifier. Confusion matrices for hard classifications are cleaned and saved, along with histograms of classification probabilities for all soft classifications.
