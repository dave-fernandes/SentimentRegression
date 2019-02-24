# Sentiment Regression Analysis
The TensorFlow code here performs a regression analysis to predict the star rating for a given textual review. This is in contrast to the usual sentiment classification \(positive\/negative\). The model also predicts the helpfulness rating for the review.

## Model
The model is a 2 layer fully connected network using an pre-trained transformer to compute a feature vector for the text.
* The feature vector is computed using the BERT language representation model [\(Devlin, Chang, Lee, \& Toutanova, 2018\)](https://arxiv.org/abs/1810.04805).
* The fully connected network has...

## Files
* `extract_features.py`, `modeling.py` and `tokenization.py` are used to pre-compute the feature vectors. The latter two files are exact copies from https://github.com/google-research/bert and `extract_features.py` is a copy that has been modified to output only the sentence embeddings, and save to a TFRecord format rather than JSON.
* `Preprocessing.ipynb` is a Jupyter notebook used to clean the data, compute the feature vectors, and save the data to TFRecord files that can be used to create TensorFlow Datasets.
* `Regression.ipynb` is a Jupyter notebook containing the regression model.

## Implementation Notes
* Tested with Python 3.6.7 and TensorFlow 1.12.0
