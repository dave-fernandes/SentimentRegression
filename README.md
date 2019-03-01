# Sentiment Regression Analysis
The TensorFlow code here performs a _regression_ analysis to predict the star rating for a textual product review. This is a little different from the usual sentiment _classification_ analysis where the review is simply flagged as positive or negative.

## Data
This analysis uses Amazon product review data obtained from https://www.kaggle.com/snap/amazon-fine-food-reviews/version/2

## Model
The model uses a pair of feature vectors computed from a pre-trained transformer to independently characterize the review text and summary\/title line. Subsequent layers process the summary and main review text independently. Finally, these two components are fed into a fully connected network.
* The feature vector is computed using the BERT language representation model [\(Devlin, Chang, Lee, \& Toutanova, 2018\)](https://arxiv.org/abs/1810.04805). The last four layers of the transformer (728 hidden units each) are combined into a 3072 element vector. Only the sentence embedding is used, not the word-piece (token) embeddings.
* The two 3072-element feature vectors are fed into a set of 1D-convolutional layers that process each of the vectors independently, but using the same set of weights. This convolutional approach reduces the number of parameters in the model while allowing the review-text to contribute a different weight from the summary-text to the final regression value.
* The output from the last convolutional layer (two 15-element vectors) is fed into a fully connected network with a final linear layer to produce the regression output value.
* The best regression performance was obtained when all convolutional layers and fully connected layers used _tanh_ activations. (The output layer has no activation.) Best performance was also obtained with no batch normalization.

## Results
A random subset of 5000 predictions from the test set are plotted below against the actual review scores. Actual scores have had gaussian "noise" added for visualization purposes. (Otherwise, the values would all be plotted on lines at integral values of 1-5.) A 1:1 reference line has been plotted for comparison. Although the data are not distributed normally, the Pearson correlation coefficient is a useful metric for comparing models; r = 0.83 for the full test set. We can also use the non-parametric Kendall tau measure (tau = 0.68).

![alt text](https://github.com/dave-fernandes/SentimentRegression/blob/master/images/score_scatter_plot.png "Scatter plot of predicted versus actual review scores.")

A box plot showing mean (yellow line) and interquartile ranges (boxes) is shown below with a reference 1:1 line.

![alt text](https://github.com/dave-fernandes/SentimentRegression/blob/master/images/score_box_plot.png "Box plot of predicted versus actual review scores.")

Finally, we also attempted to find a regression for the helpfulness score for reviews (ratio of up-votes to total votes). You can run the notebook to see these results plotted; however, the regression coefficients were fairly low (r = 0.45, tau = 0.33).

## Discussion
The BERT sentence embeddings allowed us to obtain a reasonably good regression for review score from the written product review and summary line. This approach of using the BERT transformer to compute feature vectors as a preprocessing stage vastly reduced the amount of computation needed to run experiments as compared to a fine-tuning approach where the BERT model is a component of the full regression model.

If we look into cases where the model did not perform well, we find that many discrepancies between model and actual score are due to the user misunderstanding or incorrectly using the star-rating system and giving 1-star to a product they really liked or 5-stars to a product they did not like.

Another class of discrepancies is due to the model misinterpreting sarcasm. For example, the following 5-star review obtains a low predicted score:
* Summary: "BEWARE - May cause your dog to ignore you."
* Review: "If you want your dog to ignore you for a while, this item (with the treat holder)is an ideal solution!I ..."
And the following 1-star review obtains a high predicted score:
* Summary: "Love getting expired food!"
* Review: "I love getting a case of cereal in Oct, 2012 when the product expired in JUNE OF 2012! ..."

And another common cause of discrepancies is when the customer provides a mostly positive text review but then gives a 1-star rating due to price. This last class of error might be addressed by fine-tuning the BERT model since the words "price" or "expensive" often occur in the summary.

## Files
* `extract_features.py`, `modeling.py` and `tokenization.py` are used to pre-compute the feature vectors. The latter two files are exact copies from https://github.com/google-research/bert and `extract_features.py` is a copy that has been modified to output only the sentence embeddings, and save to a TFRecord format rather than JSON.
* `Preprocessing.ipynb` is a Jupyter notebook used to clean the data, compute the feature vectors, and save the data to TFRecord files.
* `Regression.ipynb` is a Jupyter notebook containing the regression model, training and evaluation code.

## Implementation Notes
* Tested with Python 3.6.7 and TensorFlow 1.12.0
* Feature vector extraction was performed on a Google Cloud TPU using a free account. See https://cloud.google.com/tpu/docs/quickstart
