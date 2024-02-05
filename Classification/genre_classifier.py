"""
The aim of this script is to be able to predict the genre of the wav file
that is passed into it.

2 possible approaches to this problem are contrasted, as well as an ensemble of both models.

1. Use the spectogram of the wav file and use a CNN with 2DConv to classify the genre.
2. Use the raw wav file and use a CNN with 1DConv to classify the genre.
3. Ensemble model.
"""