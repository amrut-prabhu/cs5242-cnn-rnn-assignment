from nn.layers import *
from nn.model import Model
from nn.initializers import *

def MySentimentNet(word_to_idx):
    """Construct a RNN model for sentiment analysis

    # Arguments:
        word_to_idx: A dictionary giving the vocabulary. It contains V entries,
            and maps each string to a unique integer in the range [0, V).
    # Returns
        model: the constructed model
    """
    vocab_size = len(word_to_idx)
    
    embedding = 500
    units = 70
    
    model = Model()
    model.add(Linear2D(vocab_size, embedding, name='embedding', initializer=Gaussian(std=0.01)))
    model.add(BiRNN(in_features=embedding, units=units, initializer=Gaussian(std=0.01)))
    model.add(Linear2D(2*units, 50, name='linear1', initializer=Gaussian(std=0.01)))
    model.add(TemporalPooling()) # defined in layers.py
    model.add(Linear2D(50, 2, name='linear2', initializer=Gaussian(std=0.01)))
    
    return model