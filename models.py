from keras.models import Sequential, Model
from keras.layers import Input, LSTM, Concatenate, Dense, BatchNormalization, LeakyReLU
from keras.activations import tanh
from keras.optimizers import Adam

def create_model(config):
    '''Builds an LSTM-MLP model of minimum 2 layers sequentially from a given config dictionary'''
    underlying_history = Input((config['lstm_timesteps'],1))
    bs_vars = Input((config['num_features'],))

    model = Sequential()

    model.add(LSTM(
        units = config['lstm_units'],
        activation = tanh,
        input_shape = (config['lstm_timesteps'], 1),
        return_sequences = True
    ))

    for _ in range(config['lstm_timesteps'] - 2):
        model.add(LSTM(
            units = config['lstm_units'],
            activation = tanh,
            return_sequences = True
        ))
    
    model.add(LSTM(
        units = config['lstm_units'],
        activation = tanh,
        return_sequences = False
    ))

    layers = Concatenate()([bs_vars, model(underlying_history)])
    
    for _ in range(config['mlp_layers'] - 1):
        layers = Dense(config['mlp_units'])(layers)
        layers = BatchNormalization(momentum=config['bn_momentum'])(layers)
        layers = LeakyReLU()(layers)

    #create the output layer
    layers = Dense(1, activation='relu')(layers)
    
    #compile the model
    model = Model(inputs=[underlying_history, bs_vars], outputs=layers)
    model.compile(loss='mse', optimizer=Adam(lr=config['learning_rate']))

    return model