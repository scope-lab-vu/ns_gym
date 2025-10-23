def build_model_tf(num_hidden_layers, state_size):
    """
    build the model for specific number of hidden layers
    """
    import tensorflow as tf
    from tensorflow import keras
    from keras.layers import Input, Dense
    from keras.models import Model
    from tensorflow.keras.optimizers import Adam # type: ignore
    # tensors
    global layer
    state = Input(shape=(state_size,))

    # create num_hidden_layers layers
    for i in range(int(num_hidden_layers)):
        if i == 0:
            layer = keras.layers.Dense(units=64, activation=tf.nn.relu, name='Layer'+str(i))(state)
        else:
            layer = keras.layers.Dense(units=64, activation=tf.nn.relu, name='Layer'+str(i))(layer)


    # outputs
    pi_hat = keras.layers.Dense(units=3, activation=tf.nn.softmax, name='pi_hat')(layer)
    v_hat = keras.layers.Dense(units=3, name='v_hat')(layer)

    # construct the model
    model = Model(inputs=state, outputs=[pi_hat, v_hat])
    model.compile(loss={'pi_hat': 'kl_divergence', 'v_hat': 'mse'}, optimizer=Adam())
    return model
