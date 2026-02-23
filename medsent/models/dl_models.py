import tensorflow as tf
from tensorflow.keras import layers, models

def build_rnn_model(rnn_type, max_words, max_len, embedding_matrix, num_classes,
                    rnn_units=128, dense_units=64, dropout=0.5, lr=0.001, optimizer='adam'):
    inp = layers.Input(shape=(max_len,))
    emb = layers.Embedding(max_words, embedding_matrix.shape[1],
                           weights=[embedding_matrix], trainable=False)(inp)
    if rnn_type == 'lstm':
        rnn = layers.Bidirectional(layers.LSTM(rnn_units, return_sequences=True))
    elif rnn_type == 'gru':
        rnn = layers.Bidirectional(layers.GRU(rnn_units, return_sequences=True))
    else:  # 'rnn'
        rnn = layers.Bidirectional(layers.SimpleRNN(rnn_units, return_sequences=True))
    x = rnn(emb)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(dense_units, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inp, outputs=out)

    if optimizer == 'adam':
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate=lr)
    elif optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_deep_model(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=64, patience=3):
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                   restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )
    return model, history