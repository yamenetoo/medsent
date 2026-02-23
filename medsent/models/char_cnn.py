import tensorflow as tf
from tensorflow.keras import layers, models

def build_char_cnn(vocab_size, max_len, embedding_dim=50, num_filters=100, kernel_sizes=[3,4,5], num_classes=2):
    inp = layers.Input(shape=(max_len,))
    x = layers.Embedding(vocab_size, embedding_dim)(inp)
    x = layers.Reshape((max_len, embedding_dim, 1))(x)
    conv_blocks = []
    for k in kernel_sizes:
        conv = layers.Conv2D(num_filters, (k, embedding_dim), activation='relu')(x)
        conv = layers.MaxPooling2D((max_len - k + 1, 1))(conv)
        conv = layers.Flatten()(conv)
        conv_blocks.append(conv)
    x = layers.Concatenate()(conv_blocks)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model