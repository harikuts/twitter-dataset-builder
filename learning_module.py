import itertools
import tensorflow as tf
import argparse
import io
import numpy as np

SEQ_LEN = 2

def load_corpus(filename):
    corpus = []
    with io.open(filename, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            corpus.append(line.split())
    return corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", "-c", required=True, action="store", \
        help = "Corpus of tweets to load.")
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    
    # Tokenizer time!
    data = ' '.join(list(itertools.chain.from_iterable(corpus)))

    # Tokenize
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts([data])
    reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    vocab_size = len(tokenizer.word_index) + 1

    # Assemble sequences
    sequences = []
    for tweet in corpus:
        # Process it, I guess
        sequence_data = tokenizer.texts_to_sequences([tweet])[0]
        # Compile sequences
        if len(sequence_data) >= SEQ_LEN+1:
            for i in range(SEQ_LEN, len(sequence_data)):
                words = sequence_data[i-SEQ_LEN:i+1]
                sequences.append(words)
                print("Sequence added:", words, ' '.join([reverse_word_map[word] for word in words]))
    sequences = np.array(sequences)

    # Set x and y
    X = []
    y = []
    for i in sequences:
        X.append(i[0])
        y.append(i[1])
    X = np.array(X)
    y = tf.keras.utils.to_categorical((np.array(y)), num_classes=vocab_size)
    print(X[:10])
    print(y[:10])

    # CREATE MODEL

    def get_LSTM(vocab_size):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Embedding(vocab_size, 10, input_length=1))
        model.add(tf.keras.layers.LSTM(1024, return_sequences=True))
        model.add(tf.keras.layers.LSTM(1024))
        model.add(tf.keras.layers.Dense(2048, activation="relu"))
        model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
        return model

    # Callbacks, but I don't think they're required but they might help?
    # from tensorflow.keras.callbacks import ModelCheckpoint
    # from tensorflow.keras.callbacks import ReduceLROnPlateau
    # from tensorflow.keras.callbacks import TensorBoard
    # checkpoint = ModelCheckpoint("nextword1.h5", monitor='loss', verbose=1,
    #     save_best_only=True, mode='auto')
    # reduce = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=3, min_lr=0.0001, verbose = 1)
    # logdir='logsnextword1'
    # tensorboard_Visualization = TensorBoard(log_dir=logdir)

    model = get_LSTM(vocab_size)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    model.fit(X, y, epochs=150, batch_size=64)#, callbacks=[checkpoint, reduce, tensorboard_Visualization])