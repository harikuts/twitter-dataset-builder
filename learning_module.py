import itertools
import tensorflow as tf
import argparse
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Required for server job compatibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from tensorflow.compat.v1.nn.rnn_cell import CoupledInputForgetGateLSTMCell
def load_corpus(filenames):
    corpus = []
    for filename in filenames:
        with io.open(filename, 'r', encoding="utf-8") as f:
            for line in f.readlines():
                corpus.append(line.split())
    return corpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", "-c", required=True, action="store", \
        help = "Corpus of tweets to load.", nargs='+')
    parser.add_argument("--epochs", "-e", default=20, type=int, action="store", \
        help = "Number of epochs to train.")
    parser.add_argument("--seq", "-s", default=1, type=int, action="store", \
        help = "Number of previous words to predict the next word.")
    args = parser.parse_args()

    if args.seq < 1 or args.epochs < 1:
        print("Both SEQ and EPOCHS must be set to values above 1.")

    SEQ_LEN = args.seq

    corpus = load_corpus(args.corpus)
    
    # # Tokenizer time!
    all_words = list(itertools.chain.from_iterable(corpus))

    # # Tokenize
    # tokenizer = tf.keras.preprocessing.text.Tokenizer()
    # tokenizer.fit_on_texts([data])
    # reverse_word_map = dict(map(reversed, tokenizer.word_index.items()))
    # vocab_size = len(tokenizer.word_index) + 1

    unique_words = np.unique(all_words)
    unique_word_index = dict((c, i) for i, c in enumerate(list(unique_words)))
    index_word_map = dict((i, c) for i, c in enumerate(list(unique_words)))

    # Assemble sequences
    # sequences = []
    # for tweet in corpus:
    #     # Process it, I guess
    #     sequence_data = tokenizer.texts_to_sequences([tweet])[0]
    #     # Compile sequences
    #     if len(sequence_data) >= SEQ_LEN+1:
    #         for i in range(SEQ_LEN, len(sequence_data)):
    #             words = sequence_data[i-SEQ_LEN:i+1]
    #             sequences.append(words)
    #             print("Sequence added:", words, ' '.join([index_word_map[word] for word in words]))
    # sequences = np.array(sequences)

    sequences = []
    for phrase in corpus:
        if len(phrase) >= SEQ_LEN+1:
            for i in range(SEQ_LEN, len(phrase)):
                sequence = phrase[i-SEQ_LEN:i+1]
                sequences.append(sequence)

    # # Set x and y
    # X = []
    # y = []
    # for i in sequences:
    #     X.append(i[:SEQ_LEN])
    #     y.append(i[-1])
    # X = np.array(X)
    # y = tf.keras.utils.to_categorical((np.array(y)), num_classes=vocab_size)
    # print(X[-10:])
    # print(y[-10:])

    # Set X and y with manual OHE
    X = np.zeros((len(sequences), SEQ_LEN, len(unique_words)), dtype=bool)
    y = np.zeros((len(sequences), len(unique_words)), dtype=bool)
    for i, sequence in enumerate(sequences):
        prev_words = sequence[:-1]
        next_word = sequence[-1]
        # print(prev_words, next_word)
        for j, prev_word in enumerate(prev_words):
            X[i, j, unique_word_index[prev_word]] = 1
        y[i, unique_word_index[next_word]] = 1
    # print(X[-10:])
    # print(y[-10:])

    # CREATE MODEL

    def get_LSTM(vocab_size):
        model = tf.keras.Sequential()
        # model.add(tf.keras.layers.Embedding(vocab_size, 10, input_length=SEQ_LEN))
        # model.add(tf.keras.layers.LSTM(64, return_sequences=True))
        # model.add(tf.keras.layers.LSTM(128))
        # model.add(tf.keras.layers.Dense(256, activation="relu"))
        # model.add(tf.keras.layers.LSTM(128, input_shape=(SEQ_LEN, vocab_size), return_sequences=True))
        # model.add(tf.keras.layers.Dense(128, input_shape=(SEQ_LEN, vocab_size)))
        # model.add(tf.keras.layers.Dropout(0.5))
        # model.add(tf.keras.layers.LSTM(256, kernel_regularizer))
        model.add(tf.keras.layers.LSTM(512, input_shape=(SEQ_LEN, vocab_size), kernel_regularizer=tf.keras.regularizers.l2(0.01), bias_regularizer=tf.keras.regularizers.l2(0.001)))
        # model.add(tf.keras.layers.LSTM(512, input_shape=(SEQ_LEN, vocab_size)))
        # model.add(CoupledInputForgetGateLSTMCell(670, input_shape=(SEQ_LEN, vocab_size)))
        # model.add(tf.keras.layers.Dense(128))
        # model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(vocab_size, activation="softmax"))
        return model

    model = get_LSTM(len(unique_words))
    print(model.summary)
    model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.RMSprop(lr=0.001), metrics=['accuracy'])
    run = model.fit(X, y, epochs=args.epochs, validation_split=0.25, batch_size=128, shuffle=True)
    history = run.history
    print(history.keys())
    # import pdb
    # pdb.set_trace()
    try:
        plt.plot(history['accuracy'])
    except KeyError:
        plt.plot(history['acc'])
    try:
        plt.plot(history['val_accuracy'])
    except KeyError:
        plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("accuracy.png")
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show("loss.png")