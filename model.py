import os
import time
from textwrap import wrap

import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU,
    Add,
    Attention,
    Dense,
    Embedding,
    LayerNormalization,
    Reshape,
    StringLookup,
    TextVectorization,
)
from tensorflow.keras.callbacks import EarlyStopping # To prevent overfitting 

vocab_size = 20000  # Vocabulary size for word embedding
attention_dim = 512  # Size of dense layer in attention mechanism
word_embedding_dim = 128  # Dimensionality of word embeddings


feature_extractor = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    include_top=False, weights="imagenet"
)
image_height = 299
image_width = 299
image_channels = 3
feature_shape = (8, 8, 1536)  

BUFFER_SIZE = 1000

def get_image_label(example):
    caption = example["captions"]["text"][0]  
    img = example["image"]
    img = tf.image.resize(img, (image_height, image_width))
    img = img / 255.0  
    return {"image_tensor": img, "caption": caption}

# Loading the COCO Captions dataset
trainds = tfds.load("coco_captions", split="train")

# Preprocessing
trainds = trainds.map(get_image_label, num_parallel_calls=tf.data.AUTOTUNE).shuffle(BUFFER_SIZE).prefetch(buffer_size=tf.data.AUTOTUNE)

# Text preprocessing
def add_start_end_token(data):
    start = tf.convert_to_tensor("<start>")
    end = tf.convert_to_tensor("<end>")
    data["caption"] = tf.strings.join([start, data["caption"], end], separator=" ")
    return data

trainds = trainds.map(add_start_end_token)

max_len = 64  # Maximum length of captions

# Function for text preprocessing
def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs, r"[!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{|}~]", "")

# Tokenization
tokenizer = TextVectorization(max_tokens=vocab_size, standardize=standardize, output_sequence_length=max_len)
tokenizer.adapt(trainds.map(lambda x: x["caption"]))

# Word to index and index to word
word_to_index = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary())
index_to_word = StringLookup(mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True)

BATCH_SIZE = 32

def create_ds_fn(data):
    img_tensor = data["image_tensor"]
    caption = tokenizer(data["caption"])
    target = tf.roll(caption, -1, 0)
    zeros = tf.zeros([1], dtype=tf.int64)
    target = tf.concat((target[:-1], zeros), axis=-1)
    return (img_tensor, caption), target

batched_ds = (
    trainds.map(create_ds_fn)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# Image Encoder
feature_extractor.trainable = False
image_input = Input(shape=(image_height, image_width, image_channels))  
image_features = feature_extractor(image_input)
x = Reshape((feature_shape[0] * feature_shape[1], feature_shape[2]))(image_features)
encoder_output = Dense(attention_dim, activation="relu")(x)
encoder = tf.keras.Model(inputs=image_input, outputs=encoder_output)

# Caption Decoder
word_input = Input(shape=(max_len,), name="words")  
embed_x = Embedding(vocab_size, attention_dim)(word_input)
decoder_gru = GRU(attention_dim, return_sequences=True, return_state=True)
gru_output, gru_state = decoder_gru(embed_x)

# Attention mechanism
decoder_attention = Attention()
context_vector = decoder_attention([gru_output, encoder_output])
addition = Add()([gru_output, context_vector])
layer_norm = LayerNormalization(axis=-1)
layer_norm_out = layer_norm(addition)
decoder_output_dense = Dense(vocab_size)
decoder_output = decoder_output_dense(layer_norm_out)

decoder = tf.keras.Model(inputs=[word_input, encoder_output], outputs=decoder_output)

# Training the model
image_caption_train_model = tf.keras.Model(inputs=[image_input, word_input], outputs=decoder_output)

# Loss function
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    mask = tf.cast(mask, dtype=tf.int32)
    sentence_len = tf.reduce_sum(mask)
    loss_ = loss_[:sentence_len]
    return tf.reduce_mean(loss_, 1)

image_caption_train_model.compile(optimizer="adam", loss=loss_function)

# Model path for saving
model_path = "image_caption_model.h5"

# To check if the model already exists
if os.path.exists(model_path):
    image_caption_train_model = tf.keras.models.load_model(model_path)
else:
    
    early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    
    # Training loop
    history = image_caption_train_model.fit(batched_ds, epochs=10, callbacks=[early_stopping])
    
    # To save the model
    image_caption_train_model.save(model_path)

# Model for caption generation
gru_state_input = Input(shape=(attention_dim,), name="gru_state_input")  
gru_output, gru_state = decoder_gru(embed_x, initial_state=gru_state_input)
context_vector = decoder_attention([gru_output, encoder_output])
addition_output = Add()([gru_output, context_vector])
layer_norm_output = layer_norm(addition_output)
decoder_output = decoder_output_dense(layer_norm_output)

decoder_pred_model = tf.keras.Model(inputs=[word_input, gru_state_input, encoder_output], outputs=[decoder_output, gru_state])

min_seq_len = 5  # Minimum length of generated caption
def predict_caption(filename):
    gru_state = tf.zeros((1, attention_dim))
    img = tf.image.decode_jpeg(tf.io.read_file(filename), channels=image_channels)
    img = tf.image.resize(img, (image_height, image_width))
    img = img / 255.0
    features = encoder(tf.expand_dims(img, axis=0))
    dec_input = tf.expand_dims([word_to_index("<start>")], 1)
    result = []
    for i in range(max_len):
        predictions, gru_state = decoder_pred_model([dec_input, gru_state, features])
        top_probs, top_idxs = tf.math.top_k(predictions[0][0], k=10, sorted=False)
        chosen_id = tf.random.categorical([top_probs], 1)[0].numpy()
        predicted_id = top_idxs.numpy()[chosen_id][0]
        result.append(tokenizer.get_vocabulary()[predicted_id])
        if predicted_id == word_to_index("<end>"):
            return img, result
        dec_input = tf.expand_dims([predicted_id], 1)
    return img, result

# Testing
filename = "running-dogs.jpg"  
for i in range(2):
    image, caption = predict_caption(filename)
    print(" ".join(caption[:-1]) + ".")
    
plt.imshow(image)
plt.axis("off")