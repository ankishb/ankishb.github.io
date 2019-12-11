---
layout: post
title: "text-classification"
date: 2019-08-29
tag: deep-learning text
---

    In this post, we look into various method of text classification, using classical approach, deep learning, graph based method. The purpose of this post is to point out few important points and give a brief explanation of them.

## Outline :
1. Text preprocessing
2. TF-IDF Approach
3. Gensem word2vec approach
4. Pretrained(Glove) embedding
5. Keras model using above embedding matrix
6. Transformer model
7. Tree-based Model
8. Doc2Vec Model


## Important point to search for
1. Why tokenization of `gensim` and `keras` are so different (mainly keras has produced a very long word dict, why??)
2. Use `yield` for big corpus
3. Clean Webpage text using Beautiful soup


```bash

#!/bin/bash
clear
echo "Hello friend"
read -p "what's your name: " name
echo "$name" >> name_list.txt

clear
echo -p "Hello $name \nYour name is added in my known person list"
echo "====================="
cat name_list.txt
echo "====================="

echo "GoodBye $name"
sleep 2


chmod +x script.sh
./script.sh
```




```python
BertForSequenceClassification(
  (bert): BertModel(
    (embeddings): BertEmbeddings(
      (word_embeddings): Embedding(28996, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (token_type_embeddings): Embedding(2, 768)
      (LayerNorm): BertLayerNorm()
      (dropout): Dropout(p=0.1)
    )
    (encoder): BertEncoder(
      (layer): ModuleList(
        (0): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (1): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (2): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (3): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (4): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (5): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (6): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (7): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (8): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (9): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (10): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
        (11): BertLayer(
          (attention): BertAttention(
            (self): BertSelfAttention(
              (query): Linear(in_features=768, out_features=768, bias=True)
              (key): Linear(in_features=768, out_features=768, bias=True)
              (value): Linear(in_features=768, out_features=768, bias=True)
              (dropout): Dropout(p=0.1)
            )
            (output): BertSelfOutput(
              (dense): Linear(in_features=768, out_features=768, bias=True)
              (LayerNorm): BertLayerNorm()
              (dropout): Dropout(p=0.1)
            )
          )
          (intermediate): BertIntermediate(
            (dense): Linear(in_features=768, out_features=3072, bias=True)
          )
          (output): BertOutput(
            (dense): Linear(in_features=3072, out_features=768, bias=True)
            (LayerNorm): BertLayerNorm()
            (dropout): Dropout(p=0.1)
          )
        )
      )
    )
    (pooler): BertPooler(
      (dense): Linear(in_features=768, out_features=768, bias=True)
      (activation): Tanh()
    )
  )
  (dropout): Dropout(p=0.1)
  (classifier): Linear(in_features=768, out_features=2, bias=True)
)


```



```python

Scaled Dot-product attention

class DotProdSelfAttention(Layer):
    """The self-attention layer as in 'Attention is all you need'.
    paper reference: https://arxiv.org/abs/1706.03762
    
    """
    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(DotProdSelfAttention, self).__init__(*kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(min_ndim=2)
        self.supports_masking = True

    def build(self, input_shape):
        assert len(input_shape) == 3
        input_dim = input_shape[-1]
        # We assume the output-dim of Q, K, V are the same
        self.kernels = dict.fromkeys(['Q', 'K', 'V'])
        for key, _ in self.kernels.items():
            self.kernels[key] = self.add_weight(shape=(input_dim, self.units),
                                                initializer=self.kernel_initializer,
                                                name='kernel_{}'.format(key),
                                                regularizer=self.kernel_regularizer,
                                                constraint=self.kernel_constraint)
        if self.use_bias:
            raise NotImplementedError
        super(DotProdSelfAttention, self).build(input_shape)
        
    def call(self, x):
        Q = K.dot(x, self.kernels['Q'])
        K_mat = K.dot(x, self.kernels['K'])
        V = K.dot(x, self.kernels['V'])
        attention = K.batch_dot(Q, K.permute_dimensions(K_mat, [0, 2, 1]))
        d_k = K.constant(self.units, dtype=K.floatx())
        attention = attention / K.sqrt(d_k)
        attention = K.batch_dot(K.softmax(attention, axis=-1), V)
        return attention
    
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)
      
        
    

The Encoder Block

def encoder(input_tensor):
    """One encoder as in Attention Is All You Need
    """
    # Sub-layer 1
    # Multi-Head Attention
    multiheads = []
    d_v = embed_size // n_heads
    for i in range(n_heads):
        multiheads.append(DotProdSelfAttention(d_v)(input_tensor))
    multiheads = concatenate(multiheads, axis=-1)
    multiheads = Dense(embed_size)(multiheads)
    multiheads = Dropout(0.1)(multiheads)
    
    # Residual Connection
    res_con = add([input_tensor, multiheads])
    # Didn't use layer normalization, use Batch Normalization instead here
    res_con = BatchNormalization(axis=-1)(res_con)
    
    # Sub-layer 2
    # 2 Feed forward layer
    ff1 = Dense(64, activation='relu')(res_con)
    ff2 = Dense(embed_size)(ff1)
    output = add([res_con, ff2])
    output = BatchNormalization(axis=-1)(output)
    
    return output

Positional Encoding

# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/position.py
def positional_signal(hidden_size: int, length: int,
                      min_timescale: float = 1.0, max_timescale: float = 1e4):
    """
    Helper function, constructing basic positional encoding.
    The code is partially based on implementation from Tensor2Tensor library
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """

    if hidden_size % 2 != 0:
        raise ValueError(
            f"The hidden dimension of the model must be divisible by 2."
            f"Currently it is {hidden_size}")
    position = K.arange(0, length, dtype=K.floatx())
    num_timescales = hidden_size // 2
    log_timescale_increment = K.constant(
        (np.log(float(max_timescale) / float(min_timescale)) /
         (num_timescales - 1)),
        dtype=K.floatx())
    inv_timescales = (
            min_timescale *
            K.exp(K.arange(num_timescales, dtype=K.floatx()) *
                  -log_timescale_increment))
    scaled_time = K.expand_dims(position, 1) * K.expand_dims(inv_timescales, 0)
    signal = K.concatenate([K.sin(scaled_time), K.cos(scaled_time)], axis=1)
    return K.expand_dims(signal, axis=0)

# https://github.com/kpot/keras-transformer/blob/master/keras_transformer/position.py
class AddPositionalEncoding(Layer):
    """
    Injects positional encoding signal described in section 3.5 of the original
    paper "Attention is all you need". Also a base class for more complex
    coordinate encoding described in "Universal Transformers".
    """

    def __init__(self, min_timescale: float = 1.0,
                 max_timescale: float = 1.0e4, **kwargs):
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.signal = None
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['min_timescale'] = self.min_timescale
        config['max_timescale'] = self.max_timescale
        return config

    def build(self, input_shape):
        _, length, hidden_size = input_shape
        self.signal = positional_signal(
            hidden_size, length, self.min_timescale, self.max_timescale)
        return super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.signal

Transformer Encoder model

def model_transformer(embedding_matrix, n_encoder=3):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=True)(inp)
    # Add positional encoding
    x = AddPositionalEncoding()(x)
    x = Dropout(0.1)(x)
    for i in range(n_encoder):
        x = encoder(x)
    # These are my own experiments
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    conc = Dense(64, activation="relu")(conc)
    conc = Dropout(0.1)(conc)
    outp = Dense(1, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam',
                  metrics=['accuracy'])
    return model

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
class BaseDataGenerator(Sequence):
    """A data generator"""
    def __init__(self, list_IDs, batch_size=128, shuffle=True):
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """number of steps in one epoch"""
        # Here is the trick
        return len(self.list_IDs) // (self.batch_size * 2**2)

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size: (index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' 
        X = train_X[list_IDs_temp, :]
        y = train_y[list_IDs_temp]
        return X, y

Train and Predict

Here I used early stopping and model checkpoint to load the best_val model

# https://www.kaggle.com/strideradu/word2vec-and-gensim-go-go-go
def train_pred(model, epochs=2):
    # learning schedule callback
#     loss_history = LossHistory()
#     lrate = BatchLRScheduler(step_decay)
#     callbacks_list = [loss_history, lrate]
#     es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
#     model_path = 'keras_models.h5'
#     mc = ModelCheckpoint(filepath=model_path, monitor='val_loss', save_best_only=True)
#     callbacks = [es, mc]
#     train_generator = BaseDataGenerator(list(np.arange(train_X.shape[0])), batch_size=512)
#     model.fit_generator(train_generator,
#                         epochs=epochs,
#                         validation_data=(val_X, val_y),)
#                         callbacks=callbacks)
#     model = load_model(model_path)
    model.fit(train_X, train_y, batch_size=512,
              epochs=epochs,
              validation_data=(val_X, val_y),)

    pred_val_y = model.predict([val_X], batch_size=1024, verbose=0)
    pred_test_y = model.predict([test_X], batch_size=1024, verbose=0)
    return pred_val_y, pred_test_y

Main part: load, train, pred and blend

train_X, val_X, test_X, train_y, val_y, word_index = load_and_prec()
vocab = []
for w,k in word_index.items():
    vocab.append(w)
    if k >= max_features:
        break
embedding_matrix_1 = load_glove(word_index)
# embedding_matrix_2 = load_fasttext(word_index)
embedding_matrix_3 = load_para(word_index)

Train shape :  (1306122, 3)
Test shape :  (56370, 2)

100%|██████████| 222029/222029 [00:00<00:00, 590289.80it/s]

-0.0053247833 0.49346462 para

100%|██████████| 222029/222029 [00:00<00:00, 608759.48it/s]

Create New Embeddings

## Simple average: http://aclweb.org/anthology/N18-2031

# We have presented an argument for averaging as
# a valid meta-embedding technique, and found experimental
# performance to be close to, or in some cases 
# better than that of concatenation, with the
# additional benefit of reduced dimensionality  


## Unweighted DME in https://arxiv.org/pdf/1804.07983.pdf

# “The downside of concatenating embeddings and 
#  giving that as input to an RNN encoder, however,
#  is that the network then quickly becomes inefficient
#  as we combine more and more embeddings.”
  
# embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_2, embedding_matrix_3], axis = 0)
embedding_matrix = np.mean([embedding_matrix_1, embedding_matrix_3], axis = 0)
np.shape(embedding_matrix)

(95000, 300)

Train and Predict
```



```python
def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
    

class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


def model_lstm_atten(embedding_matrix):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = AttentionWithContext()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
x = Attention(maxlen)(x)
x = Dense(64, activation="relu")(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
```


```python


Extra feature part taken from https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb

def parallelize_apply(df,func,colname,num_process,newcolnames):
    # takes as input a df and a function for one of the columns in df
    pool =Pool(processes=num_process)
    arraydata = pool.map(func,tqdm(df[colname].values))
    pool.close()
    
    newdf = pd.DataFrame(arraydata,columns = newcolnames)
    df = pd.concat([df,newdf],axis=1)
    return df

def parallelize_dataframe(df, func):
    df_split = np.array_split(df, 4)
    pool = Pool(4)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df

def count_regexp_occ(regexp="", text=None):
    """ Simple way to get the number of occurence of a regex"""
    return len(re.findall(regexp, text))


# some fetures 
# ['','','india/n','','quora','','sex','','','country/countries','china','','','chinese','','']
def add_features(df):
    df['question_text'] = df['question_text'].progress_apply(lambda x:str(x))
    df["lower_question_text"] = df["question_text"].apply(lambda x: x.lower())
    # df = parallelize_apply(df,sentiment,'question_text',4,['sentiment','subjectivity']) 
    # df['sentiment'] = df['question_text'].progress_apply(lambda x:sentiment(x))
    df['total_length'] = df['question_text'].progress_apply(len)
    df['capitals'] = df['question_text'].progress_apply(lambda comment: sum(1 for c in comment if c.isupper()))
    df['caps_vs_length'] = df.progress_apply(lambda row: float(row['capitals'])/float(row['total_length']),
                                axis=1)
    df['num_words'] = df.question_text.str.count('\S+')
    df['num_unique_words'] = df['question_text'].progress_apply(lambda comment: len(set(w for w in comment.split())))
    df['words_vs_unique'] = df['num_unique_words'] / df['num_words'] 
    return df

def load_and_prec():
    if debug:
        train_df = pd.read_csv("../input/train.csv")[:80000]
        test_df = pd.read_csv("../input/test.csv")[:20000]
    else:
        train_df = pd.read_csv("../input/train.csv")
        test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ###################### Add Features ###############################
    #  https://github.com/wongchunghang/toxic-comment-challenge-lstm/blob/master/toxic_comment_9872_model.ipynb
    '''
    train = add_features(train_df)
    test = add_features(test_df)
    '''
    
    train = parallelize_dataframe(train_df, add_features)
    test = parallelize_dataframe(test_df, add_features)
    
    # lower
    train_df["question_text"] = train_df["question_text"].apply(lambda x: x.lower())
    test_df["question_text"] = test_df["question_text"].apply(lambda x: x.lower())

    # Clean the text
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_text(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_text(x))
    
    # Clean numbers
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: clean_numbers(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: clean_numbers(x))
    
    # Clean speelings
    train_df["question_text"] = train_df["question_text"].progress_apply(lambda x: replace_typical_misspell(x))
    test_df["question_text"] = test_df["question_text"].apply(lambda x: replace_typical_misspell(x))
    
    ## fill up the missing values
    train_X = train_df["question_text"].fillna("_##_").values
    test_X = test_df["question_text"].fillna("_##_").values


    
    features = train[['num_unique_words','words_vs_unique']].fillna(0)
    test_features = test[['num_unique_words','words_vs_unique']].fillna(0)
    
    # doing PCA to reduce network run times
    ss = StandardScaler()
    pc = PCA(n_components=5)
    ss.fit(np.vstack((features, test_features)))
    features = ss.transform(features)
    test_features = ss.transform(test_features)
    
    ###########################################################################

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    
#     # Splitting to training and a final test set    
#     train_X, x_test_f, train_y, y_test_f = train_test_split(list(zip(train_X,features)), train_y, test_size=0.2, random_state=SEED)    
#     train_X, features = zip(*train_X)
#     x_test_f, features_t = zip(*x_test_f)    
    
    #shuffling the data
 
    np.random.seed(SEED)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    features = features[trn_idx]
    
    return train_X, test_X, train_y, features, test_features, tokenizer.word_index
#     return train_X, test_X, train_y, x_test_f,y_test_f,features, test_features, features_t, tokenizer.word_index
#     return train_X, test_X, train_y, tokenizer.word_index

start = time.time()
# fill up the missing values
# x_train, x_test, y_train, word_index = load_and_prec()
x_train, x_test, y_train, features, test_features, word_index = load_and_prec() 
# x_train, x_test, y_train, x_test_f,y_test_f,features, test_features,features_t, word_index = load_and_prec() 
print(time.time()-start)

```
