---
layout: post
title: "cnn-tips-and-tricks"
date: 2018-12-23
tags: CNN Attention
---




## Max-Pooling VS Avg-Pooling
	- Max pooling helps to extract imp information from the spaical region of path. For example if image patch is of size 3X3 then max pooling will give 1 pixel having max intenity that tell about edge.
	- Avg-Pooling mix up everything


> In imagine recognition, pooling also provides basic invariance to translating (shifting) and rotation. When you are pooling over a region, the output will stay approximately the same even if you shift/rotate the image by a few pixels, because the max operations will pick out the same value regardless.

## In sequence data-set, while using LSTM, the output at each time-step can be helpful in further prediction, which means the output at each roll-out of LSTM, can be imp then taking Max-Pool on all those value can be concatenate among the feature for further prediction




lstm1, state_h, state_c = LSTM(1, return_state = True)(inputs1) 


    - The LSTM hidden state output for the last time step.
    - The LSTM hidden state output for the last time step (again).
    - The LSTM cell state for the last time step.


 LSTM(1, return_sequences=True)(inputs1)

 	- return output at ecah roll-out, Helpful, when stacking another LSTM layer on top of first one.




## Attention

## text Classification with various models[https://ahmedbesbes.com/overview-and-benchmark-of-traditional-and-deep-learning-models-in-text-classification.html]

Hierarchical Attention Networks 

https://github.com/Hsankesara/DeepResearch/tree/master/Hierarchical_Attention_Network

https://medium.com/analytics-vidhya/hierarchical-attention-networks-d220318cf87e


embedding_layer = Embedding(len(word_index) + 1,embed_size,weights=[embedding_matrix], input_length=max_senten_len, trainable=False)

# Words level attention model
word_input = Input(shape=(max_senten_len,), dtype='float32')
word_sequences = embedding_layer(word_input)
word_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(word_sequences)
word_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(word_lstm)
word_att = AttentionWithContext()(word_dense)
wordEncoder = Model(word_input, word_att)

# Sentence level attention model
sent_input = Input(shape=(max_senten_num, max_senten_len), dtype='float32')
sent_encoder = TimeDistributed(wordEncoder)(sent_input)
sent_lstm = Bidirectional(LSTM(150, return_sequences=True, kernel_regularizer=l2_reg))(sent_encoder)
sent_dense = TimeDistributed(Dense(200, kernel_regularizer=l2_reg))(sent_lstm)
sent_att = Dropout(0.5)(AttentionWithContext()(sent_dense))
preds = Dense(30, activation='softmax')(sent_att)
model = Model(sent_input, preds)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])