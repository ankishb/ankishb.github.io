
import mysql.connector
from mysql.connector import Error
import tweepy
import json
from dateutil import parser
import time
import os
import subprocess

#importing file which sets env variable
subprocess.call("./settings.sh", shell = True)


consumer_key    = os.environ['CONSUMER_KEY']
consumer_secret = os.environ['CONSUMER_SECRET']
access_token    = os.environ['ACCESS_TOKEN']
access_token_secret = os.environ['ACCESS_TOKEN_SECRET']
password        = os.environ['PASSWORD']


#def connect(username, created_at, tweet, retweet_count, place , location):
def connect(data_dict):
    """
    connect to MySQL database and insert twitter data
    """
    try:
        con = mysql.connector.connect(
            host = 'localhost',
            database ='twitterdb', 
            user ='root', 
            password = password, 
            charset = 'utf8'
            )
        

        if con.is_connected():
            """ Insert twitter data """
            cursor = con.cursor()
            ############### better approach ######################
            """ data_dict is defined as follows
            data_dict = {
                'username' : username,
                'created_at':created_at,
                'tweet':tweet,
                'retweet_count':retweet_count,
                'place':place,
                'location':location
            }
            """
            add_tweet_better = ("INSERT INTO Golf "
                         "(username, created_at, tweet, retweet_count, place, location) "
                         "VALUES (%(username)s, %(created_at)s, %(tweet)s, %(retweet_count)s, %(place)s, %(location)s)")

            # Insert new employee
            cursor.execute(add_tweet_better, data_dict)
            
            cursor.execute(add_salary, data_salary)
            ############### better approach end ###################
            
            ############### Lazy approach end ###################
            add_tweet_lazy = "INSERT INTO Golf "
                    "(username, created_at, tweet, retweet_count,place, location)"
                    "VALUES (%s, %s, %s, %s, %s, %s)"

            data_lazy = (data_dict['username'], data_dict['created_at'], data_dict['tweet'], data_dict['retweet_count'], data_dict['place'], data_dict['location'])

            
            cursor.execute(add_tweet_lazy, data_lazy)
            ############### Lazy approach end ###################
            
            con.commit()


    except Error as e:
        print(e)

    cursor.close()
    con.close()

    return


# Tweepy class to access Twitter API
class Streamlistener(tweepy.StreamListener):
    

    def on_connect(self):
        print("You are connected to the Twitter API")


    def on_error(self):
        if status_code != 200:
            print("error found")
            # returning false disconnects the stream
            return False

    """
    This method reads in tweet data as Json
    and extracts the data we want.
    """
    def on_data(self,data):
        
        try:
            raw_data = json.loads(data)

            if 'text' in raw_data:
                data_dict = {}
                data_dict['username'] = raw_data['user']['screen_name']
                data_dict['created_at'] = parser.parse(raw_data['created_at'])
                data_dict['tweet'] = raw_data['text']
                data_dict['retweet_count'] = raw_data['retweet_count']

                if raw_data['place'] is not None:
                    place = raw_data['place']['country']
                    print(place)
                else:
                    place = None
                data_dict['place'] = place

                data_dict['location'] = raw_data['user']['location']

                #insert data just collected into MySQL database
                #connect(username, created_at, tweet, retweet_count, place, location)
                connect(data_dict)
                print("Tweet colleted at: {} ".format(str(created_at)))
        except Error as e:
            print(e)


if __name__== '__main__':

    # # #Allow user input
    # track = []
    # while True:

    #   input1  = input("what do you want to collect tweets on?: ")
    #   track.append(input1)

    #   input2 = input("Do you wish to enter another word? y/n ")
    #   if input2 == 'n' or input2 == 'N':
    #       break
    
    # print("You want to search for {}".format(track))
    # print("Initialising Connection to Twitter API....")
    # time.sleep(2)

    # authentification so we can access twitter
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)

    # create instance of Streamlistener
    listener = Streamlistener(api = api)
    stream = tweepy.Stream(auth, listener = listener)

    track = ['golf', 'masters', 'reed', 'mcilroy', 'woods']
    #track = ['nba', 'cavs', 'celtics', 'basketball']
    # choose what we want to filter by
    stream.filter(track = track, languages = ['en'])









import mysql.connector 
from mysql.connector import Error
import os
import re
import pandas as pd 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from wordcloud import WordCloud, STOPWORDS
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob



class TweetObject():


    def __init__(self, host, database, user):
        self.password = os.environ['PASSWORD']
        self.host = host
        self.database = database
        self.user = user
        


    def MySQLConnect(self,query):
        """
        Connects to database and extracts
        raw tweets and any other columns we
        need
        Parameters:
        ----------------
        arg1: string: SQL query
        Returns: Pandas Dataframe
        ----------------
        """

        try:
            con = mysql.connector.connect(host = self.host, database = self.database, \
                user = self.user, password = self.password, charset = 'utf8')

            if con.is_connected():
                print("Successfully connected to database")

                cursor = con.cursor()
                query = query
                cursor.execute(query)

                data = cursor.fetchall()
                # store in dataframe
                df = pd.DataFrame(data,columns = ['date', 'tweet'])



        except Error as e:
            print(e)
        
        cursor.close()
        con.close()

        return df



    def clean_tweets(self, df):
    
        """
        Takes raw tweets and cleans them
        so we can carry out analysis
        remove stopwords, punctuation,
        lower case, html, emoticons.
        This will be done using Regex
        ? means option so colou?r matches
        both color and colour.
        """

        # Do some text preprocessing
        stopword_list = stopwords.words('english')
        ps = PorterStemmer()
        df["clean_tweets"] = None
        df['len'] = None
        for i in range(0,len(df['tweet'])):
            # get rid of anythin that isnt a letter

            exclusion_list = ['[^a-zA-Z]','rt', 'http', 'co', 'RT']
            exclusions = '|'.join(exclusion_list)
            text = re.sub(exclusions, ' ' , df['tweet'][i])
            text = text.lower()
            words = text.split()
            words = [word for word in words if not word in stopword_list]
             # only use stem of word
            #words = [ps.stem(word) for word in words]
            df['clean_tweets'][i] = ' '.join(words)


        # Create column with data length
        df['len'] = np.array([len(tweet) for tweet in data["clean_tweets"]])
            


        return df



    def sentiment(self, tweet):
        """
        This function calculates sentiment
        on our cleaned tweets.
        Uses textblob to calculate polarity.
        Parameters:
        ----------------
        arg1: takes in a tweet (row of dataframe)
        """

        # need to improce
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 1
        elif analysis.sentiment.polarity == 0:
            return 0
        else:
            return -1




    def save_to_csv(self, df):
        """
        Save cleaned data to a csv for further
        analysis.
        Parameters:
        ----------------
        arg1: Pandas dataframe
        """
        try:
            df.to_csv("clean_tweets.csv")
            print("\n")
            print("csv successfully saved. \n")

        
        except Error as e:
            print(e)
        



    def word_cloud(self, df):
        plt.subplots(figsize = (12,10))
        wordcloud = WordCloud(
                background_color = 'white',
                width = 1000,
                height = 800).generate(" ".join(df['clean_tweets']))
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()





if __name__ == '__main__':

    t = TweetObject( host = 'localhost', database = 'twitterdb', user = 'root')

    data  = t.MySQLConnect("SELECT created_at, tweet FROM `TwitterDB`.`Golf`;")
    data = t.clean_tweets(data)
    data['Sentiment'] = np.array([t.sentiment(x) for x in data['clean_tweets']])
    t.word_cloud(data)
    t.save_to_csv(data)
    
    pos_tweets = [tweet for index, tweet in enumerate(data["clean_tweets"]) if data["Sentiment"][index] > 0]
    neg_tweets = [tweet for index, tweet in enumerate(data["clean_tweets"]) if data["Sentiment"][index] < 0]
    neu_tweets = [tweet for index, tweet in enumerate(data["clean_tweets"]) if data["Sentiment"][index] == 0]

    #Print results
    print("percentage of positive tweets: {}%".format(100*(len(pos_tweets)/len(data['clean_tweets']))))
    print("percentage of negative tweets: {}%".format(100*(len(neg_tweets)/len(data['clean_tweets']))))
    print("percentage of neutral tweets: {}%".format(100*(len(neu_tweets)/len(data['clean_tweets']))))
view raw
part9_Twitter.py hosted with ❤ by GitHub




#########################
import random, os, sys
import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
import tensorflow as tf
from keras.engine.topology import Layer

try:
    from dataloader import TokenList, pad_to_longest
    # for transformer
except: pass

class LayerNormalization(Layer):
    def __init__(self, eps=1e-6, **kwargs):
        self.eps = eps
        super(LayerNormalization, self).__init__(**kwargs)
    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer=Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer=Zeros(), trainable=True)
        super(LayerNormalization, self).build(input_shape)
    def call(self, x):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
    def compute_output_shape(self, input_shape):
        return input_shape

class ScaledDotProductAttention():
    def __init__(self, d_model, attn_dropout=0.1):
        self.temper = np.sqrt(d_model)
        self.dropout = Dropout(attn_dropout)
    def __call__(self, q, k, v, mask):
        attn = Lambda(lambda x:K.batch_dot(x[0],x[1],axes=[2,2])/self.temper)([q, k])
        if mask is not None:
            mmask = Lambda(lambda x:(-1e+10)*(1-x))(mask)
            attn = Add()([attn, mmask])
        attn = Activation('softmax')(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x:K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn

class MultiHeadAttention():
    # mode 0 - big martixes, faster; mode 1 - more clear implementation
    def __init__(self, n_head, d_model, d_k, d_v, dropout, mode=0, use_norm=True):
        self.mode = mode
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.dropout = dropout
        if mode == 0:
            self.qs_layer = Dense(n_head*d_k, use_bias=False)
            self.ks_layer = Dense(n_head*d_k, use_bias=False)
            self.vs_layer = Dense(n_head*d_v, use_bias=False)
        elif mode == 1:
            self.qs_layers = []
            self.ks_layers = []
            self.vs_layers = []
            for _ in range(n_head):
                self.qs_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.ks_layers.append(TimeDistributed(Dense(d_k, use_bias=False)))
                self.vs_layers.append(TimeDistributed(Dense(d_v, use_bias=False)))
        self.attention = ScaledDotProductAttention(d_model)
        self.layer_norm = LayerNormalization() if use_norm else None
        self.w_o = TimeDistributed(Dense(d_model))

    def __call__(self, q, k, v, mask=None):
        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        if self.mode == 0:
            qs = self.qs_layer(q)  # [batch_size, len_q, n_head*d_k]
            ks = self.ks_layer(k)
            vs = self.vs_layer(v)

            def reshape1(x):
                s = tf.shape(x)   # [batch_size, len_q, n_head * d_k]
                x = tf.reshape(x, [s[0], s[1], n_head, d_k])
                x = tf.transpose(x, [2, 0, 1, 3])  
                x = tf.reshape(x, [-1, s[1], d_k])  # [n_head * batch_size, len_q, d_k]
                return x
            qs = Lambda(reshape1)(qs)
            ks = Lambda(reshape1)(ks)
            vs = Lambda(reshape1)(vs)

            if mask is not None:
                mask = Lambda(lambda x:K.repeat_elements(x, n_head, 0))(mask)
            head, attn = self.attention(qs, ks, vs, mask=mask)  
                
            def reshape2(x):
                s = tf.shape(x)   # [n_head * batch_size, len_v, d_v]
                x = tf.reshape(x, [n_head, -1, s[1], s[2]]) 
                x = tf.transpose(x, [1, 2, 0, 3])
                x = tf.reshape(x, [-1, s[1], n_head*d_v])  # [batch_size, len_v, n_head * d_v]
                return x
            head = Lambda(reshape2)(head)
        elif self.mode == 1:
            heads = []; attns = []
            for i in range(n_head):
                qs = self.qs_layers[i](q)   
                ks = self.ks_layers[i](k) 
                vs = self.vs_layers[i](v) 
                head, attn = self.attention(qs, ks, vs, mask)
                heads.append(head); attns.append(attn)
            head = Concatenate()(heads) if n_head > 1 else heads[0]
            attn = Concatenate()(attns) if n_head > 1 else attns[0]

        outputs = self.w_o(head)
        outputs = Dropout(self.dropout)(outputs)
        if not self.layer_norm: return outputs, attn
        # outputs = Add()([outputs, q]) # sl: fix
        return self.layer_norm(outputs), attn




class PositionwiseFeedForward():
    def __init__(self, d_hid, d_inner_hid, dropout=0.1):
        self.w_1 = Conv1D(d_inner_hid, 1, activation='relu')
        self.w_2 = Conv1D(d_hid, 1)
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout)
    def __call__(self, x):
        output = self.w_1(x) 
        output = self.w_2(output)
        output = self.dropout(output)
        output = Add()([output, x])
        return self.layer_norm(output)

class EncoderLayer():
    def __init__(self, d_model, d_inner_hid, n_head, d_k, d_v, dropout=0.1):
        self.self_att_layer = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn_layer  = PositionwiseFeedForward(d_model, d_inner_hid, dropout=dropout)
    def __call__(self, enc_input, mask=None):
        output, slf_attn = self.self_att_layer(enc_input, enc_input, enc_input, mask=mask)
        output = self.pos_ffn_layer(output)
        return output, slf_attn


def GetPosEncodingMatrix(max_len, d_emb):
    pos_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)] 
        if pos != 0 else np.zeros(d_emb) 
            for pos in range(max_len)
            ])
    pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2]) # dim 2i
    pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2]) # dim 2i+1
    return pos_enc

def GetPadMask(q, k):
    ones = K.expand_dims(K.ones_like(q, 'float32'), -1)
    mask = K.cast(K.expand_dims(K.not_equal(k, 0), 1), 'float32')
    mask = K.batch_dot(ones, mask, axes=[2,1])
    return mask

def GetSubMask(s):
    len_s = tf.shape(s)[1]
    bs = tf.shape(s)[:1]
    mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
    return mask

class Transformer():
    def __init__(self, len_limit, embedding_matrix, d_model=embed_size, \
              d_inner_hid=512, n_head=10, d_k=64, d_v=64, layers=2, dropout=0.1, \
              share_word_emb=False, **kwargs):
        self.name = 'Transformer'
        self.len_limit = len_limit
        self.src_loc_info = False # True # sl: fix later
        self.d_model = d_model
        self.decode_model = None
        d_emb = d_model

        pos_emb = Embedding(len_limit, d_emb, trainable=False, \
                            weights=[GetPosEncodingMatrix(len_limit, d_emb)])

        i_word_emb = Embedding(max_features, d_emb, weights=[embedding_matrix]) # Add Kaggle provided embedding here

        self.encoder = Encoder(d_model, d_inner_hid, n_head, d_k, d_v, layers, dropout, \
                               word_emb=i_word_emb, pos_emb=pos_emb)

        
    def get_pos_seq(self, x):
        mask = K.cast(K.not_equal(x, 0), 'int32')
        pos = K.cumsum(K.ones_like(x, 'int32'), 1)
        return pos * mask

    def compile(self, active_layers=999):
        src_seq_input = Input(shape=(None, ))
        x = Embedding(max_features, embed_size, weights=[embedding_matrix])(src_seq_input)
        
        # LSTM before attention layers
        x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x) 
        
        x, slf_attn = MultiHeadAttention(n_head=3, d_model=300, d_k=64, d_v=64, dropout=0.1)(x, x, x)
        
        avg_pool = GlobalAveragePooling1D()(x)
        max_pool = GlobalMaxPooling1D()(x)
        conc = concatenate([avg_pool, max_pool])
        conc = Dense(64, activation="relu")(conc)
        x = Dense(1, activation="sigmoid")(conc)   
        
        
        self.model = Model(inputs=src_seq_input, outputs=x)
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

s2s = Transformer(64, embedding_matrix, layers=1)
s2s.compile()
model = s2s.model
model.summary()










######## connecting mysql with pandas ########
import pandas as pd

# Create dataframe
data=pd.DataFrame({
    'book_id':[12345,12346,12347],
    'title':['Python Programming','Learn MySQL','Data Science Cookbook'],
    'price':[29,23,27]
})

from sqlalchemy import create_engine

# create sqlalchemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user="root",
                               pw="12345",
                               db="employee"))

# Insert whole DataFrame into MySQL
data.to_sql('book_details', con = engine, if_exists = 'append', chunksize = 1000)
"""
Now let’s take a closer look at what each of these parameters is doing in our code.

book_details 
    It is the name of table into which we want to insert our DataFrame.
con = engine 
    It provides the connection details (recall that we created engine using our authentication details in the previous step).
if_exists = 'append' 
    It checks whether the table we specified already exists or not, and then appends the new data (if it does exist) or creates a new table (if it doesn’t).
chunksize 
    It writes records in batches of a given size at a time. By default, all rows will be written at once.
"""


####### Read data from sql server #######
# Import module
import pymysql

# create connection
connection = pymysql.connect(host='localhost',
                             user='root',
                             password='12345',
                             db='employee')

# Create cursor
my_cursor = connection.cursor()

# Execute Query
my_cursor.execute("SELECT * from employee")

# Fetch the records
result = my_cursor.fetchall()

for i in result:
    print(i)

# Close the connection
connection.close()






import pymysql


try:
    # Connect to the database
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='12345',
        db='employee'
    )


    cursor=connection.cursor()

    # Create a new record
    sql = "INSERT INTO `employee` (`EmployeeID`, `Ename`, `DeptID`, `Salary`, `Dname`, `Dlocation`) VALUES (%s, %s, %s, %s, %s, %s)"
    cursor.execute(sql, (1009,'Morgan',1,4000,'HR','Mumbai'))

    # connection is not autocommit by default. So we must commit to save our changes.
    connection.commit()

    # Execute query
    sql = "SELECT * FROM `employee`"
    cursor.execute(sql)
    # Fetch all the records
    result = cursor.fetchall()
    for i in result:
        print(i)

except Error as e:
    print(e)

finally:
    # close the database connection using close() method.
    connection.close()



######### Insertion row by row #########

# creating column list for insertion
cols = "`,`".join([str(i) for i in data.columns.tolist()])

# Insert DataFrame recrds one by one.
for i,row in data.iterrows():
    sql = "INSERT INTO `book_details` (`" +cols + "`) VALUES (" + "%s,"*(len(row)-1) + "%s)"
    cursor.execute(sql, tuple(row))

    # the connection is not autocommitted by default, so we must commit to save our changes
    connection.commit()