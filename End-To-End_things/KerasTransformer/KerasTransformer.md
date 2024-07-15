>> Class of transformer using keras

Based on this documentation ---> https://arxiv.org/pdf/1706.03762

```ruby
class Transformer():
    def __init__(self, hparams=None, **kwargs):
        super().__init__(**kwargs)
        #hyper params
        self.voc_size = hparams["voc_size"]
        self.max_query_length = hparams["max_query_length"]
        ... #?
        #input
        self.input_k_v = ks.Input(shape=(2,)) #key-value pairs
        self.input_q = ks.Input(shape=(self.max_query_length,)) #query
        #output (shifted to right input)
        self.output_k_v = ks.Input(shape=(2,))
        self.output_q = ks.Input(shape=(self.max_query_length,))
        #embedded input
        self.input_k_v_embedded = self.embedding(self.input_k_v)
        self.input_q_embedded = self.embedding(self.input_q)
        #embedded output
        self.output_k_v_embedded = self.embedding(self.output_k_v)
        self.output_q_embedded = self.embedding(self.output_q)
    def regularization_lstm(self, layer): #lstm
        return ks.layers.LSTM(4, activation="gelu", dropout=0.4, kernel_initializer="random_normal", bias_initializer="zeros")(layer)
    def regularization_dropout(self, layer): #dropout
        return ks.layers.Dropout(0.4)(layer)
    def batch_normalization(self, layer): #batch norm
        return ks.layers.BatchNormalization(epsilon=10e-5, beta_initializer="zeros")(layer)
    def embedding(self, layer): #positional embedding
        return ks.layers.Embedding(self.voc_size, 4)(layer)
    def attn(self, input_k_v, input_q): #attention
        attn = ks.layers.MultiHeadAttention(2, 4)(input_q, input_k_v)
        attn_norm = ks.layers.LayerNormalization(epsilon=10e-7)(attn)
        attn_block = ks.layers.Add()([attn, attn_norm])
        return attn_block
    def ffn(self, layer): #feed-forward net
        ffn = ks.layers.Dense(24, activation="gelu", kernel_initializer="random_normal")(layer)
        ffn = ks.layers.Dense(24, activation="gelu", kernel_initializer="random_normal")(ffn)
        ffn = ks.layers.Dense(4, activation="gelu")(ffn)
        ffn_norm = ks.layers.LayerNormalization(epsilon=10e-7)(ffn)
        ffn_block = ks.layers.Add()([ffn, ffn_norm])
        return ffn_block
    def encoder_layer(self): #encoder layer
        return self.batch_normalization(self.regularization_dropout(self.ffn(self.attn(self.input_k_v_embedded, self.input_q_embedded))))
    def decoder_layer(self): #decoder layer
        return self.batch_normalization(self.regularization_lstm(self.ffn(self.attn(self.encoder_layer(), self.attn(self.output_k_v_embedded, self.output_q_embedded)))))
    def struct(self):
        #4 layers of encoder (adjustable)
        encoder = ks.layers.Add()([self.encoder_layer(),
                                   self.encoder_layer(),
                                   self.encoder_layer(),
                                   self.encoder_layer(),])
        #4 layers of decoder (adjustable)
        decoder = ks.layers.Add()([self.decoder_layer(),
                                   self.decoder_layer(),
                                   self.decoder_layer(),
                                   self.decoder_layer(),])
        encoder_decoder = ks.layers.Add()([encoder, decoder])
        linear = ks.layers.Dense(96, activation="linear")(encoder_decoder)
        #output probability of next token
        output = ks.layers.Dense(1)(linear) #softmax didn't give best perfomance...
        transformer = ks.Model([self.input_k_v, self.input_q, self.output_k_v, self.output_q], output)
        return transformer
```

Model architecture

```ruby
ks.utils.plot_model(mdl, "arch.png", show_layer_activations=True)
```

![architecture]()
