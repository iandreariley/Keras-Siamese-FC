class SiameseFC(object):
    X_SHAPE = (255, 255, 3)
    Z_SHAPE = (127, 127, 3)
    
    def __init__(self):
        conv_layers = self._conv_layers()
        x = self._input(self.X_SHAPE)
        z = self._input(self.Z_SHAPE)
        x_features = self._apply_layers(x, conv_layers)
        z_features = self._apply_layers(z, conv_layers)
        scores = Flatten()(self._cross_correlation([x_features, z_features]))
        print(scores)
        self.model = Model(inputs=[x, z], outputs=scores)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        
    def predict(self, x):
        return self.model.predict(x)
        
    def _input(self, shape):
        return Input(shape=shape)
    
    def _conv_layer(self, filters, kernel_dim, stride_len):
        return [Conv2D(filters, kernel_dim, strides=stride_len,
                  padding='valid', activation='relu', kernel_initializer='glorot_normal')]

    def _conv_block(self, filters, kernel_dim, stride_len):
        batch_norm = [BatchNormalization(axis=3)]
        return self._conv_layer(filters, kernel_dim, stride_len) + batch_norm

    def _max_pool(self):
        return [MaxPool2D(pool_size=3, strides=2, padding='valid')]

    def _conv_layers(self):
        layers = []
        layers += self._conv_block(48, 11, 2)
        layers += self._max_pool()
        layers += self._conv_block(128, 5, 1)
        layers += self._max_pool()
        layers += self._conv_block(48, 3, 1)
        layers += self._conv_block(48, 3, 1)
        layers += self._conv_layer(32, 3, 1)
        return layers

    def _apply_layers(self, x, layers):
        out = x
        for layer in layers:
            out = layer(out)
        return out

    def _add_dimension(self, t):
        return tf.reshape(t, (1,) + t.shape)

    def _cross_correlation_fn(self, inputs):
        x = inputs[0]
        x = tf.reshape(x, [1] + x.shape.as_list())
        z = inputs[1]
        z = tf.reshape(z, z.shape.as_list() + [1])
        return tf.nn.convolution(x, z, padding='VALID', strides=(1,1))

    def _cross_correlation(self, inputs):
        # Note that dtype MUST be specified, otherwise TF will assert that the input and output structures are the same,
        # which they most certainly are NOT.
        return tf.map_fn(self._cross_correlation_fn, inputs, dtype=tf.float32, infer_shape=False)
