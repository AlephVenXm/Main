>> Idea of Dynamic learning rate but with log10 function

```ruby
import keras as ks, tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
```

```ruby
@tf.function
def DynamicRate(loss):
    return tf.experimental.numpy.log10(loss)

class DRMSprop(ks.optimizers.RMSprop):
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0, epsilon=1e-7, centered=False, weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None, name="rmsprop", **kwargs):
        super().__init__(learning_rate, rho, momentum, epsilon, centered, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
    def scale_loss(self, loss):
        self.learning_rate = DynamicRate(loss)
        #*this one is only to track changes of rate
        LRATE_traceback.append(self.learning_rate.numpy())
        return super().scale_loss(loss)
```

```ruby
RMSprop_ = ks.optimizers.RMSprop(0.1)
X = tf.cast(tf.linspace(-1000.0, 1000.0, 100), tf.float64)
Y = X + np.random.uniform(-100.0, 100.0)
mdl_RMSprop = ks.Sequential([ks.layers.Dense(1, input_shape=[1])])
mdl_RMSprop.compile(optimizer=RMSprop_, loss=ks.losses.MeanSquaredError(), run_eagerly=True)
EPOCHS = 10
stats_RMSprop = mdl_RMSprop.fit(X, Y, epochs=EPOCHS)
```

```ruby
LRATE_traceback = []
DRMSprop_ = DRMSprop(0.1)
X = tf.cast(tf.linspace(-1000.0, 1000.0, 100), tf.float64)
Y = X + np.random.uniform(-100.0, 100.0)
mdl_DRMSprop = ks.Sequential([ks.layers.Dense(1, input_shape=[1])])
mdl_DRMSprop.compile(optimizer=DRMSprop_, loss=ks.losses.MeanSquaredError(), run_eagerly=True)
EPOCHS = 10
stats_DRMSprop = mdl_DRMSprop.fit(X, Y, epochs=EPOCHS)
```

```ruby
fig, axis = plt.subplots(2, 2)
axis[0, 0].plot(range(EPOCHS), stats_RMSprop.history["loss"], color="blue")
axis[0, 0].set_title("RMSprop")
axis[0, 1].plot(range(40), [0.1]*40, color="blue")
axis[0, 1].set_title("Static rate")
axis[1, 0].plot(range(EPOCHS), stats_DRMSprop.history["loss"], color="red")
axis[1, 0].set_title("DynamicRMSprop")
axis[1, 1].plot(range(40), LRATE_traceback, color="red")
axis[1, 1].set_title("Dynamic changes of rate")
fig.tight_layout(pad=0.5)
plt.show()
```

![graph](https://github.com/AlephVenXm/Main/blob/main/OptimizingCalculations/DLR_log10_graph.png)
