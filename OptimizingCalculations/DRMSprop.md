>> Dynamic version of Root Mean Square Propagation or RMSprop

```ruby
import keras as ks, tensorflow as tf, numpy as np
import matplotlib.pyplot as plt
```

Class for DynamicRMSprop

```ruby
class DRMSprop(ks.optimizers.RMSprop):
    def __init__(self, learning_rate=0.001, rho=0.9, momentum=0, epsilon=1e-7, centered=False, weight_decay=None, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, loss_scale_factor=None, gradient_accumulation_steps=None, name="rmsprop", **kwargs):
        super().__init__(learning_rate, rho, momentum, epsilon, centered, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
    def scale_loss(self, loss):
        loss_len = len(str(int(loss.numpy())))
        #*a really dumb way to change rate dynamically
        #*thinking about adding tf.function for a better calculation of rate
        self.learning_rate = loss_len / 8.0 * 10.0
        return super().scale_loss(loss)
```

Test...

```ruby
#*setting dynamic optimizer
RMSprop = ks.optimizers.RMSprop(0.1)
DRMSprop = DRMSprop(0.1)

#*some data for test
X = tf.cast(tf.linspace(-1000.0, 1000.0, 100), tf.float64)
Y = X + np.random.uniform(-100.0, 100.0)

#*one layer RMSprop model
mdl_RMSprop = ks.Sequential([ks.layers.Dense(1, input_shape=[1])])
mdl_RMSprop.compile(optimizer=RMSprop, loss=ks.losses.MeanSquaredError(), run_eagerly=True)

#*one layer DynamicRMSprop model
mdl_DRMSprop = ks.Sequential([ks.layers.Dense(1, input_shape=[1])])
mdl_DRMSprop.compile(optimizer=DRMSprop, loss=ks.losses.MeanSquaredError(), run_eagerly=True)

#*training
EPOCHS = 10
print('-'*24, "RMSprop", '-'*24)
stats_RMSprop = mdl_RMSprop.fit(X, Y, epochs=EPOCHS)
print('-'*24, "DRMSprop", '-'*24)
stats_DRMSprop = mdl_DRMSprop.fit(X, Y, epochs=EPOCHS)
```

Plotting results

```ruby
fig, axis = plt.subplots(2, sharex=True)
axis[0].plot(range(EPOCHS), stats_RMSprop.history["loss"], color="blue")
axis[0].set_title("RMSprop")
axis[1].plot(range(EPOCHS), stats_DRMSprop.history["loss"], color="red")
axis[1].set_title("DynamicRMSprop")
plt.show()
```

Compare of RMSprop and DynamicRMSprop

![graph](https://github.com/AlephVenXm/Main/blob/main/OptimizingCalculations/DRMSprop_graph.png)
