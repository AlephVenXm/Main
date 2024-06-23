>> Some work with optimizing neural nets

Made dynamic learning rate for optimizers

Trying to figure out how to make it more stable

```ruby
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
```

Function and Class for dynamic learning rate

```ruby
@tf.function
def DynamicOptimizer(loss, size):
    while loss >= tf.pow(tf.cast(10.0, tf.float64), size):
        loss /= 10.0
    return loss

class DynamicRate(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate):
        self.initial_learning_rate = initial_learning_rate
    def __call__(self, loss):
        loss_size = tf.cast(tf.size(loss), tf.float64)
        dynamic_change = DynamicOptimizer(tf.cast(loss, tf.float64), loss_size)
        return self.initial_learning_rate * dynamic_change
```

Constructing model and passing some data... Adam as optimizer

```ruby
DynamicDescent = keras.optimizers.Adam(learning_rate=DynamicRate(0.1))

X = tf.cast(tf.linspace(0.0, 10.0, 100), tf.float64)
Y = X * 2

mdl = keras.Sequential([keras.layers.Dense(1, input_shape=[1])])
mdl.compile(optimizer=DynamicDescent, loss=keras.losses.MeanSquaredError())

EPOCHS = 10
stats = mdl.fit(X, Y, epochs=EPOCHS)
```
Plotting loss statistic of neural net

```ruby
plt.plot(range(EPOCHS), stats.history["loss"], color="red")
```

![img](https://github.com/AlephVenXm/Main/blob/main/OptimizingCalculations/DLR_graph.png)
