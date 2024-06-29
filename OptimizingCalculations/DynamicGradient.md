>> Idea of DynamicLearningRate in DynamicGradient optimizer

```ruby
class DynamicGradient(ks.optimizers.Optimizer):
    def __init__(self, alpha=10e2, beta=10e3, learning_rate=0.0, name="DynamicGradient", **kwargs):
        super().__init__(learning_rate, name=name, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = 0.0
    @tf.function
    def scale_loss(self, loss):
        alpha = self.alpha
        beta = self.beta
        dynamic_lr = tf.experimental.numpy.log10(loss*alpha) / beta
        self.learning_rate = dynamic_lr
        return super().scale_loss(loss)
    def update_step(self, gradient, variable, learning_rate):
        increment = learning_rate * gradient
        variable.assign_add(-increment)
```

![graph](https://github.com/AlephVenXm/Main/blob/main/OptimizingCalculations/DynamicGradient_graph.png)
