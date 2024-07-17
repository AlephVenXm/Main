>> Code of DynamicGradient optimizer with in-built Adaptive Neural Net and test

note: this optimizer doesnt use moments, velocities, epsilons, rhos, average gradients and etc.

it also starts from 0.0 learning rate

```ruby
class DynamicGradient(ks.optimizers.Optimizer):
    def __init__(self, alpha=10e2, beta=10e3, learning_rate=0.0, name="DynamicGradient", **kwargs):
        super().__init__(learning_rate, name=name, **kwargs)
        self.learning_rate = 0.0
        self.alpha = tf.Variable(alpha)
        self.beta = tf.Variable(beta)
    def log10(self, loss): #Main function
        return tf.experimental.numpy.log10(loss*self.alpha) / self.beta
    def adapt(self, loss): #Train step for AdaNN
        with tf.GradientTape() as tape:
            adjust = 1/-self.log10(loss)
        d_alpha, d_beta = tape.gradient(adjust, [self.alpha, self.beta])
        self.alpha.assign_add(1/-d_beta)
        self.beta.assign_add(d_alpha)
    def scale_loss(self, loss): #Changing lr based on current loss
        self.adapt(loss)
        self.learning_rate = self.log10(loss)
        return super().scale_loss(loss)
    def update_step(self, gradient, variable, learning_rate):
        increment = learning_rate * gradient
        variable.assign_add(-increment)
```

p.s. still looking for better optimization of alpha and beta params...

![graph](https://github.com/AlephVenXm/Main/blob/main/OptimizingCalculations/DynamicGradient%20with%20in-built%20AdaNN%20sine%20function%20test.png)

![gif](https://github.com/AlephVenXm/Main/blob/main/OptimizingCalculations/test_sine.gif)
