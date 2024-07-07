>> This is idea of adding neural net in optimizer to auto-correct its alpha and beta params

AdaNN ---> controlling net

```ruby
class AdaNN(tf.Module):
    def __init__(self, alpha, beta, **kwargs):
        super().__init__(**kwargs)
        self.alpha = tf.Variable(alpha)
        self.beta = tf.Variable(beta)
    def __call__(self, loss):
        return tf.experimental.numpy.log10(loss*self.alpha) / self.beta
    def train(self, loss):
        with tf.GradientTape() as tape:
            adjust = -self(loss) * 0.99 #gamma
        d_alpha, d_beta = tape.gradient(adjust, [self.alpha, self.beta])
        self.alpha.assign_add(-d_alpha)
        self.beta.assign_add(-d_beta)
```

DynamicGradient ---> main net

```ruby
class DynamicGradient(ks.optimizers.Optimizer):
    def __init__(self, alpha=10e2, beta=10e3, learning_rate=0.0, name="DynamicGradient", **kwargs):
        super().__init__(learning_rate, name=name, **kwargs)
        self.learning_rate = 0.0
        self.adann = AdaNN(alpha, beta)
    def scale_loss(self, loss):
        self.adann.train(loss)
        dynamic_lr = self.adann(loss)
        self.learning_rate = dynamic_lr
        return super().scale_loss(loss)
    def update_step(self, gradient, variable, learning_rate):
        increment = learning_rate * gradient
        variable.assign_add(-increment)
```

p.s. still needs a little bit of adjustments to make it fully independent from starting alpha and beta
