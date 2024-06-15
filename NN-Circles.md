>> Neural Net playground with circles

In this playground you can easily test different models to work with images

![Test](https://github.com/AlephVenXm/Main/blob/main/test.png)

For example:

mdl = Sequential([

    layers.Rescaling(1./255, input_shape=(RESOLUTION, RESOLUTION, 3,)),
    
    layers.Conv2D(16, 3, activation="relu", padding="same"),
    
    layers.MaxPooling2D(),
    
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    
    layers.MaxPooling2D(),
    
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    
    layers.MaxPooling2D(),
    
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    
    layers.MaxPooling2D(),
    
    layers.Flatten(),
    
    layers.Dense(128, activation="leaky_relu"),
    
    layers.Dense(6)
    
])

mdl.compile(

    optimizer=ks.optimizers.Adam(),
    
    loss=ks.losses.MeanSquaredError(),
    
    metrics=["accuracy"]
    
)

This model is used in playground notebook as example
You can easily change it however you want
As example: try changing relu to gelu, changing leaky_relu to softmax, Adam to RMSprop or Adagrad and etc.

>> Link to notebook ---> https://colab.research.google.com/drive/1Bpgnyy2vVwV6_M8mmCYfOCymYeixkc-x?usp=sharing
