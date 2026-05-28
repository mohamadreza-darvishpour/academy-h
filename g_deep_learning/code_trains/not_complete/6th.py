
# let's visualize layer names and layer indices to see how many layers
# we should freeze:

for i, layer in enumerate(base_model.layers):
    print(i, layer.name)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:

for layer in model.layers[:249]:
    layer.trainable = False

for layer in model.layers[249:]:
    layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate

from tensorflow.keras.optimizers import SGD

model.compile(
    optimizer=SGD(lr=0.0001, momentum=0.9),
    loss='categorical_crossentropy'
)

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers

model.fit(...)