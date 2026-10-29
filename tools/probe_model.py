import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

model = tf.keras.models.load_model("models/vgg16_b4b5.h5", compile=False)

print("input:", model.input_shape)
print("output:", model.output_shape)
print("top-level layers:", len(model.layers))
for layer in model.layers[:6]:
    print("  ", layer.name, layer.__class__.__name__, "trainable=", layer.trainable)

names = [l.name for l in model.layers]
for probe in ("block1_conv1", "block3_conv1", "block4_conv1", "block4_conv3", "block5_conv1", "block5_conv3", "block5_pool", "global_average_pooling2d", "dense"):
    print(f"{probe:26s} top-level={probe in names}")

trainable = sum(int(w.shape.__len__() and __import__("numpy").prod(w.shape)) for w in model.trainable_weights)
total = sum(int(__import__("numpy").prod(w.shape)) for w in model.weights)
print("trainable params:", trainable)
print("total params:", total)
