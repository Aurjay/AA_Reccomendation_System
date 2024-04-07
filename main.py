import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

(xs, ys),_ = datasets.mnist.load_data()

xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)


model = Sequential([layers.Dense(128, activation='relu'),
                     layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(10)])
model.build(input_shape=(None, 28*28))
model.summary()

optimizer = optimizers.SGD(lr=0.01)
acc_meter = metrics.Accuracy()

for step, (x,y) in enumerate(db):

    with tf.GradientTape() as tape:
        x = tf.reshape(x, (-1, 28*28))
        out = model(x)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.square(out-y_onehot)
        loss = tf.reduce_sum(loss) / 32


    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    if step % 400==0:
        print('Loss:', float(loss), ', Acc:', acc_meter.result().numpy())
        acc_meter.reset_states()
