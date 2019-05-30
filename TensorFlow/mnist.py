import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

x = tf.placeholder(tf.float32, shape=(None,784))
y = tf.placeholder(tf.float32, shape=(None,10))
w = tf.Variable(tf.zeros((784,10)))
b = tf.Variable(tf.zeros(10))
y_pred = tf.nn.softmax(tf.matmul(x, w) + b)
loss = -tf.reduce_sum(y*tf.log(y_pred))
optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
c_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(c_pred, tf.float32))

mnist = read_data_sets(r"E:\Project\Edge\Python\TensorFlow\data\mnist", one_hot=True)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(1001):
        x_, y_ = mnist.train.next_batch(100)
        session.run(optimizer, feed_dict={x:x_, y:y_})
        if i % 100 == 0:
            print(session.run(acc, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
