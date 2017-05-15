import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

tf.set_random_seed(0)

# Download the images and label them as test and training set
mnist = read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# Read image into X: First dimension None for indexing the image
X = tf.placeholder(tf.float32, [None, 28, 28, 1])

# Correct answer will go here
Y_ = tf.placeholder(tf.float32, [None, 10])

# Weights for neurons
W = tf.Variable(tf.zeros([784, 10]))

# Biases used by every neuron
b = tf.Variable(tf.zeros([10]))

max_accu = -1

# Flatten the image into single line of pixel
# -1 for preserving the number of elements
XX = tf.reshape(X, [-1, 784])

# Neural Network Model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)

# Assumption for loss function: Cross entropy between measured and desired values
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# Measuring the accuracy of the trained model
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Training and learning rate: 0.003
train_step = tf.train.GradientDescentOptimizer(0.003).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Call the above functions in the loop for training over 100 images in each run
def training_step(i, update_test_data, update_train_data):
    global max_accu
    # training on 100 images with 100 labels
    batch_X, batch_Y = mnist.train.next_batch(100)

    if update_train_data:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
        print(str(i) + ": accuracy:" + str(a) + " loss:" + str(c))

    # Compute test values
    if update_test_data:
        a,c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_: mnist.test.labels})
        max_accu = max(max_accu, a)
        print(str(i) + ": ******* epoch " + str(i*100//mnist.train.images.shape[0]+1) + " ***** test_accuracy:" + str(a) + " test_loss: " + str(c))

    sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})

def main():
    for i in range(2000):
        training_step(i, True, True)
    print(max_accu)

if __name__ == "__main__":
    main()
