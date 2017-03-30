from tensorflow.examples.tutorials.mnist import input_data
import datetime
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


import tensorflow as tf

sess = tf.InteractiveSession()

#--------------------------------
# Create softmax regration model
#--------------------------------

# Generate placeholder variables to represent the input tensors.
# x is images placeholder
# y_ is Labels placeholder (target output)
x = tf.placeholder("float", shape=[None, 784])
y_ = tf.placeholder("float", shape=[None, 10])

# Creare weight and bias variable. W is a matrix of 748*10, b is a 10
# dimensional vector. Initial values are o.
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

#regression model
y = tf.nn.softmax(tf.matmul(x,W) + b)

#-------------------------------
# use entropy as loss function
#-------------------------------

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#--------------------------------------------------------------
# Our goal is minimize loss function. Start trainning the model
#--------------------------------------------------------------
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


starttime =  datetime.datetime.now()
print('Start:\t\t'+starttime.strftime("%Y-%m-%d %X"))
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

endtime =  datetime.datetime.now()
print('End:\t\t'+endtime.strftime("%Y-%m-%d %X"))
duringtime = endtime-starttime
print ('Spend Time:\t'+str(duringtime))


#----------
# Evaluate
#----------
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print (accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

