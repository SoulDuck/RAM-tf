import numpy as np
import tensorflow as tf


a=np.array([[1,1,1],[2,2,2]])
print np.shape(a)
b=np.array(3)
print np.shape(b)
tf_a=tf.Variable(a)
tf_b=tf.Variable(b)
tf_c=tf.multiply(tf_a , tf_b)

sess=tf.Session()
init= tf.global_variables_initializer()
sess.run(init)
print sess.run(tf_c)
