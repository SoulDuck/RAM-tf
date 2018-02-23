import numpy as np
import tensorflow as tf

a=np.array([[0.1,0.1,0.1],[0.5,0.6,1]])
a=a+0.5
a=a.astype(int)
print a
exit()
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


a=np.zeros([2,2,5])
for i in range(5):
    a[:,:,i]=i
a=a*range(5,10)
print a

a=np.sum(a , axis=2)
print a


print map(lambda (a,b): a*b , [(1,3) , (2,4) , (3,6)] )