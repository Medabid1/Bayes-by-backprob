import tensorflow as tf
from bayes_backprob import Config, BBB
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)        
  
        
conf = Config()

data = tf.placeholder(tf.float32, [None, 784])
target = tf.placeholder(tf.float32, [None, 10])
nn = BBB(data, target, conf, is_training=True)


with tf.Session() as sess:
    tf.global_variables_initializer().run()
    
    for i in range(300000):
        xs, ys = mnist.train.next_batch(conf.batch_size)
        
        _, loss_value, step, lr = sess.run(
                [nn.optimize, nn.loss, nn.global_step, nn.learning_rate], 
                feed_dict={nn.data: xs, nn.target: ys})
        
        if i % 10 == 0:
            print("After %d training step(s),  loss on training batch is %g." % (i, loss_value))
            print(sess.run(nn.accuracy1(), feed_dict={nn.data: xs, nn.target: ys}))                
