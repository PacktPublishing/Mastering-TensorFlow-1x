import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('job_name','ps','name of the job, default ps')
tf.flags.DEFINE_integer('task_index',0,'index of the job, default 0')

def main(_):
    mnist = input_data.read_data_sets('/home/armando/datasets/mnist', one_hot=True)

    ps = [
            'localhost:9001',  # /job:ps/task:0
         ]
    workers = [
            'localhost:9002',  # /job:worker/task:0
            'localhost:9003',  # /job:worker/task:1
            'localhost:9004',  # /job:worker/task:2
            ]
    clusterSpec = tf.train.ClusterSpec({'ps': ps, 'worker': workers})

    config = tf.ConfigProto()
    config.allow_soft_placement = True

    #server = tf.train.Server(clusterSpec,
    #                         job_name=FLAGS.job_name,
    #                         task_index=FLAGS.task_index,
    #                         config=config
    #                         )

    if FLAGS.job_name=='ps':
        #print(config.device_count['GPU'])
        config.device_count['GPU']=0
        server = tf.train.Server(clusterSpec,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index,
                                 config=config
                                 )
        server.join()
        sys.exit('0')
    elif FLAGS.job_name=='worker':
        config.gpu_options.per_process_gpu_memory_fraction = 0.2
        server = tf.train.Server(clusterSpec,
                                 job_name=FLAGS.job_name,
                                 task_index=FLAGS.task_index,
                                 config=config
                                 )
        is_chief = (FLAGS.task_index==0)

        worker_device='/job:worker/task:{}'.format(FLAGS.task_index)
        device_func = tf.train.replica_device_setter(worker_device=worker_device,
                                                     cluster=clusterSpec
                                                     )
    # the default values are: ps_device='/job:ps',worker_device='/job:worker'
        with tf.device(device_func):

            global_step = tf.train.get_or_create_global_step()
            #tf.Variable(0,name='global_step',trainable=False)
            x_test = mnist.test.images
            y_test = mnist.test.labels

            # parameters
            n_outputs = 10  # 0-9 digits
            n_inputs = 784  # total pixels

            learning_rate = 0.01
            n_epochs = 50
            batch_size = 100
            n_batches = int(mnist.train.num_examples/batch_size)
            n_epochs_print=10

            # input images
            x_p = tf.placeholder(dtype=tf.float32,
                                 name='x_p',
                                 shape=[None, n_inputs])
            # target output
            y_p = tf.placeholder(dtype=tf.float32,
                                 name='y_p',
                                 shape=[None, n_outputs])

            w = tf.Variable(tf.random_normal([n_inputs, n_outputs],
                                             name='w'
                                             )
                            )
            b = tf.Variable(tf.random_normal([n_outputs],
                                             name='b'
                                             )
                            )
            logits = tf.matmul(x_p,w) + b
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_p,
                                                                    logits=logits
                                                                    )
            loss_op = tf.reduce_mean(cross_entropy)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            train_op = optimizer.minimize(loss_op,global_step=global_step)
            correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y_p, 1))
            accuracy_op = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        sv = tf.train.Supervisor(is_chief=is_chief,
                                 init_op = tf.global_variables_initializer(),
                                 global_step=global_step)



        with sv.prepare_or_wait_for_session(server.target) as mts:
            lstep = 0

            for epoch in range(n_epochs):
                for batch in range(n_batches):
                    x_batch, y_batch = mnist.train.next_batch(batch_size)
                    feed_dict={x_p:x_batch,y_p:y_batch}
                    _,loss,gstep=mts.run([train_op,loss_op,global_step],
                                         feed_dict=feed_dict)
                    lstep +=1
                if (epoch+1)%n_epochs_print==0:
                    print('worker={},epoch={},global_step={}, local_step={}, loss = {}'.
                          format(FLAGS.task_index,epoch,gstep,lstep,loss))
            feed_dict={x_p:x_test,y_p:y_test}
            accuracy = mts.run(accuracy_op, feed_dict=feed_dict)
            print('worker={}, final accuracy = {}'.format(FLAGS.task_index,accuracy))
    sv.stop()

if __name__ == '__main__':

  tf.app.run()
