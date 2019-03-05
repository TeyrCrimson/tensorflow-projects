import pandas as pd
import numpy as np
import pylab as plt
import datetime as dt
import os
import urllib.request, json
import tensorflow as tf
import math
import io


batch_size = 16
layer_ids = ['hidden1','hidden2','hidden3','hidden4','hidden5','out']
layer_sizes = [1869,930,460,300,100,10,5]
beta = 10**-3

tf.reset_default_graph()

#inputs and labels
train_inputs = tf.placeholder(tf.float32, shape=[None, layer_sizes[0]], name='train_inputs')
train_labels = tf.placeholder(tf.float32, shape=[None, layer_sizes[-1]], name='train_labels')

#weights and bias definitions
for idx, lid in enumerate(layer_ids):
    with tf.variable_scope(lid):
        w = tf.get_variable('weights', shape=[layer_sizes[idx], layer_sizes[idx+1]],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('bias', shape=[layer_sizes[idx+1]],
                            initializer=tf.random_uniform_initializer(-0.1,0.1))

#calculate logits
h = train_inputs
for lid in layer_ids:
    with tf.variable_scope(lid, reuse=True):
        w, b = tf.get_variable('weights'), tf.get_variable('bias')
        if lid != 'out':
            h = tf.nn.relu(tf.matmul(h,w)+b, name=lid+'_output')
        else:
            h = tf.nn.xw_plus_b(h,w,b,name=lid+'_output')
        h = tf.nn.dropout(h, 0.9, name=lid+'_dropout')

tf_predictions = tf.nn.softmax(h, name='predictions')

#calculate loss
tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2
                         (labels=train_labels, logits=h), name='loss')
l2_loss = beta * tf.add_n(
        [tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
tf_loss = tf_loss + l2_loss

#optimizer
tf_learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
optimizer = tf.train.RMSPropOptimizer(tf_learning_rate)
grads_and_vars = optimizer.compute_gradients(tf_loss)
tf_loss_minimize = optimizer.minimize(tf_loss)


with tf.name_scope('performance'):
    tf_loss_ph= tf.placeholder(tf.float32, shape=None, name='loss_summary')
    tf_loss_summary = tf.summary.scalar('loss', tf_loss_ph)

    tf_accuracy_ph = tf.placeholder(tf.float32, shape=None, name='accuracy_summary')
    tf_accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy_ph)

#gradient norm summary
for g,v in grads_and_vars:
    if 'hidden5' in v.name and 'weights' in v.name:
        with tf.name_scope('gradients'):
            tf_last_grad_norm = tf.sqrt(tf.reduce_mean(g**2))
            tf_gradnorm_summary = tf.summary.scalar('grad_norm', tf_last_grad_norm)
            break

#merge summaries
performance_summaries = tf.summary.merge([tf_loss_summary, tf_accuracy_summary])


def accuracy(predictions, labels):
    return np.sum(np.argmax(predictions,axis=1)==np.argmax(labels,axis=1))*100.0/labels.shape[0]

#generates prediction images for Tensorboard
def gen_plot(data, labels, sess):
    plt.figure()
    classification = sess.run(tf_predictions, feed_dict={train_inputs: np.asmatrix(data)})
    classification = np.squeeze(classification, axis=0)
    pred, = plt.plot(range(len(labels)), classification, label='Prediction')
    actual, = plt.plot(range(len(labels)), labels, label='Actual')
    plt.legend(handles=[pred, actual])
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    image = tf.image.decode_png(buf.getvalue(), channels=4)

    image = tf.expand_dims(image, 0)

    summary_op = tf.summary.image("plot", image)
    return summary_op
    
    

def main():
    csvX = pd.read_csv("Movie_features.csv")
    csvY = pd.read_csv("Movie_labels.csv")
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        dataX = tf.constant(csvX).eval()
        dataY = tf.constant(csvY).eval()
    size = math.trunc(len(dataX)*0.8)
    start, end = 0,size
    x_train = dataX[start:end]
    y_train = dataY[start:end]
    x_test = dataX[end:]
    y_test = dataY[end:]

    n_classes = 5
    n_train = len(x_train)
    n_test = len(x_test)
    n_epochs = 150

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.9

    session = tf.InteractiveSession(config=config)

    if not os.path.exists('summaries'):
        os.mkdir('summaries')
        print('Directory made.')
    if not os.path.exists(os.path.join('summaries', 'first_final')):
        os.mkdir(os.path.join('summaries', 'first_final'))
        print('Directory made.')

    summ_writer = tf.summary.FileWriter(os.path.join('summaries', 'first_final'), session.graph)

    tf.global_variables_initializer().run()

    accuracy_per_epoch = []

    idx = np.arange(n_train)
    with tf.device('/device:GPU:1'):
        for epoch in range(n_epochs):
            loss_per_epoch = []
            np.random.shuffle(idx)
            x_train = x_train[idx]
            y_train = y_train[idx]

            for start, end in zip(range(0, n_train, batch_size), range(batch_size, n_train, batch_size)):
                batch_x = x_train[start:end]
                batch_y = y_train[start:end]
                if start == 0:
                    l,_,gn_summ = session.run([tf_loss, tf_loss_minimize, tf_gradnorm_summary],
                                              feed_dict={train_inputs: batch_x,
                                                         train_labels: batch_y,
                                                         tf_learning_rate: 10**-3})
                    summ_writer.add_summary(gn_summ, epoch)
                else:
                    l,_ = session.run([tf_loss, tf_loss_minimize],
                                      feed_dict={train_inputs: batch_x,
                                                 train_labels: batch_y,
                                                 tf_learning_rate: 10**-3})
                loss_per_epoch.append(l)

            print('Average loss in epoch %d: %.5f'%(epoch, np.mean(loss_per_epoch)))

            avg_loss = np.mean(loss_per_epoch)

            accuracy_per_epoch = []
            for start, end in zip(range(0, n_test, batch_size), range(batch_size, n_test, batch_size)):
                tbatch_x = x_test[start:end]
                tbatch_y = y_test[start:end]
                test_batch_predictions = session.run(
                    tf_predictions,feed_dict={train_inputs: tbatch_x})
                accuracy_per_epoch.append(accuracy(test_batch_predictions, tbatch_y))

            print('\tAverage Test Accuracy in epoch %d: %.5f\n'%(epoch, np.mean(accuracy_per_epoch)))
            avg_test_accuracy = np.mean(accuracy_per_epoch)

            summ = session.run(performance_summaries, feed_dict={tf_loss_ph: avg_loss,
                                                                 tf_accuracy_ph: avg_test_accuracy})

            summ_writer.add_summary(summ,epoch)

        for k in range(5):
            summary_op = gen_plot(x_test[k], y_test[k], session)
            summary = session.run(summary_op)
            summ_writer.add_summary(summary)

        summ_writer.close()
        session.close()

if __name__ == '__main__':
    main()
                                          
            
            
        

        



