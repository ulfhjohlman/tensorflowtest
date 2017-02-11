import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

N_DATA = 100 # from both A&B
N_CLASSES = 2
DIMENSIONS = 3
BATCH_SIZE = 10
LEARNING_RATE = 0.1
HIDDEN_1 = 4
HIDDEN_2 = 2
TRAINING_ITTERATIONS = 1000
STD = 0.2

def gen_data():
    theta = np.random.rand(N_DATA).astype(np.float32) * 2 * math.pi
    rho = np.random.rand(N_DATA).astype(np.float32) * 2 * math.pi
    r1 = np.random.normal(size=N_DATA, loc=1, scale=STD).astype(np.float32)
    r2 = np.random.normal(size=N_DATA, loc=1, scale=STD).astype(np.float32)

    x_data_a = np.multiply(r1,np.sin(theta))
    y_data_a = np.zeros(N_DATA)
    z_data_a = r1 * np.cos(theta)
    targets_a = np.zeros(N_DATA)

    x_data_b = np.zeros(N_DATA)
    y_data_b = np.multiply(r2,np.sin(rho))
    z_data_b = r2 * np.cos(rho) - np.ones(N_DATA)
    targets_b = np.ones(N_DATA)


    data_a = np.transpose(np.concatenate(([x_data_a],[y_data_a],[z_data_a],[targets_a]),0))
    data_b = np.transpose(np.concatenate(([x_data_b],[y_data_b],[z_data_b],[targets_b]),0))
    data_and_targets = np.concatenate((data_a, data_b),0)
    np.random.shuffle(data_and_targets)
    data, targets = data_and_targets[:,0:3],data_and_targets[:,3]
    return data, targets

def placeholder_inputs(batch_size):
  input_placeholder = tf.placeholder(tf.float32, shape=(batch_size, DIMENSIONS))
  target_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return input_placeholder, target_placeholder

def fill_feed_dict(data, targets, input_pl, target_pl, epoch):
    epoch = epoch % ((N_DATA*2)/BATCH_SIZE)
    data_batch = data[int(BATCH_SIZE*epoch):int(BATCH_SIZE*(epoch+1)),:]
    target_batch = targets[int(BATCH_SIZE*epoch):int(BATCH_SIZE*(epoch+1))]
    feed_dict = {
        input_pl: data_batch,
        target_pl: target_batch,
    }
    return feed_dict


def inference(input_pl, h1, h2):
    weights1 = tf.Variable(tf.truncated_normal([DIMENSIONS, h1],
                    stddev=1.0 / math.sqrt(float(DIMENSIONS))), name='weights1')
    biases1 = tf.Variable(tf.zeros([h1]), name='biases1')
    hidden1 = tf.nn.relu(tf.matmul(input_pl, weights1) + biases1)

    weights2 = tf.Variable(tf.truncated_normal([h1, h2],
                    stddev=1.0 / math.sqrt(float(DIMENSIONS))), name='weights2')
    biases2 = tf.Variable(tf.zeros([h2]), name='biases2')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights2) + biases2)

    weights3 = tf.Variable(tf.truncated_normal([h2, N_CLASSES],
                            stddev=1.0 / math.sqrt(float(h2))), name='weights3')
    biases3 = tf.Variable(tf.zeros([N_CLASSES]), name='biases3')
    logits = tf.matmul(hidden2, weights3) + biases3
    return logits

def loss(logits, targets):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  targets = tf.to_int64(targets)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=targets, logits=logits, name='Crossentropy')
  return tf.reduce_mean(cross_entropy, name='Crossentropy_mean')


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data, targets):
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = N_DATA*N_CLASSES // BATCH_SIZE
  num_examples = steps_per_epoch * BATCH_SIZE
  for step in range(steps_per_epoch):
    feed_dict = fill_feed_dict(data, targets,
                               images_placeholder,
                               labels_placeholder, step)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def training(data,targets):
    input_pl, target_pl = placeholder_inputs(BATCH_SIZE)

    logits = inference(input_pl ,HIDDEN_1, HIDDEN_2)
    lossfunc = loss(logits,target_pl)
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(lossfunc, global_step=global_step)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(TRAINING_ITTERATIONS):
        feed_dict = fill_feed_dict(data, targets, input_pl, target_pl, i)
        _, loss_value = sess.run([train_op, lossfunc], feed_dict=feed_dict)
        if (i + 1) % 1000 == 0 or (i + 1) == TRAINING_ITTERATIONS:
          # Evaluate against the training set.
          eval_correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, target_pl, 1), tf.int32))
          print('Training Data Eval:')
          do_eval(sess,
                  eval_correct,
                  input_pl,
                  target_pl,
                  data, targets)
          """# Evaluate against the validation set.
          print('Validation Data Eval:')
          do_eval(sess,
                  eval_correct,
                  input_pl,
                  target_pl,
                  data, targets)
          """
          # Evaluate against the test set.
          print('Test Data Eval:')
          data, targets = gen_data()
          do_eval(sess,
                  eval_correct,
                  input_pl,
                  target_pl,
                  data, targets)

    return

def plot(data,targets):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    class1 = targets==0
    class2 = targets==1
    ax.scatter(data[class1,0],data[class1,1],data[class1,2],c = 'r')
    ax.scatter(data[class2,0],data[class2,1],data[class2,2],c = 'b')
    plt.show()
    return

if __name__ == '__main__':
    data, targets = gen_data()
    training(data,targets)
    plot(data,targets)
