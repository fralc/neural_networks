import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging


def get_saved_idx(save_path, suffix='epoch'):
    """
    Retrieves the last (greatest) id of a checkpoint folder
    Args:
        save_path: path of the chekcpoint folder 
        suffix: suffix of checkpoint

    Returns:

    """
    idxs = []
    for l in os.listdir(save_path):
        if l.startswith(suffix):
            idxs.append(int(l.split('-')[1].split('.')[0]))
    return int(max(idxs))


def get_weights(input_size, hidden_sizes, n_classes):
    """
    Creates tensoflow variables for a MLP 
    Args:
        input_size: input size of data
        hidden_sizes: hiddel layer sizes
        n_classes: output size

    Returns:
        a dictionary with variable weights (even bias) with sorted names as keys

    """

    sizes = [input_size] + hidden_sizes + [n_classes]
    weights = {}
    for e, _ in enumerate(sizes[:-1]):
        w_key_actual = 'w' + str(e)
        b_key_actual = 'b' + str(e)
        with tf.variable_scope("weights"):
            weights[w_key_actual] = tf.get_variable(
                name=w_key_actual,
                dtype=tf.float32,
                shape=[sizes[e],
                       sizes[e + 1]],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.,
                                                                           uniform=True))
            weights[b_key_actual] = tf.get_variable(
                name=b_key_actual,
                dtype=tf.float32,
                shape=[sizes[e + 1]],
                initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.,
                                                                           uniform=True))
    return weights


def mlp(x, w, last_sigmoid=True):
    """
    Create tensorflow nodes relative to a MLP with structure described by weights
    Args:
        x: input variable. Its shape has to be congruent with weight w['w0'] and b['b0']
        w: weights dictionary. It can be obtained by get_saved_idx
        last_sigmoid: whether to consider sigmoid activation for the output layer

    Returns:

    """
    layers = [tf.sigmoid(tf.nn.bias_add(tf.matmul(x, w['w0']), w['b0']))]
    for i in range(1, len(w.items()) / 2):
        w_key_actual = 'w' + str(i)
        b_key_actual = 'b' + str(i)
        if last_sigmoid:
            layers.append(tf.sigmoid(tf.nn.bias_add(tf.matmul(layers[-1], w[w_key_actual]), w[b_key_actual])))
        else:
            layers.append(tf.nn.bias_add(tf.matmul(layers[-1], w[w_key_actual]), w[b_key_actual]))
    layers[-1] = tf.abs(layers[-1])
    return layers[-1]


def train(
        training_epochs,
        batchsize,
        x_plh,
        y_plh,
        x_train,
        y_train,
        x_test,
        y_test,
        pred,
        cost,
        optimizer,
        augment_fun=None,
        show_step=100,
        save_step=100,
        save_path='',
        reload_path=False,
        show_function=None
):
    """
    Train a Tensorflow network providing train and test data, model input and output placeholders,
    graph nodes, training hyperparameters, visualization options, and checkpoint saving options.
    Args:
        training_epochs: number of epochs
        batchsize: batch size
        x_plh: placeholder for model input variable
        y_plh: placeholder for model output variable
        x_train: training model input; the first index is for samples @type numpy.ndarray
        y_train: training model output; the first index is for samples @type numpy.ndarray
        x_test: test model input; the first index is for samples @type numpy.ndarray
        y_test: test model output; the first index is for samples @type numpy.ndarray
        pred: model prediction operator (tensorflow graph node)
        cost: cost function operator (tensorflow graph node)
        optimizer: optimizer with minimization operator (tensorflow graph node)
        augment_fun: augmentation function
        show_step: number of steps for printing results
        save_step: number of steps for saving results
        save_path: model weights saving path
        reload_path: boolean to pre-load weights
        show_function: function to apply at the show step. Its arguments should present with the same name

    Returns:

    """

    # Do not load variables relative to Optimizer
    # all_vars_dict = {v.name: v for v in tf.all_variables() if 'Adam' not in v.name}
    all_vars_dict = [v for v in tf.all_variables() if 'Adam' not in v.name]
    saver = tf.train.Saver(all_vars_dict)
    # saver = tf.train.Saver()

    with tf.Session() as sess:

        # Initialize parameters and check if they should be loaded
        sess.run(tf.global_variables_initializer()) # verify that it is not needed when parameters are loaded;
                                                    # then do it just when not loading
        global_step = 0
        if os.path.isdir(save_path) and os.listdir(save_path) != [] and reload_path:
            try:
                global_step = get_saved_idx(save_path, suffix='epochs-')
                complete_save_path = os.path.join(save_path, 'epochs-' + str(global_step))
                saver.restore(sess, save_path=complete_save_path)
                print('Parameters restored from {}'.format(complete_save_path))
            except ValueError:
                print('It has not been possible to restore parameters. They have been randomly initialized.')
        elif not os.path.exists(save_path):
            os.makedirs(save_path)

        # Training of the network
        for epoch in range(training_epochs):
            global_step += 1
            batch_idx = np.random.randint(0, x_train.shape[0], batchsize)
            batch_x = x_train[batch_idx]
            batch_y = y_train[batch_idx]
            if augment_fun is not None:
                batch_x = augment_fun(batch_x)
            c_train, _ = sess.run([cost, optimizer], feed_dict={x_plh: batch_x, y_plh: batch_y})

            # Saving weights
            if global_step % save_step == 0:
                path = saver.save(sess=sess, save_path=os.path.join(save_path, 'epochs'), global_step=global_step)
                print('Data saved on {}'.format(path))

            # Preparing and printing results
            if epoch % show_step == 0:
                batch_idx = np.random.randint(0, x_test.shape[0], batchsize)
                x_batch = x_test[batch_idx]
                y_batch = y_test[batch_idx]
                estimate, c_test = sess.run([pred, cost], feed_dict={x_plh: x_batch, y_plh: y_batch})
                print('Epoch {:>5} '
                      '| train cost = {:>10.8f} '
                      '| test cost = {:>10.8f} '
                      '| std_ratio = {:>10.8f}'.format(epoch, c_train, c_test, np.std(estimate) / np.std(batch_y)))

                # Apply show function (its arguments should be at this namespace level with the same name)
                if show_function is not None:
                    show_func_args = show_function.__code__.co_varnames[:show_function.__code__.co_argcount]
                    show_function(*[eval(arg) for arg in show_func_args])

def get_node_vars(pred, pattern=None):
    """
    Returns a list of tensorflow variables from a graph node
    Args:
        pred: graph node @type: tensorflow.python.framework.ops.Tensor
        pattern: string pattern for variable names selection @type: str

    Returns:
        a list of tensorflow variables @type list

    """

    get_node_var = lambda node: [var for var in node.op.inputs]

    new_var_list = get_node_var(pred)
    var_list = new_var_list

    while True:
        temp_var_list = []
        for v in new_var_list:
            temp_var_list += get_node_var(v)
        if not temp_var_list:
            break
        var_list += temp_var_list
        new_var_list = temp_var_list

    if pattern is not None:
        for v in var_list:
            if pattern in v.name:
                break
        if pattern in v.name:
            return v
        else:
            print('{} has not been found within the node.'.format(pattern))
    else:
        return var_list

def predict(
        x_test,
        pred,
        weights_path,
        sess=None,
        x_plh_name='x_plh'
):
    """
    Perform a prediction for a x_test provided a prediction tensorflow node and the model parameters.
    Args:
        x_test: input @type numpy.ndarray
        pred: tensorflow prediction node @type tensorflow.python.framework.ops.Tensor
        weights_path: path of weights @type str
        sess: tensorflow session @type tensorflow.python.client.session.Session
        x_plh_name: placeholder for feeding input in model @type tensorflow.python.framework.ops.Tensor

    Returns:
    model estimate @type numpy.ndarray

    """

    # Retrieve x placeholder (with standard name x_plh_name)
    x_plh = get_node_vars(pred, pattern=x_plh_name)
    if x_plh is None:
        raise ValueError('x_plh placeholder has not been found in pred.')

    saver = tf.train.Saver()
    if sess is None:
        with tf.Session() as sess:
            # Load parameters
            # sess.run(tf.global_variables_initializer())
            try:
                global_step = get_saved_idx(weights_path, suffix='epochs')
                complete_save_path = os.path.join(weights_path, 'epochs-' + str(global_step))
                saver.restore(sess, save_path=complete_save_path)
                logging.info('Parameters restored from {}'.format(complete_save_path))
            except OSError:
                print('It has not been possible to restore parameters.')
                return

            # Making prediction
            estimate = sess.run(pred, feed_dict={x_plh: x_test})
    else:
        estimate = sess.run(pred, feed_dict={x_plh: x_test})

    return estimate