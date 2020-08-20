'''
MIT License

Copyright (c) 2018 Wentao Yuan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import argparse
import datetime
import importlib
import models
import os
import tensorflow as tf
import time
from data_util import lmdb_dataflow, get_queued_data
from termcolor import colored
from tf_util import add_train_summary
from visu_util import plot_pcd_three_views


def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')
    inputs_pl = tf.placeholder(tf.float32, (1, None, 3), 'inputs')
    npts_pl = tf.placeholder(tf.int32, (args.batch_size,), 'num_points')
    gt_pl = tf.placeholder(tf.float32, (args.batch_size, args.num_gt_points, 3), 'ground_truths')

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs_pl, npts_pl, gt_pl, alpha)
    add_train_summary('alpha', alpha)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')
    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    df_train, num_train = lmdb_dataflow(
        args.lmdb_train, args.batch_size, args.num_input_points, args.num_gt_points, is_training=True)
    train_gen = df_train.get_data()
    df_valid, num_valid = lmdb_dataflow(
        args.lmdb_valid, args.batch_size, args.num_input_points, args.num_gt_points, is_training=False)
    valid_gen = df_valid.get_data()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    saver = tf.train.Saver()

    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
        writer = tf.summary.FileWriter(args.log_dir)
    else:
        sess.run(tf.global_variables_initializer())
        if os.path.exists(args.log_dir):
            delete_key = input(colored('%s exists. Delete? [y (or enter)/N]'
                                       % args.log_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % args.log_dir)
                os.makedirs(os.path.join(args.log_dir, 'plots'))
        else:
            os.makedirs(os.path.join(args.log_dir, 'plots'))
        with open(os.path.join(args.log_dir, 'args.txt'), 'w') as log:
            for arg in sorted(vars(args)):
                log.write(arg + ': ' + str(getattr(args, arg)) + '\n')     # log of arguments
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))  # bkp of model def
        os.system('cp train.py %s' % args.log_dir)                         # bkp of train procedure
        writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    total_time = 0
    train_start = time.time()
    init_step = sess.run(global_step)
    for step in range(init_step+1, args.max_step+1):
        epoch = step * args.batch_size // num_train + 1
        ids, inputs, npts, gt = next(train_gen)
        start = time.time()
        feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: True}
        _, loss, summary = sess.run([train_op, model.loss, train_summary], feed_dict=feed_dict)
        total_time += time.time() - start
        writer.add_summary(summary, step)
        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                  (epoch, step, loss, total_time / args.steps_per_print))
            total_time = 0
        if step % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            num_eval_steps = num_valid // args.batch_size
            total_loss = 0
            total_time = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                ids, inputs, npts, gt = next(valid_gen)
                feed_dict = {inputs_pl: inputs, npts_pl: npts, gt_pl: gt, is_training_pl: False}
                loss, _ = sess.run([model.loss, model.update], feed_dict=feed_dict)
                total_loss += loss
                total_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                          (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))
            total_time = 0
            if step % args.steps_per_visu == 0:
                all_pcds = sess.run(model.visualize_ops, feed_dict=feed_dict)
                for i in range(0, args.batch_size, args.visu_freq):
                    plot_path = os.path.join(args.log_dir, 'plots',
                                            'epoch_%d_step_%d_%s.png' % (epoch, step, ids[i]))
                    pcds = [x[i] for x in all_pcds]
                    plot_pcd_three_views(plot_path, pcds, model.visualize_titles)
        if step % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))

    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='data/shapenet/valid.lmdb')
    parser.add_argument('--log_dir', default='log/pcn_emd')
    parser.add_argument('--model_type', default='pcn_emd')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=3000)
    parser.add_argument('--num_gt_points', type=int, default=16384)
    parser.add_argument('--base_lr', type=float, default=0.0001)
    parser.add_argument('--lr_decay', action='store_true')
    parser.add_argument('--lr_decay_steps', type=int, default=50000)
    parser.add_argument('--lr_decay_rate', type=float, default=0.7)
    parser.add_argument('--lr_clip', type=float, default=1e-6)
    parser.add_argument('--max_step', type=int, default=300000)
    parser.add_argument('--steps_per_print', type=int, default=100)
    parser.add_argument('--steps_per_eval', type=int, default=1000)
    parser.add_argument('--steps_per_visu', type=int, default=3000)
    parser.add_argument('--steps_per_save', type=int, default=100000)
    parser.add_argument('--visu_freq', type=int, default=5)
    args = parser.parse_args()

    train(args)
