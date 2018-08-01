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


class TrainProvider:
    def __init__(self, args, is_training):
        df_train, self.num_train = lmdb_dataflow(args.lmdb_train, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=True)
        batch_train = get_queued_data(df_train.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_gt_points, 3]])
        df_valid, self.num_valid = lmdb_dataflow(args.lmdb_valid, args.batch_size,
                                                 args.num_input_points, args.num_gt_points, is_training=False)
        batch_valid = get_queued_data(df_valid.get_data(), [tf.string, tf.float32, tf.float32],
                                      [[args.batch_size],
                                       [args.batch_size, args.num_input_points, 3],
                                       [args.batch_size, args.num_gt_points, 3]])
        self.batch_data = tf.cond(is_training, lambda: batch_train, lambda: batch_valid)


def train(args):
    is_training_pl = tf.placeholder(tf.bool, shape=(), name='is_training')
    global_step = tf.Variable(0, trainable=False, name='global_step')
    alpha = tf.train.piecewise_constant(global_step, [10000, 20000, 50000],
                                        [0.01, 0.1, 0.5, 1.0], 'alpha_op')

    provider = TrainProvider(args, is_training_pl)
    ids, inputs, gt = provider.batch_data
    num_eval_steps = provider.num_valid // args.batch_size

    model_module = importlib.import_module('.%s' % args.model_type, 'models')
    model = model_module.Model(inputs, gt, alpha)
    add_train_summary('alpha', alpha)

    if args.lr_decay:
        learning_rate = tf.train.exponential_decay(args.base_lr, global_step,
                                                   args.lr_decay_steps, args.lr_decay_rate,
                                                   staircase=True, name='lr')
        learning_rate = tf.maximum(learning_rate, args.lr_clip)
        model.add_train_summary('learning_rate', learning_rate)
    else:
        learning_rate = tf.constant(args.base_lr, name='lr')

    trainer = tf.train.AdamOptimizer(learning_rate)
    train_op = trainer.minimize(model.loss, global_step)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    if args.restore:
        saver.restore(sess, tf.train.latest_checkpoint(args.log_dir))
    else:
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
        os.system('cp models/%s.py %s' % (args.model_type, args.log_dir))   # bkp of model def
        os.system('cp train.py %s' % args.log_dir)                         # bkp of train procedure

    train_summary = tf.summary.merge_all('train_summary')
    valid_summary = tf.summary.merge_all('valid_summary')
    writer = tf.summary.FileWriter(args.log_dir, sess.graph)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    total_time = 0
    train_start = time.time()
    step = sess.run(global_step)
    while not coord.should_stop():
        step += 1
        epoch = step * args.batch_size // provider.num_train + 1
        start = time.time()
        _, loss, summary = sess.run([train_op, model.loss, train_summary],
                                    feed_dict={is_training_pl: True})
        total_time += time.time() - start
        writer.add_summary(summary, step)
        if step % args.steps_per_print == 0:
            print('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                  (epoch, step, loss, total_time / args.steps_per_print))
            total_time = 0
        if step % args.steps_per_eval == 0:
            print(colored('Testing...', 'grey', 'on_green'))
            total_loss = 0
            total_time = 0
            sess.run(tf.local_variables_initializer())
            for i in range(num_eval_steps):
                start = time.time()
                loss, _ = sess.run([model.loss, model.update],
                                   feed_dict={is_training_pl: False})
                total_loss += loss
                total_time += time.time() - start
            summary = sess.run(valid_summary, feed_dict={is_training_pl: False})
            writer.add_summary(summary, step)
            print(colored('epoch %d  step %d  loss %.8f - time per batch %.4f' %
                          (epoch, step, total_loss / num_eval_steps, total_time / num_eval_steps),
                          'grey', 'on_green'))
            total_time = 0
        if step % args.steps_per_visu == 0:
            model_id, pcds = sess.run([ids[0], model.visualize_ops],
                                      feed_dict={is_training_pl: True})
            model_id = model_id.decode('utf-8')
            plot_path = os.path.join(args.log_dir, 'plots',
                                     'epoch_%d_step_%d_%s.png' % (epoch, step, model_id))
            plot_pcd_three_views(plot_path, pcds, model.visualize_titles)
        if step % args.steps_per_save == 0:
            saver.save(sess, os.path.join(args.log_dir, 'model'), step)
            print(colored('Model saved at %s' % args.log_dir, 'white', 'on_blue'))
        if step >= args.max_step:
            break
    print('Total time', datetime.timedelta(seconds=time.time() - train_start))
    coord.request_stop()
    coord.join(threads)
    sess.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_train', default='data/shapenet/train.lmdb')
    parser.add_argument('--lmdb_valid', default='data/shapenet/valid.lmdb')
    parser.add_argument('--log_dir', default='log/pcn_cd')
    parser.add_argument('--model_type', default='pcn_cd')
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_input_points', type=int, default=2048)
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
    args = parser.parse_args()

    train(args)
