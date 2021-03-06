from datetime import datetime
import os
import re
import sys

import numpy as np
import tensorflow as tf

from layers import Dense
import plot
from utils import composeAll, print_

class VAE():
    "Variational Autoencoder"

    DEFAULTS = {
        "batch_size": 128,
        "learning_rate": 1E-3,
        "dropout": 1.0,
        "lambda_l2_reg": 0.0,
        "nonlinearity": tf.nn.elu,
        "squashing": tf.nn.sigmoid
    }

    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, meta_graph=None,
                 save_graph_def=True, log_dir="./log"):
        self.architecture = architecture
        self.__dict__.update(VAE.DEFAULTS, **d_hyperparams)
        self.sesh = tf.Session()

        if not meta_graph: # new model
            self.datetime = datetime.now().strftime(r"%y%m%d_%H%M")
            assert len(self.architecture) > 2, \
                "Architecture must have more layers!"

            handles = self._buildGraph()
            for handle in handles:
                tf.add_to_collection(VAE.RESTORE_KEY, handle)
            self.sesh.run(tf.initialize_all_variables())

        else: # restore saved model
            model_datetime, model_name = os.path.basename(meta_graph).split("_vae_")
            self.datetime = "{}_reloaded".format(model_datetime)
            *model_architecture, _ = re.split("_|-", model_name)
            self.architecture = [int(n) for n in model_architecture]

            # FIXME ココらへん何やってるんだろ？
            meta_graph = os.path.abspath(meta_graph)
            tf.train.import_meta_graph(meta_graph + ".meta").restore(
                self.sesh, meta_graph)
            handles = self.sesh.graph.get_collection(VAE.RESTORE_KEY)

        (self.x_in, self.dropout_, self.z_mean, self.z_log_sigma,
         self.x_reconstructed, self.z_, self.x_reconstructed_,
         self.cost, self.global_step, self.train_op) = handles

        if save_graph_def:
            self.logger = tf.train.SummaryWriter(log_dir, self.sesh.graph)

    @property
    def step(self):
        return self.global_step.eval(session=self.sesh)


    def _buildGraph(self):
        x_in = tf.placeholder(tf.float32, shape=[None, self.architecture[0]],
                              name="x")
        dropout = tf.placeholder_with_default(1., shape=[], name="dropout")

        # encoding / "recognition": q(z|x)
        encoding = [Dense("encoding", hidden_size, dropout, self.nonlinearity)
                    for hidden_size in reversed(self.architecture[1:-1])]
        h_encoded = composeAll(encoding)(x_in)

        # latent distribution parameterized by hidden encoding
        # z ~ N(z_mean, np.exp(z_log_sigma)**2)
        z_mean = Dense("z_mean", self.architecture[-1], dropout)(h_encoded)
        z_log_sigma = Dense("z_log_sigma", self.architecture[-1], dropout)(h_encoded)

        z = self.sampleGaussian(z_mean, z_log_sigma)

        # decoding / "generative": p(x|z)
        decoding = [Dense("decoding", hidden_size, dropout, self.nonlinearity)
                    for hidden_size in self.architecture[1:-1]]
        # final reconstruction: restore original dims, squash outputs [0, 1] ???
        decoding.insert(0, Dense("x_decoding", self.architecture[0], dropout, self.squashing))
        x_reconstructed = tf.identity(composeAll(decoding)(z), name="x_reconstructed")

        # reconstruction loss
        rec_loss = VAE.crossEntropy(x_reconstructed, x_in)

        # Kullback-Leibler divergense
        kl_loss = VAE.kullbackLeibler(z_mean, z_log_sigma)

        with tf.name_scope("l2_regularization"):
            regularizers = [tf.nn.l2_loss(var) for var in self.sesh.graph.get_collection(
                "trainable_variables") if "weights" in var.name]
            l2_reg = self.lambda_l2_reg * tf.add_n(regularizers)

        with tf.name_scope("cost"):
            cost = tf.reduce_mean(rec_loss + kl_loss, name="vae_cost")
            cost += l2_reg

        # optimization
        global_step = tf.Variable(0, trainable=False)
        with tf.name_scope("Adam_optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            tvars = tf.trainable_variables()
            grads_and_vars = optimizer.compute_gradients(cost, tvars)
            clipped = [(tf.clip_by_value(grad, -5, 5), tvar)
                        for grad, tvar in grads_and_vars]
            train_op = optimizer.apply_gradients(clipped, global_step=global_step,
                                                 name="minimize_cost")

        # ops to directly explore latent space, defaults to prior z ~ N(0, I)
        # FIXME x_reconstructedとx_reconstructed_の違いって何？
        with tf.name_scope("latent_in"):
            z_ = tf.placeholder_with_default(tf.random_normal([1, self.architecture[-1]]),
                                             shape=[None, self.architecture[-1]],
                                             name="latent_in")
        x_reconstructed_ = composeAll(decoding)(z_)

        return (x_in, dropout, z_mean, z_log_sigma, x_reconstructed,
                z_, x_reconstructed_, cost, global_step, train_op)

    def sampleGaussian(self, mu, log_sigma):
        with tf.name_scope("sample_gaussian"):
            epsilon = tf.random_normal(tf.shape(log_sigma), name="epsilon")
            return mu + epsilon * tf.exp(log_sigma) # N(mu, I * sigma**2)

    @staticmethod
    def crossEntropy(obs, actual, offset=1E-7):
        with tf.name_scope("cross_entropy"):
            # bound by clipping to avoid nan
            obs_ = tf.clip_by_value(obs, offset, 1 - offset)
            return - tf.reduce_sum(actual * tf.log(obs_) +
                                    (1 - actual) * tf.log(1 - obs_), 1)

    @staticmethod
    def l1_loss(obs, actual):

        with tf.name_scope("l1_loss"):
            return tf.reduce_sum(tf.abs(obs - actual), 1)

    @staticmethod
    def l2_loss(obs, actual):
        with tf.name_scope("l2_loss"):
            return tf.reduce_sum(tf.square(obs - actual), 1)

    @staticmethod
    def kullbackLeibler(mu, log_sigma):
        with tf.name_scope("KL_divergence"):
            return - 0.5 * tf.reduce_sum(1 + 2 * log_sigma - mu ** 2 -
                                         tf.exp(2 * log_sigma), 1)

    def encode(self, x):
        feed_dict = {self.x_in: x}
        return self.sesh.run([self.z_mean, self.z_log_sigma], feed_dict=feed_dict)

    def decode(self, zs=None):
        feed_dict = dict()
        if zs is not None:
            is_tensor = lambda x: hasattr(x, "eval")
            zs = (self.sesh.run(zs) if is_tensor(zs) else zs)
            feed_dict.update({self.z_: zs})
        return self.sesh.run(self.x_reconstructed_, feed_dict=feed_dict)

    def vae(self, x):
        return self.decode(self.sampleGaussian(*self.encode(x)))

    def train(self, X, max_iter=np.inf, max_epochs=np.inf, cross_validate=True,
              verbose=True, save=True, outdir="./out", plots_outdir="./png",
              plot_latent_over_time=False):
        if save:
            saver = tf.train.Saver(tf.all_variables())

        try:
            err_train = 0
            now = datetime.now().isoformat()[11:]
            print("------- Training begin: {} -------\n".format(now))

            if plot_latent_over_time:
                BASE = 2
                INCREMENT = 0.5
                pow_ = 0

            while True:
                x, _ = X.train.next_batch(self.batch_size)
                feed_dict = {self.x_in: x, self.dropout_: self.dropout}
                fetches = [self.x_reconstructed, self.cost, self.global_step, self.train_op]
                x_reconstructed, cost, i, _ = self.sesh.run(fetches, feed_dict)

                err_train += cost

                if plot_latent_over_time:
                    while int(round(BASE**pow_)) == i:
                        plot.exploreLatent(self, nx=30, ny=30, ppf=True, outdir=plots_outdir,
                                           name="explore_ppf30_{}".format(pow_))

                        names = ("train", "validation", "test")
                        datasets = (X.train, X.validation, X.test)
                        for name, dataset in zip(names, datasets):
                            plot.plotInLatent(self, dataset.images, dataset.labels, range_=
                                              (-6, 6), title=name, outdir=plots_outdir,
                                              name="{}_{}".format(name, pow_))

                        print("{}^{} = {}".format(BASE, pow_, i))
                        pow_ += INCREMENT

                if i%1000 == 0 and verbose:
                    print("round {} --> avg cost: ".format(i), err_train / i)

                if i%2000 == 0 and verbose:# and i >= 10000:
                    # visualize `n` examples of current minibatch inputs + reconstructions
                    plot.plotSubset(self, x, x_reconstructed, n=10, name="train",
                                    outdir=plots_outdir)

                    if cross_validate:
                        x, _ = X.validation.next_batch(self.batch_size)
                        feed_dict = {self.x_in: x}
                        fetches = [self.x_reconstructed, self.cost]
                        x_reconstructed, cost = self.sesh.run(fetches, feed_dict)

                        print("round {} --> CV cost: ".format(i), cost)
                        plot.plotSubset(self, x, x_reconstructed, n=10, name="cv",
                                        outdir=plots_outdir)

                if i >= max_iter or X.train.epochs_completed >= max_epochs:
                    print("final avg cost (@ step {} = epoch {}): {}".format(
                        i, X.train.epochs_completed, err_train / i))
                    now = datetime.now().isoformat()[11:]
                    print("------- Training end: {} -------\n".format(now))

                    if save:
                        outfile = os.path.join(os.path.abspath(outdir), "{}_vae_{}".format(
                            self.datetime, "_".join(map(str, self.architecture))))
                        saver.save(self.sesh, outfile, global_step=self.step)
                    try:
                        self.logger.flush()
                        self.logger.close()
                    except(AttributeError): # not logging
                        continue
                    break

        except(KeyboardInterrupt):
            print("final avg cost (@ step {} = epoch {}): {}".format(
                i, X.train.epochs_completed, err_train / i))
            now = datetime.now().isoformat()[11:]
            print("------- Training end: {} -------\n".format(now))
            sys.exit(0)
