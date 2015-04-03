import logging

from blocks.algorithms import GradientDescent, RMSProp, Momentum
from blocks.bricks import MLP, Linear, Rectifier, Identity
from blocks.bricks.parallel import Fork
#from blocks.datasets.mnist import MNIST
from fuel.datasets.mnist import MNIST
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.extensions import FinishAfter, Timing, Printing
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.roles import add_role, PARAMETER
from blocks.utils import shared_floatx
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams


from utils import *


"""
A WIP implementation of a generative autoencoder.

We use deterministic encoder and decoder, and we use an additional penalty term
on latent encodings that measures goodness of fit to a prior distribution (iid std normal).

We use the Kolmogorov-Smirnov test for the prior cost.
This is the maximum distance at any point between
the CDFs of the prior and empirical distributions for this cost.

This is computed seperately on every component of the latent space.
"""


# TODO: do I have my axes right????

# TODO: add regularization (e.g. denoising or ZBA)


def main():
    regularization = 'ZBA' # TODO (currently NO regularization)
    lr = .001
    step_rule = RMSProp(learning_rate=lr, decay_rate=0.95)
    #step_rule = Momentum(learning_rate=lr, momentum=0.9)
    batch_size = 1000
    nvis, nhid, nlat = 784, 200, 144
    theano_rng = MRG_RandomStreams(134663)

    # Initialize prior
    prior_mu = shared_floatx(numpy.zeros(nlat), name='prior_mu')
    prior_log_sigma = shared_floatx(numpy.zeros(nlat), name='prior_log_sigma')

    # Initialize encoding network
    encoder = MLP(activations=[Rectifier(), Identity()],
                           dims=[nvis, nhid, nlat],
                           weights_init=IsotropicGaussian(std=0.001),
                           biases_init=Constant(0))
    encoder.initialize()

    # Initialize decoding network
    decoder = MLP(activations=[Rectifier(), Identity()],
                           dims=[nlat, nhid, nvis],
                           weights_init=IsotropicGaussian(std=0.001),
                           biases_init=Constant(0))
    decoder.initialize()

    # Encode / decode
    x = tensor.matrix('features')
    x.tag.test_value = np.random.randn(batch_size, 784).astype("float32")
    z = encoder.apply(x)
    z.name = 'z'
    x_hat = decoder.apply(z)
    x_hat.name = 'x_hat'


    # Compute cost
    # TODO: go back to matrix version when T.sort grad is fixed!
    if 0:
        z_fn = F([x], z)
        z_np = z_fn(x.tag.test_value)
        print "z_shape=", z_np.shape
        z_sorted = T.sort(z)
        prior_cdf_values = T.erf(z_sorted)
        empirical_cdf_values = 1./batch_size * T.arange(batch_size)
        ks_diffs = T.maximum(T.sqrt((prior_cdf_values - empirical_cdf_values)**2),
                             T.sqrt((prior_cdf_values - empirical_cdf_values + 1./batch_size)**2))
        prior_cost = T.sum(T.max(ks_diffs, 1))
        prior_cost.name = 'prior_cost'
    else:
        z_fn = F([x], z)
        z_np = z_fn(x.tag.test_value)
        print "z_shape=", z_np.shape
        prior_cdf_values = [T.erf(T.sort(z[:,i])) for i in range(nlat)]
        empirical_cdf_values = 1./batch_size * T.arange(batch_size)
        ks = [T.max(T.maximum(T.sqrt((prior_cdf_values[i] - empirical_cdf_values)**2),
                              T.sqrt((prior_cdf_values[i] - empirical_cdf_values + 1./batch_size)**2)))
                        for i in range(nlat)]
        prior_cost = T.sum(ks)
        prior_cost.name = 'prior_cost'
    reconstruction_cost = ((x - x_hat)**2).mean() * nvis
    reconstruction_cost.name = 'reconstruction_cost'
    cost = (reconstruction_cost + prior_cost).mean()
    cost.name = 'cost'

    # Datasets and data streams
    binary = False
    mnist_train = MNIST(
        'train', start=0, stop=50000, binary=binary, sources=('features',))
    train_loop_stream = DataStream(
        dataset=mnist_train,
        iteration_scheme=SequentialScheme(mnist_train.num_examples, batch_size))
    train_monitor_stream = DataStream(
        dataset=mnist_train,
        iteration_scheme=SequentialScheme(mnist_train.num_examples, batch_size))
    mnist_valid = MNIST(
        'train', start=50000, stop=60000, binary=binary, sources=('features',))
    valid_monitor_stream = DataStream(
        dataset=mnist_valid,
        iteration_scheme=SequentialScheme(mnist_valid.num_examples, batch_size))
    mnist_test = MNIST('test', binary=binary, sources=('features',))
    test_monitor_stream = DataStream(
        dataset=mnist_test,
        iteration_scheme=SequentialScheme(mnist_test.num_examples, batch_size))

    # Get parameters
    computation_graph = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(computation_graph.variables)

    # Training loop
    algorithm = GradientDescent(cost=cost, params=params, step_rule=step_rule)
    monitored_quantities = [cost, reconstruction_cost, prior_cost]
    main_loop = MainLoop(
        model=None, data_stream=train_loop_stream, algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=10000),
            DataStreamMonitoring(
                monitored_quantities, train_monitor_stream, prefix="train"),
            DataStreamMonitoring(
                monitored_quantities, valid_monitor_stream, prefix="valid"),
            DataStreamMonitoring(
                monitored_quantities, test_monitor_stream, prefix="test"),
            Printing()])
    main_loop.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
