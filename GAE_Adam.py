import logging

from blocks.algorithms import GradientDescent, RMSProp, Momentum, Adam
from blocks.bricks import MLP, Linear, Rectifier, Identity
from blocks.bricks.parallel import Fork
#from blocks.datasets.mnist import MNIST
from fuel.datasets.mnist import MNIST
from blocks.datasets.streams import DataStream
from blocks.datasets.schemes import SequentialScheme
from blocks.extensions import FinishAfter, Timing, Printing, DumpAlgorithmParams
from blocks.extensions.saveload import Checkpoint, Dump
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.initialization import Constant, IsotropicGaussian
from blocks.main_loop import MainLoop
from blocks.roles import add_role, PARAMETER
from blocks.utils import shared_floatx
from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams

#from blocks import * # TODO

from utils import *


"""
A WIP implementation of a generative autoencoder.

We use deterministic encoder and decoder, and we use an additional penalty term
on latent encodings that measures goodness of fit to a prior distribution (iid std normal).

We use the Kolmogorov-Smirnov test for the prior cost.
This is the maximum distance at any point between
the CDFs of the prior and empirical distributions for this cost.

X This is computed seperately on every component of the latent space.
^ NOT ANY MORE! that doesn't make sense
^ NOW, we instead look at the norm of the points, and compare this to (an approximate) CDF of the norm according to the prior
    This ends up being another Gaussian with mean = sqrt(nlat - 1) and std = 1
    We assume that the reconstruction error will encourage the encoder to spread points out 
      along each hypersphere of a given norm (so we don't have to include this in the criterion)
"""


# TODO: do I have my axes right????

# TODO: add regularization (e.g. denoising or ZBA)


try: filename = os.path.basename(__file__)[:-3] # this script's name (minus '.py')
except: pass
subdir = 'GAE/'
path = '/Tmp/kruegerd/' + subdir
fs_path = '/data/lisatmp/kruegerd/' + subdir
if not os.path.exists(path): os.makedirs(path)
if not os.path.exists(fs_path): os.makedirs(fs_path)
log_me(subdir+filename)


#def main():
if 1:
    regularization = 'ZBA' # TODO (currently NO regularization)
    #lr = .001
    step_rule = Adam()
    #step_rule = RMSProp(learning_rate=lr, decay_rate=0.95)
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
    x.tag.test_value = np.random.randn(batch_size, 784).astype("float32") # doesn't work (need to change theano flags?)
    #ztest = T.dot(theano.shared(np.random.randn(batch_size, batch_size)), x)
    #print ztest.tag.test_value.shape
    z = encoder.apply(x)
    z.name = 'z'
    x_hat = decoder.apply(z)
    x_hat.name = 'x_hat'


    def cdf(x, mu):
        return .5*(1 + T.erf((x - mu) / 2**.5))

    mu = (nlat - 1)**.5

    empirical_cdf_values = 1./batch_size * T.arange(batch_size)

    # Compute cost
    # TODO: go back to matrix version when T.sort grad is fixed!
    # TODO: currently it only works when flattening the matrix...
    if 0: # TODO...
        z_fn = F([x], z)
        z_np = z_fn(x.tag.test_value)
        print "z_shape=", z_np.shape
        z_sorted = T.sort(z)
        prior_cdf_values = cdf(z_sorted)
        ks_diffs = T.maximum(T.abs_(prior_cdf_values - empirical_cdf_values),
                             T.abs_(prior_cdf_values - empirical_cdf_values + 1./batch_size))
        prior_cost = T.sum(T.max(ks_diffs, 1))
        #print prior_cost.dtype
        prior_cost.name = 'prior_cost'
    else:
        z_fn = F([x], z)
        z_np = z_fn(x.tag.test_value)
        print "z_shape=", z_np.shape
        z_norms = T.sqrt(T.sum(z**2, 1))
        prior_cdf_values = [cdf(T.sort(z_norms), mu) for i in range(nlat)]
        prior_cost = T.max(T.maximum(T.abs_(prior_cdf_values - empirical_cdf_values),
                              T.abs_(prior_cdf_values - empirical_cdf_values + 1./batch_size)))
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
        # TODO: lr adjustment in the loop
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=10000),
            DataStreamMonitoring(
                monitored_quantities, train_monitor_stream, prefix="train"),
            DataStreamMonitoring(
                monitored_quantities, valid_monitor_stream, prefix="valid"),
            DataStreamMonitoring(
                monitored_quantities, test_monitor_stream, prefix="test"),
            Printing(),
            #Checkpoint('GAE_Momentum.pkl'),
            DumpAlgorithmParams(save_path=path + filename, every_n_epochs=10)])
    main_loop.run()


    close()
    # get samples
    nsamples = 49
    seed = np.random.randn(nsamples, 144).astype("float32")
    samples_fn = F([z], x_hat)
    samples_np = samples_fn(seed)
    mimshows(samples_np.reshape((-1, 28, 28)))

    # plot learning curves
    log = main_loop.log
    vcosts = []
    tcosts = []
    for i in range(0, len(log), 50):
        ent = log[i]
        vcosts.append(ent['valid_cost'])
        tcosts.append(ent['train_cost'])
    figure()
    plot(vcosts); plot(tcosts)


    # change learning_rate and resume training
    lr = algorithm.step_rule.learning_rate.get_value()
    multiplier = .5
    algorithm.step_rule.learning_rate.set_value(array(multiplier * lr, dtype=float32))
    main_loop.run()














