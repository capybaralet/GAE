#TODO: add a list that contains names of all utils here (for reference)
#TODO: add docstrings
#TODO: add paths
#TODO: define test matrices for checking things!

"""
Some syntax I'd like to add to python (I think this is easy to do, although dangerous)

list.prepend
l1[l2]
if a is in l:

try without except (assumes except: pass)

strings should support item assignment (e.g. str[4] = 'r')
    oh wait, nvm, they are immutable... actually, I think I am used to numpy and
    think python should be pass-by-value, not pass-by-reference, as a result...
    but it is neither... see here: http://stupidpythonideas.blogspot.ca/2013/11/does-python-pass-by-value-or-by.html

"""
#from __future__ import absolute_import

import sys
import os
import time
import cPickle as pickle
import subprocess
import collections
import warnings
import ipdb

import numpy
np = numpy
import numpy.random
shuffle = numpy.random.shuffle
permutation = numpy.random.permutation
import scipy
from scipy.io import wavfile as wav
from scipy.io import loadmat, savemat

import matplotlib.pyplot as plt
import pylab
from pylab import *

# causes problems when you specify the GPU# now...
import theano
import theano.tensor as T
from theano import shared as ts
from theano import function as F
from theano.tensor.shared_randomstreams import RandomStreams as RS
trng = RS(123)


from dk_reconstruct import vocoder_synth

# import theano.printing.debugprint as print_graph


# problems when local version of pylearn exists!
# for some reason, this doesn't solve it... now it will only look at the copy in repo
#sys.path.insert(0, '/u/kruegerd/repo/pylearn2/')
#from pylearn2.utils import serial
#sys.path.insert(0, '')

# This is how I was doing it before...
#
#    fl = sys._getframe(1).f_locals
#    fl['sys'] = sys

#_______________________________________________________________________________
#from numpy import array as A

def A(xx):
    return np.array(xx).astype("float32")

#_______________________________________________________________________________
# for quick testing:

vec3 = A([-1,-2,-3])
vec5 = A(range(5))
arr23 = A([[1,2,3], [4,5,6]])
arr79 = A(range(63)).reshape((7,9))
arr235 = A([[[1,2,3], [4,5,6]], [[0,1,2], [4,5,6]], [[1,2,3], [0,5,6]], [[-1,2,3], [4,5,6]], [[1,2,3], [-4,5,6]]]).transpose((1,2,0))

def tso(list):
    return theano.shared(np.ones(list))

#_______________________________________________________________________________
# data paths:
tmpp = '/Tmp/kruegerd/'
ltmpp = '/data/lisatmp/kruegerd/'
sgp = '/u/kruegerd/TTS_current/speechgeneration/'
dldp = '/data/lisa/data/'
timp = '/data/lisa/data/timit/readable/'
lcs = '_learning_curve.npy'


#_______________________________________________________________________________
# experiment logging

def log_me(filepath):
    # TODO: timestamp, and deal with resaving results for ongoing experiments...
    proc = subprocess.Popen(['hostname'], stdout=subprocess.PIPE)
    lines = proc.stdout.readlines()
    hostname = lines[0][:-1]
    fil = open('/u/kruegerd/exp_log.txt', 'a')
    fil.write('\n' + hostname + '\n' + "filepath=" + filepath)
    fil.close()

#_______________________________________________________________________________
# PLOTS

# made for visualization with matplotlib, esp FFTs.
def logsquash(arr):
    return np.log(arr - np.min(arr) + 1)

def mimshow(img, range=None, squash=False):
    # TODO: remove figure!
    #plt.figure()
    if squash:
        img = logsquash(img)
    if range is not None:
        vmin = range[0]
        vmax = range[1]
        plt.imshow(img, interpolation='none', cmap="Greys", vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, interpolation='none', cmap="Greys")

def mimsave(path, img, range=None, squash=False):
    plt.figure()
    mimshow(img, range, squash)
    plt.savefig(path)
    plt.close()
"""
# TODO: should take an expression and make a plot...
def eplot(exp):
    xx = np.linspace()...
"""

def fplot(xx):
    plt.figure()
    plt.plot(xx)

def mplots(list_of_vecs, maxnrows=10, background_vecs=None):
    plt.figure()
    nrows = min(int(len(list_of_vecs) ** .5), maxnrows)
    print "nrows=", nrows
    for i in range(nrows**2):
        plt.subplot(nrows, nrows, i+1)
        plt.plot(list_of_vecs[i])
        if background_vecs is not None:
            for vec in background_vecs:
                plt.plot(vec)

def mimshows(list_of_mats, maxnrows=10, normalize=True):
    plt.figure()
    nrows = min(int(len(list_of_mats) ** .5), maxnrows)
    print "nrows=", nrows
    if normalize:
        vmin = np.min(list_of_mats)
        vmax = np.max(list_of_mats)
    for i in range(nrows**2):
        plt.subplot(nrows, nrows, i+1)
        plt.imshow(list_of_mats[i], interpolation='none', cmap="Greys", vmin=vmin, vmax=vmax)


#_______________________________________________________________________________
# OTHER CODE (MINE!)

def nprint(obj):
    print '\n \n'
    print obj
    print '\n \n'

# FIXME (use frames???)
def dprint(str):
    print str,"=", locals()[str]


def rlen(x):
    return range(len(x))


def log_p_x_beta(x, a, b):
    return np.log( x**(a-1) * (1-x)**(b-1) / scipy.special.beta(a,b) )

def theano_beta_fn(a, b):
    return T.gamma(a) * T.gamma(b) / T.gamma(a+b)

def theano_log_p_x_beta(x, a, b):
    return T.log( x**(a-1) * (1-x)**(b-1) / theano_beta_fn(a,b) )


def segment_ddm(dmat, length, overlap=0):
    """Segment a design matrix dataset into shorter examples"""
    return np.vstack(segment_axis(dmat, length, overlap, 1))

def my_slice(arr, frame_len, subframe_len, start_ind=0):
    """
    Slices an array with concatenated frames returning an array
        with the same semantics, but with subframes of the frames slices out.
    Assumes the frames are concatenated in the last dimension.
    """
    shape = arr.shape
    ndims = len(shape)
    tmp = arr.reshape(shape[:-1] + (-1, frame_len)).transpose(range(ndims+1)[::-1])
    return tmp[start_ind:start_ind + subframe_len].transpose(range(ndims+1)[::-1]).reshape(arr.shape[:-1] + (-1,))

def replace_slice(arr, replacement_arr, frame_len, subframe_len, start_ind=0):
    """
    Like my_slice, slices an array with concatenated frames returning an array
        with the same semantics, but with subframes of the frames REPLACED by replacement_arr.
    Assumes the frames are concatenated in the last dimension.
    """
    shape = arr.shape
    ndims = len(shape)
    tmp = arr.reshape(shape[:-1] + (-1, frame_len)).transpose(range(ndims+1)[::-1])
    replacement_tmp = replacement_arr.reshape(shape[:-1] + (-1, subframe_len)).transpose(range(ndims+1)[::-1])
    tmp[start_ind:start_ind + subframe_len] = replacement_tmp
    return tmp.transpose(range(ndims+1)[::-1]).reshape(arr.shape[:-1] + (-1,))


def try_save(path, arr):
    try:
        np.save(path, arr)
    except:
        subprocess.call('ls' + path) # FIXME: complete savepath won't necessarily exist!
        time.sleep(1)
        np.save(path, arr)

def try_save_obj(path, obj):
    try:
        obj.save(path)
    except:
        subprocess.call('ls' + path) # FIXME: complete savepath won't necessarily exist!
        time.sleep(1)
        obj.save(path)

def incorporate(list, list_or_element):
    """ If its a list, we want to add the elements,
        If its not, we want to add it"""
    if isinstance(list_or_element, list):
        list.extend(list_or_element)
    else:
        list.append(list_or_element)

def time_dhm(seconds):
    m = seconds / 60
    h = m / 60
    m = m % 60
    d = h / 24
    h = h % 24
    return str(d)+' days, '+str(h)+' hours, '+str(m)+' minutes'


def sigmoidd(x):
    return 1. / (1 + np.exp(-x))

def softmaxx(vec):
    expd = np.exp(vec)
    summ = np.sum(expd)
    return expd/summ

# "spherical softmax"
def ssoftmaxx(vec):
    vec = vec**2
    summ = np.sum(vec)
    return np.log(vec/summ)

def numnans(arr):
    return np.sum(np.isnan(arr))

def isin_name(var, substr):
    if substr in var.name:
        return True
    else:
        return False

def isin(str, substr):
    if substr in str:
        return True
    else:
        return False

#def shortest(arr):
#    return min(arr.shape)


# should make another copy for arrays!
def trim(list):
    """trim a list of lists down to the shortest length among them"""
    minlen = np.inf
    for i in list:
        if len(i) < minlen:
            minlen = len(i)
    mylist = []
    for i in list:
        mylist.append(i[:minlen])
    return mylist



#_______________________________________________________________________________
# OTHERS' code


def moving_average(values,window):
    weigths = np.repeat(1.0, window)/window
    #including valid will REQUIRE there to be enough datapoints.
    #for example, if you take out valid, it will start @ point one,
    #not having any prior points, so itll be 1+0+0 = 1 /3 = .3333
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array


def shared_normal(num_rows, num_cols, scale=1):
    return theano.shared(numpy.random.normal(
        scale=scale, size=(num_rows, num_cols)).astype(theano.config.floatX))

def shared_zeros(*shape):
    return theano.shared(numpy.zeros(shape, dtype=theano.config.floatX))




# modified from http://stackoverflow.com/questions/2459295/invertible-stft-and-istft-in-python
def stft(x, amp_phase=0, fs=1, framesz=320., hop=160.):
    """
     x - signal
     fs - sample rate
     framesz - frame size
     hop - hop size (frame size = overlap + hop size)
    """
    framesamp = int(framesz*fs)
    hopsamp = int(hop*fs)
    w = scipy.hamming(framesamp)
    X = scipy.array([scipy.fft(w*x[i:i+framesamp]) 
                     for i in range(0, len(x)-framesamp, hopsamp)])
    Xamp = np.abs(X)
    Xphase = np.angle(X)
    if amp_phase:
        return X, Xamp, Xphase
    else:
        return X




def istft(X, fs, T, hop): 
    """ T - signal length """ # why do I need it!!??!?
    length = T*fs
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    # calculate the inverse envelope to scale results at the ends.
    env = scipy.zeros(T*fs)
    w = scipy.hamming(framesamp)
    for i in range(0, len(x)-framesamp, hopsamp):
        env[i:i+framesamp] += w
    env[-(length%hopsamp):] += w[-(length%hopsamp):]
    env = np.maximum(env, .01)
    return x/env # right side is still a little messed up...

def istft(X, fs=1, hop=160.): #FIXME?
    """ T - signal length """
    T = (X.shape[0]+1)*hop
    length = T
    x = scipy.zeros(T*fs)
    framesamp = X.shape[1]
    hopsamp = int(hop*fs)
    for n,i in enumerate(range(0, len(x)-framesamp-1, hopsamp)):
        x[i:i+framesamp] += scipy.real(scipy.ifft(X[n]))
    # calculate the inverse envelope to scale results at the ends.
    env = scipy.zeros(T*fs)
    w = scipy.hamming(framesamp)
    for i in range(0, len(x)-framesamp, hopsamp):
        env[i:i+framesamp] += w
    env[-(length%hopsamp):] += w[-(length%hopsamp):]
    env = np.maximum(env, .01)
    return x/env # right side is still a little messed up...

def abs_stft(x):
    stftd = stft(x)
    stftd[:,:160] = np.abs(stftd[:,:160])
    stftd[:,160:] = np.zeros_like(stftd[:,160:])
    return stftd

def stftd(x): # FIXME!
    return istft(abs_stft(x))



# from https://github.com/vdumoulin/sheldon
from numpy.lib.stride_tricks import as_strided
def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.

    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces. 

            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value

    endvalue : object
        The value to use for end='pad'


    Examples
    --------
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').

    use as_strided

    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be "\
                          "positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' "\
                          "mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)


# from http://www.iro.umontreal.ca/~memisevr/code/logreg.py
def onehot(x,numclasses=None):
    """ Convert integer encoding for class-labels (starting with 0 !)
        to one-hot encoding.
        The output is an array who's shape is the shape of the input array plus
        an extra dimension, containing the 'one-hot'-encoded labels.
    """
    if x.shape==():
        x = x[None]
    if numclasses is None:
        numclasses = x.max() + 1
    result = numpy.zeros(list(x.shape) + [numclasses], dtype="int")
    z = numpy.zeros(x.shape)
    for c in range(numclasses):
        z *= 0
        z[numpy.where(x==c)] = 1
        result[...,c] += z
    return result

# from http://glowingpython.blogspot.ca/2011/07/prime-factor-decomposition-of-number.html
from math import floor
def factor(n):
 result = []
 for i in range(2,n+1): # test all integers between 2 and n
  s = 0;
  while n/float(i) == floor(n/float(i)): # is n/i an integer?
   n = n/float(i)
   s += 1
  if s > 0:
   for k in range(s):
    result.append(i) # i is a pf s times
   if n == 1:
    return result


#_______________________________________________________________________________
# WIPs


# WIP I need to figure out the scope better
# the idea is to use a global variable to turn all print statements on or off.
printing_on = 1
def mprint(str):
    if printing_on:
        print(str)
    else:
        pass

#TODO
def contains(container, obj): 
    """test if obj is in some"""
    return "this is a WIP"



"""
# check if filename exists, if it does, load and return
# otherwise, make object from object_maker, and save as filename, and return


def saveload(filename, object_maker):
    import os
    if os.path.isfile(filename)
    return

if os.path.isfile(filename):
    return np.load(filename)
else:
    np.save(filename, object)
"""

""" WIP
def frame_size(shapes, strides):
    # both inputs are lists, in order from top to bottom
    fss = [1]
    fs = 1
    le = len(shapes)
    for i in range(le-1):
        fs += shapes[i]
        #fss.append(fs)
        fs += (shapes[i+1]-1)*strides[i]
"""


