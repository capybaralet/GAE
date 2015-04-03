# GAE

Requires Blocks and Fuel.  Modified from vdumoulin's blocks implementation of VAE.

This is a project about making generative autoencoders.

In contrast with VAE, we use deterministic encoder/decoder, and
we enforce a prior over the latents by encouraging the encoding of the dataset
to appear Gaussian according to some measure.  

In practice, we use minibatches instead of the whole dataset.




