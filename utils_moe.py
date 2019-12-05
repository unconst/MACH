"""Utilities for creating Sparsely-Gated Mixture-of-Experts Layers.
See "Outrageously Large Neural Networks"
https://arxiv.org/abs/1701.06538
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import math
import six
from six.moves import range  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin

from tensor2tensor.layers import common_layers
from tensor2tensor.layers.vq_discrete import DiscreteBottleneck

import tensorflow as tf

DEFAULT_DEV_STRING = "existing_device"

import io
import matplotlib.pyplot as plt
import tensorflow as tf

def _networkx(components):
    G = nx.DiGraph()

    node_labels = {}
    node_sizes = []
    for c in components:
        G.add_node(c.name)
        node_labels[c.name] = str(c.name)
        node_sizes.append(0.1 + c.revenue)

    edge_labels = {}
    for parent in components:
        for child in components:
            G.add_edge(parent.name, child.name)
            edge_labels[(parent.name, child.name)] = "%.3f" % parent.weights[child.name]

    forceatlas2 = ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution=True,  # Dissuade hubs
                            linLogMode=False,  # NOT IMPLEMENTED
                            adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence=1.0,

                            # Performance
                            jitterTolerance=1.0,  # Tolerance
                            barnesHutOptimize=True,
                            barnesHutTheta=1.2,
                            multiThreaded=False,  # NOT IMPLEMENTED

                            # Tuning
                            scalingRatio=2.0,
                            strongGravityMode=False,
                            gravity=1.0,

                            # Log
                            verbose=False)

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=2000)
    pos_higher = {}
    y_off = 0.2
    for k, v in positions.items():
        pos_higher[k] = (v[0], v[1]+y_off)

    nx.draw_networkx_nodes(G, positions, with_labels=True, node_size=node_sizes, node_color="blue", alpha=0.4)
    nx.draw_networkx_edges(G, positions, arrowstyle='->', arrowsize=15, edge_color="green", edge_labels=edge_labels, alpha=0.05, label_pos=0.3)
    nx.draw_networkx_labels(G, pos_higher, node_labels)
    nx.draw_networkx_edge_labels(G, pos_higher, edge_labels=edge_labels, with_labels=True, label_pos=0.3)

def metagraph_plot(components, tblogger, step, hparams):
    figure = plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.savefig(hparams.log_dir + "/" + run_prefix + str('/metagraph'))
    _networkx(components)
    tblogger.log_plot('metagraph', step)

def add_scope(scope=None, scope_fn=None):
  """Return a decorator which add a TF name/variable scope to a function.
  Note that the function returned by the decorator accept an additional 'name'
  parameter, which can overwrite the name scope given when the function is
  created.
  Args:
    scope (str): name of the scope. If None, the function name is used.
    scope_fn (fct): Either tf.name_scope or tf.variable_scope
  Returns:
    fct: the add_scope decorator
  """
  def decorator(f):

    @functools.wraps(f)
    def decorated(*args, **kwargs):
      name = kwargs.pop("name", None)  # Python 2 hack for keyword only args
      with scope_fn(name or scope or f.__name__):
        return f(*args, **kwargs)

    return decorated

  return decorator


def add_var_scope(scope=None):
  return add_scope(scope, scope_fn=tf.compat.v1.variable_scope)


def add_name_scope(scope=None):
  return add_scope(scope, scope_fn=tf.name_scope)


def _add_variable_proxy_methods(var, proxy_tensor):
  """Proxy methods of underlying variable.
  This enables our custom getters to still work with, e.g., batch norm.
  Args:
    var: Variable to proxy
    proxy_tensor: Tensor that is identity of var
  """
  proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
  proxy_tensor.assign_sub = var.assign_sub
  proxy_tensor.assign = var.assign
  proxy_tensor.initialized_value = var.initialized_value


class Parallelism(object):
  """Helper class for creating sets of parallel function calls.
  The purpose of this class is to replace this code:
      e = []
      f = []
      for i in range(len(devices)):
        with tf.device(devices[i]):
          e_, f_ = func(a[i], b[i], c)
          e.append(e_)
          f.append(f_)
  with this code:
      e, f = expert_utils.Parallelism(devices)(func, a, b, c)
  """

  def __init__(self,
               device_names_or_functions,
               reuse=True,
               caching_devices=None,
               daisy_chain_variables=False,
               ps_devices=None):
    """Create a Parallelism.
    Args:
      device_names_or_functions: A list of length n, containing device names
        or device functions (see `tf.device`)
      reuse: True or None.  Whether to reuse variables created in the first
        replica in the subsequent replicas.
      caching_devices: Either `None`, or a list of length n containing device
        names.
      daisy_chain_variables: a boolean - if true, then copies variables in a
        daisy chain between devices.
      ps_devices: list<str>, list of devices for experts.
    Returns:
      a Parallelism.
    """
    assert device_names_or_functions
    self._devices = device_names_or_functions
    self._n = len(device_names_or_functions)
    self._reuse = reuse
    self._caching_devices = self._maybe_repeat(caching_devices)
    self._daisy_chain_variables = daisy_chain_variables
    self._ps_devices = ps_devices or [""]

  def __call__(self, fn, *args, **kwargs):
    """A parallel set of function calls (using the specified devices).
    Args:
      fn: a function or a list of n functions.
      *args: additional args.  Each arg should either be not a list, or a list
         of length n.
      **kwargs: additional keyword args.  Each arg should either be not a
         list, or a list of length n.
    Returns:
      either a single list of length n (if fn does not return a tuple), or a
      tuple of lists of length n (if fn returns a tuple).
    """
    # Construct lists or args and kwargs for each function.
    if args:
      my_args = transpose_list_of_lists(
          [self._maybe_repeat(arg) for arg in args])
    else:
      my_args = [[] for _ in range(self.n)]
    my_kwargs = [{} for _ in range(self.n)]
    for k, v in six.iteritems(kwargs):
      vals = self._maybe_repeat(v)
      for i in range(self.n):
        my_kwargs[i][k] = vals[i]

    # Construct lists of functions.
    fns = self._maybe_repeat(fn)

    # Now make the parallel call.
    outputs = []
    cache = {}
    tensor_to_var = {}
    for i in range(self.n):

      def daisy_chain_getter(getter, name, *args, **kwargs):
        """Get a variable and cache in a daisy chain."""
        device_var_key = (self._devices[i], name)
        if device_var_key in cache:
          # if we have the variable on the correct device, return it.
          return cache[device_var_key]
        if name in cache:
          # if we have it on a different device, copy it from the last device
          last_device_v = cache[name]
          var = tensor_to_var[last_device_v]
          v = tf.identity(last_device_v)
        else:
          var = getter(name, *args, **kwargs)
          v = var.read_value()

        # keep track of the original variable
        tensor_to_var[v] = var
        _add_variable_proxy_methods(tensor_to_var[v], v)
        # update the cache
        cache[name] = v
        cache[device_var_key] = v
        return v

      # Variable scope will not reset caching_device on reused variables,
      # so we make a custom getter that uses identity to cache the variable.
      # pylint: disable=cell-var-from-loop
      def caching_getter(getter, name, *args, **kwargs):
        """Cache variables on device."""
        key = (self._caching_devices[i], name)
        if key in cache:
          return cache[key]

        v = getter(name, *args, **kwargs)
        with tf.device(self._caching_devices[i]):
          ret = v.read_value()
        _add_variable_proxy_methods(v, ret)
        cache[key] = ret
        return ret

      if self._daisy_chain_variables:
        custom_getter = daisy_chain_getter
      elif self._caching_devices[i]:
        custom_getter = caching_getter
      else:
        custom_getter = None
      # pylint: enable=cell-var-from-loop
      with tf.name_scope("parallel_%d" % i):
        with tf.variable_scope(
            tf.get_variable_scope() if self._reuse else "parallel_%d" % i,
            reuse=True if i > 0 and self._reuse else None,
            caching_device=self._caching_devices[i],
            custom_getter=custom_getter):
          # TODO(noam, epot, avaswani)
          # Allows for passing no device in case you want to default to the
          # existing device. This is needed when we put all experts on a single
          # device, for example in local_moe.
          if self._devices[i] != DEFAULT_DEV_STRING:
            with tf.device(self._devices[i]):
              outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
          else:
            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
    if isinstance(outputs[0], tuple):
      outputs = list(zip(*outputs))
      outputs = tuple([list(o) for o in outputs])
    return outputs

  @property
  def n(self):
    return self._n

  @property
  def devices(self):
    return self._devices

  @property
  def ps_devices(self):
    return self._ps_devices

  def _maybe_repeat(self, x):
    """Utility function for processing arguments that are singletons or lists.
    Args:
      x: either a list of self.n elements, or not a list.
    Returns:
      a list of self.n elements.
    """
    if isinstance(x, list):
      assert len(x) == self.n
      return x
    else:
      return [x] * self.n


def _rowwise_unsorted_segment_sum(values, indices, n):
  """UnsortedSegmentSum on each row.
  Args:
    values: a `Tensor` with shape `[batch_size, k]`.
    indices: an integer `Tensor` with shape `[batch_size, k]`.
    n: an integer.
  Returns:
    A `Tensor` with the same type as `values` and shape `[batch_size, n]`.
  """
  batch, k = tf.unstack(tf.shape(indices), num=2)
  indices_flat = tf.reshape(indices, [-1]) + tf.div(tf.range(batch * k), k) * n
  ret_flat = tf.unsorted_segment_sum(
      tf.reshape(values, [-1]), indices_flat, batch * n)
  return tf.reshape(ret_flat, [batch, n])


def _normal_distribution_cdf(x, stddev):
  """Evaluates the CDF of the normal distribution.
  Normal distribution with mean 0 and standard deviation stddev,
  evaluated at x=x.
  input and output `Tensor`s have matching shapes.
  Args:
    x: a `Tensor`
    stddev: a `Tensor` with the same shape as `x`.
  Returns:
    a `Tensor` with the same shape as `x`.
  """
  return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))


def _prob_in_top_k(
    clean_values, noisy_values, noise_stddev, noisy_top_values, k):
  """Helper function to NoisyTopKGating.
  Computes the probability that value is in top k, given different random noise.
  This gives us a way of backpropagating from a loss that balances the number
  of times each expert is in the top k experts per example.
  In the case of no noise, pass in None for noise_stddev, and the result will
  not be differentiable.
  Args:
    clean_values: a `Tensor` of shape [batch, n].
    noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
      normally distributed noise with standard deviation noise_stddev.
    noise_stddev: a `Tensor` of shape [batch, n], or None
    noisy_top_values: a `Tensor` of shape [batch, m].
       "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
    k: an integer.
  Returns:
    a `Tensor` of shape [batch, n].
  """
  batch = tf.shape(clean_values)[0]
  m = tf.shape(noisy_top_values)[1]
  top_values_flat = tf.reshape(noisy_top_values, [-1])
  # we want to compute the threshold that a particular value would have to
  # exceed in order to make the top k.  This computation differs depending
  # on whether the value is already in the top k.
  threshold_positions_if_in = tf.range(batch) * m + k
  threshold_if_in = tf.expand_dims(
      tf.gather(top_values_flat, threshold_positions_if_in), 1)
  is_in = tf.greater(noisy_values, threshold_if_in)
  if noise_stddev is None:
    return tf.to_float(is_in)
  threshold_positions_if_out = threshold_positions_if_in - 1
  threshold_if_out = tf.expand_dims(
      tf.gather(top_values_flat, threshold_positions_if_out), 1)
  # is each value currently in the top k.
  prob_if_in = _normal_distribution_cdf(clean_values - threshold_if_in,
                                        noise_stddev)
  prob_if_out = _normal_distribution_cdf(clean_values - threshold_if_out,
                                         noise_stddev)
  prob = tf.where(is_in, prob_if_in, prob_if_out)
  return prob


def cv_squared(x):
  """The squared coefficient of variation of a sample.
  Useful as a loss to encourage a positive distribution to be more uniform.
  Epsilons added for numerical stability.
  Returns 0 for an empty Tensor.
  Args:
    x: a `Tensor`.
  Returns:
    a `Scalar`.
  """
  epsilon = 1e-10
  float_size = tf.to_float(tf.size(x)) + epsilon
  mean = tf.reduce_sum(x) / float_size
  variance = tf.reduce_sum(tf.squared_difference(x, mean)) / float_size
  return variance / (tf.square(mean) + epsilon)


def _gates_to_load(gates):
  """Compute the true load per expert, given the gates.
  The load is the number of examples for which the corresponding gate is >0.
  Args:
    gates: a `Tensor` of shape [batch_size, n]
  Returns:
    a float32 `Tensor` of shape [n]
  """
  return tf.reduce_sum(tf.to_float(gates > 0), 0)


def update_hparams_for_vq_gating(hparams):
  """VQ Gating hparams."""
  hparams.add_hparam("z_size", 4)
  hparams.add_hparam("noise_dev", 0.5)
  # Bottleneck kinds supported: dense, vae, dvq.
  hparams.add_hparam("bottleneck_kind", "dvq")
  hparams.add_hparam("num_blocks", 1)
  hparams.add_hparam("num_residuals", 1)
  # Reshape method for DVQ: slice, project
  hparams.add_hparam("beta", 0.25)
  hparams.add_hparam("epsilon", 1e-5)
  hparams.add_hparam("decay", 0.999)
  hparams.add_hparam("ema", False)  # default is false until ema is implemented
  hparams.add_hparam("random_top_k", 1)
  hparams.add_hparam("soft_em", False)
  hparams.add_hparam("num_samples", 10)
  hparams.add_hparam("gating_type", "vq")
  hparams.add_hparam("use_scales", int(True))
  hparams.add_hparam("residual_centroids", int(False))


def _my_top_k(x, k):
  """GPU-compatible version of top-k that works for very small constant k.
  Calls argmax repeatedly.
  tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
  seems not to be, so if we use tf.nn.top_k, then both the top_k and its
  gradient go on cpu.  Once this is not an issue, this function becomes
  obsolete and should be replaced by tf.nn.top_k.
  Args:
    x: a 2d Tensor.
    k: a small integer.
  Returns:
    values: a Tensor of shape [batch_size, k]
    indices: a int32 Tensor of shape [batch_size, k]
  """
  if k > 10:
    return tf.nn.top_k(x, k)
  values = []
  indices = []
  depth = tf.shape(x)[1]
  for i in range(k):
    values.append(tf.reduce_max(x, 1))
    argmax = tf.argmax(x, 1)
    indices.append(argmax)
    if i + 1 < k:
      x += tf.one_hot(argmax, depth, -1e9)
  return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


def vq_gating(x,
              num_experts,
              k,
              bneck,
              hparams=None,
              name="vq_gating"):
  """VQ gating.
  Args:
    x: input Tensor with shape [batch_size, input_size]
    num_experts: an integer
    k: an integer - number of experts per example
    bneck: a bottleneck object
    hparams: optional hparams
    name: an optional string
  Returns:
    gates: a Tensor with shape [batch_size, num_experts]
    load: a Tensor with shape [num_experts]
  """
  with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

    if hparams.use_scales:
      scales = tf.get_variable(
          "scales", [num_experts],
          tf.float32,
          initializer=tf.ones_initializer())
      scales = tf.nn.softmax(scales)
      hparams.scales = scales
    input_size = x.get_shape().as_list()[-1]
    batch_size = common_layers.shape_list(x)[0]

    if k > 1:
      # first project into two dense layers, chop and discretize, and gate
      # TODO(avaswani): Maybe scale the embeddings flowing out of the experts.
      # We might want to do this to match the computation being done by topk
      x = tf.layers.dense(x, input_size * k)
      # x goes from [batch_size, input_size*k] to [batch_size*k, input_size]
      x = tf.reshape(x, [batch_size * k, input_size])
    inputs = tf.expand_dims(x, axis=1)
    inputs = tf.expand_dims(inputs, axis=1)
    # VQ hparams
    hparams.z_size = int(math.log(num_experts, 2))
    hparams.hidden_size = input_size
    hparams.top_k = k
    d = bneck.discrete_bottleneck(inputs)
    centroids = None
    exp_discrete = d["discrete"]
    embed_lookup = d["embed"]
    extra_loss = d["loss"]
    if hparams.residual_centroids:
      centroids = embed_lookup(exp_discrete)  # gives the centroids
    top_k_indices = tf.squeeze(exp_discrete, axis=1)
    tf.summary.histogram("discrete_counts", top_k_indices)
    # if k > 1, then we need to reshape top_k_indices from [batch_size*k, 1]
    # to [batch_size, k]
    if k > 1:
      top_k_indices = tf.reshape(top_k_indices, [batch_size, k])
    # get the top k gates
    top_k_gates = tf.ones([batch_size, k])
    # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
    # positions corresponding to all but the top k experts per example.
    gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices,
                                          num_experts)
    # Compute count per expert from the gates.
    # gates has shape [batch_size, num_experts]
    # count per expert has shape [num_experts, 1]
    count_per_expert = tf.reduce_sum(gates, axis=0)
    if hparams.use_scales:
      scale_loss = tf.reduce_mean(tf.to_float(count_per_expert) * scales)
      extra_loss += scale_loss
    if common_layers.should_generate_summaries():
      tf.summary.histogram("vq_loss", extra_loss)
      tf.summary.historgram("scale_loss", scale_loss)
    return gates, extra_loss, centroids


def noisy_top_k_gating(x,
                       num_experts,
                       train,
                       k=2,
                       initializer=tf.zeros_initializer(),
                       noisy_gating=True,
                       noise_epsilon=1e-2,
                       name=None):
  """Noisy top-k gating.
  See paper: https://arxiv.org/abs/1701.06538.
  Args:
    x: input Tensor with shape [batch_size, input_size]
    num_experts: an integer
    train: a boolean - we only add noise at training time.
    k: an integer - number of experts per example
    initializer: an initializer
    noisy_gating: a boolean
    noise_epsilon: a float
    name: an optional string
  Returns:
    gates: a Tensor with shape [batch_size, num_experts]
    load: a Tensor with shape [num_experts]
  """
  with tf.variable_scope(name, default_name="noisy_top_k_gating"):
    input_size = x.get_shape().as_list()[-1]
    w_gate = tf.get_variable(
        "w_gate", [input_size, num_experts], tf.float32, initializer)
    if noisy_gating:
      w_noise = tf.get_variable("w_noise",
                                [input_size, num_experts], tf.float32,
                                initializer)
    clean_logits = tf.matmul(x, w_gate)
    if noisy_gating:
      raw_noise_stddev = tf.matmul(x, w_noise)
      noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + noise_epsilon) *
                      (tf.to_float(train)))
      noisy_logits = clean_logits + (
          tf.random_normal(tf.shape(clean_logits)) * noise_stddev)
      logits = noisy_logits
      if common_layers.should_generate_summaries():
        tf.summary.histogram("noisy_logits", noisy_logits)
        tf.summary.histogram("noise_stddev", noise_stddev)
    else:
      logits = clean_logits
    top_logits, top_indices = _my_top_k(logits, min(k + 1, num_experts))
    # top k logits has shape [batch, k]
    top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
    top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
    top_k_gates = tf.nn.softmax(top_k_logits)
    # This will be a `Tensor` of shape `[batch_size, n]`, with zeros in the
    # positions corresponding to all but the top k experts per example.
    gates = _rowwise_unsorted_segment_sum(top_k_gates, top_k_indices,
                                          num_experts)
    if noisy_gating and k < num_experts:
      load = tf.reduce_sum(
          _prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits,
                         k), 0)
    else:
      load = _gates_to_load(gates)
    if common_layers.should_generate_summaries():
      tf.summary.histogram("importance", tf.reduce_sum(gates, 0))
      tf.summary.histogram("load", load)
    return gates, load


class PadRemover(object):
  """Helper to remove padding from a tensor before sending to the experts.
  The padding is computed for one reference tensor containing the padding mask
  and then can be applied to any other tensor of shape [dim_origin,...].
  Ex:
      input = [
        [tok1, tok2],
        [tok3, tok4],
        [0, 0],
        [0, 0],
        [tok5, tok6],
        [0, 0],
      ]
      output = [
        [tok1, tok2],
        [tok3, tok4],
        [tok5, tok6],
      ]
  """

  def __init__(self, pad_mask):
    """Compute and store the location of the padding.
    Args:
      pad_mask (tf.Tensor): Reference padding tensor of shape
        [batch_size,length] or [dim_origin] (dim_origin=batch_size*length)
        containing non-zeros positive values to indicate padding location.
    """
    self.nonpad_ids = None
    self.dim_origin = None

    with tf.name_scope("pad_reduce/get_ids"):
      pad_mask = tf.reshape(pad_mask, [-1])  # Flatten the batch
      # nonpad_ids contains coordinates of zeros rows (as pad_mask is
      # float32, checking zero equality is done with |x| < epsilon, with
      # epsilon=1e-9 as standard, here pad_mask only contains positive values
      # so tf.abs would be redundant)
      self.nonpad_ids = tf.to_int32(tf.where(pad_mask < 1e-9))
      self.dim_origin = tf.shape(pad_mask)[:1]

  def remove(self, x):
    """Remove padding from the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_origin,...]
    Returns:
      a tensor of shape [dim_compressed,...] with dim_compressed <= dim_origin
    """
    with tf.name_scope("pad_reduce/remove"):
      x_shape = x.get_shape().as_list()
      x = tf.gather_nd(
          x,
          indices=self.nonpad_ids,
      )
      if not tf.executing_eagerly():
        # This is a hack but for some reason, gather_nd return a tensor of
        # undefined shape, so the shape is set up manually
        x.set_shape([None] + x_shape[1:])
    return x

  def restore(self, x):
    """Add padding back to the given tensor.
    Args:
      x (tf.Tensor): of shape [dim_compressed,...]
    Returns:
      a tensor of shape [dim_origin,...] with dim_compressed >= dim_origin. The
      dim is restored from the original reference tensor
    """
    with tf.name_scope("pad_reduce/restore"):
      x = tf.scatter_nd(
          indices=self.nonpad_ids,
          updates=x,
          shape=tf.concat([self.dim_origin, tf.shape(x)[1:]], axis=0),
      )
    return x


@add_name_scope("map_ids")
def map_ids(x, indices, map_fn):
  """Apply a function to each coordinate ids of a multidimensional tensor.
  This allows to process each sequence of a batch independently. This is
  similar to tf.map_fn but with tensor where the batch dim has been flatten.
  Warning: The indices ids have to be contiguous and ordered in memory as the
  output vector for each of the ids are simply concatenated after being
  processed.
  Ex: if your indices are [0,2,2,1,2,0], the output will contains the processed
  rows in the following order: [0,0,1,2,2,2]
  Args:
    x (Tensor): The tensor to be dispatched of shape [length,...]
    indices (Tensor): A int32 tensor of size [length, 1] containing the batch
      coordinate of x
    map_fn (fct): Function called for every ids of the original tensor. Take
      as input a tensor of same rank than x and from shape [length_id,...] with
      length_id <= length. Isn't called if length_id == 0
  Returns:
    a tensor of same shape as x, where each elements has been processed
  """
  indices = tf.reshape(indices, [-1])

  t_i = tf.constant(0)
  # batch_coordinates start at 0
  t_batch_size = tf.reduce_max(indices) + 1

  # ta_stack_out will store the intermediate results for each individual id
  # As alternative to tf.TensorArray, scatter_update could potentially be used
  # but that would require an additional mutable tensor.
  ta_stack_out = tf.TensorArray(
      x.dtype,
      size=t_batch_size,
  )

  # Then we iterate over each sequence individually and compute the
  # transformation for each id
  while_condition = lambda t_i, *args: tf.less(t_i, t_batch_size)
  def body(t_i, ta_stack_out):
    """Loop body."""
    # Gather the ids
    current_ids = tf.to_int32(tf.where(tf.equal(indices, t_i)))
    t_row = tf.gather_nd(x, indices=current_ids)

    # TODO(epot): Should not call map_fn if t_row size is 0

    # Apply transformation to each id
    # Restore batch_dim=1 as most function expect [batch_dim, length, ...] as
    # input
    t_row = tf.expand_dims(t_row, axis=0)
    t_row = map_fn(t_row)
    t_row = tf.squeeze(t_row, axis=0)  # Squeeze for concatenation
    ta_stack_out = ta_stack_out.write(t_i, t_row)

    return [tf.add(t_i, 1), ta_stack_out]  # ++i

  # Run the loop, equivalent to:
  # stack_out = []
  # while i < batch_size:
  #   stack_out.expand(map_fn(x[indices==i]))
  _, ta_stack_out = tf.while_loop(while_condition, body, [t_i, ta_stack_out])

  # Merge all results
  return ta_stack_out.concat()


def transpose_list_of_lists(lol):
  """Transpose a list of equally-sized python lists.
  Args:
    lol: a list of lists
  Returns:
    a list of lists
  """
  assert lol, "cannot pass the empty list"
  return [list(x) for x in zip(*lol)]



def flatten_all_but_last(a):
  """Flatten all dimensions of a except the last."""
  ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
  if not tf.executing_eagerly():
    ret.set_shape([None] + a.get_shape().as_list()[-1:])
  return ret


def reduce_by_device(parallelism, data, reduce_fn):
  """Reduces data per device.
  This can be useful, for example, if we want to all-reduce n tensors on k<n
  devices (like during eval when we have only one device).  We call
  reduce_by_device() to first sum the tensors per device, then call our usual
  all-reduce operation to create one sum per device, followed by
  expand_by_device, to create the appropriate number of pointers to these
  results.  See all_reduce_ring() below for an example of how this is used.
  Args:
    parallelism: a expert_utils.Parallelism object
    data: a list of Tensors with length parallelism.n
    reduce_fn: a function taking a list of Tensors.  e.g. tf.add_n
  Returns:
    device_parallelism: a Parallelism object with each device listed only once.
    reduced_data: A list of Tensors, one per device.
  """
  unique_devices = []
  device_to_data = {}
  for dev, datum in zip(parallelism.devices, data):
    if dev not in device_to_data:
      unique_devices.append(dev)
      device_to_data[dev] = [datum]
    else:
      device_to_data[dev].append(datum)
  device_parallelism = Parallelism(unique_devices)
  grouped_data = [device_to_data[dev] for dev in unique_devices]
  return device_parallelism, device_parallelism(reduce_fn, grouped_data)


def expand_by_device(original_parallelism, device_parallelism, data):
  """Opposite of reduce_by_device().
  Args:
    original_parallelism: a expert_utils.Parallelism object.
    device_parallelism: a expert_utils.Parallelism object.
    data: a list of tensors with length device_parallelism.n
  Returns:
    a list of Tensors with length original_parallelism.n
  """
  device_to_datum = {
      device_parallelism.devices[i]: data[i]
      for i in range(device_parallelism.n)}
  return [device_to_datum[d] for d in original_parallelism.devices]


def all_reduce_ring(x, parallelism, maybe_reduce=True, use_bfloat16=True):
  """Compute the sum of all Tensors and put the result everywhere.
  Assumes that the devices are connected in a ring.
  Args:
    x: a list of Tensors with length parallelism.n
    parallelism: a expert_utils.Parallelism object.
    maybe_reduce: a boolean - first reduce per device.
    use_bfloat16: a boolean - saves bandwidth but loses precision
  Returns:
    a list of Tensors with length parallelism.n
  """
  if parallelism.n == 1:
    return x

  if maybe_reduce:
    original_parallelism = parallelism
    parallelism, x = reduce_by_device(parallelism, x, tf.add_n)

  if parallelism.n == 1:
    y = x
  else:
    # first shard the input:
    x_flat = parallelism(tf.reshape, x, [[-1]] * parallelism.n)
    # [device, shard]
    x_split = parallelism(
        common_layers.approximate_split, x_flat, parallelism.n, 0)
    def _step(source_replica, target_replica, x_split, op="plus_eq"):
      """Helper function - one step of summing or copying.
      If op == "plus_eq", then adds source_replica into target_replica
      If op == "copy", then copies source_replica onto target_replica
      These operations happen for all shards.  The replica numbers are offset
      by the shard numbers to keep all physical links busy.
      Args:
        source_replica: an integer
        target_replica: an integer
        x_split: a list of lists of tensors
        op: a string
      """
      for shard in range(parallelism.n):
        source_device = (shard + source_replica) % parallelism.n
        target_device = (shard + target_replica) % parallelism.n
        source = x_split[source_device][shard]
        if use_bfloat16:
          with tf.device(parallelism.devices[source_device]):
            source = tf.to_bfloat16(source)
        with tf.device(parallelism.devices[target_device]):
          source = tf.to_float(source)
          if op == "plus_eq":
            x_split[target_device][shard] += source
          else:
            assert op == "copy"
            x_split[target_device][shard] = tf.identity(source)
    center = parallelism.n // 2

    # accumulate everything towards the center.
    for i in reversed(range(center, parallelism.n - 1)):
      _step(i + 1, i, x_split, op="plus_eq")
    for i in range(center):
      _step(i, i + 1, x_split, op="plus_eq")
    # copy everything away from the center.
    for i in range(center, parallelism.n - 1):
      _step(i, i + 1, x_split, op="copy")
    for i in reversed(range(center)):
      _step(i + 1, i, x_split, op="copy")
    x_concat = parallelism(tf.concat, x_split, 0)
    y = parallelism(common_layers.reshape_like_all_dims, x_concat, x)
  if maybe_reduce:
    y = expand_by_device(original_parallelism, parallelism, y)
  return y


class SparseDispatcher(object):
  """Helper for implementing a mixture of experts.
  The purpose of this class is to create input minibatches for the
  experts and to combine the results of the experts to form a unified
  output tensor.
  There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
  The class is initialized with a "gates" Tensor, which specifies which
  batch elements go to which experts, and the weights to use when combining
  the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
  The inputs and outputs are all two-dimensional [batch, depth].
  Caller is responsible for collapsing additional dimensions prior to
  calling this class and reshaping the output to the original shape.
  See common_layers.reshape_like().
  Example use:
  gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
  inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
  experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
  The preceding code sets the output for a particular example b to:
  output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
  This class takes advantage of sparsity in the gate matrix by including in the
  `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
  """

  def __init__(self, num_experts, gates):
    """Create a SparseDispatcher.
    Args:
      num_experts: an integer.
      gates: a `Tensor` of shape `[batch_size, num_experts]`.
    Returns:
      a SparseDispatcher
    """
    self._gates = gates
    self._num_experts = num_experts

    where = tf.to_int32(tf.where(tf.transpose(gates) > 0))
    self._expert_index, self._batch_index = tf.unstack(where, num=2, axis=1)
    self._part_sizes_tensor = tf.reduce_sum(tf.to_int32(gates > 0), [0])
    self._nonzero_gates = tf.gather(
        tf.reshape(self._gates, [-1]),
        self._batch_index * num_experts + self._expert_index)

  @add_name_scope()
  def dispatch(self, inp):
    """Create one input Tensor for each expert.
    The `Tensor` for a expert `i` contains the slices of `inp` corresponding
    to the batch elements `b` where `gates[b, i] > 0`.
    Args:
      inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
    Returns:
      a list of `num_experts` `Tensor`s with shapes
        `[expert_batch_size_i, <extra_input_dims>]`.
    """
    inp = tf.gather(inp, self._batch_index)
    return tf.split(inp, self._part_sizes_tensor, 0, num=self._num_experts)

  @add_name_scope()
  def combine(self, expert_out, multiply_by_gates=True):
    """Sum together the expert output, weighted by the gates.
    The slice corresponding to a particular batch element `b` is computed
    as the sum over all experts `i` of the expert output, weighted by the
    corresponding gate values.  If `multiply_by_gates` is set to False, the
    gate values are ignored.
    Args:
      expert_out: a list of `num_experts` `Tensor`s, each with shape
        `[expert_batch_size_i, <extra_output_dims>]`.
      multiply_by_gates: a boolean
    Returns:
      a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
    """
    # see comments on convert_gradient_to_tensor
    stitched = common_layers.convert_gradient_to_tensor(
        tf.concat(expert_out, 0))
    if multiply_by_gates:
      stitched *= tf.expand_dims(self._nonzero_gates, 1)
    combined = tf.unsorted_segment_sum(stitched, self._batch_index,
                                       tf.shape(self._gates)[0])
    return combined

  def expert_to_gates(self):
    """Gate values corresponding to the examples in the per-expert `Tensor`s.
    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
          and shapes `[expert_batch_size_i]`
    """
    return tf.split(
        self._nonzero_gates, self._part_sizes_tensor, 0, num=self._num_experts)

  def expert_to_batch_indices(self):
    """Batch indices corresponding to the examples in the per-expert `Tensor`s.
    Returns:
      a list of `num_experts` one-dimensional `Tensor`s with type `tf.int64`
          and shapes `[expert_batch_size_i]`
    """
    return tf.split(
        self._batch_index, self._part_sizes_tensor, 0, num=self._num_experts)

  @property
  def part_sizes(self):
    return self._part_sizes_tensor



def local_moe(x,
              train,
              expert_fn,
              num_experts,
              k=1,
              loss_coef=1e-2,
              hparams=None,
              pass_x=True,
              pass_gates=False,
              additional_dispatch_params=None,
              name=None):
  """Call a local mixture of experts.
  Args:
    x: a tensors with shape [... , input_size]
    train: a boolean scalar.
    expert_fn: a function.
    num_experts: an integer - number of experts
    k: an integer - how many experts to use for each batch element
    loss_coef: a scalar - multiplier on load-balancing losses
    hparams: optional hparams for vq gating
    pass_x: a boolean. If true, x will also be dispatched to the experts.
    pass_gates: a boolean. If true, gates will be passed to experts. Might be
      necessary when dealing with sparse encoder-encoder decoder attention
    additional_dispatch_params: The extra tensors that need to be sent to each
      expert. Examples include batch batch coordinates (see
      common_attention.local_expert_attention)
    name: a string
  Returns:
    y: a tensor.  Has the same shape as x, except for the last dimension,
      which is output_size.
    extra_training_loss: a scalar.  This should be added into the overall
      training loss of the model.  The backpropagation of this loss
      encourages all experts to be approximately equally used across a batch.
  """
  with tf.variable_scope(name, default_name="local_moe"):
    centroids = None
    x_flat = flatten_all_but_last(x)
    if True:
      tf.logging.info("Using noisy top_k with k = {}".format(k))
      # The gates indicate which batch elements go to which tensors.
      # load is a measure of approximately how many examples go to each expert
      gates, load = noisy_top_k_gating(
          x_flat,
          num_experts,
          train,
          k,
          initializer=tf.zeros_initializer(),
          noisy_gating=True,
          noise_epsilon=1e-2)
      importance = tf.reduce_sum(gates, 0)
      loss = (cv_squared(importance) + cv_squared(load))
    loss *= loss_coef
    # Shuffle data between datashards and experts.
    dispatcher = SparseDispatcher(num_experts, gates)
    # Set up expert_fn arguments
    expert_kwargs = {}
    if pass_x:
      expert_kwargs["x"] = dispatcher.dispatch(x_flat)
    if pass_gates:
      expert_kwargs["gates"] = dispatcher.expert_to_gates()
    for key, val in six.iteritems(additional_dispatch_params or {}):
      val = flatten_all_but_last(val)
      expert_kwargs[key] = dispatcher.dispatch(val)

    ep = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=None)
    expert_outputs = ep(expert_fn, **expert_kwargs)

    y_flat = dispatcher.combine(expert_outputs)
    if centroids is not None:
      centroids = tf.squeeze(centroids, axis=[1, 2])
      y_flat += centroids
    y = common_layers.reshape_like(y_flat, x)
    return y, loss

def ffn_expert_fn(input_size,
                  hidden_sizes,
                  output_size,
                  hidden_activation=tf.nn.relu):
  """Returns a function that creates a feed-forward network.
  Use this function to create the expert_fn argument to distributed_moe.
  Args:
    input_size: an integer
    hidden_sizes: a list of integers
    output_size: an integer
    hidden_activation: a unary function.
  Returns:
    a unary function
  """
  def my_fn(x):
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    for i in range(1 + len(hidden_sizes)):
      w = tf.Variable(tf.random.truncated_normal(layer_sizes[i:i+2], stddev=0.01))
      x = tf.matmul(x, w)
      if i < len(hidden_sizes):
        x = hidden_activation(x)
      if layer_sizes[i] != input_size:
        x *= (layer_sizes[i] / float(input_size))**-0.5
    return x
  return my_fn
