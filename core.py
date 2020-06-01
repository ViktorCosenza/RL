import scipy.signal
import toolz
from torch import nn

def standarize(xs):
    return (xs - xs.mean()) / xs.std()

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def create_model(input_size, output_size, hidden_layers=[16], activate_every=2):
    layers = [nn.Linear(in_shape, out_shape) for (in_shape, out_shape) in toolz.sliding_window(2, hidden_layers)]
    layers = [*toolz.concat([*ls, nn.ReLU()] for ls in toolz.partition(activate_every, layers))]
    layers = [nn.ReLU(), *layers]
    return nn.Sequential(
        nn.Linear(input_size, hidden_layers[0]),
        *layers,
        nn.Linear(hidden_layers[-1], output_size)
    )

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs