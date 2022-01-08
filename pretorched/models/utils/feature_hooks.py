from collections import defaultdict, OrderedDict
from functools import partial


class FeatureHooks:

    def __init__(self, hooks, named_modules):
        # setup feature hooks
        modules = dict(named_modules)
        for h in hooks:
            hook_name = h['name']
            m = modules[hook_name]
            hook_fn = partial(self._collect_output_hook, hook_name)
            if h['type'] == 'forward_pre':
                m.register_forward_pre_hook(hook_fn)
            elif h['type'] == 'forward':
                m.register_forward_hook(hook_fn)
            else:
                assert False, "Unsupported hook type"
        self._feature_outputs = defaultdict(OrderedDict)

    def _collect_output_hook(self, name, *args):
        x = args[-1]  # tensor we want is last argument, output for fwd, input for fwd_pre
        if isinstance(x, tuple):
            x = x[0]  # unwrap input tuple
        self._feature_outputs[x.device][name] = x

    def get_output(self, device=None):
        if device is None:
            devs = list(self._feature_outputs.keys())
            assert len(devs) == 1
            device = devs[0]
        output = tuple(self._feature_outputs[device].values())
        self._feature_outputs[device] = OrderedDict()  # clear after reading
        return output
