import ray
import ray.util.multiprocessing as mp


def ray_map(func, iterable):
    ray.init(ignore_reinit_error=True)
    return mp.Pool().map(func, iterable)


def ray_remote_map(func, iter):
    ray.init(ignore_reinit_error=True)
    return ray.get(list(map(ray.remote(func).remote, iter)))
