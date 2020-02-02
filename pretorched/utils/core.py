import atexit
import functools
import hashlib
import os
import re
import shutil
import signal
import sys
from collections import defaultdict
from importlib import import_module
from multiprocessing import Process, Queue, cpu_count
from operator import itemgetter
from typing import Any, AnyStr, Callable, Collection

import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class cache(object):
    """Computes attribute value and caches it in the instance.

    This decorator allows you to create a property which can be computed once and
    accessed many times. Sort of like memoization.

    """

    def __init__(self, method, name=None):
        self.method = method
        self.name = name or method.__name__
        self.__doc__ = method.__doc__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.method(obj)
        setattr(obj, self.name, value)
        return value


def lazy_property(fn):
    """Decorator that makes a property lazy-evaluated."""
    attr_name = '_' + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return _lazy_property


class HTML(object):
    """Utility functions for generating html."""

    @staticmethod
    def head():
        return """
            <!DOCTYPE html>
            <html>
            <head>
              <meta name="viewport" content="width=device-width, initial-scale=1">
              <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
              <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.2.1/jquery.min.js"></script>
              <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
            </head>
            """

    @staticmethod
    def element(elem, inner='', id_='', cls_='', attr=''):
        if id_ != '':
            id_ = ' id="{}"'.format(id_)
        if cls_ is not '':
            cls_ = ' class="{}"'.format(cls_)
        if attr is not '':
            attr = ' {}'.format(attr)
        return ('<{}{}{}{}>{}</{}>'
                .format(elem, id_, cls_, attr, inner, elem))

    @staticmethod
    def div(inner='', id_='', cls_='', attr=''):
        return HTML.element('div', inner, id_, cls_, attr)

    @staticmethod
    def container(content):
        return HTML.div(content, cls_='container')

    @staticmethod
    def ul(li_list, ul_class='', li_class='', li_attr=''):
        inner = '\n\t'.join(
            [HTML.element('li', li, cls_=li_class, attr=li_attr) for li in li_list])
        return HTML.element('ul', '\n\t' + inner + '\n', cls_=ul_class)

    @staticmethod
    def ol(li_list, ol_class='', li_class='', li_attr=''):
        inner = '\n\t'.join(
            [HTML.element('li', li, cls_=li_class, attr=li_attr) for li in li_list])
        return HTML.element('ol', '\n\t' + inner + '\n', cls_=ol_class)

    @staticmethod
    def img(src='', style=''):
        return HTML.element('img', attr='src="{}"; style="{}";'.format(src, style))

    @staticmethod
    def video(src='', preload='auto', onmouseover='this.play();',
              onmouseout='this.pause();', style=''):
        return HTML.element('video', attr='src="{}" onmouseover="{}" onmouseout="{}" style="{}"'
                            .format(src, onmouseover, onmouseout, style))

    @staticmethod
    def a(inner='', href='', data_toggle=''):
        return HTML.element('a', inner=inner, attr='href="{}" data-toggle="{}"'.format(href, data_toggle))

    @staticmethod
    def p(inner=''):
        return HTML.element('p', inner=inner)

    @staticmethod
    def panel(label, category, li):
        return HTML.div(cls_='panel panel-default', inner='\n'.join([
            HTML.div(cls_='panel-heading',
                     inner=HTML.element('h4', cls_="panel-title",
                                        inner=HTML.a(data_toggle='collapse', href='#{}'.format(label),
                                                     inner='{} (n={})'.format(category, len(li))))),
            HTML.div(id_='{}'.format(label), cls_='panel-collapse collapse',
                     inner=HTML.ul(HTML.li, ul_class='list-group', li_class='list-group-item', li_attr="style=\"overflow: auto;\""))]))

    @staticmethod
    def panel_group(html):
        return HTML.div(cls_='panel-group', inner='\n'.join([
            HTML.panel(label, ground_truth, predictions)
            for (label, ground_truth), predictions in html.items()]))

    @staticmethod
    def format_div(header, im_name, gif_name):
        html = """
            <h4>{}</h4>
            <img style="float: left;" src="{}"/>
            <img style="float: left;" src="{}"/>
        """
        return html.format(header, im_name, gif_name)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def make_html_gallery(root, tmpl='$gallery{}.html', size=256, max_per_page=100):
    print(root)
    filenames = sorted([x for x in os.listdir(root) if is_image_file(x)])

    def make_page(filenames, size):
        html = '\n'.join([
            HTML.head(),
            HTML.div(
                inner='\n'.join([
                    HTML.img(src=f, style=f'height: {size}px; width: {size}px;')
                    for f in filenames
                ])
            )])
        return html

    for i, fnames in enumerate(chunk(filenames, max_per_page)):
        with open(os.path.join(root, tmpl.format(i)), 'w') as f:
            f.write(make_page(fnames, size))


def get_grad_hook(name):
    def hook(m, grad_in, grad_out):
        print((name, grad_out[0].data.abs().mean(), grad_in[0].data.abs().mean()))
        print((grad_out[0].size()))
        print((grad_in[0].size()))

        print((grad_out[0]))
        print((grad_in[0]))

    return hook


def softmax(scores):
    es = np.exp(scores - scores.max(axis=-1)[..., None])
    return es / es.sum(axis=-1)[..., None]


def log_add(log_a, log_b):
    return log_a + np.log(1 + np.exp(log_b - log_a))


def class_accuracy(prediction, label):
    cf = confusion_matrix(prediction, label)
    cls_cnt = cf.sum(axis=1)
    cls_hit = np.diag(cf)
    cls_acc = cls_hit / cls_cnt.astype(float)
    mean_cls_acc = cls_acc.mean()
    return cls_acc, mean_cls_acc


def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sort(arr):
    """Return indices and sorted array."""
    return zip(*sorted(enumerate(arr), key=itemgetter(1)))


def format_checkpoint(ckpt_file):
    pth_file = ckpt_file.replace('.tar', '')
    pth_file += '.pth' if not pth_file.endswith('.pth') else ''
    checkpoint = torch.load(ckpt_file, map_location=lambda storage, loc: storage)
    print(checkpoint.keys())
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    torch.save(state_dict, pth_file)
    hashval = hashsha256(pth_file)[:8]
    name, ext = os.path.splitext(pth_file)
    hashed_pth_file = name + f'-{hashval}' + ext
    print(f'Copying {pth_file} to {hashed_pth_file}')
    shutil.copyfile(pth_file, hashed_pth_file)


def hashsha256(filename):
    hashfunc = hashlib.sha256()
    with open(filename, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            hashfunc.update(byte_block)
    return hashfunc.hexdigest()


def format_tar(tar_file, ext='.tar.gz'):
    hashval = hashsha256(tar_file)[:8]
    fname = tar_file.replace(ext, f'-{hashval}' + ext)
    os.rename(tar_file, fname)


def format_hash(pth_file, ext='.pth'):
    hashval = hashsha256(pth_file)[:8]
    if torch.hub.HASH_REGEX.search(pth_file):
        fname = re.sub(torch.hub.HASH_REGEX, f'-{hashval}.', pth_file)
    else:
        fname = pth_file.replace(ext, f'-{hashval}' + ext)
    print(f'Copying {pth_file} to {fname}')
    shutil.copyfile(pth_file, fname)


def func_args(func) -> Collection[str]:
    """Return the arguments of `func`."""
    try:
        code = func.__code__
    except AttributeError:
        if isinstance(func, functools.partial):
            return func_args(func.func)
        else:
            code = func.__init__.__code__
    return code.co_varnames[:code.co_argcount]


def class_args(cls) -> Collection[str]:
    """Return the arguments of the class init method."""


def has_arg(func, arg) -> bool:
    "Check if `func` accepts `arg`."
    return arg in func_args(func)


def split_kwargs_by_func(func, kwargs):
    "Split `kwargs` between those expected by `func` and the others."
    args = func_args(func)
    func_kwargs = {a: kwargs.pop(a) for a in args if a in kwargs}
    return func_kwargs, kwargs


def autoimport_eval(term):
    """
    Used to evaluate an arbitrary command-line constructor specifying
    a class, with automatic import of global module names.
    """

    class DictNamespace(object):
        def __init__(self, d):
            self.__d__ = d

        def __getattr__(self, key):
            return self.__d__[key]

    class AutoImportDict(defaultdict):
        def __init__(self, wrapped=None, parent=None):
            super().__init__()
            self.wrapped = wrapped
            self.parent = parent

        def __missing__(self, key):
            if self.wrapped is not None:
                if key in self.wrapped:
                    return self.wrapped[key]
            if self.parent is not None:
                key = self.parent + '.' + key
            if key in __builtins__:
                return __builtins__[key]
            mdl = import_module(key)
            # Return an AutoImportDict for any namespace packages
            if hasattr(mdl, '__path__'):  # and not hasattr(mdl, '__file__'):
                return DictNamespace(
                    AutoImportDict(wrapped=mdl.__dict__, parent=key))
            return mdl

    return eval(term, {}, AutoImportDict())


'''
WorkerPool and WorkerBase for handling the common problems in managing
a multiprocess pool of workers that aren't done by multiprocessing.Pool,
including setup with per-process state, debugging by putting the worker
on the main thread, and correct handling of unexpected errors, and ctrl-C.

To use it,
1. Put the per-process setup and the per-task work in the
   setup() and work() methods of your own WorkerBase subclass.
2. To prepare the process pool, instantiate a WorkerPool, passing your
   subclass type as the first (worker) argument, as well as any setup keyword
   arguments.  The WorkerPool will instantiate one of your workers in each
   worker process (passing in the setup arguments in those processes).
   If debugging, the pool can have process_count=0 to force all the work
   to be done immediately on the main thread; otherwise all the work
   will be passed to other processes.
3. Whenever there is a new piece of work to distribute, call pool.add(*args).
   The arguments will be queued and passed as worker.work(*args) to the
   next available worker.
4. When all the work has been distributed, call pool.join() to wait for all
   the work to complete and to finish and terminate all the worker processes.
   When pool.join() returns, all the work will have been done.

No arrangement is made to collect the results of the work: for example,
the return value of work() is ignored.  If you need to collect the
results, use your own mechanism (filesystem, shared memory object, queue)
which can be distributed using setup arguments.

Inspired by:
    https://github.com/CSAILVision/gandissect
'''


class WorkerBase(Process):
    '''
    Subclass this class and override its work() method (and optionally,
    setup() as well) to define the units of work to be done in a process
    worker in a worker pool.
    '''

    def __init__(self, i, process_count, queue, initargs):
        if process_count > 0:
            # Make sure we ignore ctrl-C if we are not on main process.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.process_id = i
        self.process_count = process_count
        self.queue = queue
        super(WorkerBase, self).__init__()
        self.setup(**initargs)

    def run(self):
        # Do the work until None is dequeued
        while True:
            try:
                work_batch = self.queue.get()
            except (KeyboardInterrupt, SystemExit):
                print('Exiting...')
                break
            if work_batch is None:
                self.queue.put(None)  # for another worker
                return
            self.work(*work_batch)

    def setup(self, **initargs):
        '''
        Override this method for any per-process initialization.
        Keywoard args are passed from WorkerPool constructor.
        '''
        pass

    def work(self, *args):
        '''
        Override this method for one-time initialization.
        Args are passed from WorkerPool.add() arguments.
        '''
        raise NotImplementedError('worker subclass needed')


class WorkerPool(object):
    """
    Instantiate this object (passing a WorkerBase subclass type
    as its first argument) to create a worker pool.  Then call
    pool.add(*args) to queue args to distribute to worker.work(*args),
    and call pool.join() to wait for all the workers to complete.
    """

    def __init__(self, worker=WorkerBase, process_count=None, **initargs):
        global active_pools
        if process_count is None:
            process_count = cpu_count()
        if process_count == 0:
            # zero process_count uses only main process, for debugging.
            self.queue = None
            self.processes = None
            self.worker = worker(None, 0, None, initargs)
            return
        # Ctrl-C strategy: worker processes should ignore ctrl-C.  Set
        # this up to be inherited by child processes before forking.
        original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
        active_pools[id(self)] = self
        self.queue = Queue(maxsize=(process_count * 3))
        self.processes = None   # Initialize before trying to construct workers
        self.processes = [worker(i, process_count, self.queue, initargs)
                          for i in range(process_count)]
        for p in self.processes:
            p.start()
        # The main process should handle ctrl-C.  Restore this now.
        signal.signal(signal.SIGINT, original_sigint_handler)

    def add(self, *work_batch):
        if self.queue is None:
            if hasattr(self, 'worker'):
                self.worker.work(*work_batch)
            else:
                print('WorkerPool shutting down.', file=sys.stderr)
        else:
            try:
                # The queue can block if the work is so slow it gets full.
                self.queue.put(work_batch)
            except (KeyboardInterrupt, SystemExit):
                # Handle ctrl-C if done while waiting for the queue.
                self.early_terminate()

    def join(self):
        # End the queue, and wait for all worker processes to complete nicely.
        if self.queue is not None:
            self.queue.put(None)
            for p in self.processes:
                p.join()
            self.queue = None
        # Remove myself from the set of pools that need cleanup on shutdown.
        try:
            del active_pools[id(self)]
        except BaseException:
            pass

    def early_terminate(self):
        # When shutting down unexpectedly, first end the queue.
        if self.queue is not None:
            try:
                self.queue.put_nowait(None)  # Nonblocking put throws if full.
                self.queue = None
            except BaseException:
                pass
        # But then don't wait: just forcibly terminate workers.
        if self.processes is not None:
            for p in self.processes:
                p.terminate()
            self.processes = None
        try:
            del active_pools[id(self)]
        except BaseException:
            pass

    def __del__(self):
        if self.queue is not None:
            print('ERROR: workerpool.join() not called!', file=sys.stderr)
            self.join()


# Error and ctrl-C handling: kill worker processes if the main process ends.
active_pools = {}


def early_terminate_pools():
    for _, pool in list(active_pools.items()):
        pool.early_terminate()


atexit.register(early_terminate_pools)
