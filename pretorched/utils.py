import os
import re
import shutil
import hashlib

import numpy as np
from operator import itemgetter
from sklearn.metrics import confusion_matrix

import torch


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


def chunks(l, n):
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
