u"""
Miscellaneous
"""
from __future__ import absolute_import
import os
import os.path as osp
import collections
import itertools
import logging


def read_list(param):
    u""" Parse list of integers """
    return [int(p) for p in unicode(param).split(u',')]


def update(d, u):
    u"""update dict of dicts"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

# Modified to suit params in ar format and not dict
def parse_densenet_params_v2(args):
    # Parsing Initial Densenet
    if args.init_network is not None:
        if u"num_layers" in args.init_network:
            num_layers = read_list(args.init_network[u'num_layers'])
            if u"kernels" in args.init_network:
                kernels = read_list(args.init_network[u'kernels'])
                if len(kernels) == 1:
                    args.init_network[u'kernels'] = [kernels * n for n in num_layers]
                else:
                    assert len(kernels) == sum(
                        num_layers), u"The number of kernel sizes must match that of the network layers"
                    batched_kernels = []
                    left = 0
                    for nn in num_layers:
                        batched_kernels.append(kernels[left:left + nn])
                        left += nn
                        args.init_network[u"kernels"] = batched_kernels
            args.init_network[u'num_layers'] = num_layers
    # print(args.network['num_layers'])
    # print(args.network['kernels'])

     # Parsing OBJ ATTN Densenet
    if args.obj_att_network is not None:
        if u"num_layers" in args.obj_att_network:
            num_layers = read_list(args.obj_att_network[u'num_layers'])
            if u"kernels" in args.obj_att_network:
                kernels = read_list(args.obj_att_network[u'kernels'])
                if len(kernels) == 1:
                    args.obj_att_network[u'kernels'] = [kernels * n for n in num_layers]
                else:
                    assert len(kernels) == sum(
                        num_layers), u"The number of kernel sizes must match that of the network layers"
                    batched_kernels = []
                    left = 0
                    for nn in num_layers:
                        batched_kernels.append(kernels[left:left + nn])
                        left += nn
                        args.init_network[u"kernels"] = batched_kernels
            args.obj_att_network[u'num_layers'] = num_layers
            # print(args.network['num_layers'])
            # print(args.network['kernels'])

    if args.network is not None:
        if u"num_layers" in args.network:
            num_layers = read_list(args.network[u'num_layers'])
            if u"kernels" in args.network:
                kernels = read_list(args.network[u'kernels'])
                if len(kernels) == 1:
                    args.network[u'kernels'] = [kernels * n for n in num_layers]
                else:
                    assert len(kernels) == sum(
                        num_layers), u"The number of kernel sizes must match that of the network layers"
                    batched_kernels = []
                    left = 0
                    for nn in num_layers:
                        batched_kernels.append(kernels[left:left + nn])
                        left += nn
                        args.network[u"kernels"] = batched_kernels
            args.network[u'num_layers'] = num_layers
    # print(args.network['num_layers'])
    # print(args.network['kernels'])
    return args


def parse_densenet_params(args):
    if u"network" in args:
        if u"num_layers" in args[u'network']:
            num_layers = read_list(args[u'network'][u'num_layers'])
            if u"kernels" in args[u'network']:
                kernels = read_list(args[u'network'][u'kernels'])
                if len(kernels) == 1 :
                    args[u'network'][u'kernels'] = [kernels * n for n in num_layers]
                else:
                    assert len(kernels) == sum(num_layers), u"The number of kernel sizes must match that of the network layers"
                    batched_kernels = []
                    left = 0 
                    for nn in num_layers:
                        batched_kernels.append(kernels[left:left+nn])
                        left += nn
                    args[u"network"][u"kernels"] = batched_kernels
            args[u'network'][u'num_layers'] = num_layers
    # print(args['network']['num_layers'])
    # print(args['network']['kernels'])
    return args


class ColorStreamHandler(logging.StreamHandler):
    u"""Logging with colors"""
    DEFAULT = u'\x1b[0m'
    RED = u'\x1b[31m'
    GREEN = u'\x1b[32m'
    YELLOW = u'\x1b[33m'
    CYAN = u'\x1b[36m'

    CRITICAL = RED
    ERROR = RED
    WARNING = YELLOW
    INFO = GREEN
    DEBUG = CYAN

    @classmethod
    def _get_color(cls, level):
        if level >= logging.CRITICAL:
            return cls.CRITICAL
        if level >= logging.ERROR:
            return cls.ERROR
        if level >= logging.WARNING:
            return cls.WARNING
        if level >= logging.INFO:
            return cls.INFO
        if level >= logging.DEBUG:
            return cls.DEBUG
        return cls.DEFAULT

    def __init__(self, stream=None):
        logging.StreamHandler.__init__(self, stream)

    def format(self, record):
        text = logging.StreamHandler.format(self, record)
        color = self._get_color(record.levelno)
        return color + text + self.DEFAULT


def create_logger(job_name, log_file=None, debug=True):
    u"""
    Initialize global logger
    log_file: log to this file, besides console output
    return: created logger
    """
    logging.basicConfig(level=5,
                        format=u'%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt=u'%m-%d %H:%M')
    logging.root.handlers = []
    if debug:
        chosen_level = 5
    else:
        chosen_level = logging.INFO
    logger = logging.getLogger(job_name)
    formatter = logging.Formatter(fmt=u'%(asctime)s %(message)s',
                                  datefmt=u'%m/%d %H:%M')
    if log_file is not None:
        log_dir = osp.dirname(log_file)
        if log_dir:
            if not osp.exists(log_dir):
                os.makedirs(log_dir)
        # cerate file handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(chosen_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    # Colored stream handler
    sh = ColorStreamHandler()
    sh.setLevel(chosen_level)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger
