import os.path as osp
import configparser

from .GWASDataLoader import GWASDataLoader
from .GWASSimulator import GWASSimulator
from .TransethnicGWASSimulator import TransethnicGWASSimulator
from .LDWrapper import LDWrapper

config = configparser.ConfigParser()
config.read(osp.join(osp.dirname(__file__), 'config.ini'))


def set_option(key, value):
    if 'USER' in config.sections():
        config['USER'][key] = value
    else:
        config['USER'] = {key: value}

    with open(osp.join(osp.dirname(__file__), 'config.ini'), 'w') as configfile:
        config.write(configfile)
