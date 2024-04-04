import glob
import configparser

# Data structures:
from .AnnotationMatrix import AnnotationMatrix
from .LDMatrix import LDMatrix
from .GWADataLoader import GWADataLoader
from .SumstatsTable import SumstatsTable
from .SampleTable import SampleTable

# Simulation:

from .simulation.PhenotypeSimulator import PhenotypeSimulator

# Data utilities:

from .utils.data_utils import *

__version__ = '0.1.0'
__release_date__ = 'April 2024'


config = configparser.ConfigParser()
config.read(glob.glob(osp.join(osp.dirname(__file__), 'config/*.ini')))


def print_options():
    """
    Print the options stored in the configuration file
    """
    for sec in config.sections() + [config.default_section]:
        print("-> Section:", sec)
        for key in config[sec]:
            print(f"---> {key}: {config[sec][key]}")


def get_option(key):
    """
    Get the option associated with a given key
    """
    try:
        return config['USER'][key]
    except KeyError:
        return config['DEFAULT'][key]


def set_option(key, value):
    """
    Set an option in the configuration file by providing a key and a value
    """
    if 'USER' in config.sections():
        config['USER'][key] = value
    else:
        config['USER'] = {key: value}

    with open(osp.join(osp.dirname(__file__), 'config/paths.ini'), 'w') as configfile:
        config.write(configfile)
