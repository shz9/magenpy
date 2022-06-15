import os.path as osp
import glob
import configparser

# Data structures:
from magenpy.AnnotationMatrix import AnnotationMatrix
from magenpy.LDMatrix import LDMatrix
from magenpy.GWADataLoader import GWADataLoader
from magenpy.SumstatsTable import SumstatsTable
from magenpy.SampleTable import SampleTable

# Simulation:

from magenpy.simulation.GWASimulator import GWASimulator

__version__ = '0.0.2'


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


def tgp_eur_data_path():
    """
    Return the path of the attached 1000G sample data for
    European individuals (N=378) and chromosome 22 (p=15938)
    """
    return osp.join(osp.dirname(__file__), 'data/1000G_eur_chr22')
