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

__version__ = '0.1.5'
__release_date__ = 'April 2025'


def make_ascii_logo(desc=None, left_padding=None):
    """
    Generate an ASCII logo for the magenpy package.
    :param desc: A string description to be added below the logo.
    :param left_padding: Padding to the left of the logo.

    :return: A string containing the ASCII logo.
    """

    logo = r"""
 _ __ ___   __ _  __ _  ___ _ __  _ __  _   _ 
| '_ ` _ \ / _` |/ _` |/ _ \ '_ \| '_ \| | | |
| | | | | | (_| | (_| |  __/ | | | |_) | |_| |
|_| |_| |_|\__,_|\__, |\___|_| |_| .__/ \__, |
                 |___/           |_|    |___/
    """

    lines = logo.replace(' ', '\u2001').splitlines()[1:]
    lines.append("Modeling and Analysis of Genetics data in python")
    lines.append(f"Version: {__version__} | Release date: {__release_date__}")
    lines.append("Author: Shadi Zabad, McGill University")

    # Find the maximum length of the lines
    max_len = max([len(l) for l in lines])
    if desc is not None:
        max_len = max(max_len, len(desc))

    # Pad the lines to the same length
    for i, l in enumerate(lines):
        lines[i] = l.center(max_len)

    # Add separators at the top and bottom
    lines.insert(0, '*' * max_len)
    lines.append('*' * max_len)

    if desc is not None:
        lines.append(desc.center(max_len))

    if left_padding is not None:
        for i, l in enumerate(lines):
            lines[i] = '\u2001' * left_padding + l

    return "\n".join(lines)


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
