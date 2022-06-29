import configparser
import importlib
import appdirs
import pathlib

# initialize configparser to be imported and used anywhere
config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
with importlib.resources.path('nn_training', 'config.ini') as default_config_file:
    config.read_file(open(default_config_file))
    config.read([pathlib.Path(appdirs.user_config_dir('nn_training')) / 'config.ini'])
    
del configparser, importlib, appdirs, pathlib