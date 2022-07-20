#             GNU AFFERO GENERAL PUBLIC LICENSE
#                Version 3, 19 November 2007
#
# Copyright(C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

# https://github.com/fmcarlucci/JigenDG
from models import resnet

nets_map = {
    'resnet18': resnet.resnet18, 
}

def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)
    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)
    return get_network_fn
