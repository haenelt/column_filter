# -*- coding: utf-8 -*-
"""Python package for filtering cortical columns on a surface mesh. Execute via
the command line with python -m column_filter <args>."""

import argparse
import column_filter


# description
parser_description = "Gradient-based boundary (GBB) surface refinement. A " \
                     "GM/WM boundary surface (-w), a reference volume and an " \
                     "output directory (-o) are mandatory arguments. " \
                     "Optionally, a configuration file (-c) can be defined. " \
                     "Depending on the parameters in the configuration file, " \
                     "a vein mask (-v) and/or anchoring points (-a) are " \
                     "mandatory inputs as well. Additionally, a mask (-i) " \
                     "can be set to exlude regions from processing. A second " \
                     "surface (-p) can be set to apply the same deformation " \
                     "to another surface mesh."

# parse arguments from command line
parser = argparse.ArgumentParser(description=parser_description)
parser.add_argument('-c', '--config', type=str, help='configuration file', default=None)
requiredNames = parser.add_argument_group('mandatory arguments')
requiredNames.add_argument('-s', '--surf', type=str, help='input surface mesh')
args = parser.parse_args()

# run
print("-----------------------------------------------------------------------")
print("Column filter "+"(v"+str(column_filter.__version__)+")")
print("author: "+str(column_filter.__author__))
print("-----------------------------------------------------------------------")


# optional parameters
# sigma
# lambda
# ori
# phase
# and config stuff

# write overlay
# print out expected value