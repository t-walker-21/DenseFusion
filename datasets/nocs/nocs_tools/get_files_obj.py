"""

Tool to grab info from files of NOCS dataset and aggregrate them together
"""

import os
import sys
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--obj_id', type=int, required=True, help='object ID of interest')
opt = parser.parse_args()

print (opt.obj_id)