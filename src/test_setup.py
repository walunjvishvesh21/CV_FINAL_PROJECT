# # This is a Simple environment check script to verify that the required Python libraries
#  are installed and working correctly.

import os
import sys
import torch
import torchvision
import pandas as pd
import sklearn
import matplotlib
import PIL
import numpy as np

print("Python exe:", sys.executable)
print("Current working dir:", os.getcwd())
print("Torch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("Pandas:", pd.__version__)
print("NumPy:", np.__version__)
print("Setup works")
