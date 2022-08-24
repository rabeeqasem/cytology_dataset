import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import splitfolders

splitfolders.ratio("reshaped_dataset", output="reshaped_dataset",
    seed=1337, ratio=(.8, 0, .2), group_prefix=None, move=False) # default values