import sigfig as sgf
try :
    import cupy
    cpy = True
except ImportError:
    cpy = False
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import solve_library as solvel
import data_library as data
import csv
import math
import matplotlib.pyplot as plt
import os
import pathlib
import pycaso
