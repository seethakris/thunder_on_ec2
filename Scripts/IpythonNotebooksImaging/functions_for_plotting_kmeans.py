from numpy import load, size, min, max, array, shape, mean, linspace, around, arange, tile, asarray, transpose, zeros, \
    round, reshape, where, r_, convolve, ones
import seaborn as sns
import matplotlib.pyplot as plt
from thunder import Colorize
import time
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import axes3d
import matplotlib as mpl
from copy import copy
from scipy.stats.stats import pearsonr


class plotKmeans(object):
    def __init__(self, fileName):

        # self.frames_per_sec = frames_per_sec

        npzfile = load(fileName + 'kmeans_results.npz')
        print 'Files loaded are %s' % npzfile.files
        self.stimulus_on_time = npzfile['stimulus_on_time']
        self.stimulus_off_time = npzfile['stimulus_off_time']
        self.stimulus_train = npzfile['stimulus_train']
        self.kmeans_clusters = npzfile['kmeans_clusters']
        self.kmeans_clusters = self.kmeans_clusters[0:npzfile['time_experiment'], :]
        self.removeclusters = npzfile['ignore_clusters']
        self.brainmap = npzfile['brainmap']
        self.centered_cmap = npzfile['centered_cmap']
        self.img_sim = npzfile['img_sim']
        self.img_labels = npzfile['img_labels']
        self.reference_image = npzfile['reference_image']
        self.matched_pixels = npzfile['matched_pixels']

