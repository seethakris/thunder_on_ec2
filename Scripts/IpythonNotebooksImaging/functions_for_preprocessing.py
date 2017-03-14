from numpy import size, shape, mean, shape, convolve, r_, ones
import math
import matplotlib.pyplot as plt
from thunder import Colorize
import seaborn as sns
import matplotlib.patches as ptch
from thunder import Registration
import scipy
import scipy.signal
import os


class class_preprocess_data(object):
    # Create many functions for data analysis of images using thunder

    def __init__(self, tsc, time_baseline, stimulus_on_time, stimulus_off_time, stimulus_train, saveseriesdirectory,
                 multiplane=False):

        self.thundercontext = tsc
        self.image = Colorize.image
        self.time_baseline = time_baseline
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.stimulus_train = stimulus_train
        self.saveseriesdirectory = saveseriesdirectory
        self.multiplane = multiplane

    def load_and_preprocess_data(self, FileName, crop, register, img_size_crop_x1, img_size_crop_x2,
                                 img_size_crop_y1, img_size_crop_y2, medianfilter_window=1, start_frame=0,
                                 end_frame=285):

        # Load data and do a bit of preprocessing
        data = self.thundercontext.loadImages(FileName, inputFormat='tif', startIdx=start_frame, stopIdx=end_frame)

        data = data.medianFilter(size=medianfilter_window)

        if crop:  # If crop, crop the data else convert to timeseries
            print 'Cropping'
            data = self.crop_data(data, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2)

        self.saveasseries(data, 'raw_data')

        return data

    def crop_data(self, data, img_size_crop_x1, img_size_crop_x2, img_size_crop_y1, img_size_crop_y2):
        # Cropping unwanted pixels if required
        img_shape = data.first()[1].shape
        print img_shape
        # If data has many planes
        if self.multiplane:
            cropped_data = data.crop((img_size_crop_x1, img_size_crop_y1, 0),
                                     (img_shape[0] - img_size_crop_x2, img_shape[1] - img_size_crop_y2, img_shape[2]))
        else:
            cropped_data = data.crop((img_size_crop_x1, img_size_crop_y1),
                                     (img_shape[0] - img_size_crop_x2, img_shape[1] - img_size_crop_y2))
        return cropped_data

    def background_subtraction(self, data):

        bg_trace = data.meanByRegions([[(0, 0), (0, 10)]])
        subtracted_data = data.toTimeSeries().bg_subtract(bg_trace.first()[1])
        self.saveasseries(subtracted_data, 'bgubtracted_data')

        return subtracted_data, bg_trace

    def detrend_data(self, data, detrend_order=10):
        data = data.toTimeSeries().detrend(method='nonlin', order=detrend_order)
        self.saveasseries(data, 'detrended_data')
        return data

    @staticmethod
    def plot_registered_images(fig1, ind, original, corrected, pdffile):
        # Plot the means and differences for viewing
        plt.subplot(ind, 3, 1)
        plt.imshow(original, cmap='gray')
        plt.axis('off')
        plt.title('Original Image')
        plt.subplot(ind, 3, 2)
        plt.imshow(corrected, cmap='gray')
        plt.axis('off')
        plt.title('Registered Image')
        plt.subplot(ind, 3, 3)
        plt.imshow(corrected - original, cmap='gray')
        plt.axis('off')
        plt.title('Difference')
        plt.show()
        pdffile.savefig(fig1, bbox_inches='tight')

    def plotimageplanes(self, fig1, gs, img, plot_title='Habenula', gridspecs='[0,0]'):

        # If image has more than one plane, calculate number of subplots and plot
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(plot_title, fontsize=14)

    def normalize(self, data, squelch_parameter=0):
        # squelch to remove noisy pixels, normalize using user defined baseline
        print 'Baseline being used for normalizing is ...', self.time_baseline[0], ' to ', self.time_baseline[1]
        zscore_data = data.squelch(threshold=squelch_parameter).center(axis=0).zscore(axis=0)
        # zscore_data.cache()
        # zscore_data.dims
        self.saveasseries(zscore_data, 'zscore_data')
        return zscore_data

    def standardize(self, data, squelch_parameter=0, perc=10):
        # squelch to remove noisy pixels, standardize using standard deviation
        zscore_data = data.center(axis=1).toTimeSeries().normalize(baseline='mean', perc=perc)
        self.saveasseries(zscore_data, 'normalized_data')
        return zscore_data

    def loadseriesdataset(self, savefilename):
        if os.path.exists(self.saveseriesdirectory + savefilename):
            print 'Loading pre saved series dataset'
            data = self.thundercontext.loadSeries(self.saveseriesdirectory + savefilename)
            return data
        else:
            raise ValueError('No such series object exists')

    def saveasseries(self, data, savefilename):
        print 'Saving as series dataset from' + savefilename
        data.saveAsBinarySeries(self.saveseriesdirectory + savefilename, overwrite=True)

    # Performs smoothing using a hanning window on the rdd data
    @staticmethod
    def smooth_func(x, window_len=10):
        s = r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
        #         w = np.hanning(window_len)
        w = ones(window_len, 'd')
        y = convolve(w / w.sum(), s, mode='valid')
        # print 'Size of y...', shape(y)
        return y[window_len / 2:-window_len / 2 + 1]

    @staticmethod
    def detrend_data_with_scipy(x):
        return scipy.signal.detrend(x)

    @staticmethod
    def get_small_subset_for_plotting(data, number_samples=100, threshold=3):
        # Find a subset for plotting
        examples = data.subset(nsamples=number_samples, thresh=threshold)
        return examples

    def plot_traces(self, fig1, gs, plotting_data, gridspecs='[0,0]', **kwargs):
        # Data : rows are cells, column is time
        with sns.axes_style('darkgrid'):
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
            plt.plot(plotting_data.T)
            plt.plot(mean(plotting_data, 0), 'k', linewidth=2)
            if 'plot_title' in kwargs:
                plt.title(kwargs.values()[0], fontsize=14)
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax1)

    def plot_vertical_lines_onset(self):
        for ii in xrange(0, size(self.stimulus_on_time)):
            plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)

    def plot_vertical_lines_offset(self):
        for ii in xrange(0, size(self.stimulus_off_time)):
            plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)

    def plot_stimulus_patch(self, ax1):

        # Get y grid size
        y_tick = ax1.get_yticks()
        y_tick_width = y_tick[2] - y_tick[1]

        # Adjust axis to draw patch
        y_lim = ax1.get_ylim()
        ax1.set_ylim((y_lim[0], y_lim[1] + y_tick_width))

        for ii in xrange(0, size(self.stimulus_on_time)):
            # Find time of stimulus for width of patch
            time_stim = self.stimulus_off_time[ii] - self.stimulus_on_time[ii]

            # Check different cases of stimuli to create patches
            if self.stimulus_train[ii] == "Low":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2, fc='aqua')
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "Med":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='cornflowerblue')
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "High":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='blue')
                ax1.add_patch(rectangle)

            elif self.stimulus_train[ii] == "Lys":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='red')
                ax1.add_patch(rectangle)
            elif self.stimulus_train[ii] == "E3":
                rectangle = plt.Rectangle((self.stimulus_on_time[ii], y_lim[1]), time_stim, y_tick_width / 2,
                                          fc='green')
                ax1.add_patch(rectangle)

