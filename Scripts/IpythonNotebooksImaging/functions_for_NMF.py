from numpy import newaxis, squeeze, size, where, array, mean, zeros, round, reshape, float16, min, max, shape, linspace
from thunder import NMF
from thunder import Colorize
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d
from copy import copy


class class_NMF(object):
    def __init__(self, NMF_components, data, img_raw, NMF_colors, color_map, max_iterations, tolerance_level,
                 stimulus_on_time, stimulus_off_time, stimulus_train, sampling_rate):

        self.NMF_components = NMF_components
        self.max_iterations = max_iterations
        self.tolerance_level = tolerance_level
        self.color_map = color_map
        self.num_NMF_colors = NMF_components
        self.NMF_colors = NMF_colors
        self.reference = img_raw
        self.data = data
        self.sampling_rate = sampling_rate

        self.stimulus_train = stimulus_train
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.image = Colorize.image

    def run_NMF(self):
        model = NMF(k=self.NMF_components, maxIter=self.max_iterations, tol=self.tolerance_level, verbose='True').fit(
            self.data)
        imgs = model.w.pack()

        return model, imgs

    # Make maps and scatter plots of the NMF scores with colormaps for plotting
    def make_NMF_maps(self, imgs, mixing_parameter, ignore_clusters=0):

        new_NMF_colors = copy(self.NMF_colors)
        if ignore_clusters != 0:
            new_NMF_colors[:, ignore_clusters] = 0, 0, 0

        maps = Colorize(cmap=self.color_map, colors=new_NMF_colors,
                        scale=self.num_NMF_colors).transform(imgs,
                                                             background=self.reference,
                                                             mixing=mixing_parameter)

        return maps

    def plot_nmf_components(self, fig1, gs, nmf_components, ignore_clusters=0, plot_title='Habneula',
                            gridspecs='[0,0]'):
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        x = self.convert_frames_to_sec(nmf_components)

        with sns.axes_style('darkgrid'):
            if ignore_clusters != 0:
                for ii in range(size(nmf_components, 1)):
                    if ii not in ignore_clusters:
                        plt.plot(x, nmf_components[:, ii], alpha=0.5, linewidth=3, label=str(ii),
                                 color=self.NMF_colors[ii])
            else:
                for ii in range(size(nmf_components, 1)):
                    plt.plot(x, nmf_components[:, ii], alpha=0.5, linewidth=3, label=str(ii), color=self.NMF_colors[ii])

            plt.title(plot_title, fontsize=14)
            ax1.set(xlabel="Time (seconds)", ylabel="a.u")
            plt.locator_params(axis='y', nbins=4)
            ax1.legend(prop={'size': 14}, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
                       shadow=True)
            plt.axhline(y=0, linestyle='-', color='k', linewidth=1)
            ax1.locator_params(axis='y', pad=50, nbins=6)
            ax1.locator_params(axis='x', pad=50, nbins=12)
            plt.ylim((min(nmf_components) - 0.0001, max(nmf_components) + 0.0001))
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax1)

    def convert_frames_to_sec(self, nmf_components):
        frames_to_time = linspace(1, size(nmf_components, 0), size(nmf_components, 0)) / self.sampling_rate
        return frames_to_time

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

    class structtype():
        pass
