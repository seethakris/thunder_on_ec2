from numpy import std, logical_and, max, min, array, where, squeeze, delete, zeros, shape, size, round, reshape, \
    float64, transpose, repeat, mean
from thunder import KMeans
from thunder import Colorize
from copy import copy
from matplotlib.colors import ListedColormap
import palettable
import matplotlib.pyplot as plt
import seaborn as sns
import math


class class_kmeans(object):
    def __init__(self, kmeans_clusters, data, img_raw, stimulus_on_time, stimulus_off_time, stimulus_train):
        self.kmeans_clusters = kmeans_clusters
        self.data = data
        self.stimulus_on_time = stimulus_on_time
        self.stimulus_off_time = stimulus_off_time
        self.reference = img_raw
        self.stimulus_train = stimulus_train

    def run_kmeans(self):
        model = KMeans(k=self.kmeans_clusters).fit(self.data)
        img_labels = model.predict(self.data).pack()  # For masking
        sim = model.similarity(self.data)
        img_sim = sim.pack()

        return model, img_sim, img_labels

    def make_kmeans_maps(self, kmeans_clusters, img_labels, img_sim, mixing_parameter=0.8, std_threshold=0.5,
                         ignore_clusters=0, model_center=0):

        # Only plot those clusters where the standard deviation is greater than 0.1 - thus getting rid of noisy clusters

        interesting_clusters = array(where(logical_and(std(kmeans_clusters, 0) > std_threshold,
                                                       max(kmeans_clusters, 0) > 0.01)))

        print 'Standard deviation of clusters is..', std(kmeans_clusters, 0)
        print 'Interesting clusters after STD are..', interesting_clusters

        if ignore_clusters != 0:
            for ii in ignore_clusters:
                index = where(squeeze(interesting_clusters) == ii)[0]
                interesting_clusters = delete(interesting_clusters, index)
            print 'Interesting clusters after user specified clusters..', interesting_clusters

        # Update kmeans clusters with those with higher standard deviation
        kmeans_clusters_updated = zeros((shape(kmeans_clusters)))
        kmeans_clusters_updated[:, interesting_clusters] = kmeans_clusters[:, interesting_clusters]

        # Brewer colors
        newclrs_brewer = ListedColormap(sns.color_palette("Paired", n_colors=size(kmeans_clusters, 1)),
                                        name='from_list')
        # newclrs_brewer = (eval('palettable.colorbrewer.qualitative.' + string_cmap + '.mpl_colors'))
        # newclrs_brewer = ListedColormap(newclrs_brewer, name='from_list')
        newclrs_updated_brewer = self.update_colors(newclrs_brewer, ignore_clusters, interesting_clusters)

        # RGB colors

        newclrs_rgb = ListedColormap(sns.color_palette("husl", n_colors=size(kmeans_clusters, 1) + 1)[:-1],
                                     name='from_list')
        newclrs_updated_rgb = self.update_colors(newclrs_rgb, ignore_clusters, interesting_clusters)
        newclrs_updated_rgb.colors = round(newclrs_updated_rgb.colors)
        # Create maps
        brainmap = Colorize(cmap=newclrs_updated_brewer).transform(img_labels, mask=img_sim, background=self.reference,
                                                                   mixing=mixing_parameter)
        brainmap_for_finding_pixels = Colorize(cmap=newclrs_updated_rgb).transform(img_labels, mask=img_sim)

        # Count number of unique colors in the images
        # Get number of planes based on map dimensions
        if len(brainmap.shape) == 3:
            num_planes = 1
        else:
            num_planes = size(brainmap, 2)

        # Get specific color matches across animals and get mean and standard deviation
        round_clrs = round(newclrs_updated_rgb.colors)
        new_array = [tuple(row) for row in round_clrs]
        unique_clrs = (list(set(new_array)))  # Get unique combination of colors
        ## remove black color if it exists
        elem = (0, 0, 0)
        unique_clrs = [value for key, value in enumerate(unique_clrs) if elem != value]
        unique_clrs = round(unique_clrs)

        # From maps get number of pixel matches with color for each plane
        array_maps = brainmap_for_finding_pixels
        matched_pixels = zeros((size(unique_clrs, 0), num_planes))

        array_maps_plane = reshape(array_maps, (size(array_maps, 0) * size(array_maps, 1), 3))
        matched_pixels[:, 0] = [size(where((array(round(array_maps_plane)) == match).all(axis=1))) for match
                                in unique_clrs]

        # print matched_pixels, unique_clrs

        # Get brewer colors for plotting matched pixels
        elem = [0, 0, 0]
        a = unique_clrs.tolist()
        b = newclrs_updated_rgb.colors.tolist()
        c = newclrs_updated_brewer.colors.tolist()

        d1 = [value for key, value in enumerate(b) if elem != value]
        d2 = [value for key, value in enumerate(c) if elem != value]

        unique_clrs_brewer = zeros(shape(unique_clrs))
        for ii in xrange(0, size(unique_clrs, 0)):
            unique_clrs_brewer[ii, :] = d2[a.index(d1[ii])]

        return brainmap, unique_clrs_brewer, newclrs_updated_rgb, newclrs_updated_brewer, matched_pixels, \
               kmeans_clusters_updated

    @staticmethod
    def update_colors(newclrs, ignore_clusters, interesting_clusters):

        newclrs_updated = copy(newclrs)
        newclrs_updated.colors = zeros(shape(newclrs_updated.colors), dtype=float64)
        newclrs.colors = array(newclrs.colors)

        if ignore_clusters != 0:
            newclrs_updated.colors = newclrs.colors
            newclrs_updated.colors[ignore_clusters, :] = [0, 0, 0]
        else:
            newclrs_updated.colors[interesting_clusters, :] = newclrs.colors[interesting_clusters, :]

        return newclrs_updated

    def plot_kmeans_components(self, fig1, gs, kmeans_clusters, uncentered_clrs, plot_title='Hb', num_subplots=1,
                               flag_separate=1, gridspecs=[0, 0], model_center=0, removeclusters=0):

        with sns.axes_style('darkgrid'):
            if flag_separate:
                ax1 = fig1.add_subplot(2, 1, num_subplots)
            else:
                ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')

            if model_center == 1:  # Models colors according to trace
                clrs_cmap = Colorize.optimize(kmeans_clusters.T, asCmap=True)
                clrs = clrs_cmap.colors
            else:
                clrs_cmap = uncentered_clrs
                clrs = clrs_cmap.colors

            # Update kmeans if clusters were removed
            kmeans_clusters_new = copy(kmeans_clusters)
            if removeclusters != 0:
                newclrs_updated = copy(clrs_cmap)
                for index, value in enumerate(clrs_cmap.colors):
                    if index in removeclusters:
                        newclrs_updated.colors[index] = [0, 0, 0]
                clrs_cmap = newclrs_updated
                clrs = clrs_cmap.colors

                for ii in removeclusters:
                    print ii
                    kmeans_clusters_new[:, ii] = zeros(size(kmeans_clusters, 0))

            plt.gca().set_color_cycle(clrs)
            # remove those clusters that are ignored before plotting
            for ii in xrange(0, size(kmeans_clusters_new, 1)):
                plt.plot(kmeans_clusters_new[:, ii], lw=4, label=str(ii))

            plt.locator_params(axis='y', nbins=4)
            ax1.set(xlabel="Time (seconds)", ylabel="a.u")
            ax1.legend(prop={'size': 14}, loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fancybox=True,
                       shadow=True)

            plt.title(plot_title, fontsize=14)

            plt.ylim((min(kmeans_clusters_new) - 0.0001,
                      max(kmeans_clusters_new) + 0.0001))
            plt.xlim((0, size(kmeans_clusters_new, 0)))
            plt.axhline(y=0, linestyle='-', color='k', linewidth=1)
            self.plot_vertical_lines_onset()
            self.plot_vertical_lines_offset()
            self.plot_stimulus_patch(ax1)

        return clrs_cmap

    def plot_vertical_lines_onset(self):
        for ii in xrange(0, size(self.stimulus_on_time)):
            plt.axvline(x=self.stimulus_on_time[ii], linestyle='-', color='k', linewidth=2)

    def plot_vertical_lines_offset(self):
        for ii in xrange(0, size(self.stimulus_off_time)):
            plt.axvline(x=self.stimulus_off_time[ii], linestyle='--', color='k', linewidth=2)

    def plotimageplanes(self, fig1, gs, img, plot_title='Habenula', gridspecs=[0, 0]):

        # If image has more than one plane, calculate number of subplots and plot
        if len(shape(img)) > 3:
            numz = size(img, 3)
            num_subplots = int((math.ceil(numz / 2.) * 2) / 2)
            for ii in xrange(0, numz):
                ax1 = fig1.add_subplot(num_subplots, 2, ii + 1)
                ax1.imshow(img[:, :, ii])
                ax1.axis('off')
                plt.title(plot_title, fontsize=12)

        else:  # If single plane, get user defined subplot number for plotting
            ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
            ax1.imshow(img)
            ax1.axis('off')
            plt.title(plot_title, fontsize=14)

    def createbrainmap_withcmap(self, fig1, gs, colormap, img_labels, img_sim, mixing_parameter, gridspecs):
        brainmap = Colorize(cmap=colormap).transform(img_labels, mask=img_sim, background=self.reference,
                                                     mixing=mixing_parameter)
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        ax1.imshow(brainmap)
        ax1.axis('off')

    @staticmethod
    def plot_matchedpixels(fig1, gs, matched_pixels, unique_clrs, gridspecs=[0, 0]):
        ax1 = eval('fig1.add_subplot(gs' + gridspecs + ')')
        with sns.axes_style("darkgrid"):
            # print shape(matched_pixels)
            for ii in xrange(0, size(matched_pixels, 0)):
                plt.plot(ii + 1, transpose(matched_pixels[ii, :]), 'o', color=unique_clrs[ii], markersize=10)
                plt.xlim([0, size(matched_pixels, 0) + 1])

            # for ii in range(0, size(unique_clrs, 0)):
            #     plt.plot(repeat(ii + 1, size(matched_pixels, 1)), transpose(matched_pixels[ii, :]), 's',
            #              color=unique_clrs[ii], markersize=10, markeredgecolor='k', markeredgewidth=2)

            x = range(0, size(unique_clrs, 0) + 1)
            labels = [str(e) for e in x]

            plt.xticks(x, labels, rotation='vertical')
            ax1.set(xlabel="Colors", ylabel="Number of Pixels")
            sns.despine(offset=10, trim=True)

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
