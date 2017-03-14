import seaborn as sns
import matplotlib.pyplot as plt


class GetStimulusOnOffTimes(object):
    # Initiate class
    def __init__(self, experiment_type, **kwargs):
        self.experiment_type = experiment_type
        self.experiment_parameters = kwargs
        CheckArguments(self.experiment_type,
                       self.experiment_parameters)  # Check arguments given by user for relevant experiment

    def get_stimulus_parameters(self):

        # For stimulus with alternate pulses of light
        if self.experiment_type == '2Stimx3':
            stimulus_on_time = [46, 86, 126, 166, 206, 246]
            stimulus_off_time = [65, 105, 145, 185, 225, 265]
            stimulus_train = self.experiment_parameters['light_type'] * (len(stimulus_on_time) / 2)
            color_mat = ['#00FFFF', '#FF0000', '#0000FF', '#FF1493', '#3090C7', '#800000']  # blue-red alternates

        elif self.experiment_type == '1Stimx4New':
            stimulus_on_time = [46, 86, 126, 166]
            stimulus_off_time = [65, 105, 145, 185]
            stimulus_train = self.experiment_parameters['light_type'] * (len(stimulus_on_time))
            color_mat = sns.color_palette(str(self.experiment_parameters['light_type']).strip("'[]'") + 's',
                                          len(stimulus_on_time) + 2)[2:]

        elif self.experiment_type == 'BlueRedx2':
            stimulus_on_time = [43, 83, 123, 163]
            stimulus_off_time = [64, 104, 144, 184]
            stimulus_train = ['Blue', 'Red', 'Blue', 'Red']
            color_mat = ['#00FFFF', '#FF0000', '#0000FF', '#FF1493']  # blue-red alternates

        elif self.experiment_type == 'FarRedBluex3':
            stimulus_on_time = [32, 52, 72, 92, 112, 132]
            stimulus_off_time = [42, 62, 82, 102, 122, 142]
            stimulus_train = ['Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
            color_mat = ['#FF0000', '#FF1493', '#800000', '#00FFFF', '#0000FF', '#3090C7']  # blue-red alternates

        elif self.experiment_type == '1color4stim':
            stimulus_on_time = [46, 98, 142, 194]
            stimulus_off_time = [69, 120, 164, 216]
            stimulus_train = self.experiment_parameters['light_type'] * (len(stimulus_on_time))
            color_mat = sns.color_palette(str(self.experiment_parameters['light_type']).strip("'[]'") + 's',
                                          len(stimulus_on_time) + 2)[2:]

        elif self.experiment_type == 'RedBlueHighSpeedLQ':
            stimulus_on_time = [422, 702, 982, 1262, 1542, 1822]
            stimulus_off_time = [574, 854, 1134, 1414, 1694, 2044]
            print 'number of stimulus pulses caluculated is %s' % (len(stimulus_on_time))
            stimulus_train = ['Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue']
            color_mat = sns.color_palette(["salmon", "aqua", "orangered", "dodgerblue", "maroon", "royalblue"])


        # For high speed stimuli with just one type of light
        elif self.experiment_type == 'HighSpeed13fps':
            time_end = self.experiment_parameters['time_end']
            frames_per_sec = self.experiment_parameters['frames_per_sec']
            num_frames_in_20s = frames_per_sec * 20
            frame_num = num_frames_in_20s - frames_per_sec * 2 + 10
            stimulus_off_time = []
            stimulus_on_time = []
            while frame_num < time_end - num_frames_in_20s:
                stimulus_on_time.append(frame_num)
                stimulus_off_time.append(frame_num + num_frames_in_20s)
                frame_num += num_frames_in_20s * 2
            stimulus_on_time[6] += 5
            # stimulus_on_time[5] += 10
            stimulus_off_time[6] += 5
            # stimulus_off_time[5] += 10
            print 'number of stimulus pulses caluculated is %s' % (len(stimulus_on_time))
            stimulus_train = self.experiment_parameters['light_type'] * len(stimulus_on_time)
            color_mat = sns.color_palette(str(self.experiment_parameters['light_type']).strip("'[]'") + 's',
                                          len(stimulus_on_time) + 2)[2:]


        elif self.experiment_type == 'HighSpeed30fps':
            time_end = self.experiment_parameters['time_end']
            frames_per_sec = self.experiment_parameters['frames_per_sec']
            num_frames_in_20s = frames_per_sec * 20
            frame_num = num_frames_in_20s + frames_per_sec / 7 - 25
            stimulus_off_time = []
            stimulus_on_time = []
            while frame_num < time_end - num_frames_in_20s:
                stimulus_on_time.append(frame_num)
                stimulus_off_time.append(frame_num + num_frames_in_20s)
                frame_num += num_frames_in_20s * 2
            stimulus_on_time[5] += 40
            stimulus_on_time[4] += 20
            stimulus_off_time[5] += 40
            stimulus_off_time[4] += 20

            print 'number of stimulus pulses caluculated is %s' % (len(stimulus_on_time))
            stimulus_train = self.experiment_parameters['light_type'] * len(stimulus_on_time)
            color_mat = sns.color_palette(str(self.experiment_parameters['light_type']).strip("'[]'") + 's',
                                          len(stimulus_on_time) + 2)[2:]

        self.print_and_plot_stuff(stimulus_on_time, stimulus_off_time, stimulus_train, color_mat)
        return stimulus_on_time, stimulus_off_time, stimulus_train, color_mat

    @staticmethod
    def print_and_plot_stuff(stimulus_on_time, stimulus_off_time, stimulus_train, color_mat):

        print 'For this experiment:\n Stimulus ON at : %s\n Stimulus OFF at : %s\n Type of Stimulus is : %s\n' \
              ' Colormap for PCA is:' % (str(stimulus_on_time), str(stimulus_off_time), stimulus_train)
        sns.palplot(color_mat)  # Plot colormap chosen for PCA plots
        plt.show()
        plt.close()


class RequiredArgumentsForExperimentType(object):
    def __init__(self, experiment_type):
        self.available_experiments = ['2Stimx3', '1color4stim', 'HighSpeed13fps', 'HighSpeed30fps',
                                      'RedBlueHighSpeedLQ', 'BlueRedx2', 'FarRedBluex3', '1Stimx4New']

        if experiment_type == '2Stimx3':
            self.kwargs_needed = ['light_type']
            self.length_light_type = 2
        if experiment_type == '1Stimx4New':
            self.kwargs_needed = ['light_type']
            self.length_light_type = 1
        if experiment_type == '1color4stim':
            self.kwargs_needed = ['light_type']
            self.length_light_type = 1
        if experiment_type == 'BlueRedx2':
            self.kwargs_needed = ['light_type']
            self.length_light_type = 1
        if experiment_type == 'FarRedBluex3':
            self.kwargs_needed = ['light_type']
            self.length_light_type = 1
        if experiment_type == 'HighSpeed13fps':
            self.kwargs_needed = ['light_type', 'time_end', 'frames_per_sec']
            self.length_light_type = 1
        if experiment_type == 'HighSpeed30fps':
            self.kwargs_needed = ['light_type', 'time_end', 'frames_per_sec']
            self.length_light_type = 1
        if experiment_type == 'RedBlueHighSpeedLQ':
            self.kwargs_needed = ['light_type', 'time_end', 'frames_per_sec']
            self.length_light_type = 1


# Errors to be generates
class CheckArguments(object):
    def __init__(self, experiment_type, kwargs_given):
        self.kwargs_given = kwargs_given
        self.experiment_type = experiment_type

        args_needed = RequiredArgumentsForExperimentType(self.experiment_type)
        self.available_experiments = args_needed.available_experiments
        self.kwargs_needed = args_needed.kwargs_needed
        self.length_light_type = args_needed.length_light_type
        self.check_arguments_provided()

    def check_arguments_provided(self):
        try:
            if self.experiment_type not in self.available_experiments:
                raise ExperimentTypeError(self.experiment_type).print_message()

            for key in self.kwargs_needed:
                if key not in self.kwargs_given:
                    raise NameError

            if len(self.kwargs_given['light_type']) != self.length_light_type:
                raise ValueError

        except NameError:
            print 'The additional arguments required for experiment type %s are : \n' % self.experiment_type
            for a, b in enumerate(self.kwargs_needed, 1):
                print '{} {}'.format(a, b)

        except ValueError:
            print 'Given : %s \n Required %s types of stimuli' % (self.light_type, self.length_light_type)

        else:
            print 'Correct arguments given for this experiment type'


class ExperimentTypeError(Exception):
    @staticmethod
    def print_message(experiment_type):
        print 'Error : This function helps create stimulus train and colors for these stimuli only \n'
        for a, b in enumerate(experiment_type, 1):
            print '{} {}'.format(a, b)
