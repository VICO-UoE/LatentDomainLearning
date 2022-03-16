from collections import defaultdict
import numpy as np
import os


class AverageValue(object):
    """
    Computes and stores the average and current value.
    Source:
    https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self, color, form):
        self.reset()
        self.color = color
        self.form = form


    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0


    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Log():
    """
    Meta-class that holds observables, has methods to update and report them. Automatically generates a legend. Since deep learning always works with batches and epochs, these are not stored internally, and instead taken for granted. The input arguments to methods such as Log.save_to_dat(..) reflect this.
    """

    def __init__(self, file=None):
        self.observables = {}
        self.file = file
        self.storage = defaultdict(dict)


    def register(self, name, color=None, format="{0:.0}"):
        """
        Registers a new observable by initializing an AverageValue. Optional arguments are a highlighting color and a specific format for the observable.
        args:
            name
            color (optional)
            format (optional)
        """

        self.observables[name] = AverageValue(color, format)


    def update(self, name, new_value, n=1):
        """
        Updates the value of AverageMeter with a new observation.
        args:
            name
            new_value
            n (number of observations, optional)
        """
        if n > 0:
            self.observables[name].update(new_value, n)


    def reset(self):
        """
        Resets the values of AverageMeters.
        """

        for v in self.observables.values():
            v.reset()


    def average(self, obs):
        """
        Returns average of an observable.
        args:
            obs
        """
        assert obs in self.observables.keys()
        return self.observables[obs].avg


    def legend(self):
        """
        Prints a colorized legend of the observables that are written to the console (or any alternative stream stored in self.file).
        """

        legend_str = colorize("white", "[epoch]")
        legend_str += " "
        legend_str += "[batch_id]"
        legend_str += " "

        for k, v in self.observables.items():
            legend_str += colorize(v.color, "[" + "{0}".format(k)+ "]")
            legend_str += " "
        print(legend_str, end="\n\n", file=self.file, flush=True)


    def report(self, which, epoch, batch_id=None, episode=None):
        """
        Prints the current averages of all AverageMeters specified in which (a list). Requires epoch and batch_id.
        args:
            which (list of AverageMeters to display)
            epoch
            batch_id
        """

        report_str = colorize("white", "[" + "{0}".format(epoch) + "]")
        report_str += " "
        if batch_id is not None:
            report_str += "[" + str(batch_id) + "]"
            report_str += " "
        if episode is not None:
            report_str += "[" + str(episode) + "]"
            report_str += " "

        for k, v in self.observables.items():
            if k in which:
                report_str += colorize(v.color, "[" + v.form.format(v.val) + "]")
                report_str += " "

        print(report_str,
            end="\r",
            file=self.file,
            flush=True)


    def save_to_dat(self, epoch, path, dataset=None, reset_log_values=False):
        """
        Stores the averages of the current as well as all previous epochs (for all AverageMeters in the Log) to a file log.dat in the path specified. Automatically generates a header. Requires an epoch.
        args:
            epoch
            path
        """

        # disregard all properties except AverageValue.avg ..
        self.storage[epoch] = {k: v.avg for k, v in self.observables.items()}

        # .. and bring into appropriate format
        vals_to_file = []
        for k, v in self.storage.items():
            header = ",".join(["epoch", *v])
            vals_to_file.append([k, *v.values()])

        if dataset is not None:
            file_path = os.path.join(path, "log_%s.dat" % dataset)
        else:
            file_path = os.path.join(path, "log.dat")

        np.savetxt(file_path,
            np.vstack(vals_to_file),
            delimiter=",",
            fmt=["%i"] + (len(vals_to_file[0])-1)*["%.8e"],
            header=header,
            comments="")

        if reset_log_values:
            self.reset()


def colorize(color, string):
    """
    Returns colorized strings.
    """

    if color == "red":
        return "\033[41m" + string + "\033[0m"
    elif color == "green":
        return "\033[42m" + string + "\033[0m"
    elif color == "yellow":
        return "\033[43m" + string + "\033[0m"
    elif color == "blue":
        return "\033[44m" + string + "\033[0m"
    elif color == "purple":
        return "\033[45m" + string + "\033[0m"
    elif color == "cyan":
        return "\033[46m" + string + "\033[0m"
    elif color == "white":
        return "\033[47m" + string + "\033[0m"
    else:
        return string
