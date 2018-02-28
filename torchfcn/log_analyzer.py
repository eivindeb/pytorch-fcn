import numpy as np
from matplotlib import pyplot as plt
import csv


class UnexpectedFactor(Exception):
    pass


class LogAnalyzer:
    def __init__(self, log_path):
        self.log_path = log_path
        self.data = None
        self.read_log_data()

    def read_log_data(self):
        with open(self.log_path, 'r') as log:
            reader = csv.DictReader(log)
            self.data = {"epoch_idxs": [], "validation_idxs": []}

            for i, row in enumerate(reader):
                for key, val in row.items():
                    if val != "" and val is not None:
                        if key not in self.data:
                            self.data.update({key: []})
                        if key == "filename":
                            self.data[key].append(val)
                            continue
                        if key == "valid/loss":
                            self.data["validation_idxs"].append(i)
                        elif key == "epoch" and int(val) > len(self.data["epoch_idxs"]):
                            self.data["epoch_idxs"].append(i - 1)
                        try:
                            self.data[key].append(int(val))
                        except ValueError:
                            try:
                                self.data[key].append(float(val))
                            except ValueError:
                                raise ValueError
                        except TypeError:
                            continue

        self.data = {key: np.asarray(data_list) for key, data_list in self.data.items()}

    def graph_factor(self, factor, x_axis_scale="epoch", include_validation=False, iteration_window=0, reject_outliers=True):
        if "train/{}".format(factor) not in self.data:
            raise UnexpectedFactor
        data = self.data["train/{}".format(factor)]
        if include_validation:
            try:
                valid_data = self.data["valid/{}".format(factor)]
                valid_y_values = []
                valid_x_values = []
                validation_idxs = self.data["validation_idxs"]
            except KeyError:
                print("No validation data in log")
                include_validation = False

        if x_axis_scale == "epoch":
            x_values = range(len(self.data["epoch_idxs"]))
            y_values = []
            for i, epoch_end in enumerate(self.data["epoch_idxs"]):
                epoch_start = 0 if i == 0 else self.data["epoch_idxs"][i-1]
                if include_validation:
                    validation_idxs_in_epoch = np.argwhere((epoch_start <= validation_idxs) & (validation_idxs <= epoch_end))
                    if validation_idxs_in_epoch.size > 0:
                        if validation_idxs_in_epoch.size == 1:
                            valid_y_values.append(valid_data[validation_idxs_in_epoch[0][0]])
                        else:
                            valid_y_values.append(np.average(valid_data[validation_idxs_in_epoch[0]:validation_idxs_in_epoch[-1]]))
                        valid_x_values.append(i)
                y_values.append(np.average(data[epoch_start: epoch_end]))

        elif x_axis_scale == "iteration":
            if iteration_window != 0:
                y_values = np.convolve(data, np.ones((iteration_window,))/iteration_window, mode="valid")
            else:
                y_values = data

            if reject_outliers:
                d = np.abs(y_values - np.median(y_values))
                mdev = np.median(d)
                s = d/mdev if mdev else 0
                y_values = y_values[s < 10]
            x_values = range(len(y_values))
            if include_validation:
                valid_y_values = valid_data
                valid_x_values = validation_idxs

        plt.plot(x_values, y_values, label="Training")
        if include_validation:
            plt.plot(valid_x_values, valid_y_values, label="Validation", marker="o")
        plt.title(factor)
        plt.legend()
        plt.xlabel(x_axis_scale)
        plt.show()


if __name__ == "__main__":
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_INTERVAL_WEIGHT_UPDATE-10_INTERVAL_CHECKPOINT-3000_MOMENTUM-0.99_INTERVAL_VALIDATE-60000_WEIGHT_DECAY-0.0005_LR-7.000000000000001e-12_MAX_ITERATION-800000_VCS-b'c68e4f7'_TIME-20180222-133545/log.csv"
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_MOMENTUM-0.99_INTERVAL_VALIDATE-60000_LR-2.1000000000000002e-11_INTERVAL_CHECKPOINT-5000_MAX_ITERATION-800000_WEIGHT_DECAY-0.0005_VCS-b'eb42bcf'_TIME-20180216-181256/log.csv"
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_LR-7.000000000000001e-12_MAX_ITERATION-800000_MOMENTUM-0.99_INTERVAL_CHECKPOINT-3000_INTERVAL_WEIGHT_UPDATE-10_INTERVAL_VALIDATE-100_WEIGHT_DECAY-0.0005_VCS-b'a41ae7a'_TIME-20180222-182224/log.csv"
    analyzer = LogAnalyzer(log_path)
    analyzer.graph_factor("loss", x_axis_scale="iteration", include_validation=True, iteration_window=10, reject_outliers=True)
