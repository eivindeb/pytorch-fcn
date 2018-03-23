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
            self.data = {"train": [{"data": {}}], "valid": []}

            is_validation = False

            for i, row in enumerate(reader):
                if int(row["epoch"]) > len(self.data["train"]) - 1:
                    self.data["train"].append({"data": {}})
                if row["train/loss"] != "" and is_validation:
                    is_validation = False
                    self.data["valid"][-1]["mean"] = {key: val.pop() for key, val in
                                                      self.data["valid"][-1]["data"].items() if key != "filename"}
                    self.data["valid"][-1]["end_time"] = float(row["elapsed_time"])
                elif row["valid/loss"] != "" and not is_validation:
                    is_validation = True
                    self.data["valid"].append(
                        {"data": {}, "epoch": int(row["epoch"]), "iteration": int(row["iteration"]),
                         "start_time": self.data["train"][-1]["data"]["elapsed_time"][-1], "end_time": None})
                for key, val in row.items():
                    if val != "" and val is not None:
                        if key in {"iteration", "epoch"}:
                            continue
                        else:
                            if key != "filename":
                                try:
                                    val = int(val)
                                except ValueError:
                                    try:
                                        val = float(val)
                                    except ValueError:
                                        raise ValueError
                                except TypeError:  # what case is this?
                                    continue

                            row_cat = "valid" if is_validation else "train"
                            if key not in self.data[row_cat][-1]["data"]:
                                self.data[row_cat][-1]["data"].update({key: []})

                            self.data[row_cat][-1]["data"][key].append(val)

            if is_validation:
                self.data["valid"][-1]["mean"] = {key: val.pop() for key, val in self.data["valid"][-1]["data"].items() if key != "filename"}
                self.data["valid"][-1]["end_time"] = float(row["elapsed_time"])

    def validation_metric_histogram(self, metric, validation_idx=-1):
        try:
            data = self.data["valid"][validation_idx]["data"]["valid/{}".format(metric)]
        except Exception as e:
            print("No data available for selected validation index and metric")
            raise e
        iteration = self.data["valid"][validation_idx]["iteration"]
        data = np.asarray(data)
        plt.hist(data[~np.isnan(data)], bins=100)
        plt.xlim(0, 1)
        plt.title("Histogram for {} in validation on iteration {}".format(metric, iteration))
        plt.show()

    def get_low_scoring_files(self, metric, percentile=1, mode="min", split="valid", split_idx=-1):
        data = np.asarray(self.data[split][split_idx]["data"]["{}/{}".format(split, metric)])

        N = int(data.size * percentile / 100)

        if mode == "min":
            res = np.argsort(data)[:N]
        elif mode == "max":
            res = np.argsort(data)[::-1][:N]

        return [self.data[split][split_idx]["data"]["filename"][i] for i in res]


    def graph_factor(self, factor, x_axis_scale="iteration", include_validation=False, per_class=False, iteration_window=0, reject_outliers=True):
        train_factors = [f for d in self.data["train"] for f in d["data"] if "train/{}".format(factor) in f]
        #train_factors = {"train/{}".format(factor): []}
        include_training = any(any(t_f in d["data"] for t_f in train_factors) for d in self.data["train"])
        if include_training:
            y_values = {t_f: [] for t_f in train_factors}
            x_values = {t_f: [] for t_f in train_factors}
        if include_validation:
            valid_factor = "valid/{}".format(factor)
            if not any([valid_factor in d["data"] or valid_factor in d["mean"] for d in self.data["valid"]]):
                print("No validation data available for requested factor {}".format(factor))
                include_validation = False
            else:
                if per_class:
                    if "mean_" in valid_factor:
                        base_factor = valid_factor.replace("mean_", "")
                    elif "mean" in valid_factor:
                        base_factor = valid_factor.replace("mean", "")
                    else:
                        base_factor = valid_factor
                    valid_factors = [f for f in self.data["valid"][-1]["mean"] if base_factor in f]
                    valid_factors.append(valid_factor)
                    valid_y_values = {f: [] for f in valid_factors}
                    valid_x_values = {f: [] for f in valid_factors}
                else:
                    valid_y_values = []
                    valid_x_values = []
        if not include_validation and not include_training:
            print("No data available for requested factor {}".format(factor))
            raise UnexpectedFactor

        if x_axis_scale == "epoch":
            for i, epoch in enumerate(self.data["train"]):
                if include_validation:
                    if per_class:
                        for f in valid_factors:
                            valid_epoch_data = [data["mean"][f] if f in data["mean"] else 0 for data in self.data["valid"] if data["epoch"] == i]
                            if len(valid_epoch_data) > 0:
                                valid_y_values[f].append(np.average(valid_epoch_data))
                                valid_x_values[f].append(i)
                    else:
                        valid_epoch_data = [data["mean"][valid_factor] for data in self.data["valid"] if data["epoch"] == i]  # last or average?
                        if len(valid_epoch_data) > 0:
                            valid_y_values.append(np.average(valid_epoch_data))
                            valid_x_values.append(i)
                if include_training:
                    for t_f in train_factors:
                        if t_f in epoch["data"]:
                            y_values[t_f].append(np.mean(epoch["data"][t_f]))
                            x_values[t_f].append(i)

        elif x_axis_scale == "iteration":
            if include_training:
                data = {t_f: [] for t_f in train_factors}
                for d in self.data["train"]:
                    for t_f in train_factors:
                        data[t_f].extend(d["data"][t_f])

                for t_f in train_factors:
                    data[t_f] = np.asarray(data[t_f])  # TODO: x values are slightly off due to removed nans
                    data[t_f] = data[t_f][~np.isnan(data[t_f])]
                    if iteration_window != 0:
                        y_values[t_f] = np.convolve(data[t_f], np.ones((iteration_window,))/iteration_window, mode="valid")
                    else:
                        y_values[t_f] = data[t_f]
                    x_values[t_f] = range(len(y_values[t_f]))

                if reject_outliers:
                    for t_f in train_factors:
                        d = np.abs(y_values[t_f] - np.median(y_values[t_f]))
                        mdev = np.median(d)
                        s = d/mdev if mdev else 0
                        outliers = np.where(s >= 10)[0]
                        y_values[t_f] = y_values[t_f][s < 10]
                        x_values[t_f] = [i for i in x_values[t_f] if i not in outliers]
            if include_validation:
                if per_class:
                    for f in valid_factors:
                        for d in self.data["valid"]:
                            if f in d["mean"]:
                                valid_y_values[f].append(d["mean"][f])
                                valid_x_values[f].append(d["iteration"])
                else:
                    for d in self.data["valid"]:
                        if valid_factor in d["mean"]:
                            valid_y_values.append(d["mean"][valid_factor])
                            valid_x_values.append(d["iteration"])

        if include_training:
            for t_f in sorted(train_factors):
                plt.plot(x_values[t_f], y_values[t_f], label=t_f)
        if include_validation:
            if per_class:
                for f in sorted(valid_factors):
                    plt.plot(valid_x_values[f], valid_y_values[f], label=f, marker="o")
            else:
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
