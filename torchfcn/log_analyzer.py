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
            epoch_iteration_start = 0
            elapsed_time_base = 0

            for i, row in enumerate(reader):
                if int(row["epoch"]) > len(self.data["train"]) - 1:
                    epoch_iteration_start = int(row["iteration"])
                    self.data["train"].append({"data": {}})
                if row["train/loss"] != "" and is_validation:
                    is_validation = False
                    self.data["valid"][-1]["mean"] = {key: val.pop() for key, val in
                                                      self.data["valid"][-1]["data"].items() if key != "filename"}
                    self.data["valid"][-1]["end_time"] = float(row["elapsed_time"]) + elapsed_time_base
                elif row["valid/loss"] != "" and not is_validation:
                    is_validation = True
                    self.data["valid"].append(
                        {"data": {}, "epoch": int(row["epoch"]), "iteration": int(row["iteration"]),
                         "start_time": self.data["train"][-1]["data"]["elapsed_time"][-1], "end_time": None})
                row_epoch_iteration = int(row["iteration"]) - epoch_iteration_start
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

                            if key == "elapsed_time":
                                try:
                                    if self.data[row_cat][-1]["data"][key][-1] - elapsed_time_base > val:
                                        elapsed_time_base = self.data[row_cat][-1]["data"][key][-1]
                                except:
                                    pass
                                val += elapsed_time_base

                            if key not in self.data[row_cat][-1]["data"]:
                                self.data[row_cat][-1]["data"].update({key: []})

                            try:
                                self.data[row_cat][-1]["data"][key][row_epoch_iteration] = val
                            except IndexError:
                                self.data[row_cat][-1]["data"][key].append(val)

            if is_validation:
                self.data["valid"][-1]["mean"] = {key: val.pop() for key, val in self.data["valid"][-1]["data"].items() if key != "filename"}
                self.data["valid"][-1]["end_time"] = float(row["elapsed_time"]) + elapsed_time_base

        self.total_time = self.data["train"][-1]["data"]["elapsed_time"][-1]
        self.total_iteration = sum(len(d["data"]["train/loss"]) for d in self.data["train"])

        for valid in self.data["valid"]:
            for f in ["bj", "iu"]:
                adjusted_mean = 0
                for c in ["land", "background", "unknown"]:
                    d = np.asarray(valid["data"]["valid/{}_{}".format(f, c)])
                    d = d[~np.isnan(d)]
                    adjusted_mean += np.sum(d) / d.size
                adjusted_mean /= 3
                valid["mean"].update({"valid/adj_mean_{}".format(f): adjusted_mean})

    def validation_metric_histogram(self, metric, validation_idx=-1, fig=None, return_fig=False, **kwargs):
        try:
            data = self.data["valid"][validation_idx]["data"]["valid/{}".format(metric)]
        except Exception as e:
            print("No data available for selected validation index and metric")
            raise e
        iteration = self.data["valid"][validation_idx]["iteration"]
        data = np.asarray(data)

        plt.hist(data[~np.isnan(data)], bins=100, **kwargs)
        plt.xlim(0, 1)
        if return_fig:
            return
        plt.title("Histogram for {} in validation on iteration {}".format(metric, iteration))
        plt.show()

    def share_over_threshold(self, threshold, metric="mean_bj", validation_idx=-1):
        try:
            data = self.data["valid"][validation_idx]["data"]["valid/{}".format(metric)]
        except Exception as e:
            print("No data available for selected validation index and metric")
            raise e

        share = sum(1 for i in data if i >= threshold) / len(data)

        return share


    def get_low_scoring_files(self, metric, percentile=1, mode="min", split="valid", split_idx=-1):
        data = np.asarray(self.data[split][split_idx]["data"]["{}/{}".format(split, metric)])

        N = int(data.size * percentile / 100)

        if mode == "min":
            res = np.argsort(data)[:N]
        elif mode == "max":
            res = np.argsort(data)[::-1][:N]

        return [self.data[split][split_idx]["data"]["filename"][i] for i in res]

    def get_best_valid_idx(self, factor="mean_bj", max_time=-1, max_iteration=-1):
        factor = "valid/{}".format(factor)
        best_idx, best_val = 0, 0
        for i in range(1, len(self.data["valid"])):
            if (max_time == -1 or self.data["valid"][i]["start_time"] <= max_time) and (max_iteration == -1 or self.data["valid"][i]["iteration"] <= max_iteration):
                if self.data["valid"][i]["mean"][factor] > best_val:
                    best_idx = i
                    best_val = self.data["valid"][i]["mean"][factor]

        return best_idx


    def graph_factor(self, factor, x_axis_scale="iteration", include_validation=False, per_class=False, iteration_window=0, reject_outliers=True, include_time=False, save_plot=False, data_range=(0, -1), fig=None, return_fig=False, labels=None, **plot_kwargs):
        train_factors = {"train/{}".format(factor)} #{f for d in self.data["train"] for f in d["data"] if "train/{}".format(factor) in f}
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

        else:
            if include_training:
                data = {t_f: [] for t_f in train_factors}
                data.update({"elapsed_time": []})
                for d in self.data["train"]:
                    data["elapsed_time"].extend(d["data"]["elapsed_time"])
                    for t_f in train_factors:
                        data[t_f].extend(d["data"][t_f])

                if x_axis_scale == "time":
                    start = data["elapsed_time"][0] if data_range[0] == 0 else None
                    stop = data["elapsed_time"][-1] if data_range[1] == -1 else None
                    idx_range = [0, -1]

                    if stop is None or start is None:
                        for i, v in enumerate(data["elapsed_time"]):
                            if start is None and v > data_range[0]:
                                start = v
                                idx_range[0] = i
                            if stop is None and v > data_range[1]:
                                stop = data["elapsed_time"][i-1]
                                idx_range[1] = i-1
                    if stop is None:
                        stop = data["elapsed_time"][-1]
                    data_range = [start, stop]

                for k, v in data.items():
                    if x_axis_scale == "iteration":
                        data[k] = v[data_range[0]:data_range[1]]
                    if x_axis_scale == "time":
                        data[k] = v[idx_range[0]:idx_range[1]]

                for t_f in train_factors:
                    data[t_f] = np.asarray(data[t_f])  # TODO: x values are slightly off due to removed nans
                    idxs = np.argwhere(~np.isnan(data[t_f]))
                    data[t_f] = data[t_f][~np.isnan(data[t_f])]
                    if iteration_window != 0:
                        y_values[t_f] = np.convolve(data[t_f], np.ones((iteration_window,))/iteration_window, mode="valid")
                    else:
                        y_values[t_f] = data[t_f]

                    if x_axis_scale == "iteration":
                        x_values[t_f] = idxs[:-(iteration_window - 1)] + data_range[0]
                    elif x_axis_scale == "time":
                        x_values[t_f] = np.asarray(data["elapsed_time"])[idxs[:-(iteration_window - 1)] + idx_range[0]]

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
                                if x_axis_scale == "iteration":
                                    cur_x = d["iteration"]
                                elif x_axis_scale == "time":
                                    cur_x = d["start_time"]
                                if data_range[0] <= cur_x <= data_range[1] or data_range[1] == -1 and data_range[0] <= cur_x:
                                    valid_y_values[f].append(d["mean"][f])
                                    valid_x_values[f].append(cur_x)
                else:
                    for d in self.data["valid"]:
                        if valid_factor in d["mean"]:
                            if x_axis_scale == "iteration":
                                cur_x = d["iteration"]
                            elif x_axis_scale == "time":
                                cur_x = d["start_time"]
                            if data_range[0] <= cur_x <= data_range[1] or data_range[1] == -1 and data_range[0] <= cur_x:
                                valid_y_values.append(d["mean"][valid_factor])
                                valid_x_values.append(cur_x)

        if fig is None:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
        else:
            all_axes = fig.get_axes()
            if len(all_axes) == 2:
                ax1, ax2 = all_axes[0], all_axes[1]
            else:
                ax1 = all_axes[0]
        if include_training:
            for t_f in sorted(train_factors):
                ax1.plot(x_values[t_f], y_values[t_f], label=t_f if labels is None else labels[t_f], **plot_kwargs)
        if include_validation:
            if per_class:
                for f in sorted(valid_factors):
                    ax1.plot(valid_x_values[f], valid_y_values[f], label=f if labels is None else labels[f], marker="o")
            else:
                label = "Validation" if labels is None else labels["valid/{}".format(factor)]
                if label is None:
                    line_c = fig.axes[0].lines[-1].get_color()
                    ax1.plot(valid_x_values, valid_y_values, label=None, c=line_c, ls=":", marker="o", **plot_kwargs)
                else:
                    ax1.plot(valid_x_values, valid_y_values, label=label, marker="o", **plot_kwargs)

        plt.legend()
        plt.xlabel(x_axis_scale)

        if include_time:
            if fig is None or "ax2" not in locals():
                ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            ax1_ticks = ax1.get_xticks()
            ax1_ticks = [t for t in ax1_ticks if t >= 0 and t < len(data["elapsed_time"])]
            ax2.set_xticks(ax1_ticks)
            ax2.set_xticklabels(["{0:.1f}".format(data["elapsed_time"][int(t)] / 3600) for t in ax1_ticks])
            ax2.set_xlabel("Elapsed time (h)")

        if save_plot:
            plt.title("FCN8s")
            plt.savefig("figures/{}.pdf".format(factor), format="pdf")

        plt.title(factor, y=1.12 if include_time else 1)

        if return_fig:
            return fig
        else:
            plt.show()


if __name__ == "__main__":
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_INTERVAL_WEIGHT_UPDATE-10_INTERVAL_CHECKPOINT-3000_MOMENTUM-0.99_INTERVAL_VALIDATE-60000_WEIGHT_DECAY-0.0005_LR-7.000000000000001e-12_MAX_ITERATION-800000_VCS-b'c68e4f7'_TIME-20180222-133545/log.csv"
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_MOMENTUM-0.99_INTERVAL_VALIDATE-60000_LR-2.1000000000000002e-11_INTERVAL_CHECKPOINT-5000_MAX_ITERATION-800000_WEIGHT_DECAY-0.0005_VCS-b'eb42bcf'_TIME-20180216-181256/log.csv"
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_LR-7.000000000000001e-12_MAX_ITERATION-800000_MOMENTUM-0.99_INTERVAL_CHECKPOINT-3000_INTERVAL_WEIGHT_UPDATE-10_INTERVAL_VALIDATE-100_WEIGHT_DECAY-0.0005_VCS-b'a41ae7a'_TIME-20180222-182224/log.csv"
    # FCN8s with metadata
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_INTERVAL_VALIDATE-30000_INTERVAL_WEIGHT_UPDATE-10_LR-3.5000000000000004e-12_INTERVAL_CHECKPOINT-500_WEIGHT_DECAY-0.0005_MAX_ITERATION-800000_MOMENTUM-0.99_VCS-b'0942aaa'_TIME-20180305-154053/log.csv"
    # FCN8s with metadata lower weights
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_INTERVAL_VALIDATE-30000_LR-3.5000000000000004e-12_WEIGHT_DECAY-0.0005_INTERVAL_WEIGHT_UPDATE-3_MOMENTUM-0.99_MAX_ITERATION-800000_INTERVAL_CHECKPOINT-500_VCS-b'c427d01'_TIME-20180312-142853/log.csv"
    # PSPNet
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_LR-3.5000000000000004e-12_INTERVAL_CHECKPOINT-500_MOMENTUM-0.99_INTERVAL_VALIDATE-30000_INTERVAL_WEIGHT_UPDATE-1_MAX_ITERATION-800000_WEIGHT_DECAY-0.0005_VCS-b'a3c61c5'_TIME-20180319-182455/log.csv"
    # PSPnet different weights and weight decay
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_CFG-001_MOMENTUM-0.99_MAX_ITERATION-800000_WEIGHT_DECAY-0.005_INTERVAL_VALIDATE-10000_INTERVAL_WEIGHT_UPDATE-1_INTERVAL_CHECKPOINT-500_LR-3.5000000000000004e-12_VCS-b'3823d61'_TIME-20180320-140754/log.csv"

    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180321-182203/log.csv"
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180321-180851/log.csv"

    # PSPnet pretrained
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180322-162514/log.csv"
    # PSPnet update every 16
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180323-140432/log.csv"
        # 5 min data interval
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180331-140140/log.csv"
        # islet removed
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180403-150627/log.csv"
        # islet removed and stronger weight decay
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180405-135950/log.csv"
        # with metadatas
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180408-164114/log.csv"
        # no aux
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180410-130318/log.csv"
        # aux and cfar filters
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180410-185353/log.csv"
        # aux and cfar filters only one and batch norm
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180412-153602/log.csv"
        # aux and cfar filters only one and batch norm no zero padding
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180412-215439/log.csv"
        # downsampling3
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_-downsample3_TIME-20180413-151538/log.csv"
        # pretrained3ch no freeze
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_-pretrained3ch_TIME-20180413-181424/log.csv"
        # pretrained3ch fixed first layer
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch-MODEL-PSPnet__TIME-20180414-163350/log.csv"
        # pretrained3ch frozens
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch_freeze-MODEL-PSPnet__TIME-20180415-171201/log.csv"
        # downsampling2
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/downsample2-MODEL-PSPnet__TIME-20180414-155205/log.csv"
        # pretrained1ch freeze not first last
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained1ch_freeze_not_first_last-MODEL-PSPnet__TIME-20180416-000745/log.csv"
        # pretrained3ch freeze not last
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch_freeze_not_last-MODEL-PSPnet__TIME-20180415-234848/log.csv"
        # cartesian
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/cartesian_MODEL-PSPnet_TIME-20180419-125237/log.csv"
        # downsample1.5
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/downsample15_MODEL-PSPnet_TIME-20180419-234207/log.csv"
        # small training set
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/smallset_MODEL-PSPnet_TIME-20180424-132709/log.csv"
        # range normalized
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/rangeNormalized_MODEL-PSPnet_TIME-20180425-161146/log.csv"
        # range normalized 2
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/rangeNormalize2_MODEL-PSPnet_TIME-20180429-182912/log.csv"
        # cfar newest
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/cfar_MODEL-PSPnet_TIME-20180427-202859/log.csv"
        # groupNorm
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNorm_MODEL-PSPnet_TIME-20180509-184028/log.csv"
        # groupNorm UI1
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNormUI1_MODEL-PSPnet_TIME-20180511-234007/log.csv"
        # groupNorm Fixed 16
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNormFixed16_MODEL-PSPnet_TIME-20180513-192856/log.csv"
        # final
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final_MODEL-PSPnet_TIME-20180515-140415/log.csv"
        # final 2 (lower vessel weight, lower l2, GN 32g)
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final2_MODEL-PSPnet_TIME-20180516-140732/log.csv"
        # final 3 (fixed GN initialization, lower l2, GN32g, middle vessel weight, pretrained)
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final3_MODEL-PSPnet_TIME-20180518-153626/log.csv"
        # final 4
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final4_MODEL-PSPnet_TIME-20180519-124813/log.csv"
        # final 5
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final5_MODEL-PSPnet_TIME-20180520-134557/log.csv"
        # final 6 (pretrained)
    log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final6_MODEL-PSPnet_TIME-20180521-133432/log.csv"
    # GCN
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-GCN_TIME-20180325-220204/log.csv"
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-GCN_TIME-20180325-225308/log.csv"
    # RefineNet
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-RefineNet_TIME-20180326-212928/log.csv"
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-RefineNet_TIME-20180327-122122/log.csv"
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-RefineNet_TIME-20180326-172451/log.csv"

    # test
    #log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-RefineNet_TIME-20180327-145136/log.csv"
    # FCN8s
    #log_path="/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn8s_TIME-20180401-143117/log.csv"

    #low_scoring = analyzer.get_low_scoring_files("loss", percentile=1, mode="max", split="train")
    #low_scoring = sorted(low_scoring, key=lambda x: x.split("/")[-1])
    #print(".bmp;\n".join(low_scoring))

    mode = "single"
    family = "groupnorm"
    comp = "Final"
    match_length = True

    if mode == "extension":
        kwargs = {"alpha": 0.9}
        if family == "pretrained":
            # pretrained3ch fixed first layer
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch-MODEL-PSPnet__TIME-20180414-163350/log.csv"
            analyzer = LogAnalyzer(log_path)
            # pretrained3ch frozens
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch_freeze-MODEL-PSPnet__TIME-20180415-171201/log.csv"
            analyzer2 = LogAnalyzer(log_path)
            # pretrained1ch freeze not first last
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained1ch_freeze_not_first_last-MODEL-PSPnet__TIME-20180416-000745/log.csv"
            analyzer3 = LogAnalyzer(log_path)
            # pretrained3ch freeze not last
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch_freeze_not_last-MODEL-PSPnet__TIME-20180415-234848/log.csv"
            analyzer4 = LogAnalyzer(log_path)
            legends = ["Freeze none", "Freeze all", "Freeze module 2-4", "Freeze module 1-4", "Baseline"]
        elif family == "downsample":
            # downsample1.5
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/downsample15_MODEL-PSPnet_TIME-20180419-234207/log.csv"
            analyzer = LogAnalyzer(log_path)
            # downsampling2
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/downsample2-MODEL-PSPnet__TIME-20180414-155205/log.csv"
            analyzer2 = LogAnalyzer(log_path)
            # downsample 3
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_-downsample3_TIME-20180413-151538/log.csv"
            analyzer3 = LogAnalyzer(log_path)
            legends = ["Downsample 1.5", "Downsample 2", "Downsample 3", "Baseline"]
        elif family == "rangenorm":
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/rangeNormalized_MODEL-PSPnet_TIME-20180425-161146/log.csv"
            # range normalized 2
            analyzer = LogAnalyzer(log_path)
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/rangeNormalize2_MODEL-PSPnet_TIME-20180429-182912/log.csv"
            analyzer2 = LogAnalyzer(log_path)
            legends = ["RangeNorm1", "RangeNorm2", "Baseline"]
        elif family == "baseline":
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-fcn8s_TIME-20180401-143117/log.csv"
            analyzer = LogAnalyzer(log_path)
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-GCN_TIME-20180325-220204/log.csv"
            analyzer2 = LogAnalyzer(log_path)
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180323-140432/log.csv"
            analyzer3 = LogAnalyzer(log_path)
            legends = ["FCN8s", "GCN", "PSPnet", "RefineNet"]
        elif family == "cfar":
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180410-185353/log.csv"
            analyzer = LogAnalyzer(log_path)
                # aux and cfar filters only one and batch norm
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180412-153602/log.csv"
            analyzer2 = LogAnalyzer(log_path)
                # aux and cfar filters only one and batch norm no zero padding
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180412-215439/log.csv"
            analyzer3 = LogAnalyzer(log_path)
                # cfar newest
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/cfar_MODEL-PSPnet_TIME-20180427-202859/log.csv"
            analyzer4 = LogAnalyzer(log_path)
            legends = ["3 Filters", "1 Filter", "1 Filter BatchNorm", "New", "Baseline"]
        elif family == "cartesian":
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/cartesian_MODEL-PSPnet_TIME-20180419-125237/log.csv"
            analyzer = LogAnalyzer(log_path)
            legends = ["Cartesian", "Baseline"]
        elif family == "groupnorm":
            # groupNorm
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNorm_MODEL-PSPnet_TIME-20180509-184028/log.csv"
            analyzer = LogAnalyzer(log_path)
            # groupNorm UI1
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNormUI1_MODEL-PSPnet_TIME-20180511-234007/log.csv"
            analyzer2 = LogAnalyzer(log_path)
            # groupNorm Fixed 16
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNormFixed16_MODEL-PSPnet_TIME-20180513-192856/log.csv"
            analyzer3 = LogAnalyzer(log_path)
            legends = ["GN 32g", "GN 32g GA1", "GN 16ch", "Baseline"]
        elif family == "final":
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final_MODEL-PSPnet_TIME-20180515-140415/log.csv"
            analyzer = LogAnalyzer(log_path)
            # final 6 (pretrained)
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final6_MODEL-PSPnet_TIME-20180521-133432/log.csv"
            analyzer2 = LogAnalyzer(log_path)
            # final 5
            log_path = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final5_MODEL-PSPnet_TIME-20180520-134557/log.csv"
            analyzer3 = LogAnalyzer(log_path)
            legends = ["GN 16ch, high L2, VW1000", "GN32g, pretrained, low L2, VW750", "GN32g, low L2, VW750", "Baseline"]

        log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180405-135950/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-RefineNet_TIME-20180327-122122/log.csv"
        baseline = LogAnalyzer(log_path_base)

        valid = True
        data_range = (0, analyzer.total_time + 20000 if match_length else -1)
        for f in ["mean_iu", "acc_cls_vessel", "mean_bj", "adj_mean_bj", "adj_mean_iu"]:
            if "bj" in f or "adj" in f:
                labels = {"train/{}".format(f): None, "valid/{}".format(f): legends[0]}
            else:
                labels = {"train/{}".format(f): legends[0], "valid/{}".format(f): None}
            fig = analyzer.graph_factor(f, x_axis_scale="time", include_validation=valid, iteration_window=1000,
                                        reject_outliers=False, data_range=data_range, labels=labels,
                                        return_fig=True, **kwargs)

            if "bj" in f or "adj" in f:
                labels = {"train/{}".format(f): None, "valid/{}".format(f): legends[1]}
            else:
                labels = {"train/{}".format(f): legends[1], "valid/{}".format(f): None}
            baseline_legend = 1

            if "analyzer2" in locals():
                fig = analyzer2.graph_factor(f, x_axis_scale="time", include_validation=valid, iteration_window=1000,
                                        reject_outliers=False, data_range=data_range, fig=fig, return_fig=True, labels=labels, **kwargs)
                baseline_legend += 1

            if "analyzer3" in locals():
                if "bj" in f or "adj" in f:
                    labels = {"train/{}".format(f): None, "valid/{}".format(f): legends[2]}
                else:
                    labels = {"train/{}".format(f): legends[2], "valid/{}".format(f): None}
                fig = analyzer3.graph_factor(f, x_axis_scale="time", include_validation=valid, iteration_window=1000,
                                            reject_outliers=False, data_range=data_range, fig=fig, return_fig=True, labels=labels, **kwargs)
                baseline_legend += 1
            if "analyzer4" in locals():
                if "bj" in f or "adj" in f:
                    labels = {"train/{}".format(f): None, "valid/{}".format(f): legends[3]}
                else:
                    labels = {"train/{}".format(f): legends[3], "valid/{}".format(f): None}
                fig = analyzer4.graph_factor(f, x_axis_scale="time", include_validation=valid, iteration_window=1000,
                                            reject_outliers=False, data_range=data_range, fig=fig, return_fig=True, labels=labels, **kwargs)
                baseline_legend += 1
            if "bj" in f or "adj" in f:
                labels = {"train/{}".format(f): None, "valid/{}".format(f): legends[baseline_legend]}
            else:
                labels = {"train/{}".format(f): legends[baseline_legend], "valid/{}".format(f): None}
            fig = baseline.graph_factor(f, x_axis_scale="time", include_validation=valid, iteration_window=1000,
                                        reject_outliers=False, data_range=data_range, fig=fig, return_fig=True, labels=labels, **kwargs)

            fig.show()
            fig.savefig("figures/{}_{}.pdf".format(family, f), format="pdf")
    elif mode == "baseline":
        analyzer = LogAnalyzer(log_path)


        kwargs = {"alpha": 0.7}
        factors = ["mean_iu", "acc_cls_vessel", "mean_bj", "adj_mean_bj"]
        per_class = True

            # real baseline
        log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180405-135950/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final_MODEL-PSPnet_TIME-20180515-140415/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/final5_MODEL-PSPnet_TIME-20180520-134557/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/groupNorm_MODEL-PSPnet_TIME-20180509-184028/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_-downsample3_TIME-20180413-151538/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/rangeNormalized_MODEL-PSPnet_TIME-20180425-161146/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180410-185353/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/downsample2-MODEL-PSPnet__TIME-20180414-155205/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180410-185353/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/pretrained3ch_freeze_not_last-MODEL-PSPnet__TIME-20180415-234848/log.csv"
        #log_path_base = "/home/eivind/Documents/dev/ntnu-project/ml/pytorch-fcn/examples/radar/logs/MODEL-PSPnet_TIME-20180412-215439/log.csv"

        if per_class:
            labels = {"valid/iu_vessel": {"c": 'r', "marker": "o", "ls": "-"},
                      "valid/iu_land": {"c": 'g', "marker": "o", "ls": "-"},
                      "valid/iu_unknown": {"c": 'm', "marker": "o", "ls": "-"},
                      "valid/iu_background": {"c": 'k', "marker": "o", "ls": "-"},
                      "valid/mean_iu": {"c": 'k', "marker": "o", "ls": "-"},
                    }
        else:
            labels = {""}

        baseline = LogAnalyzer(log_path_base)
        for factor in factors:
            if "bj" in factor or "adj" in factor:
                labels = {"train/{}".format(factor): None, "valid/{}".format(factor): comp}
            else:
                labels = {"train/{}".format(factor): comp, "valid/{}".format(factor): None}
            fig = analyzer.graph_factor(factor, x_axis_scale="time",include_validation=True, per_class=False, iteration_window=1000, reject_outliers=False,
                                    return_fig=True, labels=labels, **kwargs)

            if "bj" in factor or "adj" in factor:
                labels = {"train/{}".format(factor): None, "valid/{}".format(factor): "Baseline"}
            else:
                labels = {"train/{}".format(factor): "Baseline", "valid/{}".format(factor): None}
            fig = baseline.graph_factor(factor, x_axis_scale="time", include_validation=True, per_class=False, iteration_window=1000, reject_outliers=False, data_range=(0, analyzer.total_time if match_length else -1), fig=fig, return_fig=True, labels=labels, save_plot=True, **kwargs)
            fig.show()

        for factor in ["mean_bj", "mean_iu"]:
            analyzer.validation_metric_histogram(factor, validation_idx=analyzer.get_best_valid_idx("adj_" + factor), return_fig=True, alpha=0.75, label=comp)
            baseline.validation_metric_histogram(factor, validation_idx=baseline.get_best_valid_idx("adj_" + factor, max_time=analyzer.total_time), alpha=0.75, return_fig=True, label="Baseline")

            plt.legend()
            plt.title("Best validation performance on {}".format(factor))
            plt.savefig("figures/{}_valid_hist.pdf".format(factor), format="pdf")
            plt.show()

        print("New over 0.5 Mean BJ: {}".format(analyzer.share_over_threshold(0.5, validation_idx=analyzer.get_best_valid_idx("adj_mean_bj"))))
        print("Baseline over 0.5 Mean BJ: {}".format(baseline.share_over_threshold(0.5, validation_idx=baseline.get_best_valid_idx("adj_mean_bj", max_time=analyzer.total_time))))

    elif mode == "single":
        analyzer = LogAnalyzer(log_path)
        analyzer.graph_factor("loss", x_axis_scale="iteration", include_validation=False, iteration_window=1000, reject_outliers=False, include_time=False, save_plot=True, data_range=(0, -1))
        analyzer.graph_factor("mean_iu", x_axis_scale="iteration", per_class=True, include_validation=True, iteration_window=1000,reject_outliers=False, include_time=True, save_plot=False, data_range=(0, -1))
        analyzer.graph_factor("acc_cls_vessel", x_axis_scale="time", per_class=True, include_validation=True, iteration_window=1000, reject_outliers=False, save_plot=False)
        analyzer.graph_factor("mean_bj", x_axis_scale="iteration", per_class=True, include_validation=True, iteration_window=1000,reject_outliers=False, save_plot=False)
        analyzer.graph_factor("adj_mean_bj", x_axis_scale="iteration", per_class=False, include_validation=True,
                              iteration_window=1000, reject_outliers=False, save_plot=False)
        analyzer.validation_metric_histogram("acc_cls_vessel", validation_idx=-1)
        analyzer.validation_metric_histogram("mean_iu", validation_idx=-1)
        analyzer.validation_metric_histogram("mean_bj", validation_idx=-1)
    elif mode == "best valid":
        analyzer = LogAnalyzer(log_path)
        print("max time {}".format(analyzer.total_time))
        for k, v in sorted(analyzer.data["valid"][analyzer.get_best_valid_idx(factor="adj_mean_bj", max_time=175000)]["mean"].items()):
            print("{}:\t\t\t{:.4f}".format(k, v))
        print("Over 0.5: \t\t\t{:.4f}".format(analyzer.share_over_threshold(0.5, validation_idx=analyzer.get_best_valid_idx("adj_mean_bj"))))
    elif mode == "low scoring":
        analyzer = LogAnalyzer(log_path)
        factor = "acc_cls_vessel"
        low_scoring = analyzer.get_low_scoring_files(factor, percentile=1, split="train")
        low_scoring = sorted(low_scoring, key=lambda x: x.split("/")[-1])
        with open("low_scoring_files_{}.txt".format(factor), "a+") as file:
            for ls in low_scoring:
                file.write("{}.bmp\n".format(ls))
        print(".bmp;\n".join(low_scoring))
