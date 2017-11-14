from git import Repo
from os import path

root = path.expanduser('~/Projects/ntnu-project/ml/pytorch-fcn')
log_dir = "/pytorch-fcn/examples/radar/logs/MODEL-fcn32s_CFG-001_INTERVAL_VALIDATE-2_LR-4e-12_WEIGHT_DECAY-0.0005_MOMENTUM-0.99_MAX_ITERATION-100000_VCS-b'9789623'_TIME-20171113-150930"
repo = Repo("../../")
#log_dir = path.join(root, log_dir)
repo.index.add(["torchfcn/datasets/test.py"], force=True)
repo.index.commit("Test")
origin = repo.remote("origin")
origin.push()
print(repo.untracked_files)