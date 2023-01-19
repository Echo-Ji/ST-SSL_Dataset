# Datasets used in ST-SSL

We provide several datasets used in the [ST-SSL](https://github.com/Echo-Ji/ST-SSL) framework, which leverages self-supervised learning for traffic flow prediction. 

The datasets range from `{NYCBike1, NYCBike2, NYCTaxi, BJTaxi}`.

Please use Git Large File Storage ([LFS](https://git-lfs.github.com/)) to pull this repo to your computer.

You can also download the dataset at [Beihang Cloud Drive](https://bhpan.buaa.edu.cn:443/link/8FD8DF90A642DB30FA98538EFEDF61D4).

## Dataset Format

A dataset is composed of 4 files, namely `train.npz`, `val.npz`, `test.npz`, and `adj_mx.npz`.

```
|----BJTaxi\
|    |----train.npz
|    |----adj_mx.npz
|    |----test.npz
|    |----val.npz
```

Train/Val/Test data is composed of 4 `numpy.ndarray` objects:

* `x`: a 4D tensor of shape (#timeslots, #lookback_window, #nodes, #flow_types)
* `y`: a 4D tensor of shape (#timeslots, #predict_horizon, #nodes, #flow_types). `x` and `y` are processed as a `sliding window view`.

* `x_offset`: a tensor indicating offsets of `x`'s lookback window. Note that the lookback window of data `x` is not consistent in time.
* `y_offset`: a tensor indicating offsets of `y`'s predict horizon.

For all datasets, previous 2-hour flows as well as previous 3-day flows around the predicted time are used to predict the flows for the next time step.

`adj_mx.npz` is a symmetric adjacency matrix, taking the value of 0 or 1.

## Dataset Usage

You can use the following code to view the data:

```python
import numpy as np

data = np.load('./BJTaxi/train.npz')
for file in data.files:
    print(file, data[file].shape)
```

## Raw Data

All datasets are processed by us as a `sliding window view`. Raw data of **NYCBike1** and **BJTaxi** are collected from [STResNet](https://ojs.aaai.org/index.php/AAAI/article/view/10735). Raw data of **NYCBike2** and **NYCTaxi** are collected from [STDN](https://ojs.aaai.org/index.php/AAAI/article/view/4511).



