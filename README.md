# Pixels to Drive

Learning to drive from pixel inputs.
This is the repository for a software project at the Freie Universtität.

# Dependencies

See the `requirements.txt` file. The packages can be install with `$ pip install -r
requirements.txt`.

# Notebooks

Most code is written in Jupyter Notebooks. Here is a short overview. A more
detailed description can be found inside the notebooks.

1. [rosbag_to_hdf5.ipynb](notebooks/rosbag_to_hdf5.ipynb): Converts the rosbag
   files to a train and test hdf5 file
1. [train.ipynb](notebooks/train.ipynb): Train the neural network.
1. [driver.ipynb](notebooks/driver.ipynb): Connects to the car, loads, and runs the
   neural network live.

# Code

* [`data.py`](deep_car/data.py): Contains augmentation functions and other
  helper functions
* [`model.py`](deep_car/model.py): The neural network architecture is defined
  here.

# Train data collection

For collecting the data we build a little test race circuit. For the first
generation of data we were simply driving the trip manually controlling
the car with the android app. The driving time was about 1 hour. We sticked
to driving in the same direction. In the end we had 3 big files with trainig
data.
For the second generation we placed various obstacles on the driving road.
We used orange soccer balls and some big chess statues. This time we were also
driving in both directions. The driving time was about 1 1/2 hour. Again the
driving was done manually with the android app. This time we split our trainig
data into many smaller data sets.

Finally we combined the data to create our total test set.

We used the following command for recording the data:

```
   rosbag record  \
    	/manual_control/speed \
    	/manual_control/steering \
    	/model_car/yaw \
        /deepcar/resize_img80x60/compressed
```
