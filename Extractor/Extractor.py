'''
The following code implements a simple extractor
from .mat files in order to make them understandable
by DataLoader class. It then writes all extracted data
into .dat files.

Result file structure:
    RGB 32x32 Image byte array, 1 class byte,
    RGB 32x32 Image byte array, 1 class byte,
    ...
    RGB 32x32 Image byte array, 1 class byte

Last changes: 08 may 2020 by Skyme Factor
'''

import sys, time

import scipy.io as io
import numpy as np

# Loader function
def load_data_mat(filename):
    raw = io.loadmat(filename)
    X = raw['X']  # Array of [32, 32, 3, n_samples]
    y = raw['y']  # Array of [n_samples, 1]
    X = np.moveaxis(X, [3], [0])
    y = y.flatten()
    # Fix up class 0 to be 0
    y[y == 10] = 0

    return X, y

# Writer function
def write_datasets(filename, images, labels):
    f = open(filename, 'wb')
    for i in range(images.shape[0]):
        f.write(images[i].flatten().tobytes())
        f.write(labels[i].tobytes())
    f.close()


if __name__ == "__main__":
    # Printing the system info
    print("Python version:")
    print(sys.version)
    print()
    print(time.strftime("%Y-%M-%d %H:%M:%S") + " Extraction started")

    # Unpack the train_32x32.mat file
    print("Unpacking train samples")
    try:
        # Loading
        train_X, train_y = load_data_mat("train_32x32.mat")
        print(time.strftime("%Y-%M-%d %H:%M:%S") + " Train samples loaded")
        
        # Writing to file
        write_datasets("train_32x32.dat", train_X, train_y)
        print(time.strftime("%Y-%M-%d %H:%M:%S") + " Train samples were successfully written to train_32x32.dat")
    except Exception as error:
        print(error)
    
    # Unpack the train_32x32.mat file
    print("Unpacking test samples")
    try:
        # Loading
        test_X, test_y = load_data_mat("test_32x32.mat")
        print(time.strftime("%Y-%M-%d %H:%M:%S") + " Test samples loaded")
        
        # Writing to file
        write_datasets("test_32x32.dat", test_X, test_y)
        print(time.strftime("%Y-%M-%d %H:%M:%S") + " Test samples were successfully written to test_32x32.dat")
    except Exception as error:
        print(error)

    # Report about success
    print(time.strftime("%Y-%M-%d %H:%M:%S") + " Extraction complete successfully")