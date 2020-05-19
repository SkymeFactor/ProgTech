## Fully-Connected NN
#### ITMO Project
The project contains a source code for a library that implements a Fully-Connected NN and which works with svhn dataset.
This NN is strongly depended on my hand-written static library (which is also presented as a source code here)
that implements some parallel operations over matrixes in a NumPy-like style and consists of Matrix.hpp and Matrix.cpp files.
### Compilation and running
#### System requirements:
    multicore x86_64 compatible processor
    RAM 2 Gb >= (16Gb)
    bash 5.0 ~= (5.0)
    gcc 7.0 >= (8.3)
    make 3.0 >= (4.2)
    OpenMP 4.0 >= (4.5)
    python 2.7 >= (3.8)
    scipy 1.3 >= (1.4)
#### Library compilation
**Important**: In order to use the pre-compiled library which is included in the repository you should have OpenMP installed 
as far as it works in parallel. Otherwise, you can remove -fopenmp option from the makefile and run `make lib` within the 
containing folder in order to compile your own version of libMatrix.so that will NOT be using the parallelism feature.
#### NN compilation
Staying within the current directory, write `make` command.
#### Running
Make sure you've downloaded the dataset and stored it in the same folder as the project.
To do so, run `chmod u+x download_data.sh && ./download_data.sh` command.
Then, simply execute the program you've got after the compilation process.
By default: `./fnn.x86_64`

#### Released:
2020 May 19 by SkymeFactor
