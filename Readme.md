## Fully-Connected NN
#### ITMO Project
The project contains a source code for a Fully-Connected NN that works with svhn dataset. (Well, it will be. Soon)
This NN is strongly depended on my hand-written library (which is also presented as a source code here)
that implements some operations over matrixes in a NumPy-like style and consists of Matrix.hpp and Matrix.cpp files.
### Compilation and running
#### System requirements:
    processor x86_64 compatible
    bash 5.0 ~=
    gcc 7.0 >=
    make 3.0 >=
#### Library compilation
**Important**: In order to compile the library you should have OpenMP installed as far as it works in parallel.
After making sure you have it, run 'make lib' command within the containing folder.
Otherwise, you can use pre-compiled version of this library which is also included in the repository.
#### NN compilation
**Not fully supported yet. Wait until release.**
Staying within the current directory, write 'make' command.
#### Running
Make sure you've downloaded the dataset and stored it in the same folder as the project.
To do so, run 'chmod u+x download_data.sh && ./download_data.sh' command.
Then, simply execute the program you've got after the compilation process.
By default: './fnn.x86_64'