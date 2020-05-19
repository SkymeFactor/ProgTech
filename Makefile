#---------------------------------------------------------------
#	Makefile that performs a compilation of the Neural Network
#	executable which is called fnn.x86_64.
#	Given NN works with svhn dataset, so be careful and make sure
#	to download it with download_data.sh before execution.
#
#	Last changes 13 may 2020 by Skyme Factor.
#---------------------------------------------------------------
CC=g++
CFLAGS=-c -Wall -std=c++17 -fopenmp
LIBFLAGS=-shared -O3 -fpic -fopenmp
SOURCES=main.cpp ReLULayer.cpp FCLayer.cpp Model.cpp DataLoader.cpp SGDOptim.cpp SoftmaxLayer.cpp Trainer.cpp
OBJECTS=$(SOURCES:.cpp=.o)
LIBSOURCES=./MatrixLib/Matrix.cpp
LIB=libMatrix.so
EXECUTABLE=fnn.x86_64

#---------------------------------------------------------------
#	Compilation of fnn.x86_64 
#---------------------------------------------------------------
all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) -L$(PWD) -Wl,-rpath=$(PWD) $(OBJECTS) -o $@ -lMatrix -fopenmp

.cpp.o:
	$(CC) $(CFLAGS) $< -o $@

#---------------------------------------------------------------
#	Cleaning the last compilation result
#---------------------------------------------------------------
clean:
	rm -rf $(OBJECTS) $(EXECUTABLE)

#---------------------------------------------------------------
#	Compilation of libMatrix.so
#---------------------------------------------------------------
lib: $(LIB)

$(LIB): $(LIBSOURCES)
	$(CC) -std=c++17 $(LIBFLAGS) -o $@ $<

clean_lib:
	rm -rf $(LIB)