#---------------------------------------------------------------
#	Makefile that performs a compilation of the Neural Network
#	executable which is called fnn.x86_64.
#	Given NN works with svhn dataset, so be careful and make sure
#	to download it with download_data.sh before execution.
#	Last changes 17 mar 2020 by Skyme Factor.
#---------------------------------------------------------------
CC=g++
CFLAGS=-c -Wall -std=c++17
LIBFLAGS=-shared -O -fpic -fopenmp
SOURCES=main.cpp
OBJECTS=$(SOURCES:.cpp=.o)
LIBSOURCES=Matrix.cpp
LIB=libMatrix.so
EXECUTABLE=fnn.x86_64

#---------------------------------------------------------------
#	Compilation of fnn.x86_64 
#---------------------------------------------------------------
all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS)
	$(CC) -L$(PWD) -Wl,-rpath=$(PWD) $(OBJECTS) -o $@ -lMatrix

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