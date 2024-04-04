all: libchunking.so

libchunking.so: chunking.o
	g++ chunking.o -shared -o libchunking.so

chunking.o: chunking.cc
	g++ -c -fPIC chunking.cc -o chunking.o  -fopenmp -g

clean:
	rm *.o *.so
