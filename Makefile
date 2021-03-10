all : gray

gray.o : src/gray.cpp src/stb_image_write.h
	g++ -Wall -c -O3 src/gray.cpp -o gray.o

stb_image_write.o : src/stb_image_write.cpp src/stb_image_write.h
	g++ -c -O3 src/stb_image_write.cpp -o stb_image_write.o

gray : gray.o stb_image_write.o
	g++ gray.o stb_image_write.o -o gray
