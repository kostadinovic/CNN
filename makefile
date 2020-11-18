all: main

main: main.o conv_nn.o matrice.o load_mnist.o
		gcc -o main main.o conv_nn.o matrice.o load_mnist.o 

main.o: main.c conv_nn.h
		gcc -c main.c -Wall -O2 -std=c99

conv_nn.o: conv_nn.c conv_nn.h
		gcc -c conv_nn.c -Wall -O2 -std=c99

matrice.o: matrice.c matrice.h
		gcc -c matrice.c -Wall -O2 -std=c99 -lm

load_mnist.o: load_mnist.c load_mnist.h
		gcc -c load_mnist.c -Wall -O2 -std=c99

clean:
	rm -f main *.o
	clear
