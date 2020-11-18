cmake_minimum_required(VERSION 3.17)
project(apprentissageCNN C)

set(CMAKE_C_STANDARD 99)

add_executable(apprentissageCNN main.c matrice.c matrice.h load_mnist.c load_mnist.h conv_nn.c conv_nn.h)