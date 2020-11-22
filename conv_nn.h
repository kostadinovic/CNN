//
// Created by Nemanja Kostadinovic on 14/11/2020.
//

#ifndef APPRENTISSAGECNN_CONV_NN_H
#define APPRENTISSAGECNN_CONV_NN_H


#include "matrice.h"
#include "load_mnist.h"
#include <stdbool.h>

#define MAX_POOLING 0
#define MIN_POOLING 1
#define AVG_POOLING 2

typedef struct ConvolutionalLayer{
    int input_width; //largeur de l'image en entrée
    int input_height; //longueur de l'image en entrée
    int filter_size; //la taille du filtre (n*n)

    int input_channels; //nombre de channels (RGB=3) en entrée
    int output_channels; //nombre de channels (RGB=3) en sortie

    float ****filter_data; //poids du filtre, tableau de dimension 4 (input_channels*output_channels*filter_size*filter_size)
    float ****dfilter_data; //dérivé, poids mise à jour par le gradiant

    float *bias; //le biais sa taille est celle de output_channels
    bool is_fully_connected; //si la couche est entièrement connectée
    bool *connect_model;

    float ***v; //valeur en entrée de la fonction d'activation (taille de la dimension de sortie)
    float ***y; //valeur en sortie du neurone après la fonction d'activation (taille de la dimension de sortie)

    float ***d; //le gradiant local du pixel de sortie
}ConvLayer;

typedef struct PoolingLayer{
    int input_width; //largeur de l'image en entrée
    int input_height; //longueur de l'image en entrée
    int filter_size; //la taille du filtre (n*n)

    int input_channels; //nombre de channels (RGB=3) en entrée
    int output_channels; //nombre de channels (RGB=3) en sortie

    int pooling_type; //méthode de pooling (max,min,avg)
    float *bias; //biais

    float ***y; //valeur en sortie du neurone après la fonction d'activation (taille de la dimension de sortie)
    float ***d; //le gradiant local du pixel de sortie
}PoolLayer;

typedef struct FullyConnectedLayer{
    int input_neuron; //nombre de données en entrée (input data/unité de neurones)
    int output_neuron; //nombre de données en sortie (output data/unité de neurones)

    float **weight; //les poids pour chaque neurone
    float *bias; //le biais de chaque neurone (taille input_neuron)

    float *v; //valeur en entrée de la fonction d'activation (taille de la dimension de sortie)
    float *y; //valeur en sortie du neurone après la fonction d'activation (taille de la dimension de sortie)
    float *d; ////le gradiant local du pixel de sortie

    bool is_full_connect; //couche entièrement connecté ou non
}FCLayer;

//l'architecture CNN (LeNet-5) de Yann LeCun
typedef struct ConvolutionalNeuralNetwork{
    int number_of_layer; //nombre de couches
    ConvLayer *C1;
    PoolLayer *P2;
    ConvLayer *C3;
    PoolLayer *P4;
    FCLayer *FC5;

    float *error; //erreur lors du train
    float *loss; //erreur actuel LOSS
}CNN;

//train option
typedef struct training_hyperparam {
    int nbEpochs;
    float alpha; //valeur d'apprentissage
}training_hyperparam;

void network_weight(float *input, float *output, float **weight, float *bias, MatriceSize network_size);
int maxIndex(float *m, int m_size);
void trainCNN(CNN *network,	ArrayOfImage input_data, ArrayOfLabel output_data,training_hyperparam hyper_params, int train_number);
void UpdateWeightNetwork(CNN *network, training_hyperparam hyper_params, float **input);
void back_propagation(CNN* network,float* output);
void forward_propagation(CNN *network, float **input);
void network_weight(float *output, float *input, float **weight, float *bias, MatriceSize network_size);
float mult_vec(float *vec1, float *vec2, int size);
void AvgPool(float **output, MatriceSize output_size, float **input, MatriceSize input_size, int filter_size);
float dSigm(float y);
float ActSigm(float input, float bias);
void CreateCnn(CNN *cnn, MatriceSize input_size, int output_size);
FCLayer *CreateFullyConnectedLayer(int input_neuron, int output_neuron);
PoolLayer *CreatePoolingLayer(int input_width, int input_height, int filter_size, int input_channels, int output_channels, int pooling_type);
ConvLayer *CreateConvLayer(int input_width, int input_height, int filter_size, int input_channels, int output_channels);


float testCNN(CNN* cnn, ArrayOfImage input_data, ArrayOfLabel output_data,int test_num);
float testCNN2(CNN *network, ArrayOfImage input_data, ArrayOfLabel output_data, int test_num, int *tab);
void importCNN(CNN *network, const char* path);
void saveCNN(CNN* cnn, const char* filename);

void printf_cnn(CNN *network);
void clearCNN(CNN* network);

#endif //APPRENTISSAGECNN_CONV_NN_H
