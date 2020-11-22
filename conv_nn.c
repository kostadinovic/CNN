//
// Created by Nemanja Kostadinovic on 14/11/2020.
//

#include "conv_nn.h"
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "matrice.h"

//Allocate and initialize all the parameters of the Convolutional layer
ConvLayer *CreateConvLayer(int input_width, int input_height, int filter_size, int input_channels, int output_channels){

    ConvLayer* conv_layer=(ConvLayer*)malloc(sizeof(ConvLayer));
    conv_layer->input_width=input_width;
    conv_layer->input_height=input_height;
    conv_layer->filter_size=filter_size;
    conv_layer->input_channels=input_channels;
    conv_layer->output_channels=output_channels;
    conv_layer->is_fully_connected=true;

    srand((unsigned)time(NULL));
    conv_layer->filter_data=(float****)malloc(input_channels*sizeof(float***));
    for(int i=0;i<input_channels;i++){
        conv_layer->filter_data[i]=(float***)malloc(output_channels*sizeof(float**));
        for(int j=0;j<output_channels;j++){
            conv_layer->filter_data[i][j]=(float**)malloc(filter_size*sizeof(float*));
            for(int r=0;r<filter_size;r++){
                conv_layer->filter_data[i][j][r]=(float*)malloc(filter_size*sizeof(float));
                for(int c=0;c<filter_size;c++){
                    float rand_var=(((float)rand()/(float)RAND_MAX)-0.5)*2;
                    conv_layer->filter_data[i][j][r][c] = rand_var * (sqrt((float)6.0 / (float)(filter_size * filter_size * (input_channels + output_channels))));
                }
            }
        }
    }

    conv_layer->dfilter_data=(float****)malloc(input_channels*sizeof(float***));
    for(int i=0;i<input_channels;i++){
        conv_layer->dfilter_data[i]=(float***)malloc(output_channels*sizeof(float**));
        for(int j=0;j<output_channels;j++){
            conv_layer->dfilter_data[i][j]=(float**)malloc(filter_size*sizeof(float*));
            for(int r=0;r<filter_size;r++){
                conv_layer->dfilter_data[i][j][r]=(float*)calloc(filter_size,sizeof(float));
            }
        }
    }

    conv_layer->bias=(float*)calloc(output_channels,sizeof(float));

    int out_sizeW=input_width-filter_size+1;
    int out_sizeH=input_height-filter_size+1;

    conv_layer->d=(float***)malloc(output_channels*sizeof(float**));
    conv_layer->v=(float***)malloc(output_channels*sizeof(float**));
    conv_layer->y=(float***)malloc(output_channels*sizeof(float**));

    for(int i=0; i<output_channels; i++){

        conv_layer->d[i]=(float**)malloc(out_sizeH*sizeof(float*));
        //printf("DEBUG CCL boucle 1 :     i=%d\n",i);
        conv_layer->v[i]=(float**)malloc(out_sizeH*sizeof(float*));
        conv_layer->y[i]=(float**)malloc(out_sizeH*sizeof(float*));

        for(int j=0; j<out_sizeH; j++){
            //printf("DEBUG CCL boucle 2 :     i=%d et j=%d\n",i,j);
            conv_layer->d[i][j]=(float*)calloc(out_sizeW,sizeof(float));
            conv_layer->v[i][j]=(float*)calloc(out_sizeW,sizeof(float));
            conv_layer->y[i][j]=(float*)calloc(out_sizeW,sizeof(float));
        }
    }
    return conv_layer;
}

//Allocate and initialize all the parameters of the Pooling layer
PoolLayer *CreatePoolingLayer(int input_width, int input_height, int filter_size, int input_channels, int output_channels, int pooling_type){
    PoolLayer *pool_layer = (PoolLayer*)malloc(sizeof(PoolLayer));
    pool_layer->input_width = input_width;
    pool_layer->input_height = input_height;
    pool_layer->filter_size = filter_size;
    pool_layer->input_channels = input_channels;
    pool_layer->output_channels = output_channels;
    pool_layer->pooling_type = pooling_type;

    pool_layer->bias = (float*)calloc(output_channels,sizeof(float));
    pool_layer->d = (float***)malloc(output_channels*sizeof(float**));
    pool_layer->y = (float***)malloc(output_channels*sizeof(float**));

    int out_sizeW = input_width/filter_size;
    int out_sizeH = input_height/filter_size;

    for(int i=0; i<output_channels; i++){
        pool_layer->d[i] = (float**)malloc(out_sizeH*sizeof(float*));
        pool_layer->y[i] = (float**)malloc(out_sizeH*sizeof(float*));
        for(int j=0; j<out_sizeH; j++){
            pool_layer->d[i][j] = (float*)calloc(out_sizeW,sizeof(float));
            pool_layer->y[i][j] = (float*)calloc(out_sizeW,sizeof(float));
        }
    }
    return pool_layer;
}

//Allocate and initialize all the parameters of the Output (fully connected) layer
FCLayer *CreateFullyConnectedLayer(int input_neuron, int output_neuron){
    FCLayer *fc_layer = (FCLayer*)malloc(sizeof(FCLayer));

    fc_layer->input_neuron = input_neuron;
    fc_layer->output_neuron = output_neuron;
    fc_layer->weight = (float**)malloc(output_neuron*sizeof(float*));
    fc_layer->bias =(float*)calloc(output_neuron,sizeof(float));

    fc_layer->v = (float*)calloc(output_neuron, sizeof(float));
    fc_layer->y = (float*)calloc(output_neuron, sizeof(float));
    fc_layer->d = (float*)calloc(output_neuron, sizeof(float));
    fc_layer->is_full_connect = true;

    srand((unsigned)time(NULL));
    for(int i=0; i<output_neuron; i++){
        fc_layer->weight[i] = (float*)malloc(input_neuron*sizeof(float));
        for(int j=0; j<input_neuron; j++){
            float rand_var = (((float)rand()/(float)RAND_MAX)-0.5)*2;
            fc_layer->weight[i][j] = rand_var*sqrt((float)6.0/(float)(input_neuron+output_neuron));
        }
    }
    return fc_layer;
}

//Create the CNN structure and initialize randomly the weight of Convolutional layer (break the simmetry)
void CreateCnn(CNN *cnn, MatriceSize input_size, int output_size){
    int filter_size = 5;
    MatriceSize switch_input_size;
    cnn->number_of_layer = 5;

    //Convolutional Layer with input size = {28,28} for MNIST
    switch_input_size.columns = input_size.columns;
    switch_input_size.rows = input_size.rows;
    cnn->C1 = CreateConvLayer(switch_input_size.columns, switch_input_size.rows, 5, 1, 6);

    //Pooling Layer with input size = {24,24}
    switch_input_size.columns = switch_input_size.columns-filter_size+1;
    switch_input_size.rows = switch_input_size.rows-filter_size+1;
    cnn->P2 = CreatePoolingLayer(switch_input_size.columns, switch_input_size.rows, 2, 6, 6, AVG_POOLING);

    //Convolutional Layer with input size = {12,12}
    switch_input_size.columns = switch_input_size.columns/2;
    switch_input_size.rows = switch_input_size.rows/2;
    cnn->C3 = CreateConvLayer(switch_input_size.columns, switch_input_size.rows, 5, 6, 12);

    //Pooling Layer with input size = {8,8}
    switch_input_size.columns = switch_input_size.columns-filter_size+1;
    switch_input_size.rows = switch_input_size.rows-filter_size+1;
    cnn->P4 = CreatePoolingLayer(switch_input_size.columns, switch_input_size.rows, 2, 12, 12, AVG_POOLING);

    //Fully connected Layer with input size: {4,4}
    switch_input_size.columns = switch_input_size.columns/2;
    switch_input_size.rows = switch_input_size.rows/2;
    cnn->FC5 = CreateFullyConnectedLayer(switch_input_size.columns*switch_input_size.rows*12, output_size);

    cnn->error = (float*)calloc(cnn->FC5->output_neuron, sizeof(float));
}

//activation function sigma
float ActSigm(float input, float bias){
    float sum = input + bias;
    float res = ((float)1.0 / ((float)(1.0 + exp(-sum))));
    return res;
}

//derivate sigma (y is already sigma(x))
float dSigm(float y){
    return y*(1-y);
}

//AVG POOLING first use by Yann LeCun in the CNN
void AvgPool(float **output, MatriceSize output_size, float **input, MatriceSize input_size, int filter_size){
    int out_sizeW = input_size.columns / filter_size;
    int out_sizeH = input_size.rows / filter_size;

    if(output_size.columns != out_sizeW || output_size.rows != out_sizeH){
        printf("\n!! ERROR: Wrong output size in AvgPool !!\n");
    }

    for(int i=0; i<out_sizeH; i++)
        for(int j=0; j<out_sizeW; j++){
            float s = 0.0;
            for(int k=i*filter_size; k<i*filter_size+filter_size; k++) {
                for (int l = j * filter_size; l < j * filter_size + filter_size; l++) {
                    s = s + input[k][l];
                }
            }
            output[i][j]=s/(float)(filter_size*filter_size);
        }
}

//dot two vector with same size
float mult_vec(float *vec1, float *vec2, int size){
    float q = 0.0;
    for(int i=0; i<size; i++){
        q = q+ vec1[i]*vec2[i];
    }
    return q;
}

//update weight (w*a+b)
void network_weight(float *output, float *input, float **weight, float *bias, MatriceSize network_size){
    int columns = network_size.columns;
    int rows = network_size.rows;
    for(int i=0; i<rows; i++){
        output[i] = mult_vec(input,weight[i],columns)+bias[i];
    }
}

//forward propagation of network
void forward_propagation(CNN *network, float **input){

    //first layer of propagation
    MatriceSize filter_size={network->C1->filter_size,network->C1->filter_size};
    MatriceSize input_size={network->C1->input_width,network->C1->input_height};
    MatriceSize output_size={network->P2->input_width,network->P2->input_height};

    for(int i=0;i<(network->C1->output_channels);i++){

        for(int j=0;j<(network->C1->input_channels);j++){

            float **filter_out=MatConv(network->C1->filter_data[j][i], filter_size, input, input_size, VALID);
            sumMatrix(network->C1->v[i],network->C1->v[i],output_size,filter_out,output_size);

            for(int row=0; row<output_size.rows; row++){
                free(filter_out[row]);
            }
            free(filter_out); //pb stack
        }

        for(int row=0; row<output_size.rows; row++){
            for(int col=0; col<output_size.columns; col++){
                network->C1->y[i][row][col] = ActSigm(network->C1->v[i][row][col], network->C1->bias[i]);
            }
        }
    }

    //output propagate to the P2 layer
    output_size.columns = network->C3->input_width;
    output_size.rows = network->C3->input_height;
    input_size.columns = network->P2->input_width;
    input_size.rows = network->P2->input_height;

    for(int i=0; i<(network->P2->output_channels); i++){
        if(network->P2->pooling_type == AVG_POOLING){
            AvgPool(network->P2->y[i], output_size, network->C1->y[i], input_size, network->P2->filter_size);
        }
    }

    //the output of third layer fully connected
    output_size.columns = network->P4->input_width;
    output_size.rows = network->P4->input_height;

    input_size.columns = network->C3->input_width;
    input_size.rows = network->C3->input_height;
    filter_size.columns = network->C3->filter_size;
    filter_size.rows = network->C3->filter_size;

    for(int i=0; i<(network->C3->output_channels); i++){
        for(int j=0; j<(network->C3->input_channels); j++){
            float** mapout = MatConv(network->C3->filter_data[j][i], filter_size, network->P2->y[j], input_size, VALID);
            sumMatrix(network->C3->v[i], network->C3->v[i], output_size, mapout, output_size);

            for(int r=0; r<output_size.rows; r++){
                free(mapout[r]);
            }
            free(mapout);
        }
        for(int r=0; r<output_size.rows; r++){
            for(int c=0; c<output_size.columns; c++){
                network->C3->y[i][r][c] = ActSigm(network->C3->v[i][r][c], network->C3->bias[i]);
            }
        }
    }

    //propagation output of fourth layer
    input_size.columns = network->P4->input_width;
    input_size.rows = network->P4->input_height;
    output_size.columns = input_size.columns / network->P4->filter_size;
    output_size.rows = input_size.rows / network->P4->filter_size;

    for(int i=0;i<(network->P4->output_channels);i++){
        if(network->P4->pooling_type==AVG_POOLING){
            AvgPool(network->P4->y[i], output_size, network->C3->y[i], input_size, network->P4->filter_size);
        }
    }

    //preprocessing of fullyconnected layer output
    // transform the multi-dimensional vector to one-dimensional
    //reshape
    float  *fullyCo_input=(float*)malloc((network->FC5->input_neuron)*sizeof(float));
    for(int i=0; i<(network->P4->output_channels); i++){
        for(int row=0; row<output_size.rows; row++){
            for(int col=0; col<output_size.columns; col++){
                fullyCo_input[i * output_size.rows * output_size.columns + row * output_size.columns + col] = network->P4->y[i][row][col];
            }
        }
    }

    MatriceSize network_size = {network->FC5->input_neuron, network->FC5->output_neuron};
    network_weight(network->FC5->v, fullyCo_input, network->FC5->weight, network->FC5->bias, network_size);

    for(int i=0; i<network->FC5->output_neuron; i++){
        network->FC5->y[i] = ActSigm(network->FC5->v[i], network->FC5->bias[i]);
    }
    free(fullyCo_input);
}

//back propagation of network
void back_propagation(CNN *network,float *output){

    for(int i=0; i<network->FC5->output_neuron; i++){  //save the error to the network
        network->error[i] = network->FC5->y[i]-output[i];
    }

    //calcul backward from back to front 5FC LAYER
    for(int i=0; i<network->FC5->output_neuron; i++){
        network->FC5->d[i] = network->error[i]*dSigm(network->FC5->y[i]);
    }

    //fourth layer of pooling
    MatriceSize output_size = {network->P4->input_width/network->P4->filter_size, network->P4->input_height/network->P4->filter_size};
    for(int i=0; i<network->P4->output_channels; i++){
        for(int r=0; r<output_size.rows; r++){
            for(int c=0; c<output_size.columns; c++) {
                for(int j=0; j<network->FC5->output_neuron; j++){
                    int w = i*output_size.columns*output_size.rows+r*output_size.columns+c;
                    network->P4->d[i][r][c] = network->P4->d[i][r][c]+network->FC5->d[j]*network->FC5->weight[j][w];
                }
            }
        }
    }

    int filter_size_pool = network->P4->filter_size;
    MatriceSize matrix1;
    matrix1.columns = network->P4->input_width/filter_size_pool;
    matrix1.rows = network->P4->input_height/filter_size_pool;

    for(int i=0; i<network->C3->output_channels; i++){
        float **CLe = UpSamplingMatrice(network->P4->d[i], matrix1, network->P4->filter_size, network->P4->filter_size);
        for(int row=0; row<network->P4->input_height; row++){
            for(int col=0; col<network->P4->input_width; col++){
                network->C3->d[i][row][col] = CLe[row][col] * dSigm(network->C3->y[i][row][col])/(float)(network->P4->filter_size * network->P4->filter_size);
            }
        }

        for(int row=0; row<network->P4->input_height; row++){
            free(CLe[row]);
        }
        free(CLe);
    }

    output_size.columns = network->C3->input_width;
    output_size.rows = network->C3->input_height;
    MatriceSize input_size = {network->P4->input_width, network->P4->input_height};
    MatriceSize map_size = {network->C3->filter_size, network->C3->filter_size};

    for(int i=0; i<network->P2->output_channels; i++){
        for(int j=0; j<network->C3->output_channels; j++){
            float **conv = MatCorrelation(network->C3->filter_data[i][j], map_size, network->C3->d[j], input_size, FULL);
            sumMatrix(network->P2->d[i], network->P2->d[i], output_size, conv, output_size);
            for(int row=0; row<output_size.rows; row++){
                free(conv[row]);
            }
            free(conv);
        }
    }


    filter_size_pool = network->P2->filter_size;
    MatriceSize matrix;
    matrix.columns = network->P2->input_width / filter_size_pool;
    matrix.rows = network->P2->input_height / filter_size_pool;

    for(int i=0; i<network->C1->output_channels; i++){

        float **CL1e = UpSamplingMatrice(network->P2->d[i], matrix, network->P2->filter_size, network->P2->filter_size);
        for(int r=0; r<network->P2->input_height; r++){
            for(int c=0; c<network->P2->input_width; c++){
                network->C1->d[i][r][c] = CL1e[r][c] * dSigm(network->C1->y[i][r][c]) / (float)(network->P2->filter_size * network->P2->filter_size);
            }
        }

        for(int row=0; row<network->P2->input_height; row++){
            free(CL1e[row]);
        }
        free(CL1e);
    }
}

// update weight of network
void UpdateWeightNetwork(CNN *network, training_hyperparam hyper_params, float **input){

    //update weight of the first Convolutional layer
    MatriceSize d_size = {network->P2->input_height, network->P2->input_width};
    MatriceSize y_size = {network->C1->input_height, network->C1->input_width};
    MatriceSize filter_size = {network->C1->filter_size, network->C1->filter_size};
    int i,j;
    for(i=0; i<network->C1->output_channels; i++){
        for(j=0; j<network->C1->input_channels; j++){

            float **rotated_input = MatriceRotation180(input, y_size);
            float **dC1 = MatConv(network->C1->d[i], d_size, rotated_input, y_size,VALID);

            MatMultiScaler(dC1, dC1, filter_size, -1*hyper_params.alpha);
            sumMatrix(network->C1->filter_data[j][i], network->C1->filter_data[j][i], filter_size, dC1, filter_size);

            for(int r=0; r<(d_size.rows-(y_size.rows-1)); r++){
                free(dC1[r]);
            }
            free(dC1);

            for(int r=0; r<y_size.rows; r++){
                free(rotated_input[r]);
            }
            free(rotated_input);
        }
        network->C1->bias[i] = network->C1->bias[i]-hyper_params.alpha*sumMat(network->C1->d[i], d_size);
    }

    //update weight of Convolutional layer 3
    d_size.columns = network->P4->input_width;
    d_size.rows = network->P4->input_height;
    y_size.columns = network->C3->input_width;
    y_size.rows = network->C3->input_height;

    filter_size.columns = network->C3->filter_size;
    filter_size.rows = network->C3->filter_size;

    for(i=0; i<network->C3->output_channels; i++){
        for(j=0; j<network->C3->input_channels; j++){

            float **rotated_input = MatriceRotation180(network->P2->y[j], y_size);
            float **dC3 = MatConv(network->C3->d[i], d_size, rotated_input, y_size, VALID);

            MatMultiScaler(dC3, dC3, filter_size, -1.0*hyper_params.alpha);
            sumMatrix(network->C3->filter_data[j][i], network->C3->filter_data[j][i], filter_size, dC3, filter_size);

            for(int r=0; r<(d_size.rows-(y_size.rows-1)); r++){
                free(dC3[r]);
            }
            free(dC3);

            for(int r=0; r<y_size.rows; r++){
                free(rotated_input[r]);
            }
            free(rotated_input);
        }
        network->C3->bias[i] = network->C3->bias[i] - hyper_params.alpha * sumMat(network->C3->d[i], d_size);
    }

    //update weight of the fully connected layer
    float *output_layer_5=(float*)malloc((network->FC5->input_neuron)*sizeof(float));

    MatriceSize output_size;
    output_size.columns = network->P4->input_width/network->P4->filter_size;
    output_size.rows = network->P4->input_height/network->P4->filter_size;

    for(i=0; i<(network->P4->output_channels); i++){ //reshape in (1,1)
        for(int r=0; r<output_size.rows; r++){
            for(int c=0; c<output_size.columns; c++){
                output_layer_5[i*output_size.rows * output_size.columns + r*output_size.columns + c] = network->P4->y[i][r][c];
            }
        }
    }

    for(j=0; j<network->FC5->output_neuron; j++){
        for(i=0; i<network->FC5->input_neuron; i++){
            network->FC5->weight[j][i] = network->FC5->weight[j][i] - hyper_params.alpha * network->FC5->d[j] * output_layer_5[i];
        }
        network->FC5->bias[j] = network->FC5->bias[j] - hyper_params.alpha * network->FC5->d[j];
    }
    free(output_layer_5);
}

//function to train the CNN with train data and hyperparams
void trainCNN(CNN *network,	ArrayOfImage input_data, ArrayOfLabel output_data,training_hyperparam hyper_params, int train_number){

    network->loss = (float*)malloc(train_number*sizeof(float));
    int ep;
    for(ep=0;ep<hyper_params.nbEpochs;ep++){
        printf("[T][R][A][I][N][I][N][G]...    %d/%d \n",ep,hyper_params.nbEpochs);
        int n;
        for(n=0;n<train_number;n++){
            printf("Learning image :  %d / %d\n",n,train_number);

            forward_propagation(network,input_data->image[n].data);
            //printf_cnn(network);
            back_propagation(network,output_data->label[n].data);
            UpdateWeightNetwork(network,hyper_params,input_data->image[n].data);
            clearCNN(network);
            float loss=0.0;
            for(int i=0;i<network->FC5->output_neuron;i++){
                loss=loss+network->error[i]*network->error[i];
            }
            if(n==0) {
                network->loss[n] = loss / (float) 2.0;
            }else{
                network->loss[n] = network->loss[n-1]*0.99+0.01*loss/(float)2.0;
            }
        }
    }
}

//print the CNN architecture with values (for debug)
void printf_cnn(CNN *network){
    printf("C1\n");
    printf("Input size width: %d\n", network->C1->input_width);
    printf("Input size height: %d\n", network->C1->input_height);
    printf("Filter size: %d x %d\n", network->C1->filter_size, network->C1->filter_size);
    printf("Input channels : %d\n", network->C1->input_channels);
    printf("Output channels: %d\n", network->C1->output_channels);
    printf("Filter data\n");
    for(int i=0; i<network->C1->input_channels;i++){
        for(int j=0; j<network->C1->output_channels;j++){
            for(int k=0; k<network->C1->filter_size;k++){
                for(int l=0; l<network->C1->filter_size;l++){
                    printf(" %f ",network->C1->filter_data[i][j][k][l]);
                }
            }
        }
    }

    printf("\nFilter derivate data\n");
    for(int i=0; i<network->C1->input_channels;i++){
        for(int j=0; j<network->C1->output_channels;j++){
            for(int k=0; k<network->C1->filter_size;k++){
                for(int l=0; l<network->C1->filter_size;l++){
                    printf(" %f ",network->C1->dfilter_data[i][j][k][l]);
                }
            }
        }
    }

    printf("\nBias :\n");
    for(int i=0; i<network->C1->output_channels;i++){
        printf(" %f ",network->C1->bias[i]);
    }


    printf("\n Output  V:\n");
    for(int i=0; i<network->C1->output_channels;i++){
        for(int j=0; j<network->C1->output_channels;j++){
            for(int k=0; k<network->C1->output_channels;k++){
                printf(" %f ",network->C1->v[i][j][k]);
            }
        }
    }

    printf("\n Output  Y:\n");
    for(int i=0; i<network->C1->output_channels;i++){
        for(int j=0; j<network->C1->output_channels;j++){
            for(int k=0; k<network->C1->output_channels;k++){
                printf(" %f ",network->C1->y[i][j][k]);
            }
        }
    }

    printf("\n Output  d:\n");
    for(int i=0; i<network->C1->output_channels;i++){
        for(int j=0; j<network->C1->output_channels;j++){
            for(int k=0; k<network->C1->output_channels;k++){
                printf(" %f ",network->C1->d[i][j][k]);
            }
        }
    }

    printf("\n");
    printf("\nP2\n");
    printf("Input size width: %d\n", network->P2->input_width);
    printf("Input size height: %d\n", network->P2->input_height);
    printf("Filter size: %d x %d\n", network->P2->filter_size, network->P2->filter_size);
    printf("Input channels : %d\n", network->P2->input_channels);
    printf("Output channels: %d\n", network->P2->output_channels);
    printf("Pooling type: %d\n", network->P2->pooling_type);

    printf("\nBias :\n");
    for(int i=0; i<network->P2->output_channels;i++){
        printf(" %f ",network->P2->bias[i]);
    }

    printf("\nOutput  y:\n");
    for(int i=0; i<network->P2->output_channels;i++){
        for(int j=0; j<network->P2->output_channels;j++){
            for(int k=0; k<network->P2->output_channels;k++){
                printf(" %f ",network->P2->y[i][j][k]);
            }
        }
    }

    printf("\nOutput  d:\n");
    for(int i=0; i<network->P2->output_channels;i++){
        for(int j=0; j<network->P2->output_channels;j++){
            for(int k=0; k<network->P2->output_channels;k++){
                printf(" %f ",network->P2->d[i][j][k]);
            }
        }
    }


    printf("\n");
    printf("\nC3\n");
    printf("Input size width: %d\n", network->C3->input_width);
    printf("Input size height: %d\n", network->C3->input_height);
    printf("Filter size: %d x %d\n", network->C3->filter_size, network->C3->filter_size);
    printf("Input channels : %d\n", network->C3->input_channels);
    printf("Output channels: %d\n", network->C3->output_channels);
    printf("Filter data\n");
    for(int i=0; i<network->C3->input_channels;i++){
        for(int j=0; j<network->C3->output_channels;j++){
            for(int k=0; k<network->C3->filter_size;k++){
                for(int l=0; l<network->C3->filter_size;l++){
                    printf(" %f ",network->C3->filter_data[i][j][k][l]);
                }
            }
        }
    }

    printf("\nFilter derivate data\n");
    for(int i=0; i<network->C3->input_channels;i++){
        for(int j=0; j<network->C3->output_channels;j++){
            for(int k=0; k<network->C3->filter_size;k++){
                for(int l=0; l<network->C3->filter_size;l++){
                    printf(" %f ",network->C3->dfilter_data[i][j][k][l]);
                }
            }
        }
    }

    printf("\nBias :\n");
    for(int i=0; i<network->C3->output_channels;i++){
        printf(" %f ",network->C3->bias[i]);
    }


    printf("\n Output  V:\n");
    for(int i=0; i<network->C3->output_channels;i++){
        for(int j=0; j<network->C3->output_channels;j++){
            for(int k=0; k<network->C3->output_channels;k++){
                printf(" %f ",network->C3->v[i][j][k]);
            }
        }
    }

    printf("\n Output  Y:\n");
    for(int i=0; i<network->C3->output_channels;i++){
        for(int j=0; j<network->C3->output_channels-network->C3->filter_size+1;j++){
            for(int k=0; k<network->C3->output_channels-network->C3->filter_size+1;k++){
                printf(" %f ",network->C3->y[i][j][k]);
            }
        }
    }

    printf("\n Output  d:\n");
    for(int i=0; i<network->C3->output_channels;i++){
        for(int j=0; j<network->C3->output_channels;j++){
            for(int k=0; k<network->C3->output_channels;k++){
                printf(" %f ",network->C3->d[i][j][k]);
            }
        }
    }

    printf("\n");
    printf("\nP4\n");
    printf("Input size width: %d\n", network->P4->input_width);
    printf("Input size height: %d\n", network->P4->input_height);
    printf("Filter size: %d x %d\n", network->P4->filter_size, network->P4->filter_size);
    printf("Input channels : %d\n", network->P4->input_channels);
    printf("Output channels: %d\n", network->P4->output_channels);
    printf("Pooling type: %d\n", network->P4->pooling_type);

    printf("\nBias :\n");
    for(int i=0; i<network->P4->output_channels;i++){
        printf(" %f ",network->P4->bias[i]);
    }

    printf("\nOutput  y:\n");
    for(int i=0; i<network->P4->output_channels;i++){
        for(int j=0; j<network->P4->input_height/network->P4->filter_size;j++){
            for(int k=0; k<network->P4->input_height/network->P4->filter_size;k++){
                printf(" %f ",network->P4->y[i][j][k]);
            }
        }
    }

    printf("\nOutput  d:\n");
    for(int i=0; i<network->P4->output_channels;i++){
        for(int j=0; j<network->P4->input_height/network->P4->filter_size;j++){
            for(int k=0; k<network->P4->input_height/network->P4->filter_size;k++){
                printf(" %f ",network->P4->d[i][j][k]);
            }
        }
    }

    printf("\n");
    printf("\nFC5\n");
    printf("Input neuron: %d\n", network->FC5->input_neuron);
    printf("Output neuron: %d\n", network->FC5->output_neuron);
    printf("\nWeight:\n");
    for(int i=0;i<network->FC5->output_neuron;i++){
        for(int j=0;j<network->FC5->input_neuron;j++){
            printf(" %f ",network->FC5->weight[i][j]);
        }
    }
    printf("\nBias:\n");
    for(int i=0;i<network->FC5->output_neuron;i++){
        printf(" %f ", network->FC5->bias[i]);
    }

    printf("\nOutput V:\n");
    for(int i=0;i<network->FC5->output_neuron;i++){
        printf(" %f ", network->FC5->v[i]);
    }

    printf("\nOutput Y:\n");
    for(int i=0;i<network->FC5->output_neuron;i++){
        printf(" %f ", network->FC5->y[i]);
    }

    printf("\nOutput d:\n");
    for(int i=0;i<network->FC5->output_neuron;i++){
        printf(" %f ", network->FC5->d[i]);
    }
}

//return max index of vector m (for test)
int maxIndex(float *m, int m_size){
    int i=0;
    float max = -1.0;
    int maxI = 0;
    for(i=0; i<m_size; i++){
        if(m[i]>max){
            max=m[i];
            maxI=i;
        }
    }
    return maxI;
}

//function to test CNN with test data
float testCNN(CNN *network, ArrayOfImage input_data, ArrayOfLabel output_data, int test_num){
    int error=0;
    for(int n=0; n<test_num; n++){
        printf("----- Testing image : %d/%d\n",n,test_num);
        forward_propagation(network, input_data->image[n].data);

        if(maxIndex(network->FC5->y,network->FC5->output_neuron) != maxIndex(output_data->label[n].data, network->FC5->output_neuron)){
            error++;
        }
        clearCNN(network);
    }
    return (float)error/(float)test_num;
}

//function to test CNN with test data and the number of misclassified
float testCNN2(CNN *network, ArrayOfImage input_data, ArrayOfLabel output_data, int test_num, int *tab){
    int error=0;
    for(int n=0; n<test_num; n++){
        printf("----- Testing image : %d/%d\n",n,test_num);
        forward_propagation(network, input_data->image[n].data);

        if(maxIndex(network->FC5->y,network->FC5->output_neuron) != maxIndex(output_data->label[n].data, network->FC5->output_neuron)){
            error++;
            tab[maxIndex(output_data->label[n].data, network->FC5->output_neuron)]++;
        }
        //clearCNN(network);
    }
    return (float)error/(float)test_num;
}

//save CNN network architecture in file
void saveCNN(CNN *network, const char *path){
    FILE *file=NULL;
    file = fopen(path,"wb");
    if(file == NULL){
        printf("!! ERROR : Save CNN network to file <%s> failed\n",path);
    }
    for(int i=0; i<network->C1->input_channels; i++){
        for(int j=0; j<network->C1->output_channels; j++){
            for(int k=0; k<network->C1->filter_size; k++){
                fwrite(network->C1->filter_data[i][j][k], sizeof(float), network->C1->filter_size, file);
            }
        }
    }
    fwrite(network->C1->bias, sizeof(float), network->C1->output_channels, file);

    for(int i=0; i<network->C3->input_channels; i++){
        for(int j=0; j<network->C3->output_channels; j++){
            for(int k=0; k<network->C3->filter_size; k++){
                fwrite(network->C3->filter_data[i][j][k], sizeof(float), network->C3->filter_size, file);
            }
        }
    }
    fwrite(network->C3->bias, sizeof(float), network->C3->output_channels, file);

    for(int i=0; i<network->FC5->output_neuron; i++){
        fwrite(network->FC5->weight[i], sizeof(float), network->FC5->input_neuron, file);
    }
    fwrite(network->FC5->bias, sizeof(float), network->FC5->output_neuron, file);
    fclose(file);
}

//import CNN network from file
void importCNN(CNN *network, const char* path){
    FILE  *file=NULL;
    file = fopen(path,"rb");

    if(file==NULL){
        printf("!! Error: ImportCNN Open file failed! <%s>\n",path);
    }

    for(int i=0; i<network->C1->input_channels; i++){
        for(int j=0; j<network->C1->output_channels; j++){
            for(int k=0; k<network->C1->filter_size; k++){
                for(int l=0; l<network->C1->filter_size; l++){
                    float *input = (float*)malloc(sizeof(float));
                    size_t lect = fread(input,sizeof(float),1,file);
                    if(lect==0) printf("Erreur importCNN");
                    network->C1->filter_data[i][j][k][l] = *input;
                }
            }
        }
    }

    int i,j,k,l;
    for(i=0; i<network->C1->output_channels; i++){
        size_t lect = fread(&network->C1->bias[i],sizeof(float),1,file);
        if(lect==0) printf("Erreur importCNN");
    }

    for(i=0; i<network->C3->input_channels; i++){
        for(j=0; j<network->C3->output_channels; j++){
            for(k=0; k<network->C3->filter_size; k++){
                for(l=0; l<network->C3->filter_size; l++){
                    size_t lect = fread(&network->C3->filter_data[i][j][k][l],sizeof(float),1,file);
                    if(lect==0) printf("Erreur importCNN");
                }
            }
        }
    }

    for(i=0; i<network->C3->output_channels; i++){
        size_t lect = fread(&network->C3->bias[i],sizeof(float),1,file);
        if(lect==0) printf("Erreur importCNN");
    }

    for(i=0; i<network->FC5->output_neuron; i++){
        for(j=0; j<network->FC5->input_neuron; j++){
            size_t lect = fread(&network->FC5->weight[i][j],sizeof(float),1,file);
            if(lect==0) printf("Erreur importCNN");
        }
    }

    for(i=0; i<network->FC5->output_neuron; i++){
        size_t lect = fread(&network->FC5->bias[i],sizeof(float),1,file);
        if(lect==0) printf("Erreur importCNN");
    }
    fclose(file);
}

void clearCNN(CNN *network){
    for(int i=0; i<network->C1->output_channels; i++){
        for(int row=0; row<network->P2->input_height; row++){
            for(int col=0; col<network->P2->input_width; col++){
                network->C1->d[i][row][col] = (float)0.0;
                network->C1->v[i][row][col] = (float)0.0;
                network->C1->y[i][row][col] = (float)0.0;
            }
        }
    }

    for(int i=0; i<network->P2->output_channels; i++){
        for(int row=0; row<network->C3->input_height; row++){
            for(int col=0; col<network->C3->input_width; col++){
                network->P2->d[i][row][col] = (float)0.0;
                network->P2->y[i][row][col] = (float)0.0;
            }
        }
    }

    for(int i=0; i<network->C3->output_channels; i++){
        for(int row=0; row<network->P4->input_height; row++){
            for(int col=0; col<network->P4->input_width; col++){
                network->C3->d[i][row][col] = (float)0.0;
                network->C3->v[i][row][col] = (float)0.0;
                network->C3->y[i][row][col] = (float)0.0;
            }
        }
    }

    for(int i=0; i<network->P4->output_channels; i++){
        for(int row=0; row<network->P4->input_height/network->P4->filter_size; row++){
            for(int col=0; col<network->P4->input_width/network->P4->filter_size; col++){
                network->P4->d[i][row][col] = (float)0.0;
                network->P4->y[i][row][col] = (float)0.0;
            }
        }
    }

    for(int n=0; n<network->FC5->output_neuron; n++){
        network->FC5->d[n] = (float)0.0;
        network->FC5->v[n] = (float)0.0;
        network->FC5->y[n] = (float)0.0;
    }
}