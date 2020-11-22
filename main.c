#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "conv_nn.h"
#include "load_mnist.h"
#include <string.h>



int main(int argc, char *argv[]){

    if(strcmp(argv[1],"train_test") == 0){
        printf("Start :\n");
        //Read train and test data.
        ArrayOfLabel train_labels = read_label("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/train-labels-idx1-ubyte");
        ArrayOfImage train_images = read_image("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/train-images-idx3-ubyte");
        ArrayOfLabel test_labels = read_label("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/t10k-labels-idx1-ubyte");
        ArrayOfImage test_images = read_image("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/t10k-images-idx3-ubyte");
        printf("Read all data finished\n");

        const char *cnn_arch_path = "/Users/nemanja/CLionProjects/apprentissageCNN/cnn_structure.txt";
        const char *cnn_layer_path = "/Users/nemanja/CLionProjects/apprentissageCNN/cnn_layer.txt";

        //Input of network
        MatriceSize input_size = {test_images->image[0].width, test_images->image[0].hight};
        printf("--- Input size: {%d,%d}\n",input_size.columns,input_size.rows);

        //Output of network
        int output_size = test_labels->label[0].label_size;
        printf("--- Output size: %d\n",output_size);

        //Create CNN
        CNN *network = (CNN*)malloc(sizeof(CNN));
        CreateCnn(network,input_size,output_size);
        printf("--- CNN creation finished\n");

        //Hyper parameter initialization
        training_hyperparam opts;
        opts.nbEpochs=1;
        opts.alpha=1.0;

        int total_train_image = 55000;

        // Train the network
        trainCNN(network, train_images, train_labels, opts, 100);
        printf("--- Train CNN finished\n");


        //Save CNN architecture and layer to file
        saveCNN(network,cnn_arch_path);

        FILE  *file = fopen(cnn_layer_path, "wb");
        if(file == NULL) printf("ERROR : Save layer failed\n");
        fwrite(network->loss, sizeof(float), total_train_image, file);
        fclose(file);

        //Import the CNN network
        importCNN(network, cnn_arch_path);
        printf("Import CNN network finished\n");


        //Testing the CNN network on test images
        int test_images_num = 10000;
        float ratio_of_error = 1.0;
        int tab[10][100];
        //ratio_of_error = testCNN2(network, test_images, test_labels, test_images_num, &tab);
        ratio_of_error = testCNN(network, test_images, test_labels, test_images_num);

        printf("---- Accuracy = %f\n", (1-ratio_of_error));
        printf("---- Error = %f\n", ratio_of_error);
        printf("!!!The test is finished!!!\n");
    }

    if(strcmp(argv[1],"test") == 0){
        printf("Start :\n");
        //Read test data.
        ArrayOfLabel test_labels = read_label("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/t10k-labels-idx1-ubyte");
        ArrayOfImage test_images = read_image("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/t10k-images-idx3-ubyte");
        printf("Read all data finished\n");

        //Input of network
        MatriceSize input_size = {test_images->image[0].width, test_images->image[0].hight};
        printf("--- Input size: {%d,%d}\n",input_size.columns,input_size.rows);

        //Output of network
        int output_size = test_labels->label[0].label_size;
        printf("--- Output size: %d\n",output_size);

        const char *cnn_arch_path = "/Users/nemanja/CLionProjects/apprentissageCNN/cnn_structure.txt";

        //Import the CNN network
        CNN *network = (CNN*)malloc(sizeof(CNN));
        CreateCnn(network,input_size,output_size);
        importCNN(network, cnn_arch_path);
        printf("Import CNN network finished\n");

        printf("Begin the test : \n");
        //Testing the CNN network on test images
        int test_images_num = 10000;
        float ratio_of_error = 1.0;
        int tab[10] = {0};
        ratio_of_error = testCNN2(network, test_images, test_labels, 1000, &tab);
        for(int i=0; i<10;i++){
            printf("The number %d is %d times misclassified\n", i, tab[i]);
        }

        printf("\n---- Accuracy = %f\n", (1-ratio_of_error));
        printf("---- Error = %f\n", ratio_of_error);
        printf("!!!The test is finished!!!\n");

    }

    if(strcmp(argv[1],"test") != 0 && strcmp(argv[1],"train_test") != 0){
        printf("You need to specify the args 'train_test' or 'test.\n");
        printf("\ttrain_test : train and test the network\n");
        printf("\ttest: load network and test data\n");
    }

    return 0;
}
