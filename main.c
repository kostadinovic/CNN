#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "conv_nn.h"
#include "load_mnist.h"
#include <string.h>



int main(int argc, char *argv[]){

	//Read train and test data.
	ArrayOfLabel train_labels = read_label("./mnist/train-labels-idx1-ubyte");
    ArrayOfImage train_images = read_image("./mnist/train-images-idx3-ubyte");
    ArrayOfLabel test_labels = read_label("./mnist/t10k-labels-idx1-ubyte");
    ArrayOfImage test_images = read_image("./mnist/t10k-images-idx3-ubyte");

    if(strcmp(argv[1],"train_test") == 0){
        printf("Start :\n");
    
        printf("Read all data finished\n");

        printf("Train data sample\n");
        TestMnist(1);

        const char *cnn_arch_path = "./cnn_structure.txt";
        const char *cnn_layer_path = "./cnn_layer.txt";

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
        opts.alpha=1;

        int total_train_image = 60000;

        // Train the network
        trainCNN(network, train_images, train_labels, opts, total_train_image);
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
        float ratio_of_error;
        int mis_classified[10] = {0};
        int count[10] = {0};
        ratio_of_error = testCNN2(network, test_images, test_labels, test_images_num, mis_classified, count);
        for(int i=0; i<10;i++){
            printf("The number %d appear %d times and is %d times misclassified (ratio=%f)\n", i, count[i], mis_classified[i], (1.0)*mis_classified[i]/count[i]);
        }

        printf("---- Accuracy = %f\n", (1-ratio_of_error));
        printf("---- Error = %f\n", ratio_of_error);
        printf("!!!The test is finished!!!\n");
    }

    if(strcmp(argv[1],"test") == 0){
        printf("Start :\n");
        //Read test data.
        printf("Read all data finished\n");

        //Train data sample:
        printf("Train data sample\n");
        TestMnist(1);

        //Input of network
        MatriceSize input_size = {test_images->image[0].width, test_images->image[0].hight};
        printf("--- Input size: {%d,%d}\n",input_size.columns,input_size.rows);

        //Output of network
        int output_size = test_labels->label[0].label_size;
        printf("--- Output size: %d\n",output_size);

        const char *cnn_arch_path = "./cnn_structure.txt";

        //Import the CNN network
        CNN *network = (CNN*)malloc(sizeof(CNN));
        CreateCnn(network,input_size,output_size);
        importCNN(network, cnn_arch_path);
        printf("Import CNN network finished\n");

        printf("Begin the test : \n");

        //Testing the CNN network on test images
        int test_images_num = 10000;
        float ratio_of_error;
        int mis_classified[10] = {0};
        int count[10] = {0};
        ratio_of_error = testCNN2(network, test_images, test_labels, test_images_num, mis_classified, count);
        for(int i=0; i<10;i++){
            printf("The number %d appear %d times and is %d times misclassified (ratio=%f)\n", i, count[i], mis_classified[i], (1.0)*mis_classified[i]/count[i]);
        }

        printf("\n---- Accuracy = %f\n", (1-ratio_of_error));
        printf("---- Error = %f\n", ratio_of_error);
        printf("!!!The test is finished!!!\n");

    }

    if((strcmp(argv[1],"test") != 0 && strcmp(argv[1],"train_test")  != 0) || argc == 0){
        printf("You need to specify the args 'train_test' or 'test.\n");
        printf("\ttrain_test : train and test the network\n");
        printf("\ttest: load network and test data\n");
    }

    return 0;
}
