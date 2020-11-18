//
// Created by Nemanja Kostadinovic on 15/11/2020.
//

#include "load_mnist.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>


int reverse_int(int integer){
    unsigned char bits_0_7;
    unsigned char bits_8_15;
    unsigned char bits_16_23;
    unsigned char bits_24_31;

    bits_0_7   = integer & 255;         
    bits_8_15  = (integer >> 8) & 255;  
    bits_16_23 = (integer >> 16) & 255; 
    bits_24_31 = (integer >> 24) & 255; 
    return ((int)bits_0_7 << 24) + ((int)bits_8_15 << 16) + ((int)bits_16_23 << 8) + bits_24_31;
}


ArrayOfImage read_image(const char* filename){

    FILE  *file_point = NULL;
    file_point = fopen(filename,"rb");
    if(file_point == NULL){
        printf("ERROR: read_image file <%s> failed \n",filename);
        assert(file_point);
    }
    
    int magic_number = 0;     
    int total_image = 0; 
    int number_rows = 0;  
    int number_columns = 0;  
    
    size_t lect = fread((char*)&magic_number, sizeof(magic_number), 1, file_point);
    if(lect==0) printf("Erreur read image");
    magic_number = reverse_int(magic_number);
    
    size_t lect2 = fread((char*)&total_image, sizeof(total_image), 1, file_point);
    if(lect2==0) printf("Erreur read image");
    total_image = reverse_int(total_image);
    
    size_t lect3 = fread((char*)&number_rows, sizeof(number_rows), 1, file_point);
    if(lect3==0) printf("Erreur read image");
    size_t lect4 = fread((char*)&number_columns, sizeof(number_columns), 1, file_point);
    if(lect4==0) printf("Erreur read image");
    number_rows = reverse_int(number_rows);
    number_columns = reverse_int(number_columns);
    
    ArrayOfImage image_array = (ArrayOfImage)malloc(sizeof(ArrayOfImage));
    image_array->total_image = total_image;  
    image_array->image = (Image*)malloc(total_image * sizeof(Image));

    int row,column;
    for(int i=0; i<total_image; ++i){
        image_array->image[i].hight = number_rows;     
        image_array->image[i].width = number_columns;  
        image_array->image[i].data = (float**) malloc(number_rows * sizeof(float*));

        for(row = 0; row < number_rows; ++row){
            image_array->image[i].data[row] = (float*)malloc(number_columns * sizeof(float));
            for(column = 0; column < number_columns; ++column){
                unsigned char temp_pixel = 0;
                size_t lect5 = fread((char*) &temp_pixel, sizeof(temp_pixel), 1, file_point);
                if(lect5==0) printf("Erreur read image");
                image_array->image[i].data[row][column]= (float)temp_pixel/255;
            }
        }
    }

    fclose(file_point);
    return image_array;
}

ArrayOfLabel read_label(const char* filename){
    FILE *file_point=NULL;
    file_point = fopen(filename, "rb");
    if(file_point==NULL)
        printf("ERROR: read_label file <%s> failed \n",filename);
    assert(file_point);

    int magic_number = 0;
    int number_of_labels = 0;
    int label_long = 10;

    size_t lect = fread((char*)&magic_number,sizeof(magic_number),1,file_point);
    if(lect==0) printf("Erreur read label");
    magic_number = reverse_int(magic_number);

    size_t lect2 = fread((char*)&number_of_labels,sizeof(number_of_labels),1,file_point);
    if(lect2==0) printf("Erreur read image");
    number_of_labels = reverse_int(number_of_labels);

    int i;

    ArrayOfLabel labarr=(ArrayOfLabel)malloc(sizeof(ArrayOfLabel));
    labarr->total_label = number_of_labels;
    labarr->label = (Label*)malloc(number_of_labels*sizeof(Label));

    for(i = 0; i < number_of_labels; ++i){
        labarr->label[i].label_size = 10;
        labarr->label[i].data = (float*)calloc(label_long,sizeof(float));
        unsigned char temp = 0;
        size_t lect3 = fread((char*) &temp, sizeof(temp),1,file_point);
        if(lect3==0) printf("Erreur read image");
        labarr->label[i].data[(int)temp] = 1.0;
    }

    fclose(file_point);
    return labarr;
}


void TestMnist(int number){
    ArrayOfImage train_images = read_image("/Users/nemanja/CLionProjects/apprentissageCNN/mnist/train-images-idx3-ubyte");
    for(int j=0; j<number; j++){
        for (int i=0; i<784; i++) {
            printf("%1.1f ", train_images->image->data[j][i]);
            if ((i+1) % 28 == 0) putchar('\n');
        }
        printf("\n");
    }
}