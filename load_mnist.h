//
// Created by Nemanja Kostadinovic on 14/11/2020.
//

#ifndef APPRENTISSAGECNN_LOAD_MNIST_H
#define APPRENTISSAGECNN_LOAD_MNIST_H


typedef struct Image{
    int width; //number of columns
    int hight; //number of rows
    float **data;
}Image;

typedef struct ArrayImage{
    int total_image; //number of images in the dataset
    Image *image;
} *ArrayOfImage;

typedef struct Label{
    int label_size; //length
    float *data; //value of label
}Label;

typedef struct ArrayLabel{
    int total_label;
    Label *label;
} *ArrayOfLabel;

ArrayOfImage read_image(const char *filename);
ArrayOfLabel read_label(const char *filename);

void TestMnist();



#endif //APPRENTISSAGECNN_LOAD_MNIST_H
