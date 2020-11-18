//
// Created by Nemanja Kostadinovic on 14/11/2020.
//

#ifndef APPRENTISSAGECNN_MATRICE_H
#define APPRENTISSAGECNN_MATRICE_H


#define FULL 0
#define SAME 1
#define VALID 2

typedef struct Matrice2DSize{
    int columns;
    int rows;
}MatriceSize;

/*
void print_matrice(float **mat,MatriceSize mat_size);
float **MatriceRotation180(float **matrice, MatriceSize size_matrice);
float **expandMatriceEdge(float **matrice, MatriceSize size_matrice, int ad_columns, int ad_rows);
float **reduceMatriceEdge(float **matrice, MatriceSize size_matrice, int reduce_columns, int reduce_rows);
void sumMatrix(float **sum, float **mat1, MatriceSize mat_size1, float **mat2, MatriceSize mat_size2);
void MatMultiScaler(float **mult, float **matrice, MatriceSize mat_size, float scaler);
float sumMat(float **mat,MatriceSize mat_size);
void print_matrice(float **mat,MatriceSize mat_size);

float **MatCorrelation(float **filter, MatriceSize filter_size, float **input, MatriceSize input_size, int type);
float **MatConv(float **filter, MatriceSize filter_size, float **input, MatriceSize input_size, int type);
float **UpSamplingMatrice(float **matrice, MatriceSize size, int upc, int upr);
void test_matrice();

 */


float** MatriceRotation180(float** mat, MatriceSize mat_size);
void sumMatrix(float** res, float** mat1, MatriceSize mat_size1, float** mat2, MatriceSize mat_size2);

float** MatCorrelation(float** map, MatriceSize map_size, float** inputData, MatriceSize inSize, int type);

float** MatConv(float** map,MatriceSize map_size,float** inputData,MatriceSize inSize,int type);

float** UpSamplingMatrice(float** mat,MatriceSize mat_size,int upc,int upr);

float** MatEdgeExpand(float** mat,MatriceSize mat_size,int addc,int addr);

float** MatEdgeShrink(float** mat,MatriceSize mat_size,int shrinkc,int shrinkr);

void MatSaving(float** mat,MatriceSize mat_size,const char* filename);

void MatMultiScaler(float** res, float** mat, MatriceSize mat_size, float factor);

float sumMat(float** mat,MatriceSize mat_size);

char * CombineStrings(char *a, char *b);

char* IntToChar(int i);



#endif //APPRENTISSAGECNN_MATRICE_H
