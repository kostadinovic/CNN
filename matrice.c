//
// Created by Nemanja Kostadinovic on 14/11/2020.
//

#include "matrice.h"
#include <stdlib.h>
#include <stdio.h>



// rotate input matrice 180°
float **MatriceRotation180(float **matrice, MatriceSize size_matrice){
    int column;
    int row;
    int outSizeWeight = size_matrice.columns;
    int outSizeHeight = size_matrice.rows;
    float **output = (float**)(malloc(outSizeHeight*sizeof(float*)));

    for(int i=0; i<outSizeHeight; i++){
        output[i] = (float*)(malloc(outSizeWeight*sizeof(float)));
    }

    for(row=0; row<outSizeHeight; row++){
        for(column=0; column<outSizeWeight; column++){
            output[row][column] = matrice[outSizeHeight-row-1][outSizeWeight-column-1];
        }
    }
    return output;
}


// Convolution of matrix with 3 choice :
//         FULL: the size of the output is input_size+(filter_size-1), add of 0 if edge
//         SAME: the output has the same size as the input
//         VALID: the output size is shrink ==> input_size-(filter_size-1)




float** MatConv(float **filter,MatriceSize filter_size,float **input, MatriceSize input_size,int type){
    float** rotated_filter=MatriceRotation180(filter,filter_size);
    float** res=MatCorrelation(rotated_filter,filter_size,input,input_size,type);
    int i;
    for(i=0;i<filter_size.rows;i++)
        free(rotated_filter[i]);
    free(rotated_filter);
    return res;
}

//resize the matrix (unpooling)
float **UpSamplingMatrice(float **matrice, MatriceSize size, int upc, int upr){
    int columns=size.columns;
    int rows=size.rows;

    float **res=(float**)malloc((rows*upr)*sizeof(float*));

    int i;
    for(i=0; i<(rows*upr); i++){
        res[i]=(float*)malloc((columns*upc)*sizeof(float));
    }

    int k,l;
    for(int j=0; j<rows*upr; j=j+upr){
        for(i=0;i<columns*upc;i=i+upc){
            for(k=0; k<upc; k++){
                res[j][i+k]=matrice[j/upr][i/upc];
            }
        }

        for(l=1; l<upr; l++){
            for(i=0;i<columns*upc;i++){
                res[j+l][i]=res[j][i];
            }
        }
    }
    return res;
}

//sum of two matrix
void sumMatrix(float **sum, float **mat1, MatriceSize mat_size1, float **mat2, MatriceSize mat_size2){
    int i;
    int j;

    if(mat_size1.columns!=mat_size2.columns || mat_size1.rows!=mat_size2.rows){
        printf("ERROR: Size is not same!");
    }

    for(i=0; i<mat_size1.rows; i++){
        for(j=0; j<mat_size1.columns; j++){
            sum[i][j]=mat1[i][j]+mat2[i][j];
        }
    }
}

//multiplication of two matrix
void MatMultiScaler(float **mult, float **matrice, MatriceSize mat_size, float scaler){
    int i;
    int j;

    for(i=0; i<mat_size.rows; i++){
        for(j=0;j<mat_size.columns;j++){
            mult[i][j]=matrice[i][j]*scaler;
        }
    }
}


//sum of the matrice
float sumMat(float **mat,MatriceSize mat_size){
    int i;
    int j;
    float sum=0;

    for(i=0; i<mat_size.rows; i++){
        for(j=0; j<mat_size.columns; j++){
            sum=sum+mat[i][j];
        }
    }
    return sum;
}


/* print the matrice for debug!!
void print_matrice(float **mat,MatriceSize mat_size){
    printf(("\n"));
    for(int i=0; i<mat_size.rows; i++){
        for(int j=0; j<mat_size.columns; j++){
            printf(" %f ",mat[i][j]);
            if(j==mat_size.columns-1){
                printf(("\n"));
            }
        }
    }
    printf(("\n"));
}
 */



float **expandMatriceEdge(float **matrice, MatriceSize size_matrice, int ad_columns, int ad_rows){
    int i;
    int j;

    int column= size_matrice.columns;
    int row = size_matrice.rows;
    float **output = (float**)(malloc((row+2*ad_rows)*sizeof(float*)));

    for(i=0; i<(row+2*ad_rows); i++){
        output[i] = (float*)malloc((column+2*ad_columns)*sizeof(float));
    }

    for(j=0;j<row+2*ad_rows;j++){
        for(i=0;i<column+2*ad_columns;i++){
            if(j<ad_rows||i<ad_columns||j>=(row+ad_rows)||i>=(column+ad_columns)){
                output[j][i]=(float)0.0;
            }
            else
                output[j][i]=matrice[j-ad_rows][i-ad_columns];
        }
    }
    return output;
}




//contraire de expand : reduce input matrice edge
float **reduceMatriceEdge(float **matrice,MatriceSize mat_size,int shrinkc,int shrinkr){
    int i,j;
    int c=mat_size.columns;
    int r=mat_size.rows;
    float** res=(float**)malloc((r-2*shrinkr)*sizeof(float*));
    for(i=0;i<(r-2*shrinkr);i++)
        res[i]=(float*)malloc((c-2*shrinkc)*sizeof(float));


    for(j=0;j<r;j++){
        for(i=0;i<c;i++){
            if(j>=shrinkr&&i>=shrinkc&&j<(r-shrinkr)&&i<(c-shrinkc))
                res[j-shrinkr][i-shrinkc]=matrice[j][i];
        }
    }
    return res;
}


float **MatCorrelation(float **filter, MatriceSize filter_size, float **input, MatriceSize input_size, int type){

    //cut the filter
    int half_filter_sizeW; //half width of the filter
    int half_filter_sizeH; //half height of the filter

    if(filter_size.rows%2==0&&filter_size.columns%2==0){ //the filter size is even
        half_filter_sizeW=(filter_size.columns)/2;
        half_filter_sizeH=(filter_size.rows)/2;
    }else{                                      //the filter size is odd => transform to even
        half_filter_sizeW=(filter_size.columns-1)/2;
        half_filter_sizeH=(filter_size.rows-1)/2;
    }

    //by default type==FULL, the size of output is input_size+filter_size-1
    int out_sizeW=input_size.columns+(filter_size.columns-1);
    int out_sizeH=input_size.rows+(filter_size.rows-1);

    float **output=(float**)malloc(out_sizeH*sizeof(float*));
    for(int i=0;i<out_sizeH;i++){
        output[i]=(float*)calloc(out_sizeW,sizeof(float));
    }


    float **expand_input=expandMatriceEdge(input,input_size,filter_size.columns-1,filter_size.rows-1);

    for(int i=0; i<out_sizeH; i++){
        for(int j=0; j<out_sizeW; j++){
            for(int r=0; r<filter_size.rows; r++){
                for(int c=0; c<filter_size.columns; c++){
                    output[i][j]=output[i][j]+filter[r][c]*expand_input[i+r][j+c];
                }
            }
        }

    }
    MatriceSize out_size={out_sizeW,out_sizeH};

    for(int i=0;i<input_size.rows+2*(filter_size.rows-1);i++){
        free(expand_input[i]);
    }
    free(expand_input);


    switch(type){
        case FULL: {
            return output;
        }
        case SAME:{
            float **same_output=reduceMatriceEdge(output,out_size,half_filter_sizeW,half_filter_sizeW);

            for(int i=0; i<out_size.rows; i++){
                free(output[i]);
            }
            free(output);

            return same_output;
        }
        case VALID:{
            float **valid_output;

            if(filter_size.rows%2==0&&filter_size.columns%2==0){
                valid_output=reduceMatriceEdge(output,out_size,half_filter_sizeW*2-1,half_filter_sizeH*2-1);
            }else{
                valid_output=reduceMatriceEdge(output,out_size,half_filter_sizeW*2,half_filter_sizeH*2);
            }

            for(int i=0;i<out_size.rows;i++){
                free(output[i]);
            }
            free(output);
            return valid_output;
        }
        default:
            return output;
    }
}




/*  DEBUG MATRIX

void test_matrice(){
    int i,j;
    MatriceSize size1={12,12};
    MatriceSize filter_size={5,5};
    float** matrice1=(float**)malloc(size1.rows*sizeof(float*));
    for(i=0;i<size1.rows;i++){
        matrice1[i]=(float*)malloc(size1.columns*sizeof(float));
        for(j=0;j<size1.columns;j++){
            matrice1[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
        }
    }
    float** filter1=(float**)malloc(filter_size.rows*sizeof(float*));
    for(i=0;i<filter_size.rows;i++){
        filter1[i]=(float*)malloc(filter_size.columns*sizeof(float));
        for(j=0;j<filter_size.columns;j++){
            filter1[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
        }
    }
    float** filter2=(float**)malloc(filter_size.rows*sizeof(float*));
    for(i=0;i<filter_size.rows;i++){
        filter2[i]=(float*)malloc(filter_size.columns*sizeof(float));
        for(j=0;j<filter_size.columns;j++){
            filter2[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
        }
    }
    float** filter3=(float**)malloc(filter_size.rows*sizeof(float*));
    for(i=0;i<filter_size.rows;i++){
        filter3[i]=(float*)malloc(filter_size.columns*sizeof(float));
        for(j=0;j<filter_size.columns;j++){
            filter3[i][j]=(((float)rand()/(float)RAND_MAX)-0.5)*2;
        }
    }

    float** conv1=MatConv(filter1,filter_size,matrice1,size1,VALID);
    float** conv2=MatConv(filter2,filter_size,matrice1,size1,VALID);
    MatriceSize conv_size={size1.columns-(filter_size.columns-1),size1.rows-(filter_size.rows-1)};
    float** conv3=MatConv(filter3,filter_size,matrice1,size1,VALID);
    sumMatrix(conv1,conv1,conv_size,conv2,conv_size);
    sumMatrix(conv1,conv1,conv_size,conv3,conv_size);

    float** mSM = UpSamplingMatrice(filter1,filter_size,3,3);

    print_matrice(matrice1,size1);
    print_matrice(filter1,filter_size);
    print_matrice(filter2,filter_size);
    print_matrice(filter3,filter_size);
    print_matrice(conv1,conv_size);
    print_matrice(conv2,conv_size);
    print_matrice(conv3,conv_size);
    print_matrice(mSM,filter_size);
    
}
 */

