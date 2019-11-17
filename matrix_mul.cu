#include<stdio.h>
#include<stdlib.h>

__global__ void multiply_matrix(int* d_A, int* d_B, int* d_C, int width)
{
    //Calculating the Row and Column
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if(row<width && col<width)
    {
        int product_value = 0;
        for(int k=0;k<width;k++)
        {
            product_value = product_value + d_A[row*width+k] * d_B[k*width+col];
        }
        d_C[row*width+col] = product_value;
    }

}

int main()
{
    //Initializing width of the square matrix 
    int width = 3;

    //Initializing the Host matrices
    int* h_A;
    int* h_B;
    int* h_C;

    //Initializing the Device matrices
    int* d_A;
    int* d_B;
    int* d_C;

    size_t bytes = width*width*sizeof(int);

    //Allocating Memory on Host Side
    h_A = (int*)malloc(bytes);
    h_B = (int*)malloc(bytes);
    h_C = (int*)malloc(bytes);

    //initializing the Host matrices with some values 
    for(int r=0;r<width;r++)
    {
        for(int c=0;c<width;c++)
        {
            h_A[r*width+c] = r*width+c;
            h_B[r*width+c] = r*width+c;
        }
    }

    printf("Matrix A: \n");
    for(int r=0;r<width;r++)
    {
        for(int c=0;c<width;c++)
        {
            printf("%d ",h_A[r*width+c]);
        }
    }

    printf("\nMatrix B: \n");
    for(int r=0;r<width;r++)
    {
        for(int c=0;c<width;c++)
        {
            printf("%d ",h_B[r*width+c]);
        }
    }


    //Allocating memory on Device side
    cudaMalloc(&d_A,bytes);
    cudaMalloc(&d_B,bytes);
    cudaMalloc(&d_C,bytes);

    //Copy data from Host to Device
    cudaMemcpy(d_A,h_A,width,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,width,cudaMemcpyHostToDevice);

    dim3 blockSize(width*width, width*width);
    dim3 gridSize(1);

    multiply_matrix<<<gridSize,blockSize>>>(d_A,d_B,d_C,width);

    //Copy data from Device back to Host
    cudaMemcpy(h_C,d_C,width,cudaMemcpyDeviceToHost);

    printf("\nCompleted Successfully!\n");

    printf("Matrix C: \n");
    for(int r=0;r<width;r++)
    {
        for(int c=0;c<width;c++)
        {
            printf("%d ",h_C[r*width+c]);            
        }
    }

    printf("\n");
    //Memory clean-up Host
    free(h_A);
    free(h_B);
    free(h_C);

    //Memory clean-up Device
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}