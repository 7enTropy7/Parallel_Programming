// Matrix Multiplication
#include <stdio.h>
#include <stdlib.h>

__global__ void matrixMul(int* m, int* n,int* p,int size)
{
    //Calculate ow and Column
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int p_sum = 0;
    for(int i=0;i<size;i++)
    {
        p_sum += m[row * size + i] * n[i * size + column];
    }
    p[row * size + column] = p_sum;

}

int main()
{
    int n = 1<<10; //1024 or 2^10

    //Host matrix m,n,p
    int* h_m;
    int* h_n;
    int* h_p;

    //Device matrix m,n,p
    int* d_m;
    int* d_n;
    int* d_p;

    size_t bytes = n * n * sizeof(int);
    
    //Allocating memory on Host side
    h_m = (int*)malloc(bytes);
    h_n = (int*)malloc(bytes);
    h_p = (int*)malloc(bytes);

    //Initialize matrix m,n,p
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            h_m[i*n+j]=rand()%1024;
            h_n[i*n+j]=rand()%1024;
        }
    }

    //Allocating memory on Device side
    cudaMalloc(&d_m, bytes);
    cudaMalloc(&d_n, bytes);
    cudaMalloc(&d_p, bytes);

    //Copy data from Host to the Device
    cudaMemcpy(d_m, h_m, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_n, h_n, bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 16;
    dim3 block_size(threads_per_block, threads_per_block);
    dim3 grid_size(n / block_size.x, n / block_size.y);

    matrixMul <<< grid_size, block_size >>>(d_m,d_n,d_p,n);

    cudaMemcpy(h_p, d_p, bytes, cudaMemcpyDeviceToHost);
    
    printf("Completed Successfully!\n");

    //Clean-Up
    free(h_m);
    free(h_n);
    free(h_p);

    cudaFree(d_m);
    cudaFree(d_n);
    cudaFree(d_p);

    return 0;
}