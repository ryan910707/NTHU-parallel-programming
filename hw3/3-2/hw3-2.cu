#include <stdio.h>
#include <algorithm>
#include <string.h>
using namespace std;

#define INF 1073741823
#define block_size  72
#define thread_size  24

int V,E, pad_V;
int *matrix;

void read_input(char* input_file){
    FILE* file = fopen(input_file, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);
    pad_V = (V%block_size==0)?V:(V/block_size+1)*block_size;
    matrix = (int*)malloc(sizeof(int)*pad_V*pad_V);
    for(int i=0;i<pad_V;i++){
        for(int j=0;j<pad_V;j++){
            if(i==j){
                matrix[i*pad_V+j]=0;
            }
            else {
                matrix[i*pad_V+j]=INF;
            }
        }
    }
    for(int i=0;i<E;i++){
        int tmp[3];
        fread(tmp, sizeof(int), 3, file);
        matrix[tmp[0]*pad_V+tmp[1]]=tmp[2];
    }
    fclose(file);
}

void output(char* output_file){
    FILE* file = fopen(output_file, "w");
    for(int i=0;i<V;i++){
        fwrite(matrix+(i*pad_V), sizeof(int),  V, file);
    }
	fclose(file);
}

__global__ void floyd_phase1(int* device_matrix, int pad_V, int round){
    /*calculate x, y*/
    int x = threadIdx.x;  //0~23
    int y = threadIdx.y;

    /*copy device matrix needed to cache*/
    __shared__ int share_matrix[block_size][block_size];
    
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            share_matrix[3*y+i][3*x+j] = device_matrix[(round*block_size+3*y+i)*pad_V+(round*block_size+3*x+j)];
        }
    }
    __syncthreads();

    /*calculation*/
    for(int k=0;k<block_size;k++){
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                share_matrix[3*y+i][3*x+j] = min(share_matrix[3*y+i][3*x+j], share_matrix[3*y+i][k]+share_matrix[k][3*x+j]);
            }
        }
        __syncthreads();
    }

    /*write back*/
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            device_matrix[(round*block_size+3*y+i)*pad_V+(round*block_size+3*x+j)] = share_matrix[3*y+i][3*x+j];
        }
    }
}

__global__ void floyd_phase2(int* device_matrix, int pad_V, int round, int num_block){
    if(blockIdx.x==round){
        return;
    }
    /*calculate x, y*/
    int x = threadIdx.x;  //0~23
    int y = threadIdx.y;
    int start_x, start_y;

    //TODO:change to not use branch
    if(blockIdx.y==0){    //0~n-1, do col
        start_x = round*block_size;
        start_y = blockIdx.x*block_size;
    }
    else{  //n~2n-1, do row
        start_x = blockIdx.x*block_size;
        start_y = round*block_size;
    }

    /*copy device matrix needed to cache*/
    __shared__ int share_my[block_size][block_size];
    __shared__ int share_phase1[block_size][block_size];

    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            share_my[3*y+i][3*x+j] = device_matrix[(start_y+3*y+i)*pad_V+(start_x+3*x+j)];
        }
    }
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            share_phase1[3*y+i][3*x+j] = device_matrix[(round*block_size+3*y+i)*pad_V+(round*block_size+3*x+j)];
        }
    }
    __syncthreads();

    /*calculation*/
    for(int k=0;k<block_size;k++){
        for(int i=0;i<3;i++){
            for(int j=0;j<3;j++){
                share_my[3*y+i][3*x+j] = min(share_my[3*y+i][3*x+j], share_phase1[3*y+i][k]+share_my[k][3*x+j]);
            }
        }
        __syncthreads();
    }

    /*write back*/
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){
            device_matrix[(start_y+3*y+i)*pad_V+(start_x+3*x+j)] = share_my[3*y+i][3*x+j];
        }
    }
}

__global__ void floyd_phase3(int* device_matrix, int pad_V, int round){
    if(blockIdx.x==round||blockIdx.y==round){
        return;
    }
    /*calculate x, y*/
    int x = threadIdx.x;
    int y = threadIdx.y;
    int start_x = blockIdx.x*block_size;
    int start_y = blockIdx.y*block_size;

    int ans1 = device_matrix[(start_y+3*y)*pad_V+(start_x+3*x)];
    int ans2 = device_matrix[(start_y+3*y)*pad_V+(start_x+3*x+1)];
    int ans3 = device_matrix[(start_y+3*y)*pad_V+(start_x+3*x+2)];
    int ans4 = device_matrix[(start_y+3*y+1)*pad_V+(start_x+3*x)];
    int ans5 = device_matrix[(start_y+3*y+1)*pad_V+(start_x+3*x+1)];
    int ans6 = device_matrix[(start_y+3*y+1)*pad_V+(start_x+3*x+2)];
    int ans7 = device_matrix[(start_y+3*y+2)*pad_V+(start_x+3*x)];
    int ans8 = device_matrix[(start_y+3*y+2)*pad_V+(start_x+3*x+1)];
    int ans9 = device_matrix[(start_y+3*y+2)*pad_V+(start_x+3*x+2)];

    /*copy device matrix needed to cache*/
    __shared__ int share_row[block_size][block_size];
    __shared__ int share_col[block_size][block_size];
    for(int i=0;i<3;i++){
        for(int j=0;j<3;j++){ 
            share_row[3*y+i][3*x+j] = device_matrix[(start_y+3*y+i)*pad_V+(round*block_size+3*x+j)];   //load data with same row
            share_col[3*y+i][3*x+j] = device_matrix[(round*block_size+3*y+i)*pad_V+(start_x+3*x+j)];    //same col
        }
    }
    __syncthreads();

    /*calculation*/
    for(int k=0;k<block_size;k++){
        ans1 = min(ans1, share_col[3*y][k]+share_row[k][3*x]);
        ans2 = min(ans2, share_col[3*y][k]+share_row[k][3*x+1]);    
        ans3 = min(ans3, share_col[3*y][k]+share_row[k][3*x+2]);
        ans4 = min(ans4, share_col[3*y+1][k]+share_row[k][3*x]);
        ans5 = min(ans5, share_col[3*y+1][k]+share_row[k][3*x+1]);
        ans6 = min(ans6, share_col[3*y+1][k]+share_row[k][3*x+2]);
        ans7 = min(ans7, share_col[3*y+2][k]+share_row[k][3*x]);
        ans8 = min(ans8, share_col[3*y+2][k]+share_row[k][3*x+1]);
        ans9 = min(ans9, share_col[3*y+2][k]+share_row[k][3*x+2]);
    }

    /*write back*/
    device_matrix[(start_y+3*y)*pad_V+(start_x+3*x)] = ans1;
    device_matrix[(start_y+3*y)*pad_V+(start_x+3*x+1)] = ans2;
    device_matrix[(start_y+3*y)*pad_V+(start_x+3*x+2)] = ans3;
    device_matrix[(start_y+3*y+1)*pad_V+(start_x+3*x)] = ans4;
    device_matrix[(start_y+3*y+1)*pad_V+(start_x+3*x+1)] = ans5;
    device_matrix[(start_y+3*y+1)*pad_V+(start_x+3*x+2)] = ans6;
    device_matrix[(start_y+3*y+2)*pad_V+(start_x+3*x)] = ans7;
    device_matrix[(start_y+3*y+2)*pad_V+(start_x+3*x+1)] = ans8;
    device_matrix[(start_y+3*y+2)*pad_V+(start_x+3*x+2)] = ans9;
}

int main(int argc, char** argv){
    char* input_file = argv[1];
    char* output_file = argv[2];
    read_input(input_file);
    int* device_matrix;
    //pin
    cudaHostRegister(matrix, sizeof(int)*pad_V*pad_V, cudaHostRegisterDefault);
    cudaMalloc(&device_matrix, sizeof(int)*pad_V*pad_V);
    cudaMemcpy(device_matrix, matrix, sizeof(int)*pad_V*pad_V, cudaMemcpyHostToDevice);

    int num_block = pad_V/block_size;
    dim3 phase1(1,1);
    dim3 phase2(num_block, 2);
    dim3 blockPerGrid(num_block, num_block);
    dim3 threadPerBlock(thread_size, thread_size);

    printf("pad from %d to %d\n", V, pad_V);
    int total_round = num_block;
    for(int round=0;round<total_round;round++){
        floyd_phase1<<<phase1, threadPerBlock>>>(device_matrix, pad_V, round);
        floyd_phase2<<<phase2, threadPerBlock>>>(device_matrix, pad_V, round, num_block);
        floyd_phase3<<<blockPerGrid, threadPerBlock>>>(device_matrix, pad_V, round);
    }
    cudaMemcpy(matrix, device_matrix, sizeof(int)*pad_V*pad_V, cudaMemcpyDeviceToHost);
    cudaFree(device_matrix);
    output(output_file);
    free(matrix);  
}