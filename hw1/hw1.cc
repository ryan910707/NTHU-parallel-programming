#include <mpi.h>

#include <algorithm>
#include <boost/sort/spreadsort/spreadsort.hpp>
#include <cmath>
#include <compare>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <iostream>
#include <string>

void merge_array_left(float* &my, float* received, float* &ans, int my_size, int other_size){
    int l1, r1 ,ans_num;
    r1 = ans_num = my_size-1;
    l1 = other_size-1;
    while(r1>=0 && l1>=0 && ans_num>=0 ){
        if(my[r1]<=received[l1] ){
            ans[ans_num--] = received[l1--];
        }
        else {
            ans[ans_num--] = my[r1--];
        }
    }
    while(ans_num>=0 && l1>=0){
        ans[ans_num--] = my[r1--];
    }
    while(ans_num>=0 && r1>=0){
        ans[ans_num--] = received[l1--];
    }
    std::swap(my, ans);
}

void merge_array_right(float* &my, float* received, float* &ans, int my_size, int other_size){
    int l1, r1 ,ans_num;
    l1 = r1 = ans_num = 0;
    while(ans_num<my_size && l1<my_size && r1 <other_size ){
        if(my[l1]<=received[r1] ){
            ans[ans_num++] = my[l1++];
        }
        else {
            ans[ans_num++] = received[r1++];
        }
    }
    while(ans_num<my_size && l1<my_size){
        ans[ans_num++] = my[l1++];
    }
    while(ans_num<my_size && r1<other_size){
        ans[ans_num++] = received[r1++];
    }
    std::swap(my, ans);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int array_size = std::stoi(argv[1]);
    char *input_filename = argv[2];
    char *output_filename = argv[3];

    int rank_size=0, rest_size;
    int left_neighbor=-1, right_neighbor=-1;
    int left_count=0, right_count=0;
    int start;
    int round  = std::min(array_size, world_size);

    MPI_File input_file, output_file;

    if(world_size>=array_size){
        // n: 20, P; 50
        // count rank_size
        if(rank<array_size){
            rank_size++;
        }
        //count neighbor
        if(rank==0||rank>=array_size) left_neighbor = -1;
        else left_neighbor = rank-1;
        if(rank>=array_size-1) right_neighbor = -1;
        else right_neighbor = rank+1;
        
        left_count = right_count = rank_size;
        if(left_neighbor==-1) {
            left_count=-1;
        }
        if(right_neighbor==-1){
            right_count=-1;
        }
        start = rank_size*rank;
    }   
    else {
        // n:5 p:2
        rank_size = array_size/world_size;
        rest_size = array_size%world_size;
        if(rest_size!=0){
            if(rank<rest_size){
                rank_size++;
                start = rank_size*rank;
            }
            else {
                start  = (rank_size+1)*rest_size + (rank-rest_size)*rank_size;
            }
        }
        else start  = rank_size*rank;
        //count neighbor
        if(rank==0){
            left_neighbor=-1;
        }
        else {
            left_neighbor = rank-1;
        }
        if(rank==world_size-1){
            right_neighbor=-1;
        }
        else {
            right_neighbor=rank+1;
        }
        //get countd
        if(rest_size!=0 && rank==rest_size){
            left_count = rank_size+1;
            right_count = rank_size;
        }
        else if(rest_size!=0 && rank==rest_size-1){
            left_count = rank_size;
            right_count = rank_size-1;
        }
        else {
            left_count = right_count = rank_size;
        }
        if(left_neighbor==-1) {
            left_count=-1;
        }
        if(right_neighbor==-1){
            right_count=-1;
        }
    }
    if(rank_size == 0){
        left_neighbor = right_neighbor = left_count = right_count = -1;
    }
    float *my_array = new float[rank_size];
    float *received_array = new float[rank_size+1];
    float *result_array = new float[rank_size];
    
    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    if(rank_size!=0) {
        // printf("start reading %d %d %d %d %d %d %d\n", rank, start, rank_size, left_count, right_count, left_neighbor, right_neighbor);
        MPI_File_read_at(input_file, sizeof(float) *start, my_array, rank_size, MPI_FLOAT, MPI_STATUS_IGNORE);
        // for(int i=0;i<rank_size;i++){
        //     printf("%f ", my_array[i]);
        // }
        // printf("initial \n ");
    }
    MPI_File_close(&input_file);
    //sort local
    if(rank_size!=0){
        boost::sort::spreadsort::spreadsort(my_array, my_array+rank_size);
    }

    int all_sort;
    int sort = 0;
    float received_num;
    while(all_sort < world_size){
        sort = 1;
        //odd round
        if(rank&1){
            if(right_neighbor!=-1){
                MPI_Sendrecv(my_array+rank_size-1, 1, MPI_FLOAT, right_neighbor, 0, &received_num, 1, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(received_num<my_array[rank_size-1]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, right_neighbor, 0, received_array, right_count, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_array_right(my_array, received_array, result_array, rank_size, right_count);
                }
            }
        }
        else {
            if(left_neighbor!=-1){
                MPI_Sendrecv(my_array, 1, MPI_FLOAT, left_neighbor, 0, &received_num, 1, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(received_num>my_array[0]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, left_neighbor, 0, received_array, left_count, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_array_left(my_array, received_array, result_array, rank_size, left_count);
                }
            }
        }
        //even round
        if(rank&1){
            if(left_neighbor!=-1){
                MPI_Sendrecv(my_array, 1, MPI_FLOAT, left_neighbor, 0, &received_num, 1, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(received_num>my_array[0]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, left_neighbor, 0, received_array, left_count, MPI_FLOAT, left_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_array_left(my_array, received_array, result_array, rank_size, left_count);
                }
            }
        }
        else {
            if(right_neighbor!=-1){
                MPI_Sendrecv(my_array+rank_size-1, 1, MPI_FLOAT, right_neighbor, 0, &received_num, 1, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(received_num<my_array[rank_size-1]){
                    sort = 0;
                    MPI_Sendrecv(my_array, rank_size, MPI_FLOAT, right_neighbor, 0, received_array, right_count, MPI_FLOAT, right_neighbor, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    merge_array_right(my_array, received_array, result_array, rank_size, right_count);
                }
            }
        }
        MPI_Allreduce(&sort, &all_sort, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    }
    
    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    if (rank_size != 0) {
        // for(int i=0;i<rank_size;i++){
        //     printf("%f", my_array[i]);
        // }
        // printf("output from %d\n", rank);
        MPI_File_write_at(output_file, sizeof(float) * start, my_array, rank_size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&output_file);
    delete []my_array;
    delete []received_array;
    delete []result_array;

    MPI_Finalize();
    return 0;
}


