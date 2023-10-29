#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>


// Structure to hold thread-specific data
struct ThreadData {
    unsigned long long r;
    unsigned long long k;
    unsigned long long pixels;
    unsigned long long start_x;
    unsigned long long end_x;
};

// Function for each thread to perform the calculation
void* thread_work(void* arg) {
    struct ThreadData* data = (struct ThreadData*)arg;

    unsigned long long r = data->r;
    unsigned long long k = data->k;
    unsigned long long pixels = 0;

    for (unsigned long long x = data->start_x; x < data->end_x; x++) {
        unsigned long long y = ceil(sqrtl(r * r - x * x));
        pixels += y;
    }

    data->pixels = pixels;

    return NULL;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Must provide exactly 2 arguments!\n");
        return 1;
    }
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);
	int NUM_THREADS = (int)ncpus;
    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);

    pthread_t threads[NUM_THREADS];
    struct ThreadData thread_data[NUM_THREADS];
    
    for (int i = 0; i < NUM_THREADS; i++) {
        thread_data[i].r = r;
        thread_data[i].k = k;
        thread_data[i].pixels = 0;
        thread_data[i].start_x = (i * r) / NUM_THREADS;
        thread_data[i].end_x = ((i + 1) * r) / NUM_THREADS;
        pthread_create(&threads[i], NULL, thread_work, &thread_data[i]);
    }

    // Wait for all threads to finish
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    unsigned long long pixels = 0;
    for (int i = 0; i < NUM_THREADS; i++) {
        pixels += thread_data[i].pixels;
		pixels %= k; 
    }

    printf("%llu\n", (4 * pixels) % k);

    return 0;
}
