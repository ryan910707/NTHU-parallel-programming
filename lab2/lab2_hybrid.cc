#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
	int rc;
    rc = MPI_Init (&argc,&argv);
	if (rc != MPI_SUCCESS) {
		printf ("Error starting MPI program. Terminating.\n");
		MPI_Abort (MPI_COMM_WORLD, rc);
	} 

    if (argc != 3) {
        fprintf(stderr, "must provide exactly 2 arguments!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    unsigned long long r = atoll(argv[1]);
    unsigned long long k = atoll(argv[2]);
    unsigned long long pixels = 0;
	unsigned long long r_2 = r*r;
    int num_procs;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    #pragma omp parallel for reduction(+:pixels)
    for (unsigned long long x = my_rank; x < r; x += num_procs) {
        unsigned long long y = ceil(sqrtl(r_2 - x * x));
        pixels += y;
        // pixels %= k;
    }
    unsigned long long total_pixels = 0;
    MPI_Reduce(&pixels, &total_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    total_pixels%=k;
    if (my_rank == 0) {
        printf("%llu\n", (4 * total_pixels) % k);
    }

    MPI_Finalize();
    return 0;
}
