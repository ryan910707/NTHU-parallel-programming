#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
// #include <sched.h>
// #include <assert.h>
#include <png.h>
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <emmintrin.h>

int num_threads;
int odd;
int* image;
int width;
int height;
int iters;
double left;
double right;
double lower;
double upper;
double y_scale, x_scale;

void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
    FILE* fp = fopen(filename, "wb");
    // assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    // assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    // assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}


void* mandelbrot(void* arg) {
    // struct timespec start, end, temp;
    // double time_used;   
    // clock_gettime(CLOCK_MONOTONIC, &start); 
    int id = *(int*)arg;
    __m128d v_2 = _mm_set_pd1(2);
	__m128d v_4 = _mm_set_pd1(4);

    for (int j = id; j < height; j += num_threads) {
        double y0 = j * y_scale + lower;
        __m128d v_y0 = _mm_load1_pd(&y0);
        for (int i = 0; i < width; ++i) {
            if(i+1<width){
                double x0[2] = {i * x_scale + left, (i + 1) * x_scale + left};
                __m128d v_x0 = _mm_load_pd(x0);
                __m128d v_x = _mm_setzero_pd();
                __m128d v_y = _mm_setzero_pd();
                __m128d v_sq_x = _mm_setzero_pd();
                __m128d v_sq_y = _mm_setzero_pd();
                __m128i v_repeat = _mm_setzero_si128();
                __m128d v_length_squared = _mm_setzero_pd();
                int repeats = 0;
                while(repeats < iters){
                    __m128d v_cmp = _mm_cmpgt_pd(v_4, v_length_squared);
                    //if two > 4 break
                    if (_mm_movemask_pd(v_cmp) == 0)
                        break;
                    repeats++;
                    __m128d temp = _mm_add_pd(_mm_sub_pd(v_sq_x, v_sq_y), v_x0);
                    v_y = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(v_x, v_y), v_2), v_y0);
                    v_x = temp;
                    v_sq_x = _mm_mul_pd(v_x,v_x);
                    v_sq_y = _mm_mul_pd(v_y, v_y);
                    v_length_squared = _mm_or_pd(_mm_andnot_pd(v_cmp, v_length_squared), _mm_and_pd(v_cmp, _mm_add_pd(v_sq_x, v_sq_y)));
                    v_repeat = _mm_add_epi64(v_repeat, _mm_srli_epi64(_mm_castpd_si128(v_cmp), 63));
                }
                _mm_storel_epi64((__m128i*)(image + j*width+i), _mm_shuffle_epi32(v_repeat, 0b01000));
                i++;
            }
            else {
                double x0 = i * x_scale + left;
                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iters && length_squared < 4) {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                image[j * width + i] = repeats;
            }

        }
    }

    // clock_gettime(CLOCK_MONOTONIC, &end);
    // if ((end.tv_nsec - start.tv_nsec) < 0) {
    //     temp.tv_sec = end.tv_sec-start.tv_sec-1;
    //     temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
    // } else {
    //     temp.tv_sec = end.tv_sec - start.tv_sec;
    //     temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    // }
    // time_used = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;

    // printf("%f second\n", time_used); 

    pthread_exit(NULL);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    /* argument parsing */
    // assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = (int*)malloc(width * height * sizeof(int));
    // assert(image);

    pthread_t threads[num_threads];
    int thread_ids[num_threads];

    y_scale = ((upper - lower) / height);
    x_scale = ((right - left) / width);

    if(width&1){
        odd = 1;
    }

    
    
    for (int i = 0; i < num_threads; i++) {
        thread_ids[i] = i;
        pthread_create(&threads[i], NULL, mandelbrot, &thread_ids[i]);
    }

    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }
    
    /* draw and cleanup */
    write_png(filename, iters, width, height, image);
    free(image);

    return 0;
}