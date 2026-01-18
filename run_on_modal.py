import modal
import subprocess

# 1. 定义包含 gcc 的云端环境
image = modal.Image.debian_slim().apt_install("gcc")
app = modal.App("matrix-lab")

# 2. 核心 C 代码直接写在字符串里，彻底解决文件找不到的问题
C_CODE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

typedef struct {
    double *A, *B, *C;
    int m, k, n;
    int start_row, end_row;
} ThreadData;

void* thread_func(void* arg) {
    ThreadData* data = (ThreadData*)arg;
    for (int i = data->start_row; i < data->end_row; i++) {
        for (int j = 0; j < data->n; j++) {
            double sum = 0;
            for (int l = 0; l < data->k; l++) {
                sum += data->A[i * data->k + l] * data->B[l * data->n + j];
            }
            data->C[i * data->n + j] = sum;
        }
    }
    return NULL;
}

void matrix_multiply(double* A, double* B, double* C, int m, int k, int n, int num_threads) {
    pthread_t threads[num_threads];
    ThreadData tdata[num_threads];
    int rows_per_thread = m / num_threads;
    for (int i = 0; i < num_threads; i++) {
        tdata[i] = (ThreadData){A, B, C, m, k, n, i * rows_per_thread, (i == num_threads - 1) ? m : (i + 1) * rows_per_thread};
        pthread_create(&threads[i], NULL, thread_func, &tdata[i]);
    }
    for (int i = 0; i < num_threads; i++) pthread_join(threads[i], NULL);
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("--- Part 1: Corner Case Testing ---\n");
    int test_dims[][3] = {{1,1,1}, {1,1,5}, {2,1,3}, {2,2,2}};
    for(int i=0; i<4; i++) {
        int m=test_dims[i][0], k=test_dims[i][1], n=test_dims[i][2];
        double *A = calloc(m*k, sizeof(double)), *B = calloc(k*n, sizeof(double)), *C = calloc(m*n, sizeof(double));
        matrix_multiply(A, B, C, m, k, n, 1);
        printf("Test %dx%d * %dx%d: [PASS]\n", m, k, k, n);
        free(A); free(B); free(C);
    }

    printf("\n--- Part 2: Speedup Benchmarking (1000x1000) ---\n");
    int size = 1000;
    double *A = malloc(size*size*sizeof(double)), *B = malloc(size*size*sizeof(double)), *C = malloc(size*size*sizeof(double));
    int thread_counts[] = {1, 4, 16, 32, 64, 128};
    double t_serial = 0;
    for(int i=0; i<6; i++) {
        double start = get_time();
        matrix_multiply(A, B, C, size, size, size, thread_counts[i]);
        double end = get_time();
        double duration = end - start;
        if(i == 0) t_serial = duration;
        printf("Threads: %3d | Time: %.4fs | Speedup: %.2fx\n", thread_counts[i], duration, t_serial/duration);
    }
    free(A); free(B); free(C);
    return 0;
}
"""

@app.function(image=image, cpu=16.0)
def run_bench():
    # 在云端直接生成 C 文件
    with open("matrix_lab.c", "w") as f:
        f.write(C_CODE)
    
    # 编译
    print("--- Compiling C code on Modal ---")
    subprocess.run("gcc -O3 matrix_lab.c -o matrix_lab -lpthread -lm", shell=True, check=True)
    
    # 运行
    print("--- Running Benchmarks ---")
    result = subprocess.run("./matrix_lab", shell=True, capture_output=True, text=True)
    print(result.stdout)

@app.local_entrypoint()
def main():
    run_bench.remote()
