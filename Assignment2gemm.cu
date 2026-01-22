#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <vector>

// ==================== Part 1: Basic GEMM Implementation ====================

/**
 * Basic GEMM kernel: C ← α·A·B + β·C
 * A: m × k matrix
 * B: k × n matrix
 * C: m × n matrix (updated in-place)
 */
__global__ void gemm_basic(
    int m, int n, int k,
    float alpha, const float* A, const float* B,
    float beta, float* C
) {
    // Calculate the matrix position (row, col) for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < m && col < n) {
        float sum = 0.0f;
        
        // Compute dot product for matrix multiplication
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        
        // In-place update: C ← α·A·B + β·C
        int idx = row * n + col;
        C[idx] = alpha * sum + beta * C[idx];
    }
}

// ==================== Part 2: Extended GEMM Implementation ====================

/**
 * Extended GEMM kernel: C ← α·op(A)·op(B) + β·C
 * Supports optional transposition of A and/or B
 * Updates C in-place
 * 
 * @param m Rows of C, also rows of A if !transposeA, else columns of A
 * @param n Columns of C, also columns of B if !transposeB, else rows of B
 * @param k Inner dimension, columns of A if !transposeA, else rows of A
 * @param alpha Scalar coefficient
 * @param A Input matrix A
 * @param transposeA Whether to transpose A
 * @param B Input matrix B
 * @param transposeB Whether to transpose B
 * @param beta Scalar coefficient
 * @param C Input/output matrix C (updated in-place)
 */
__global__ void gemm_extended(
    int m, int n, int k,
    float alpha, const float* A, bool transposeA,
    const float* B, bool transposeB,
    float beta, float* C
) {
    // Calculate the matrix position (row, col) for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (row < m && col < n) {
        float sum = 0.0f;
        
        // Compute dot product based on transpose flags
        for (int i = 0; i < k; i++) {
            // Get A element index
            int a_idx;
            if (!transposeA) {
                // A not transposed: A is m×k, access A[row, i]
                a_idx = row * k + i;
            } else {
                // A transposed: Aᵀ is m×k, original A is k×m, access A[i, row]
                a_idx = i * m + row;
            }
            
            // Get B element index
            int b_idx;
            if (!transposeB) {
                // B not transposed: B is k×n, access B[i, col]
                b_idx = i * n + col;
            } else {
                // B transposed: Bᵀ is k×n, original B is n×k, access B[col, i]
                b_idx = col * k + i;
            }
            
            sum += A[a_idx] * B[b_idx];
        }
        
        // In-place update: C ← α·op(A)·op(B) + β·C
        int c_idx = row * n + col;
        C[c_idx] = alpha * sum + beta * C[c_idx];
    }
}

// ==================== Helper Functions ====================

/**
 * CPU reference implementation for verification
 */
void cpu_gemm(
    int m, int n, int k,
    float alpha, const float* A, bool transposeA,
    const float* B, bool transposeB,
    float beta, const float* C_in, float* C_out
) {
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            float sum = 0.0f;
            
            for (int i = 0; i < k; i++) {
                // Calculate A index
                int a_idx;
                if (!transposeA) {
                    a_idx = row * k + i;
                } else {
                    a_idx = i * m + row;
                }
                
                // Calculate B index
                int b_idx;
                if (!transposeB) {
                    b_idx = i * n + col;
                } else {
                    b_idx = col * k + i;
                }
                
                sum += A[a_idx] * B[b_idx];
            }
            
            int idx = row * n + col;
            C_out[idx] = alpha * sum + beta * C_in[idx];
        }
    }
}

/**
 * Compare two float matrices (with tolerance)
 */
bool compare_matrices(const float* mat1, const float* mat2, int size, float epsilon = 1e-5f) {
    for (int i = 0; i < size; i++) {
        if (fabs(mat1[i] - mat2[i]) > epsilon) {
            printf("Mismatch at index %d: %f vs %f (diff: %f)\n", 
                   i, mat1[i], mat2[i], fabs(mat1[i] - mat2[i]));
            return false;
        }
    }
    return true;
}

/**
 * Initialize matrix with random values
 */
void init_matrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f; // range [-1, 1]
    }
}

/**
 * Print matrix (for debugging)
 */
void print_matrix(const float* matrix, int rows, int cols, const char* name) {
    printf("%s (%d x %d):\n", name, rows, cols);
    for (int i = 0; i < std::min(rows, 5); i++) {
        for (int j = 0; j < std::min(cols, 5); j++) {
            printf("%8.4f ", matrix[i * cols + j]);
        }
        if (cols > 5) printf("...");
        printf("\n");
    }
    if (rows > 5) printf("...\n");
    printf("\n");
}

// ==================== Main Test Function ====================

int main() {
    // Set random seed
    srand(42);
    
    // Test parameters
    const int m = 256;
    const int n = 128;
    const int k = 64;
    
    const float alpha = 2.0f;
    const float beta = 0.5f;
    
    // Allocate host memory
    size_t a_size = m * k * sizeof(float);
    size_t b_size = k * n * sizeof(float);
    size_t c_size = m * n * sizeof(float);
    
    float* h_A = (float*)malloc(a_size);
    float* h_B = (float*)malloc(b_size);
    float* h_C = (float*)malloc(c_size);
    float* h_C_gpu = (float*)malloc(c_size);
    float* h_C_cpu = (float*)malloc(c_size);
    float* h_C_original = (float*)malloc(c_size);
    
    // Initialize matrices
    init_matrix(h_A, m, k);
    init_matrix(h_B, k, n);
    init_matrix(h_C, m, n);
    memcpy(h_C_original, h_C, c_size);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, a_size);
    cudaMalloc(&d_B, b_size);
    cudaMalloc(&d_C, c_size);
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, a_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice);
    
    // ========== Test 1: Basic GEMM Implementation ==========
    printf("========== Test 1: Basic GEMM (C ← α·A·B + β·C) ==========\n");
    
    // Set thread block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                  (m + blockSize.y - 1) / blockSize.y);
    
    // Launch kernel
    gemm_basic<<<gridSize, blockSize>>>(
        m, n, k, alpha, d_A, d_B, beta, d_C
    );
    cudaDeviceSynchronize();
    
    // Check CUDA error
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (basic): %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    // Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);
    
    // Compute reference on CPU
    cpu_gemm(m, n, k, alpha, h_A, false, h_B, false, beta, h_C_original, h_C_cpu);
    
    // Compare results
    bool basic_correct = compare_matrices(h_C_gpu, h_C_cpu, m * n);
    printf("Basic GEMM Test: %s\n", basic_correct ? "PASS ✓" : "FAIL ✗");
    
    // ========== Test 2: Extended GEMM (No transpose) ==========
    printf("\n========== Test 2: Extended GEMM (No transpose) ==========\n");
    
    // Restore original C
    memcpy(h_C, h_C_original, c_size);
    cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice);
    
    // Launch extended kernel (no transpose)
    gemm_extended<<<gridSize, blockSize>>>(
        m, n, k, alpha, d_A, false, d_B, false, beta, d_C
    );
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (extended no-transpose): %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);
    
    bool extended_no_transpose_correct = compare_matrices(h_C_gpu, h_C_cpu, m * n);
    printf("Extended GEMM No Transpose Test: %s\n", 
           extended_no_transpose_correct ? "PASS ✓" : "FAIL ✗");
    
    // ========== Test 3: Extended GEMM (Transpose A) ==========
    printf("\n========== Test 3: Extended GEMM (Transpose A) ==========\n");
    
    // Restore original C
    memcpy(h_C, h_C_original, c_size);
    cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice);
    
    // When A is transposed: Aᵀ dimensions are k×m
    int m_Atrans = k;  // rows of Aᵀ = columns of A
    int n_Atrans = n;  // unchanged
    int k_Atrans = m;  // columns of Aᵀ = rows of A
    
    dim3 gridSize_Atrans((n_Atrans + blockSize.x - 1) / blockSize.x,
                         (m_Atrans + blockSize.y - 1) / blockSize.y);
    
    gemm_extended<<<gridSize_Atrans, blockSize>>>(
        m_Atrans, n_Atrans, k_Atrans,
        alpha, d_A, true, d_B, false, beta, d_C
    );
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (extended A-transpose): %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);
    
    // Compute reference on CPU
    cpu_gemm(m_Atrans, n_Atrans, k_Atrans,
             alpha, h_A, true, h_B, false,
             beta, h_C_original, h_C_cpu);
    
    bool Atrans_correct = compare_matrices(h_C_gpu, h_C_cpu, m_Atrans * n_Atrans);
    printf("Extended GEMM Transpose A Test: %s\n", Atrans_correct ? "PASS ✓" : "FAIL ✗");
    
    // ========== Test 4: Extended GEMM (Transpose B) ==========
    printf("\n========== Test 4: Extended GEMM (Transpose B) ==========\n");
    
    // Restore original C
    memcpy(h_C, h_C_original, c_size);
    cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice);
    
    // When B is transposed: Bᵀ dimensions are n×k
    int m_Btrans = m;  // unchanged
    int n_Btrans = k;  // columns of Bᵀ = rows of B
    int k_Btrans = n;  // rows of Bᵀ = columns of B
    
    dim3 gridSize_Btrans((n_Btrans + blockSize.x - 1) / blockSize.x,
                         (m_Btrans + blockSize.y - 1) / blockSize.y);
    
    gemm_extended<<<gridSize_Btrans, blockSize>>>(
        m_Btrans, n_Btrans, k_Btrans,
        alpha, d_A, false, d_B, true, beta, d_C
    );
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (extended B-transpose): %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);
    
    // Compute reference on CPU
    cpu_gemm(m_Btrans, n_Btrans, k_Btrans,
             alpha, h_A, false, h_B, true,
             beta, h_C_original, h_C_cpu);
    
    bool Btrans_correct = compare_matrices(h_C_gpu, h_C_cpu, m_Btrans * n_Btrans);
    printf("Extended GEMM Transpose B Test: %s\n", Btrans_correct ? "PASS ✓" : "FAIL ✗");
    
    // ========== Test 5: Extended GEMM (Transpose both A and B) ==========
    printf("\n========== Test 5: Extended GEMM (Transpose both A and B) ==========\n");
    
    // Restore original C
    memcpy(h_C, h_C_original, c_size);
    cudaMemcpy(d_C, h_C, c_size, cudaMemcpyHostToDevice);
    
    // When both A and B are transposed
    // Aᵀ: k×m, Bᵀ: n×k
    int m_ABtrans = k;  // rows of Aᵀ = columns of A
    int n_ABtrans = n;  // unchanged
    int k_ABtrans = m;  // columns of Aᵀ = rows of A
    
    dim3 gridSize_ABtrans((n_ABtrans + blockSize.x - 1) / blockSize.x,
                          (m_ABtrans + blockSize.y - 1) / blockSize.y);
    
    gemm_extended<<<gridSize_ABtrans, blockSize>>>(
        m_ABtrans, n_ABtrans, k_ABtrans,
        alpha, d_A, true, d_B, true, beta, d_C
    );
    cudaDeviceSynchronize();
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error (extended AB-transpose): %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaMemcpy(h_C_gpu, d_C, c_size, cudaMemcpyDeviceToHost);
    
    // Compute reference on CPU
    cpu_gemm(m_ABtrans, n_ABtrans, k_ABtrans,
             alpha, h_A, true, h_B, true,
             beta, h_C_original, h_C_cpu);
    
    bool ABtrans_correct = compare_matrices(h_C_gpu, h_C_cpu, m_ABtrans * n_ABtrans);
    printf("Extended GEMM Transpose A and B Test: %s\n", ABtrans_correct ? "PASS ✓" : "FAIL ✗");
    
    // ========== Summary ==========
    printf("\n========== Test Summary ==========\n");
    printf("Basic GEMM Implementation: %s\n", basic_correct ? "PASS ✓" : "FAIL ✗");
    printf("Extended GEMM No Transpose: %s\n", extended_no_transpose_correct ? "PASS ✓" : "FAIL ✗");
    printf("Extended GEMM Transpose A: %s\n", Atrans_correct ? "PASS ✓" : "FAIL ✗");
    printf("Extended GEMM Transpose B: %s\n", Btrans_correct ? "PASS ✓" : "FAIL ✗");
    printf("Extended GEMM Transpose A and B: %s\n", ABtrans_correct ? "PASS ✓" : "FAIL ✗");
    
    // ========== Performance Test ==========
    printf("\n========== Performance Test ==========\n");
    
    const int big_m = 512;
    const int big_n = 512;
    const int big_k = 512;
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Allocate large matrices
    size_t big_size = big_m * big_n * sizeof(float);
    float* d_big_A, *d_big_B, *d_big_C;
    cudaMalloc(&d_big_A, big_m * big_k * sizeof(float));
    cudaMalloc(&d_big_B, big_k * big_n * sizeof(float));
    cudaMalloc(&d_big_C, big_size);
    
    dim3 bigBlockSize(32, 8);
    dim3 bigGridSize((big_n + bigBlockSize.x - 1) / bigBlockSize.x,
                     (big_m + bigBlockSize.y - 1) / bigBlockSize.y);
    
    // Measure kernel execution time
    cudaEventRecord(start);
    gemm_extended<<<bigGridSize, bigBlockSize>>>(
        big_m, big_n, big_k,
        1.0f, d_big_A, false, d_big_B, false, 0.0f, d_big_C
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Calculate FLOPS
    float flops = 2.0f * big_m * big_n * big_k / (milliseconds / 1000.0f);
    float gflops = flops / 1e9;
    
    printf("Matrix Size: %d x %d x %d\n", big_m, big_n, big_k);
    printf("Execution Time: %.3f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // ========== Cleanup ==========
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_big_A);
    cudaFree(d_big_B);
    cudaFree(d_big_C);
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_gpu);
    free(h_C_cpu);
    free(h_C_original);
    
    printf("\nAll tests completed!\n");
    
    return 0;
}

/*
================================================================================
CUDA GEMM Kernel Implementation
Assignment: Implement a naive GEMM kernel with transposition and in-place updates.

DESCRIPTION:
    This file contains a naive implementation of Generalized Matrix Multiplication.
    - Formula: C ← α·op(A)·op(B) + β·C
    - Supports: Optional transposition of A and B via index remapping.
    - Optimization: In-place updates to matrix C (no extra allocation for D).
    - Verification: CPU reference implementation included for correctness checking.

COMPILATION:
    nvcc -O3 matmul.cu -o gemm

USAGE:
    ./gemm

NEW CONCEPTS IMPLEMENTED (from Week 02 Discussion):
    - 2D Grid/Block layout
    - In-place memory updates
    - CUDA Error handling (cudaGetLastError)
    - Performance profiling with CUDA Events (GFLOPS calculation)
================================================================================
*/