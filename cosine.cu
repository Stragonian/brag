/*
cosine.cu : This programs preforms the cosine function needed to compute the top-K
entires within an embedded .json file used in the BRAG pipeline.

Created by Jerry B Nettrouer II <j2@inpito.org> https://www.inpito.org/projects.php

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>

NOTE:  I came up with this crazy idea of using Bash & C & the ollama REST_API
as a RAG as an alternative to the python RAGs to learn how RAG's handle data
without all the python dependencies.

This idea and project is still very experimental and I don't know how far I
plan on going with it - it may just depend on how well it works.

While still an amnionic idea and project, I'm attempting to accomplish much
of the same work that I've seen done to create a RAG using python, but instead
my hope is to avoid using python as much as possible, and try to keep the
entire BRAG pipeline and project in Bash, C, and interacting with the Ollama
REST_API, and keep the scripts and C programs about as simple as I possibly can.

NOTICE: This project is in the development stage and I make no guarantees
about its abilities or performance.

REQUIRED: jq, curl, cudatoolkit, Ollama and LLMs that run on Ollama are required
to use this BRAG pipeline.  Make sure to use the same embedding LLM that was
used in the embedding stage of your BRAG as you plan to use within your query
to process top-K entities of the pipeline, otherwise, your results might not
be all that good.

Compute cosine similarity between two vectors (double precision)
Usage: ./cosine -e embed.txt -q query.txt [-n]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// ------------------ GPU Kernel ------------------

__global__ void cosineKernel(const double *a, const double *b, double *partialSum, double *partialA, double *partialB, int n) {
    extern __shared__ double cache[];
    double *cacheSum = cache;
    double *cacheA = cache + blockDim.x;
    double *cacheB = cache + 2 * blockDim.x;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double tempSum = 0.0;
    double tempA = 0.0;
    double tempB = 0.0;

    while (tid < n) {
        double av = a[tid];
        double bv = b[tid];
        tempSum += av * bv;
        tempA += av * av;
        tempB += bv * bv;
        tid += blockDim.x * gridDim.x;
    }

    cacheSum[cacheIndex] = tempSum;
    cacheA[cacheIndex] = tempA;
    cacheB[cacheIndex] = tempB;

    __syncthreads();

    // Parallel reduction
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cacheSum[cacheIndex] += cacheSum[cacheIndex + i];
            cacheA[cacheIndex] += cacheA[cacheIndex + i];
            cacheB[cacheIndex] += cacheB[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (cacheIndex == 0) {
        partialSum[blockIdx.x] = cacheSum[0];
        partialA[blockIdx.x] = cacheA[0];
        partialB[blockIdx.x] = cacheB[0];
    }
}

// ------------------ CPU Function ------------------

double cosineCPU(const double *a, const double *b, int n) {
    double dot = 0.0, magA = 0.0, magB = 0.0;
    for (int i = 0; i < n; i++) {
        dot += a[i] * b[i];
        magA += a[i] * a[i];
        magB += b[i] * b[i];
    }
    if (magA == 0.0 || magB == 0.0)
        return 0.0;
    return dot / (sqrt(magA) * sqrt(magB));
}

// ------------------ Helper: Load Vector ------------------

double *loadVector(const char *filename, int *len) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: could not open %s\n", filename);
        exit(1);
    }

    int capacity = 1024;
    int size = 0;
    double *arr = (double *)malloc(capacity * sizeof(double));
    if (!arr) {
        fprintf(stderr, "Error: memory allocation failed\n");
        exit(1);
    }

    while (fscanf(fp, "%lf", &arr[size]) == 1) {
        size++;
        if (size >= capacity) {
            capacity *= 2;
            arr = (double *)realloc(arr, capacity * sizeof(double));
            if (!arr) {
                fprintf(stderr, "Error: memory reallocation failed\n");
                exit(1);
            }
        }
    }

    fclose(fp);
    *len = size;
    return arr;
}

// ------------------ Main ------------------

int main(int argc, char *argv[]) {
    int useGPU = 0;
    char *embedFile = NULL;
    char *queryFile = NULL;

    // Argument parsing
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0) {
            useGPU = 1;
        } else if (strcmp(argv[i], "-e") == 0 && i + 1 < argc) {
            embedFile = argv[++i];
        } else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc) {
            queryFile = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s -e embed.txt -q query.txt [-n]\n", argv[0]);
            printf("  -e   Embeddings file (required)\n");
            printf("  -q   Query file (required)\n");
            printf("  -n   Use NVIDIA GPU via CUDA (optional)\n");
            return 0;
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s -e embed.txt -q query.txt [-n]\n", argv[0]);
            return 1;
        }
    }

    if (!embedFile || !queryFile) {
        fprintf(stderr, "Error: both -e and -q arguments are required.\n");
        fprintf(stderr, "Usage: %s -e embed.txt -q query.txt [-n]\n", argv[0]);
        return 1;
    }

    int n1, n2;
    double *a = loadVector(embedFile, &n1);
    double *b = loadVector(queryFile, &n2);

    if (n1 != n2) {
        fprintf(stderr, "Error: vector lengths do not match (%d vs %d)\n", n1, n2);
        free(a);
        free(b);
        return 1;
    }

    if (!useGPU) {
        double score = cosineCPU(a, b, n1);
        printf("%.12f\n", score);
    } else {
        // ---------------- GPU Mode ----------------
        int n = n1;
        double *d_a, *d_b, *d_sum, *d_A, *d_B;
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        cudaMalloc((void **)&d_a, n * sizeof(double));
        cudaMalloc((void **)&d_b, n * sizeof(double));
        cudaMalloc((void **)&d_sum, gridSize * sizeof(double));
        cudaMalloc((void **)&d_A, gridSize * sizeof(double));
        cudaMalloc((void **)&d_B, gridSize * sizeof(double));

        cudaMemcpy(d_a, a, n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n * sizeof(double), cudaMemcpyHostToDevice);

        cosineKernel<<<gridSize, blockSize, 3 * blockSize * sizeof(double)>>>(d_a, d_b, d_sum, d_A, d_B, n);

        double *h_sum = (double *)malloc(gridSize * sizeof(double));
        double *h_A = (double *)malloc(gridSize * sizeof(double));
        double *h_B = (double *)malloc(gridSize * sizeof(double));

        cudaMemcpy(h_sum, d_sum, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_A, d_A, gridSize * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_B, d_B, gridSize * sizeof(double), cudaMemcpyDeviceToHost);

        double totalSum = 0.0, totalA = 0.0, totalB = 0.0;
        for (int i = 0; i < gridSize; i++) {
            totalSum += h_sum[i];
            totalA += h_A[i];
            totalB += h_B[i];
        }

        double result = (totalA == 0.0 || totalB == 0.0) ? 0.0 : totalSum / (sqrt(totalA) * sqrt(totalB));
        printf("%.12f\n", result);

        // Cleanup
        free(h_sum); free(h_A); free(h_B);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_sum); cudaFree(d_A); cudaFree(d_B);
    }

    free(a);
    free(b);
    return 0;
}
