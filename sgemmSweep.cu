
/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.  Users and possessors of this source code
 * are hereby granted a nonexclusive, royalty-free license to use this code
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.   This source code is a "commercial item" as
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer  software"  and "commercial computer software
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 *
 * Any use of this source code in individual and commercial software must
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */


#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <cublas.h>
#include <stdint.h>

#define SWEEP_SUCCESS ((void*)1)
#define SWEEP_FAILURE ((void*)0)
#define MAX_DEVICES 256 //256 devices is enough for anyone...

unsigned int testIDs[MAX_DEVICES];
unsigned int testedDevices = 0;
int iterations = 1;
unsigned int speedSetting = 32;
int deviceCount = 0;
pthread_mutex_t lock;
pthread_cond_t condvar;
pthread_t devThreads[MAX_DEVICES];
float elapsedTimes[MAX_DEVICES];

volatile int terminatingDevice = -1;

__global__ void floatMemset(float* ptr, unsigned int length, float value)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (; idx < length; idx += stride)
    ptr[idx] = value;
}

void* sgemmSweep(void* devID)
{
    int device = (intptr_t)devID;
    printf("device = %d\n", device);
    float *A, *B, *C, alpha = 1.0, beta = 1.0;
    unsigned int i, j, k;
    if (cudaSetDevice(device) != cudaSuccess)
    {
        fprintf(stderr, "cudaSetDevice(%d) failed\n", device);
        return SWEEP_FAILURE;
    }
    if (cublasInit() != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "Error: cublasInit failed from device %u\n",device);
        return SWEEP_FAILURE;
    }
    struct cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, device);
    unsigned int iterSize = ((unsigned int)(sqrt((properties.totalGlobalMem-(200*(1<<20)))/24))) & ~(speedSetting-1);
    printf("iterSize = %u\n", iterSize);
    //  printf("Performing %d iterations with increment size %d on device %d...\n", iterations, speedSetting, device);
    for (int curIter = 0; curIter < iterations; curIter++)
    {
        for (i = 128; i < iterSize; i+= speedSetting)
        {
            if (terminatingDevice != -1)
            {
                cublasFree(A);
                cublasFree(B);
                cublasFree(C);
                return SWEEP_SUCCESS;
            }
            printf("Device %d: i = %d\n",device, i);
            float* c_h = (float*)malloc(sizeof(float) * i*i);
            if (!c_h)
            {
                fprintf(stderr, "ERROR: malloc of c_h failed. Aborting.\n");
                terminatingDevice = device;
                return SWEEP_FAILURE;
            }
            if (cudaMalloc((void**)&A, i*i*sizeof(float)) != cudaSuccess)
            {
                fprintf(stderr, "Error: cublasAlloc(A) failed at i = %d\n", i);
                terminatingDevice = device;
                free(c_h);
                return SWEEP_FAILURE;
            }
            if (cudaMalloc((void**)&B, i*i*sizeof(float)) != cudaSuccess)
            {
                fprintf(stderr, "Error: cublasAlloc(B) failed at i = %d\n", i);
                terminatingDevice = device;
                free(c_h);
                return SWEEP_FAILURE;
            }

            if (cudaMalloc((void**)&C, i*i*sizeof(float)) != cudaSuccess)
            {
                fprintf(stderr, "Error: cublasAlloc(C) failed at i = %d\n", i);
                terminatingDevice = device;
                free(c_h);
                return SWEEP_FAILURE;
            }
            floatMemset<<<i/128, 128>>>(A, i*i, 1.0);
            floatMemset<<<i/128, 128>>>(B, i*i, 2.0);
            floatMemset<<<i/128, 128>>>(C, i*i, 3.0);

            if (cudaThreadSynchronize() != cudaSuccess)
            {
                fprintf(stderr, "Error: cudaThreadSynchronize returned %s\n", cudaGetErrorString(cudaGetLastError()));
                terminatingDevice = device;
                free(c_h);
                return SWEEP_FAILURE;
            }
            float result = 2.0 * i + 3.0;

      cublasSgemm('n', 'n', i, i, i, alpha, A, i, B, i, beta, C, i);
      if (cublasGetError() != CUBLAS_STATUS_SUCCESS)
      {
          fprintf(stderr, "Error: cublasSgemm failed!\n");
          terminatingDevice = device;
          free(c_h);
          return SWEEP_FAILURE;
      }
      cudaMemcpy(c_h, C, sizeof(float)*i*i, cudaMemcpyDeviceToHost);
      for (j = 0; j < i; j++)
      {
          for (k = 0; k < i; k++)
          if (c_h[j*i+k] != result)
          {
              fprintf(stderr, "Error: cublasSgemm returned an invalid result at location %d,%d in iteration %d on device %d\n", j, k, i, device);
              printf("%f\n", c_h[j*i+k]);
              terminatingDevice = device;
              free(c_h);
              return SWEEP_FAILURE;
          }
      }
      free(c_h);
      cublasFree(A);
      cublasFree(B);
      cublasFree(C);

    }

      //      printf("Finished iteration %d\n", curIter);
      }
      printf("Device %d completed successfully\n", device);
      return SWEEP_SUCCESS;
}

int main (int argc, char** argv)
{
    int i;

    if (argc < 2)
    {
        fprintf(stderr, "usage: %s <speed setting> <iterations>\nSpeed settings:\n0 = iterate by 32 (default)\n1 = iterate by 64\n2 = iterate by 128 (fastest)\n", argv[0]);
        return 1;
    }
    switch (argc)
    {
        case 3:
        sscanf(argv[2], "%d", &iterations);
        case 2:
        unsigned int speed = 0;
        sscanf(argv[1], "%u", &speed);
        if (speed == 2)
        speedSetting = 128;
        else if (speed == 1)
        speedSetting = 64;
    }

    cudaGetDeviceCount(&deviceCount);
    printf("deviceCount = %d\n", deviceCount);
    for (i = 0; i < deviceCount; i++)
    {
        struct cudaDeviceProp properties;
        if (cudaGetDeviceProperties(&properties, i) != cudaSuccess)
        {
            printf("Could not retrieve properties of device %d\n", i);
            exit(1);
        }

        printf("Testing device %d: %s\n", i, properties.name);
        if ((properties.major != 9999 && properties.minor != 9999)
            //&&
            //((properties.major >= 1 && properties.minor >= 3) ||
            // (properties.major >= 2))
            // && !properties.kernelExecTimeoutEnabled
           )
        {
            testIDs[testedDevices++] = i;
        }
    }
    if (testedDevices == 0)
    {
        printf("No suitable NVIDIA GPUs found. Aborting...\n");
        exit(1);
    }

    for (i = 0; i < testedDevices; i++)
    {
        pthread_create(&devThreads[i], NULL,
                       (sgemmSweep),(void*)((intptr_t)testIDs[i]));
    }
void* returnVal = 0;
for (int i = 0; i < testedDevices; i++)
{
    pthread_join(devThreads[i], &returnVal);
    if (returnVal != SWEEP_SUCCESS)
    {
        printf("ERROR: Failed with device %d. sgemmSweep FAILED.\n", terminatingDevice);
        exit(1);
    }
}
printf("sgemmSweep PASSED.\n");
}

