TARGET=cuda_dgemmSweep
CUDA_INSTALL_PATH ?=/usr/local/cuda

CC=gcc
CXX=g++
CUDACC=nvcc

CUDA_INCLUDES := -I. -I${CUDA_INSTALL_PATH}/include 
CUDALIB := -L${CUDA_INSTALL_PATH}/lib64 -lcuda  -lcudart -lcublas
CFLAGS= -arch sm_13 -DSM_13 -O3
CFLAGS_SM20= -arch sm_20 -DSM_20 -O3

default: ${TARGET}
all: cuda_dgemmSweep cuda_dgemmSweep_sm20

cuda_dgemmSweep: dgemmSweep.cu
	${CUDACC} ${CFLAGS} -o $@ $^ ${CUDALIB}

cuda_dgemmSweep_sm20: dgemmSweep.cu
	${CUDACC} ${CFLAGS_SM20} -o $@ $^ ${CUDALIB}

clean:
	rm -vf ./cuda_dgemmSweep
	rm -vf ./cuda_dgemmSweep_sm20
