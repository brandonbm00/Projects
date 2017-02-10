#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

static void HandleError (cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
                printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
                exit(1);
        }
}
#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))





void printDevProp(cudaDeviceProp devProp)
{
    printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %lu\n",  devProp.totalGlobalMem);
    printf("Total shared memory per block: %lu\n",  devProp.sharedMemPerBlock);
    printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of block:  %d\n", i, devProp.maxThreadsDim[i]);
    for (int i = 0; i < 3; ++i)
        printf("Maximum dimension %d of grid:   %d\n", i, devProp.maxGridSize[i]);
    printf("Clock rate:                    %d\n",  devProp.clockRate);
    printf("Total constant memory:         %lu\n",  devProp.totalConstMem);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ?"Yes" : "No"));
    return;
}

def _get_block_dimensions(int DIM, int TMAX) {
    // Want 1024 threads per block
    int xdim;

    if (DIM < TMAX) {
        ret = TMAX;        
    }
    else {
        ret = ceil(DIM / TMAX) 
    }
    return ret
}


def _get_gid_2d_2d() {
    return ((blockIdx.x * blockDim.x) + threadIdx.x) + (blockDim.x * gridDim.x) * blockIdx.y;
}

def _get_gid_2d_2d_T() {
    return ((blockIdx.y * blockDim.y) + threadIdx.y) + (blockDim.y * gridDim.y) * blockIdx.x;
}

// Kernel function

__global__ void mmult(double a[], double b[], double c[], int INDIM) {
    int GID = _get_gid_2d_2d()
    int* ROW = &a[blockIdx.y * blockDim.y + threadIdx.y];
    int* COL = &b[blockIdx.x * blockDim.x + threadIdx.x];

    int TOT
    for (int i = 0; i < INDIM; i++) {
        TOT += ROW[i]*COL[i];
    }
    
    c[GID] = TOT; 
     
}

__global__ void transpose_2d_double(double mat[], int N, int M) {
    int GID  = _get_gid_2d_2d(); 
    int L    = N * M - 1 
 
    int GIDX;
    if GID == L {
        GIDX = L;
    }
    else {
        GIDX = (GID * M) % L;
    }

    val = mat[GID]; 
    __syncthreads();
    mat[GIDX] = val;
   
 
}


int main(int argc, char* argv[]) {
    // Initial Machinery to select the GPU
    // ___________________________________ 
    cudaDeviceProp prop; // This is a blank struct at this point    
    int dev;
    memset(&prop, 0, sizeof(cudaDeviceProp)); // Initialize the struct

    prop.multiProcessorCount = 13;
    cudaChooseDevice(&dev, &prop);
    HANDLE_ERROR(cudaSetDevice(dev));
    cudaGetDeviceProperties(&prop, dev);
   
     
    printf("Device Properties: -------------------------------- \n");
    printDevProp(prop);
    printf("End Device Properties: ---------------------------- \n");
 
    // Get dimensions of matrices from command line


    printf("Begin. --------------------------------------------\n");  
    if (argc < 4) {
       printf("Please enter four args: A_n, A_m, B_n, B_m \n");
       return 1;
    }

    int A_n = atoi(argv[1]);
    int A_m = atoi(argv[2]);
    int B_n = atoi(argv[3]);
    int B_m = atoi(argv[4]);

    if (A_m != B_n) {
        printf("Inner matrix dimensions A_m = %d and B_n = %d must match \n", A_m, B_n);
        return 2;
    }

    printf("Matrices will be A = (%d x %d), B = (%d x %d) \n", A_n, A_m, B_n, B_m);
    printf("Final matrix will be C = (%d x %d) \n", A_n, B_m);

    // instantiate matrices
    
    double* A = (double *)malloc(A_n*A_m*sizeof(double));
    double* B = (double *)malloc(B_n*B_m*sizeof(double));

    double* A_gpu;
    double* B_gpu;


    for (int i = 0; i < A_n*A_m; i++) {
        A[i] = double(i);  
    }
    for (int i = 0; i < B_n*B_m; i++) {
        B[i] = double(i);  
    }
     


    // Copy data down to GPU
    if ( cudaSuccess != cudaMalloc((void**)&A_gpu, A_n*A_m*sizeof(double)) ) {
        printf("cudaMalloc Failed...\n");
        exit(1);
    }
    if ( cudaSuccess != cudaMalloc((void**)&B_gpu, B_n*B_m*sizeof(double)) ) {
        printf("cudaMalloc Failed...\n");
        exit(1);
    }

    cudaMemcpy(A_gpu, A, A_n*A_m*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(B_gpu, B, B_n*B_m*sizeof(double), cudaMemcpyHostToDevice);

    // Size the grid
    // Want 32 X 32 Blocks 
 
    block_x = _get_block_dimensions(A_n, 32);
    block_y = _get_block_dimensions(B_m, 32);

    dim3 block = (block_x, block_y, 1)
    # HERE

    free(A);
    free(B);
    cudaFree(A_gpu);
    cudaFree(B_gpu);
    return 0;
}
