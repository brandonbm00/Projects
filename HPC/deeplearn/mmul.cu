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

#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))

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

    return 0;
}
