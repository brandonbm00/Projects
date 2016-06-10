/*                                                                |
 *             HEAT2 - PARALLEL FINITE DIFFERENCE SOLVER          |
 *________________________________________________________________|
 *
 *      Computes a finite difference solution for Laplace's
 *      equation using a two dimensional periodic initial 
 *      condition and the ADI Method
 *
 *      INPUT PARAMETERS: N (sqrt gridsize, int), T (terminal time, double) 
 *                        dt(timestep size, double)
 *
 * Written by Brandon B. Miller
 */


#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include "magma.h"
#include "magma_types.h"
#include "magma_lapack.h"

static void HandleError (cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
                printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
                exit(1);
        }
}
#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))

__device__ int get_global_idx_2D_2D_nrml() {
	// Compute a global index based on thread and block dimension
	// Used in a 2D Grid of 2D blocks
        int N = gridDim.x * blockDim.x;
        int id = threadIdx.x    +  N*threadIdx.y + blockDim.x*blockIdx.x + blockDim.y*blockIdx.y*N;
	//       R THREAD SHIFT      D ROW SHIFT       R BLOCK SHIFT           D BLOCK SHIFT
        return id;
}
__device__ int get_global_idx_2D_2D_rvrs() {
	// Computes the transposed index. Used in transpose function
        int N = gridDim.x * blockDim.x;
        int id = threadIdx.y + N*threadIdx.x + blockDim.y*blockIdx.y + blockDim.x*blockIdx.x*N;
        return id;
}

__device__ int get_global_idx_1D_1D_nrml() {
	// Returns a global index in a 1D grid of 1D blocks 
	int N = gridDim.x * blockDim.x;
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	return id;
}

__global__ void tnsps_sq_array(double arr[], double tmp[], int N) {
	// Transpose a square array
	// Obtain symmetric indices
        int nmidx = get_global_idx_2D_2D_nrml();
        int rvidx = get_global_idx_2D_2D_rvrs();
        tmp[rvidx] = arr[nmidx];
	__syncthreads();
	arr[nmidx] = tmp[nmidx];

}

__global__ void make_grid(double grid[], double dx) {
	// Function to initialize the grid
	int id = get_global_idx_2D_2D_nrml();
	int N = gridDim.x * blockDim.x;
	// Avoid the boundaries
	if (id < N*N) {
		if (id < N) {
			grid[id] = -1.0 + threadIdx.x*dx; 
		}
		if (id > N*(N-1)) {
			grid[id] = 1.0 - threadIdx.x*dx;
		}
		if (id % N == 0) {
			grid[id] = -1.0 + threadIdx.y*dx;
		}
		if (id % (N-1) == 0) {
			grid[id] = 1.0 - threadIdx.y*dx;
		}

		if (id > N && id < N*(N-1) && id % N !=0 && id % (N-1) != 0) {
		grid[id] = -((-1.0+threadIdx.x*dx)*(-1.0*threadIdx.y*dx))+cos(11.0*M_PI*(-1.0+threadIdx.x*dx)/2.0)*sin(8.0*M_PI*(-1.0 + threadIdx.y*dx));
		}
	}
}


__global__ void trid_mult(double udg[], double dia[], double ldg[], double grd[], int N, int tnsps) {
	extern __shared__ double l_vec[]; // A vector 5N * sizeof(double) long 	
	// Tridiagonal matrix multiplier - this was a bad idea	
	// Copy the vector and three diagonals into shared memory
	// Uses N*5 double memory locations in shared memory 
	// These are: input vector, upper diagonal, diagonal, lower diagonal, and
	// a location for the results
	
	
	int id = get_global_idx_2D_2D_nrml();


	int tid = threadIdx.x;
	int off = blockDim.x; // Offset
	l_vec[tid]         = grd[ id]; // Fill the local vector with the target vector
	l_vec[tid + off]   = udg[tid]; // Offset by blockDim.x and place the diagonals 
	l_vec[tid + off*2] = dia[tid]; //  
	l_vec[tid + off*3] = ldg[tid]; //
	// Leave the last N locations for
	// results

	// Define these as intermediate locations
	// in shared memory for readability
	double* vec_pt = &l_vec[0];
	double* udg_pt = &l_vec[off];
	double* dia_pt = &l_vec[off*2];
	double* ldg_pt = &l_vec[off*3];
	double* res_pt = &l_vec[off*4]; // Results

	__syncthreads();


	// Do everything but the first and last elements	
	if (tid > 0 && tid < N - 1) {
		res_pt[tid] = ldg_pt[tid-1]*vec_pt[tid-1] + dia_pt[tid]*vec_pt[tid] + udg_pt[tid]*vec_pt[tid+1];
	}
	// Do the first and last elements as special cases
	// to avoid segfault. Can't think of a cleaner way
	if (tid == 0)   {
		res_pt[tid] = vec_pt[tid]*dia_pt[tid] + vec_pt[tid+1]*udg_pt[tid];	
	}	

	if (tid == N-1) {
		res_pt[tid] = ldg_pt[tid-1]*vec_pt[tid-1] + dia_pt[tid]*vec_pt[tid];
	}
	// Place the results
	grd[id] = res_pt[tid];	
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

	float tym;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	cudaEventRecord(start, 0);


        // ___________________________________


        // Initial Machinery to read in params
        // __________________________________

        FILE* inputfile;
        FILE* outputfile;

	int N;
	double dt;
	double nu;
	double T;
	

        if (argc != 3) {
                printf("Incorrect usage: only enter the input and output filenames\n");
                return 0;
        }
        inputfile = fopen(argv[1], "r");
        if (!inputfile) {
                printf("Unable to open input file \n");
                return 0;
        }
        fscanf(inputfile, "%d", &N);
        fscanf(inputfile, "%lf", &dt);
	fscanf(inputfile, "%lf", &T);
	fscanf(inputfile, "%lf", &nu);
	double dx = 2.0 / abs((double)N - 1);
	int nsteps = roundf(T / dt);	
	double cnst = nu*dt/2.0/dx/dx;

	outputfile = fopen(argv[2], "wb");	
	fwrite(&N, sizeof(int), 1, outputfile);
	double* x_axis = (double *)malloc(N * sizeof(double));
	for (int i = 0; i < N; i++) {
		x_axis[i] = -1.0 + dx*i;
	}
	fwrite(x_axis, sizeof(double), N, outputfile);
	free(x_axis);	


        // __________________________________


	
        // __________________________________
	//
	//	 Instantiation of Memory
        // __________________________________

	double* main_grid = (double *)malloc(N*N*sizeof(double));
	double* main_grid_d;
	double* main_grid_d_tmp;

	double* Bu = (double *)calloc(N,sizeof(double)); // Matrix Diagonals(B Matrix) 
	double* Bd = (double *)calloc(N,sizeof(double));
	double* Bl = (double *)calloc(N,sizeof(double));

	double* Bu_d;
	double* Bd_d;
	double* Bl_d;

	double* A;
	double* A_d;
      
        // __________________________________
	//
	//   Instantiation of B Diagonals
        // __________________________________

	
	for (int i=0; i < N-2; i++) {
                Bu[i+1]       =  cnst;
                Bl[i]         =  cnst;
                Bd[i+1]       =  1.0 - 2*cnst;
	}

	Bd[0] = 1.0;
        Bd[(int)N - 1] = 1.0;


	
        // __________________________________
	//
	//	 Instantiation of Grid
        // __________________________________
	
	for (int i = 0; i < N; i++) {
        	main_grid[i]         =   -1.0+i*dx; // Top -> T(x, 1, t)
        	main_grid[N*(N-1)+i] =    1.0-i*dx; // Bottom -> T(x, -1, t)
        	main_grid[N*i]       =   -1.0+i*dx; // Left -> T(-1, x, t)
        	main_grid[N*i+(N-1)] =    1.0-i*dx; // Right -> T(1, x, t) 
        }


	// Fill in the initial condition        
        for (int i = 1; i < N-1; i++) {
                for (int j = 1; j < N-1; j++) {
                        main_grid[i*N + j] = -((-1.0+dx*i)*(-1.0+dx*j))+cos(11.0*M_PI*(-1.0 +i*dx)/2.0)*sin(8.0*M_PI*(-1.0 + dx*j));
                }
        }

	if ( cudaSuccess != cudaMalloc((void**)&main_grid_d, N*N*sizeof(double)) ) {
		printf("cudaMalloc Failed...\n");
		exit(1);
	}
	if ( cudaSuccess != cudaMalloc((void**)&main_grid_d_tmp, N*N*sizeof(double)) ) {
		printf("cudaMalloc Failed...\n");
		exit(1);
	}

	// Allocate memory for diagonals

	if ( cudaSuccess != cudaMalloc((void**)&Bu_d, N*sizeof(double)) ) {
		printf("cudaMalloc Failed...\n");
		exit(1);
	}
	if ( cudaSuccess != cudaMalloc((void**)&Bd_d, N*sizeof(double)) ) {
		printf("cudaMalloc Failed...\n");
		exit(1);
	}
	if ( cudaSuccess != cudaMalloc((void**)&Bl_d, N*sizeof(double)) ) {
		printf("cudaMalloc Failed...\n");
		exit(1);
	}

	// Place the diagonals into global memory on the device	
	cudaMemcpy(main_grid_d, main_grid, N*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Bu_d, Bu, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Bd_d, Bd, N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(Bl_d, Bl, N*sizeof(double), cudaMemcpyHostToDevice);


	// We will process columns individually using a block per column
	int  blkSide = N / 32;
	dim3 bdim(blkSide, blkSide, 1); 
	dim3 tdim(32, 32, 1);


	// Initial MAGMA machinery	
	magma_init();
	magma_int_t *piv, info;
	magma_int_t m = N;
	magma_int_t n = N - 2; // Will start 1 row in and finish 1 early - these are  
	magma_int_t err;       // the boundaries that we want to avoid updating
     
	err = magma_dmalloc_cpu(&A, m*m);


      // __________________________________
      //
      //   Instantiation of A Matrix
      // __________________________________
      // 
	for (int i = 0; i < N*N; i++) {
		A[i] = 0.0;
	}

	for (int i = 1; i < N - 1; i++) {
		A[i*N + i] = 1.0 + 2*cnst;
		A[i*N+i+1] = -cnst;
		A[i*N+i-1] = -cnst;
	}
	// This is basically a transpose for this particular matrix
	A[0] = 1.0;
	A[N*N-1] = 1.0;
	A[1] = A[N];
	A[N*N-2] = A[N*(N-1) - 1];
	A[N] = 0.0;
	A[N*(N-1) - 1] = 0.0;
	

      // Make room for A on the device
	err = magma_dmalloc(&A_d, m*m);
	if (err) {
		printf("Malloc Error! \n");
		exit(1);
	}

	piv = (magma_int_t*)malloc(m*sizeof(magma_int_t));
	magma_dsetmatrix(m, m, A, m, A_d, m);


  //    make_grid<<<bdim, tdim>>>(main_grid_d, dx);
      cudaMemcpy(main_grid, main_grid_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);


      exit(1);
      // MAIN LOOP
      for (int step = 0; step <= nsteps ; step++) {

            	// FIRST EXPLICIT STEP
            	// We want to hit the columns first - transpose, use trid (which acts on r
   	 	// contiguous memory arrays) and then transpose back
    		tnsps_sq_array<<<bdim, tdim>>>(main_grid_d, main_grid_d_tmp, N);
    		trid_mult<<<N, N, N*5*sizeof(double)>>>(Bu_d, Bd_d, Bl_d, main_grid_d, N, 1);
    		tnsps_sq_array<<<bdim, tdim>>>(main_grid_d, main_grid_d_tmp, N);


    		// FIRST IMPLICIT STEP
    		// From here we can directly solve the rows since they're already
    		// contiguous in memory
      		magma_dgesv_gpu(m, n, A_d, m, piv, &main_grid_d[N], m, &info);

    		// SECOND EXPLICIT STEP
    		// Next, use B on the rows again
    		trid_mult<<<N, N, N*5*sizeof(double)>>>(Bu_d, Bd_d, Bl_d, main_grid_d, N, 1);
      
    		// SECOND IMPLICIT STEP
    		// More or less the inverse action of the first implicit step
    		tnsps_sq_array<<<bdim, tdim>>>(main_grid_d, main_grid_d_tmp, N);
	      	magma_dgesv_gpu(m, n, A_d, m, piv, &main_grid_d[N], m, &info);
    		tnsps_sq_array<<<bdim, tdim>>>(main_grid_d, main_grid_d_tmp, N);


    		// Write the data	
    		if (step % (nsteps / 4) == 0) {
    			printf("Print step %d \n", step);
    			cudaMemcpy(main_grid, main_grid_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    			fwrite(main_grid, sizeof(double), N*N, outputfile);
      		}
     	 }


      // Closing machinery
	free(piv);
	magma_finalize();
	cudaMemcpy(main_grid, main_grid_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);

	free(A);	

	free(Bu);
	free(Bd);
	free(Bl);

	cudaFree(A_d);

	cudaFree(Bu_d);
	cudaFree(Bd_d);
	cudaFree(Bl_d);


	cudaFree(main_grid_d_tmp);
	cudaFree(main_grid_d);
	free(main_grid);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tym, start, stop);
	printf("Elapsed Time %3.1f milliseconds \n", tym);	
	return 0;
}
