/*                                                                |
 *              HEAT - PARALLEL FINITE DIFFERENCE SOLVER          |
 *________________________________________________________________|
 *
 *      Computes a finite difference solution for Laplace's
 *      equation using a two dimensional periodic initial 
 *      condition
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

static void HandleError (cudaError_t err, const char* file, int line) {
        if (err != cudaSuccess) {
                printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
                exit(1);
        }
}
#define HANDLE_ERROR( err ) (HandleError(err, __FILE__, __LINE__))


__device__ int getGlobalIdx_2D_2D() {
	// Returns a row-major index on a 2D grid of 2D Blocks
	int N = gridDim.x * blockDim.x;
	int threadID =  threadIdx.x   +  N*threadIdx.y + blockDim.x*blockIdx.x + blockDim.y*N*blockIdx.y; 
	//             R THREAD SHIFT     D ROW SHIFT        R BLOCK SHIFT           D BLOCK SHIFT
	return threadID;
}

__global__ void diff_kern(double grid[], int N, double cnst)  {
	// Kernel called should allocate a (32 + 2) x (32 + 2) size grid
	// To store the elements to be updated and the border elements
	// Only the central 32x32 will be updated in parallel by threads

	extern __shared__ double l_grid[];
	// This shared array is twice the required size so the back half
	// can be used as the update destination. This is to avoid race
	// condition updating the grid wrong. Just like a temporary array.
	// Then I copy the data out of the back half of it at the end. m
		

	int dim  = blockDim.x + 2;  // New size of the subdomain
	int tidx = threadIdx.x + 1; // Offset 1 index in for B.C.
	int tidy = threadIdx.y + 1; // Offset 1 index in for B.C.

	int id = getGlobalIdx_2D_2D(); // Where am I on the main grid?
	l_grid[tidy * dim + tidx] = grid[id]; // Put the global val in shared


	// This structure controls the boundary elements for the subdomain
	// The outer edge elements should all move "outwards" one more and 
	// pick up the edge elements from the next domain over
	// Happens in all 4 directions for all subdomains but avoid
	// updating boundaries with an IF later

	if (tidx == 1) {
		l_grid[tidy * dim + tidx - 1] = grid[id - 1];
	}
	if (tidy == 1) {
		l_grid[tidy * dim + tidx - dim] = grid[id - N]; 	
	}
	if (tidx == dim - 2) {
		l_grid[tidy * dim + tidx + 1] = grid[id + 1];
	}
	if (tidy == dim - 2) {
		l_grid[tidy * dim + tidx + dim] = grid[id + N];
	}	

	
	__syncthreads();

	// Do not update the boundary elements!
	// FINITE DIFFERENCE
	if (id > N && id < N*(N-1) && id % N !=0 && id % (N-1) != 0) {
		l_grid[(tidy * dim + tidx) + dim*dim] = 0.0 + 
                                             cnst * (l_grid[(tidy - 1)*dim + tidx]
				                  +  l_grid[(tidy + 1)*dim + tidx] 
                                                  +  l_grid[tidy*dim + (tidx - 1)]
					          +  l_grid[tidy*dim + (tidx + 1)]	
					          -  4.0 * l_grid[tidy*dim + tidx])
				                  +  l_grid[tidy  *  dim  +  tidx]; 

		grid[N*N-id] = l_grid[(tidy * dim + tidx) + dim*dim];				   	
	}
	// FIXME - There is an infuriating bug somewhere!
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
	
	outputfile = fopen(argv[2], "wb");
	fwrite(&N, sizeof(int), 1, outputfile);
	double* x_axis = (double *)malloc(N* sizeof(double));
	for (int i = 0; i < N; i ++) {
		x_axis[i] = -1.0 + dx*i;
	}
	fwrite(x_axis, sizeof(double), N, outputfile);	
	free(x_axis);

        // __________________________________


	
        // __________________________________
	//
	//	 Instantiation of Grid
        // __________________________________

	double* main_grid = (double *)malloc(N*N*sizeof(double));
	double* main_grid_d;



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

	
	cudaMemcpy(main_grid_d, main_grid, N*N*sizeof(double), cudaMemcpyHostToDevice);

	int  blkSize = 32  * 32;   // Use 1024 threads per block
	int  blkSide = N   / 32;   // Number of blocks per "side"
	int  sbgd_sz = 34  * 34;   // Total elements in a subgrid

	
	

        // __________________________________
	//
	double cnst = nu*dt/dx/dx;
	dim3 dim_blk(blkSide, blkSide);	
	dim3 dim_trd(32, 32);	


	// MAIN LOOP
	for (int step = 0; step < nsteps; step++) {
		// Call the kernel once per timestep to propagate the system forward in time
		diff_kern<<<dim_blk, dim_trd, (sbgd_sz)*sizeof(double) * 2>>>(main_grid_d, N, cnst);

//		if (step >= 0) {
		if (step % (nsteps / 4) == 0) {
			cudaMemcpy(main_grid, main_grid_d, N*N*sizeof(double), cudaMemcpyDeviceToHost);

			printf("Main Grid step %d: \n", step);
			for (int i = 0; i < N; i++) {
				for (int j = 0; j < N; j++) {
					printf("%lf,", main_grid[i*N + j]);	
				}
				printf("\n");
			}
			printf("main grid 129: %lf \n", main_grid[129]);
			fwrite(main_grid, sizeof(double), N*N, outputfile);
			printf("main grid 129: %lf \n", main_grid[129]);
		}
	}
	
	// Closing machinery
	cudaFree(main_grid_d);
	free(main_grid);
	fclose(outputfile);

	cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&tym, start, stop);
        printf("Elapsed Time %3.1f milliseconds \n", tym);


	return 0;
}
