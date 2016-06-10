/*                                                                |
 *              MCQUAD - CUDA MONTE CARLO INTEGRATOR              |
 *________________________________________________________________|
 *
 *      Performs a Monte Carlo integration of the form 
 *      \int_{0}^{\inf} \exp{-x} g(x) dx for g(x) = ln(x) 
 *      
 *      Draws N samples for each integration according to the density
 *      -> g(x).
 *
 *      INPUT PARAMETERS: 
 *	ntrials (total nuber of trials),
 *      nsamps  (samples taken per trial)
 *
 * Written by Brandon B. Miller
 */



#include <stdlib.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>


__global__ void setup_kernel(int N, long int seed, curandState_t *state) {
	// Set up the RNG for each sample thread	
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < N) {
		// Each RNG state is different
		// It augomatically increments itself
		curand_init(seed, id, 0, &state[id]);
	}
}


__global__ void do_trials(int ntrials, int nsamps, double* results, curandState_t *state) {
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	// We will have each thread do a trial. So we need to launch 
	// Ntrials blocks each with one thread.  
	if (id < ntrials) {
		double sum = 0;	
		for (int sample = 0; sample < nsamps; sample++) { 
			sum += cos(-log(curand_uniform_double(&state[id]))); 	

		}
		results[id] = sum / nsamps; // Answer
	}
}


static void HandleError (cudaError_t err, const char* file, int line) {
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(1);
	}
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
	cudaSetDevice(dev);
	cudaGetDeviceProperties(&prop, dev);


	// ___________________________________


	// Initial Machinery to read in params
	// __________________________________

	float tym;	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop );
	cudaEventRecord(start, 0);
	int nsamps;
	int ntrials;
	FILE* inputfile;
	FILE* outputfile;
	
	if (argc != 3) {
		printf("Incorrect usage: only enter the input and output filenames\n");
		return 0;
	}
	inputfile = fopen(argv[1], "r");
	if (!inputfile) {
		printf("Unable to open input file \n");
		return 0;
	}
	fscanf(inputfile, "%d", &nsamps);
	fscanf(inputfile, "%d", &ntrials);
	// __________________________________

	double* results = (double *)malloc(ntrials * sizeof(double));	

	// Random number generation
	curandState_t* dev_states;	
	double* dev_results; // will contain final random numbers
	
	if ( cudaSuccess != cudaMalloc((void**)&dev_results, ntrials*sizeof(double)) ) {
		printf("cudaMalloc Failed...");
		exit(1);
	}
	// THERE IS NOW AN NTRIALS LENGTH ARRAY IN GLOBAL MEM ON THE DEVICE 
	if ( cudaSuccess != cudaMalloc((void**)&dev_states, ntrials*sizeof(curandState_t)) ) {
		printf("cudaMalloc Failed...");
		exit(1);
	}
	// dev_states is an array containing an RNG state to be used for each trial
	// We will index into it uniquely based on thread and blockID within the kernel
	setup_kernel<<<ntrials, 1>>>(nsamps, time(NULL), dev_states);
	
	// FIXME - Launch a block for each trial with one thread each - SLOW	
	do_trials<<<ntrials, 1>>>(ntrials, nsamps, dev_results, dev_states); 

	// Retrieve results
	cudaMemcpy(results, dev_results, ntrials*sizeof(double), cudaMemcpyDeviceToHost);

	outputfile = fopen(argv[2], "wb");
	fwrite(results, sizeof(double), ntrials, outputfile);

	// Closing machinery
	cudaFree(dev_results);
	cudaFree(dev_states);
	free(results);
	fclose(outputfile);
	fclose(inputfile);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&tym, start, stop);
	printf("Elapsed time %3.1f milliseconds", tym);
	return 0;
}
