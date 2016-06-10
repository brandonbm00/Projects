/*											    |
 *			MCQUAD - PARALLEL MONTE-CARLO SAMPLER				    |		
 *__________________________________________________________________________________________|
 *
 *	Performs total_trials Monte-Carlo integrations of \int_{0}^{\inf} \exp{-x} g(x) dx
 *	for g(x) = ln(x).
 *	
 *	Draws N samples for each integration according to the density g(x). 
 *	
 *	INPUT PARAMETERS: total_trials (#trials, integer), N (samples, integer)
 *	
 *
 * Written by Brandon B. Miller
 */ 


#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include "func.h"

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);
	int nprocs;
	int rank;		
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank  );

	double precision = MPI_Wtick();
	double starttime = MPI_Wtime();


	// Preliminary Machinery - read in parameters
	// and broadcast them to the  working   procs
	int N; 
	int total_trials;
	FILE* inputfile;
	FILE* outfile;

	if (rank == 0) {	
		if (argc != 3) {		
			printf("Incorrect usage: only enter the input and output filenames \n");
			MPI_Finalize();
			return 0;
		}
		inputfile = fopen(argv[1], "r");
		if (!inputfile) {
			printf("Unable to open input file \n");
			MPI_Finalize();
			return 0;
		}
		fscanf(inputfile, "%d", &N);
		fscanf(inputfile, "%d", &total_trials);
		fclose(inputfile);

	}
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&total_trials, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	srand48(time(NULL));

	double *mytrials;
	double *results;
	
		
	
	int remainder = total_trials % nprocs; 
	int ntrials = (total_trials - remainder) / nprocs;


	int *counts = (int *)malloc(nprocs * sizeof(int));
	int *offsets = (int *)malloc(nprocs *sizeof(int));

	for (int i = 1; i < nprocs; i++) {
		counts[i] = ntrials;
		offsets[i] = ntrials * i + remainder;	
	}

	counts[0] = ntrials + remainder;
	offsets[0] = 0;

	double q = 0;
	if (rank == 0) {

		results = (double *)malloc(total_trials * sizeof(double));

		mytrials = (double *)malloc((ntrials + remainder)  * sizeof(double));	
		for (int trial = 0; trial < ntrials + remainder; trial++) {	
			for (int i = 0; i < N; i++) {	
				double x = drand48();
				q += func(-log(x));
			}
			mytrials[trial] = q / (double)N;
			q = 0.0;
		}
	}
	else {
		mytrials = (double *)malloc(ntrials * sizeof(double));
		for (int trial = 0; trial < ntrials; trial++) {	
			for (int i = 0; i < N; i++) {	
				double x = drand48();
				q += func(-log(x));
			}
			mytrials[trial] = q / (double)N;
			q = 0.0;
		}	
	}
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Gatherv(mytrials, counts[rank], MPI_DOUBLE, results, counts, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	if (rank == 0) {
		outfile = fopen(argv[2], "wb");
		fwrite(results, sizeof(double), total_trials, outfile);
		fclose(outfile);
		free(results);
		double elapsed_time = MPI_Wtime() - starttime;
		printf("Execution time = %le seconds with precision %le seconds\n", elapsed_time, precision);
	}
	MPI_Finalize();
	return 0;
}























