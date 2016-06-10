#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "trid.h"

#define M_PI  3.141592654


void dgtsv_(int*, int*, double*, double*, double*, double*, int*, int*);


void tnsps_sqr_arr(double array[], int N) {
	double* tmp = (double *)calloc(N*N,sizeof(double));
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			tmp[j*N + i] = array[i*N + j];	
		}
	}	

	for (int i = 0; i < N*N; i++) {
		array[i] = tmp[i];
	}

	free(tmp);
}




int main(int argc, char* argv[]) {
	MPI_Init(NULL, NULL);
	int nprocs;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank  );

	double precision = MPI_Wtick();
	double starttime = MPI_Wtime();

	int N;
	double nu;
	double dt;
	double T;
	FILE* inputfile;
	FILE* outfile;
	
	if (rank == 0) {	
		if (argc != 3) {
			printf("Incorrect usage: only enter the input and output filenames\n");
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
		fscanf(inputfile, "%lf", &dt);	
		fscanf(inputfile, "%lf", &T);
		fscanf(inputfile, "%lf", &nu);


		outfile = fopen(argv[2], "wb");
		fwrite(&N, sizeof(int), 1, outfile);
		double* x_axis = (double *)calloc(N,sizeof(double));
		double dx = 2 / (N - 1.0);
	
		for (int i = 0; i < N; i++) {
			x_axis[i] = -1.0 + dx*i;	
		}
		fwrite(x_axis, sizeof(double), N, outfile);
	}
	
	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&nu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double dx = 2 / (N - 1.0);
	double cnst = nu*dt/2.0/dx/dx;
	int nsteps = roundf(T / dt);
	int lo;
	int hi;


	int remainder = N % nprocs;
	int nrows = (N - remainder) / nprocs;
	int myrowcount;



	// Initialize the grid
	
	double* main_grid;
	double* mydomain;

	int *counts = (int *)calloc(nprocs,sizeof(int));
	int *offsets = (int *)calloc(nprocs,sizeof(int));

	for (int i = 1; i < nprocs; i++) {
		counts[i] = nrows * N;
		offsets[i] = (nrows * i + remainder) * N;
	}

	counts[0] = (nrows + remainder) * N;
	offsets[0] = 0;
		
	if (rank == 0) {
		main_grid = (double *)calloc(N*N,sizeof(double));
		myrowcount = nrows + remainder;	
		
	
               // Fill in the boundary elements
               
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
               

	mydomain = (double *)calloc((nrows + remainder)*N,sizeof(double));


	// Controls such that the boundary rows are not updated
	// ____________________________________________________
	}

	if (rank != 0) {
		mydomain = (double *)calloc(nrows*N,sizeof(double));
		myrowcount = nrows;
	}


	if (rank == 0) {
		 lo = 1;
		 hi = myrowcount;
	}
	else if (rank == nprocs) {
		 lo = 0;
		 hi = myrowcount - 1;
	}
	else {
		 lo = 0;
		 hi = myrowcount;
	}
	// ____________________________________________________


	// Main loop
	// .........
	// These will be the diagonals of the tridiagonal
	// matrices
	double* Bu = (double *)calloc(N,sizeof(double));
	double* Bd = (double *)calloc(N,sizeof(double));
	double* Bl = (double *)calloc(N,sizeof(double));	
	double* Au;
	double* Ad;
	double* Al;


	for (int i=0; i < N-2; i++) {
               	Bu[i+1]       =  cnst;
               	Bl[i]         =  cnst;
               	Bd[i+1]       =  1.0 - 2*cnst;
	}	
		
	Bd[(int)N - 1] = 1.0;	
	Bd[0]	       = 1.0;
	int info = 0;
	int one = 1;
	int dim = (int)N;

	if (rank == 0) {
		tnsps_sqr_arr(main_grid, N);
	}

	for (int step = 0; step <= nsteps; step++) {
		printf("entering iteration %d \n", step);
		MPI_Scatterv(main_grid, counts, offsets, MPI_DOUBLE, mydomain, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		// First explicity step


		for (int row = lo; row < hi; row++) {
			trid(Bu, Bl, Bd, &mydomain[N*row], N);
		}
		// End first explicit step


		// The data is collected by the master process
		// and transposed, then scattered in the exact
		// same way as before
		MPI_Gatherv(mydomain, counts[rank], MPI_DOUBLE, main_grid, counts, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);


		//__________________________________________________________
		if (rank == 0) {
			tnsps_sqr_arr(main_grid, N);
		}
		//__________________________________________________________


		MPI_Scatterv(main_grid, counts, offsets, MPI_DOUBLE, mydomain, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// First implicit step	
		Au = (double *)calloc(N,sizeof(double));
		Ad = (double *)calloc(N,sizeof(double));
		Al = (double *)calloc(N,sizeof(double));	
			
		for (int i=0; i < N-2; i++) {
        	        Au[i+1]       =  -cnst;    // Upper Diagonal
       	        	Al[i]         =  -cnst;    // Lower Diagonal
                	Ad[i+1]       =  1.0 + 2*cnst; // Main  Diagonal
		}	
		Ad[0]	       = 1.0;
		Ad[(int)N - 1] = 1.0;

		int nrhs = hi - lo;
		dgtsv_(&dim, &nrhs, Al, Ad, Au, &mydomain[N*lo], &dim, &info);
		free(Au);
		free(Ad);
		free(Al);		


		// End first implicit step

		MPI_Gatherv(mydomain, counts[rank], MPI_DOUBLE, main_grid, counts, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		MPI_Scatterv(main_grid, counts, offsets, MPI_DOUBLE, mydomain, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// Second explicit step
		for (int row = lo; row < hi; row++) {
			trid(Bu, Bl, Bd, &mydomain[N*row], N);
		}
		// End second explicit step		

		MPI_Gatherv(mydomain, counts[rank], MPI_DOUBLE, main_grid, counts, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);


		if (rank == 0) {
			tnsps_sqr_arr(main_grid, N);
		}
		MPI_Scatterv(main_grid, counts, offsets, MPI_DOUBLE, mydomain, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
					
		// Second implicit step
		Au = (double *)calloc(N,sizeof(double));
		Ad = (double *)calloc(N,sizeof(double));
		Al = (double *)calloc(N,sizeof(double));	
		for (int i=0; i < N-2; i++) {
        	        Au[i+1]       =  -cnst;    // Upper Diagonal
       	        	Al[i]         =  -cnst;    // Lower Diagonal
                	Ad[i+1]       =  1.0 + 2*cnst; // Main  Diagonal
		}	
		Ad[0]	       = 1.0;
		Ad[(int)N - 1] = 1.0;
		dim = (int)N;
	
		nrhs = hi - lo;
		dgtsv_(&dim, &nrhs, Al, Ad, Au, &mydomain[N*lo], &dim, &info);
		free(Au);
		free(Ad);
		free(Al);		

		// End second implicit step				
		MPI_Gatherv(mydomain, counts[rank], MPI_DOUBLE, main_grid, counts, offsets, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	
		//__________________________________________________________
		if (rank == 0) {
			tnsps_sqr_arr(main_grid, N);

			}
		//__________________________________________________________

		if (step % (nsteps / 4) == 0) {
			if (rank == 0) {
				fwrite(main_grid, sizeof(double), N*N, outfile);
			}	
		}




	}
	free(mydomain);
	free(counts);
	free(offsets);


	if (rank == 0) {
		fclose(outfile);
		free(main_grid);
		double elapsed_time = MPI_Wtime() - starttime;
		printf("Execution time = %le seconds with precision %le seconds \n", elapsed_time, precision);
	}
	
	MPI_Finalize();
	return 0;
}
