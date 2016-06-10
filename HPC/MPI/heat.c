/*								  |
 *		HEAT - PARALLEL FINITE DIFFERENCE SOLVER          |
 *________________________________________________________________|
 *
 *	Computes a finite difference solution for Laplace's
 *	equation using a two dimensional periodic initial 
 *	condition
 *
 *	INPUT PARAMETERS: N (sqrt gridsize, int), T (terminal time, double) 
 *			  dt(timestep size, double)
 *
 * Written by Brandon B. Miller
 */


#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define M_PI  3.141592654

// Function to determine looping indices 
//  and communication patterns for procs
void subdomain(int rank, int nprocs, int N, int dest[]) { 
	int type;
	int P  = sqrt(nprocs);
	int side = (N / P) + 2; // Number of points on the side of a subdomain
	int iidx;
	int ifdx;
	int jidx;
	int jfdx;	
	int up;
	int right;
	int down;
	int left;
	
	if (rank < P) { 
	// Top row of procs 
		if      (rank % P == 0)    {
			type = 0; // Upper Left Corner	
			jidx = 2;
			jfdx = side - 2;
			iidx = 2;
			ifdx = side - 2;
			up   = 0;
			right= 1;
			down = 1;
			left = 0; 
		}
		else if (rank % P == P - 1) {
			type = 2; // Upper Right Corner
			jidx = 1;
			jfdx = side - 3;
			iidx = 2;
			ifdx = side - 2; 
			up = 0;
			right = 0;
			down = 1;
			left = 1;
		}
		else                       {
			type = 1; // Upper Central
			jidx = 1;
			jfdx = side - 2;
			iidx = 2;
			ifdx = side - 2;
			up = 0;
			right = 1;
			down = 1;
			left = 1;
		}
	}
	else if (rank > P*P - P - 1) {
	// Bottom row of procs			
		if      (rank % P == 0)    {
			type = 6; // Lower Left Corner
			jidx = 2;
			jfdx = side - 2;
			iidx = 1;
			ifdx = side - 3;
			up = 1;
			right = 1;
			down = 0;
			left = 0;
	
		}
		else if (rank % P == P - 1) {
			type = 8; // Lower Right Corner 
			jidx = 1;
			jfdx = side - 3;
			iidx = 1;
			ifdx = side - 3;
			up = 1;
			right = 0;
			down = 0; 
			left = 1;
		}
		else                       {
			type = 7; // Lower Central
			jidx = 1;
			jfdx = side - 2;
			iidx = 1;
			ifdx = side - 3;
			up = 1;
			right = 1;
			down = 0;
			left = 1;
		}

	}
	else {
	// Only three cases left: Right Central, Left Central, 
	// and "Central Central"	
		if      (rank % P == 0)     {
			type = 3; // Left Central
			jidx = 2; 
			jfdx = side - 2;
			iidx = 1;
			ifdx = side - 2;
			up = 1;
			right = 1;
			down = 1;
			left = 0;
		}
		else if (rank % P == P - 1) {
			type = 5; // Right Central
			jidx = 1;
			jfdx = side - 3;
			iidx = 1;
			ifdx = side - 2;
			up = 1;
			right = 0;
			down = 1;
			left = 1;
		}
		else                        {
			type = 4; // Central Central
			jidx = 1;
			jfdx = side - 2;
			iidx = 1;
			ifdx = side - 2;
			up = 1;
			right = 1;
			down = 1;
			left = 1;
		}
	

	}

	if (P == 1) {
		up 	= 0;
		right 	= 0;
		down 	= 0;
		left 	= 0;
	}
	
	dest[0] = type;	
	dest[1] = iidx;
	dest[2] = ifdx;
	dest[3] = jidx;
	dest[4] = jfdx;

	dest[5] = 0;
	for (int i = 0; i < (int)sqrt(nprocs); i++) {
		for (int j = 0; j < (int)sqrt(nprocs); j++) {
			if ((i % 2) == (j % 2)) {
				if (rank == (i*(int)sqrt(nprocs) + j)) {
					dest[5] = 1; // Checkerboard Color
				}
			}	
		}
	}

	dest[6] = up;	
	dest[7] = right;
	dest[8] = down;
	dest[9] = left;
	dest[10] = ifdx - iidx;

}


int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);		
	int nprocs;
	int rank;
	MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank  );	

	double precision = MPI_Wtick();
	double starttime = MPI_Wtime();

	// Preliminary Machinery - read in parameters
	// and broadcast them to  the  working  procs	
	int	 N;
	double   dt;
	double   T;
	double   nu;
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

		
		double dx = 2.0 / abs((double)N - 1);

		outfile = fopen(argv[2], "wb");
		fwrite(&N, sizeof(int), 1, outfile);

		double* x_axis = (double *)malloc(N * sizeof(double));
		for (int i = 0; i < N; i++) {
			x_axis[i] = -1.0 + dx*i;
		}
		fwrite(x_axis, sizeof(double), N, outfile); 
	}

	MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&T, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&nu, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);


	int nsteps = roundf(T / dt);
	double dx = 2.0 / abs((double)N - 1);
	double dy = 2.0 / abs((double)N - 1);
	int ppp = N*N / nprocs; // Points Per Proc

	// Initialize send and recieve buffers for all processes
	double* ubufr = (double *)calloc(sqrt(ppp),sizeof(double));
	double* rbufr = (double *)calloc(sqrt(ppp),sizeof(double));
	double* dbufr = (double *)calloc(sqrt(ppp),sizeof(double));
	double* lbufr = (double *)calloc(sqrt(ppp),sizeof(double));

	
	double* ubufs = (double *)calloc(sqrt(ppp),sizeof(double));
	double* rbufs = (double *)calloc(sqrt(ppp),sizeof(double));
	double* dbufs = (double *)calloc(sqrt(ppp),sizeof(double));
	double* lbufs = (double *)calloc(sqrt(ppp),sizeof(double));




	double* subgrid = (double *)malloc(ppp * sizeof(double)) ;
	double* main_grid;
	double* rearr_main_grid;


	int type[11];// This static array holds the numberpad-type 
		     // of the process in question and sets up 
		     // the indices of the subdomain to loop through
		     // such that the boundaries are not updated
	if (rank == 0) {
		// Instantiate the grid and scatter it to the worker procs
		main_grid = (double *)malloc(N * N * sizeof(double));
		rearr_main_grid = (double *)malloc(N * N * sizeof(double));
		
		for (int i = 0; i < N; i++) {
			main_grid[i]		 = -1.0+i*dx;
			main_grid[N*(N-1)+i] 	 = 1.0-i*dx;
			main_grid[N*i] 		 = -1.0+i*dx;	
			main_grid[N*i+(N-1)] 	 = 1.0-i*dx;
		}
		
		for (int i = 1; i < N-1; i++) {
                	for (int j = 1; j < N-1; j++) {
                        	main_grid[i*N + j] = -((-1.0+dx*i)*(-1.0+dx*j))+cos(11.0*M_PI*(-1.0 +i*dx)/2.0)*sin(8.0*M_PI*(-1.0 + dx*j));
                        }
                }


      		// Recast the grid such that it is scattered correctly
		// This is like unrolling a rank-4 tensor

		int cnt=0;
		for (int i = 0; i < sqrt(nprocs); i++){
			for (int j = 0; j < sqrt(nprocs); j++) {
				for (int k = 0; k < sqrt(ppp); k++) {
					for (int l = 0; l < sqrt(ppp); l++) {
						rearr_main_grid[cnt] = main_grid[N*i*(int)sqrt(ppp) + j*(int)sqrt(ppp) + k*N + l]; 		
						cnt++;
					}
				}
			}
		}	
	}	

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Scatter(rearr_main_grid, ppp, MPI_DOUBLE, subgrid, ppp, MPI_DOUBLE, 0, MPI_COMM_WORLD);	

	

	// Now that each process has been assigned a subdomain of the problem
	// we need to associate with each processor information about which 
	// subdomain elements must be left untouched during the updating due 
	// to being part of the boundary conditions - the subdomain function 
	// does this by returning a set of looping indices used in the main loop
	// for updating each subdomain
	//
	// In addition the subdomain function returns a checkerboarding color
	// which is used to organize the communication pattern
	subdomain(rank, nprocs, N, type); 
	int typ = type[0];
	int xlo = type[1]; // Lower X index
	int xhi = type[2]; // Upper X index
	int ylo = type[3]; // Lower Y index
	int yhi = type[4]; // Upper Y index
	int col = type[5]; // Red  or Black 
	int up  = type[6]; // Do I send and recieve up?
	int right = type[7];//Do I send and recieve up?
	int down = type[8]; //Do I send and recieve up?
	int left = type[9]; //Do I send and recieve up?
	int span = type[10]; // What is the span of my x indices? (for linear indexing)

	// Main loop

	double constx = nu*dt/dx/dx;
	double consty = nu*dt/dy/dy;
	int sd = sqrt(ppp) + 2; // Subdomain sidelength

	// Pad the subdomain on all sides with 
	// room to carry boundary data from other
	// procs - we won't loop over these elements 
	// during the update process 
	
	
	int cnt=0;
	double* nstd_subgrid = (double *)calloc(sd*sd, sizeof(double));
	for (int i = 1; i < sd - 1; i++) {
		for (int j = 1; j < sd - 1; j++) {
			nstd_subgrid[j*sd + i] = subgrid[(j-1)*(sd-2) + (i-1)];
			cnt++;
		}

	}	

	for (int step = 0; step <= nsteps; step++) {
		if (rank == 0) {
			printf("On step %d \n", step);
		}	
		// __________________________________________________________________________


		// Fill up the buffers with the appropriate edges of the grid
		// Must be careful because we need to index one layer of elements
		// into the sides of the grid from every direction
		for (int i = 1; i < sd - 1; i++) {
			if (up == 1) {
				// Upwards buffer
				ubufs[i-1] = nstd_subgrid[i + sd];  	
			}	
			if (right == 1) {
				// Rightwards buffer
				rbufs[i-1] = nstd_subgrid[sd-1 + i*sd - 1]; 
			}
			if (down == 1) {
				// Downards buffer
				dbufs[i-1] = nstd_subgrid[sd*(sd-1) + i - sd];
			}
			if (left == 1) {
				// Leftwards buffer
				lbufs[i-1] = nstd_subgrid[sd*i + 1];
			}
		}	
		// __________________________________________________________________________
				

		// __________________________________________________________________________
		// 	
		// 			-- BEGIN COMMUNICATIONS
		// 
		// __________________________________________________________________________

		for (int color = 0; color <= 1; color++) {
			if (col == color) {
				if (up == 1) {
					MPI_Send(ubufs, sd-2, MPI_DOUBLE, rank - (int)sqrt(nprocs), 0, MPI_COMM_WORLD);
				}
				if (right == 1) {
					MPI_Send(rbufs, sd-2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
				}		
				if (down == 1) {
					MPI_Send(dbufs, sd-2, MPI_DOUBLE, rank + (int)sqrt(nprocs), 0, MPI_COMM_WORLD);
				}
				if (left == 1) {
					MPI_Send(lbufs, sd-2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
				}	
			}
			else {
				if (down == 1) {
					MPI_Recv(dbufr, sd-2, MPI_DOUBLE, rank + (int)sqrt(nprocs), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (left == 1) {
					MPI_Recv(lbufr, sd-2, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				if (up == 1) {
					MPI_Recv(ubufr, sd-2, MPI_DOUBLE, rank - (int)sqrt(nprocs), 0, MPI_COMM_WORLD,  MPI_STATUS_IGNORE);
				}	
				if (right == 1) {
					MPI_Recv(rbufr, sd-2, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
			}
		}
		// __________________________________________________________________________
		//
		//			-- END OF COMMUNICATIONS --			
		//			
		// __________________________________________________________________________


		// Now that the buffers are full of the boundary elements from the 
		// bordering processors, we need to insert them correctly into the 
		// borders of the matrices inside which the subgrids are nested
	


		for (int i = 1; i < sd - 1; i++) {
			if (up == 1){
				nstd_subgrid[i] = ubufr[i-1]; // Upper Buffer 
			}
			if (right == 1){
				nstd_subgrid[i*sd + sd-1] = rbufr[i-1]; // Right Buffer
			}
			if (down == 1) {
				nstd_subgrid[sd*(sd-1) + i] = dbufr[i-1]; // Bottom Buffer
			}
			if (left == 1) {
				nstd_subgrid[i*sd] = lbufr[i-1]; 	
			}
		}


		// With the domains fully in place we can difference the grid
		// and propagate the system one step forward in time
		double* tmp = (double *)calloc(sd*sd, sizeof(double));

		for (int p = 0; p < sd*sd; p++) {
			tmp[p] = nstd_subgrid[p];
		}

		MPI_Barrier(MPI_COMM_WORLD);	
		for (int x = xlo; x <= xhi; x++) {
			for (int y = ylo; y <= yhi; y++) {
				tmp[x*sd + y] = nstd_subgrid[x*sd + y] + constx * (nstd_subgrid[x*sd + (y-1)] - 2.0 * nstd_subgrid[x*sd + y] + nstd_subgrid[x*sd + (y+1)]) + consty * (nstd_subgrid[(x+1)*sd + y] - 2.0 * nstd_subgrid[x*sd + y] + nstd_subgrid[(x-1)*sd + y]); 
			}		
		}	
		

		// Copy back to subgrid
		for (int p = 0; p < sd*sd; p++) {
			nstd_subgrid[p] = tmp[p];
		}
		free(tmp);

		// Reassemble the grid by inverting the method that was used
		// to scatter it	
		MPI_Barrier(MPI_COMM_WORLD); 	
		if (step % (nsteps / 4) == 0) {
			// Pull the subgrid out of the nested subgrid
			for (int i = 1; i < sd - 1; i++) {
				for (int j = 1; j < sd - 1; j++) {
				        subgrid[(i-1)*(sd-2) + (j-1)] = nstd_subgrid[i*sd + j] + (double)rank/100.0;
				}
			}

			MPI_Gather(subgrid, ppp, MPI_DOUBLE, rearr_main_grid, ppp, MPI_DOUBLE, 0, MPI_COMM_WORLD);

			// Grid reassembly into correct format 
			// with correct relative locations of 
			// matrix elements	
			if (rank == 0) {
				int cnt = 0;
				for (int wut = 0; wut < sqrt(nprocs); wut++) {
					for (int row = 0; row < (int)sqrt(ppp); row++) {
						for (int subrow = 0; subrow < (int)sqrt(nprocs); subrow++) {
							for (int item = 0; item < (int)sqrt(ppp); item++) {
								main_grid[cnt] = rearr_main_grid[row*(int)sqrt(ppp) + subrow*ppp + item + ppp*wut*(int)sqrt(nprocs)];
								cnt++;
							}
						}
					}
				}
				fwrite(main_grid, sizeof(double), N*N, outfile);
			}
		}
	}	

	if (rank == 0) {
		free(main_grid);
		free(rearr_main_grid);
	}
	free(nstd_subgrid);
	free(ubufs);
	free(rbufs);
	free(dbufs);
	free(lbufs);
	free(ubufr);
	free(rbufr);
	free(dbufr);
	free(lbufr);

	if (rank == 0) {
		fclose(outfile);
		fclose(inputfile);
		double elapsed_time = MPI_Wtime() - starttime;
		printf("Execution time = %le seconds with precision %le seconds \n", elapsed_time, precision);
	}
	MPI_Finalize();
	return 0;
}

// /projects/e20579/
// create a directory with netid as the name 










