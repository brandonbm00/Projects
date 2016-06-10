#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <lapacke.h>
#include <time.h>
#include <cblas.h>
#include "trid.h"

#define M_PI	3.14159265358979323846



int main(int argc, char* argv[]) {


	clock_t start = clock();

        // Preliminaries: Warnings and parameter input
        if (argc  != 3) {
                printf("Incorrect usage: enter the input file name, then output file name \n");
                return 0;
        }
        FILE* inputfile = fopen(argv[1], "r");
        if (!inputfile) {
                printf("Unable to open input file \n");
                return 0;
        }

        FILE* fileid = fopen(argv[2], "wb");

        // start reading input data using function fscanf here
        int N;
        double dt;
        double T;
        fscanf(inputfile ,"%d %lf %lf", &N, &dt, &T); // read an integer N from the input file
        fclose(inputfile);

	// Define initial constants 
	double nu	= 1.0 / 100.0 / M_PI;
	double dx	= 2.0 / (N - 1.0);
	int nSteps 	= roundf(T / dt);
	
	double k1A 	= 1.0 + nu * (dt / dx / dx);
	double k2A	= nu * (dt / 2) / (dx * dx);
		
	double k1B	= 1.0 - nu * (dt / dx / dx);
	double k2B	= nu * (dt / 2) / (dx * dx);
	// Create array of initial u-Vals

	double *x	= (double *)calloc(N, sizeof(double));	
	double *u	= (double *)calloc(N, sizeof(double));
	double *uhat	= (double *)calloc(N, sizeof(double));

	for (int i=0; i < N; i++) {
                u[i] = -sin(M_PI*(-1.0 + dx*i));
		x[i] = dx*i;
        }

	fwrite(&N, sizeof(int), 1, fileid);		
	fwrite(x, sizeof(double), (int)N, fileid);
	// This is basically MatLab linspace(-1,1,N)	


	
	// Create the necessary portions of the tridiagonal matrix
	// with which we can call dgtsv_


	// Bands of Matrix "A"
	double *udgA 	= (double *)calloc(N, sizeof(double));
	double *ldgA 	= (double *)calloc(N, sizeof(double));
	double *diaA 	= (double *)calloc(N, sizeof(double));
	
	double *udgAtmp	= (double *)calloc(N, sizeof(double));
	double *ldgAtmp	= (double *)calloc(N, sizeof(double));
	double *diaAtmp	= (double *)calloc(N, sizeof(double));
	// Bands of Matrix "B"
	double *udgB 	= (double *)calloc(N, sizeof(double));
	double *ldgB	= (double *)calloc(N, sizeof(double));
	double *diaB 	= (double *)calloc(N, sizeof(double));

	// The leading and trailing elements of dia are 1.0

	diaA[0]   = 1.0;
	diaA[(int)N-1] = 1.0;


	// There are now (N - 2) * 3 nonzero elements to fill up 
	// split evenly amongst the three diagonals. That means
	// we can fill all the rest of the elements with careful
	// indexing of a single forloop

	for (int i=0; i < N-2; i++) {
		udgA[i+1]	= -k2A; // Upper Diagonal
		ldgA[i]   	= -k2A; // Lower Diagonal
		diaA[i+1] 	=  k1A; // Main  Diagonal

		udgAtmp[i+1] 	= -k2A; // Temporary arrays
		ldgAtmp[i] 	= -k2A;	// Because of LAPack
		diaAtmp[i+1] 	=  k1A;

		udgB[i+1]	=  k2B; 
		ldgB[i]   	=  k2B;
		diaB[i+1] 	=  k1B;		

	}



	// Main loop
	int cntr = 0;
	for (int ts = 1; ts <= nSteps; ts++) {
		if (ts % 2 == 0) {
			for (int i=1; i < (int)N - 1; i++) {	
				uhat[i] = u[i] - (dt/dx)*(u[i+1]*u[i+1]/2.0 - u[i]*u[i]/2.0);
			}
			for (int i=1; i < (int)N - 1; i++) {
				u[i] = 0.5*(uhat[i]+u[i]-(dt/dx)*(uhat[i]*uhat[i]/2.0 - uhat[i-1]*uhat[i-1]/2.0));	
			}
		}
		else {
			for (int i=1; i < (int)N - 1; i++) {
				uhat[i] = u[i] - (dt/dx)*(u[i]*u[i]/2.0 - u[i-1]*u[i-1]/2.0);
			}
			for (int i=1; i < (int)N - 1; i++) {
				u[i] = 0.5*(uhat[i]+u[i]-(dt/dx)*(uhat[i+1]*uhat[i+1]/2.0 - uhat[i]*uhat[i]/2.0)); 

			}
	
			
		}
		
		
		// Premultiply B into U
		trid(udgB, ldgB, diaB, u, N);

		//     Call to BLAS		
		//     x = A \ B   solves A*x = B   for x. 
		// So  u = A \ B*u solves A*u = B*u for u.	
		lapack_int info	= 0 ;
		lapack_int one 	= 1;
		lapack_int dim  = (int)N;
		dgtsv_(&dim, &one, ldgA, diaA, udgA, u, &dim, &info); 
		// Restore the matrix A to its original state
		for (int i=0; i < N; i++) {
			udgA[i] = udgAtmp[i];
			ldgA[i] = ldgAtmp[i];
			diaA[i] = diaAtmp[i];
	
		}
		diaA[0] 	 = 1.0;
		diaA[(int)N - 1] = 1.0;
		
		// This structure controls when to write to the file 
                // 


                if (cntr == nSteps / 5) {
                        cntr = 0;
                }
                if (cntr == 0) {
			fwrite(u, sizeof(double), (int)N, fileid);
                }
		
		cntr += 1;	


	}


	free(ldgA);
	free(udgA);
	free(diaA);
	free(ldgAtmp);
	free(udgAtmp);
	free(diaAtmp);
	free(ldgB);
	free(udgB);
	free(diaB);	

	fclose(fileid);
        printf("Time elapsed: %g seconds\n", (double)(clock()-start)/CLOCKS_PER_SEC);

	
	return 0;
}
