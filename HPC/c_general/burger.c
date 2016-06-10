#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define M_PI    3.14159265358979323846



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

	FILE* fileid = fopen(argv[2], "w");	

        // start reading input data using function fscanf here
        int N;
	double dt;
	double T;
        fscanf(inputfile ,"%d %lf %lf", &N, &dt, &T); // read an integer N from the input file
        fclose(inputfile);





	// Define initial constants 
        double nu        = 1.0 / 100.0 / M_PI;
        double dx        = 2 / (N - 1.0);
        int nSteps      = roundf(T / dt);



        // Create array of initial u-Vals
	double *x	= (double *)malloc(N * sizeof(double));
        double *u       = (double *)malloc(N * sizeof(double));
	double *uhat 	= (double *)malloc(N * sizeof(double));
        // This is the initial condition   
        for (int i=0; i < N; i++) {
                u[i] = -sin(M_PI*(-1.0 + dx*i));
		x[i] = dx*i;
        }
		
	// Main loop	
	// MaCormack method 
	
	fwrite(&N, sizeof(int), 1, fileid);
	fwrite(x, sizeof(double), (int)N, fileid);
	
	int cntr = 0; // Keeps track of when write to file
	for (int ts=1; ts <= nSteps; ts++) {
		if (ts % 2 == 0) {
			// Forward differencing step - loops over central portion of
			// The vector, i.e. elements on ( 0, N )
			for (int i=1; i < (int)N - 1; i++) {

				uhat[i] = u[i] + ((dt/dx) * ( (-(u[i+1]*u[i+1]/2.0)) + (nu*(u[i+1] - u[i]) / dx)
					+ (u[i]*u[i]/2.0) - (nu*(u[i] - u[i-1])/dx) ));	
					
			}
				
	
			for (int i=1; i < (int)N - 1; i++) {
				
				u[i] = 0.5*(uhat[i]+u[i] + (dt/dx)*( -(uhat[i]*uhat[i]/2) + nu*(uhat[i+1] - uhat[i]) / dx
				     + (uhat[i-1]*uhat[i-1]/2.0) - nu*(uhat[i]-uhat[i-1])/dx ));
			
			}	
		
	
		}
		else {
			// Backward differencing step
			for (int i=1; i < (int)N - 1; i++) {
				
				uhat[i] = u[i] + dt/dx * (-u[i]*u[i]/2 + nu*(u[i+1] - u[i])/dx
					+ u[i-1]*u[i-1]/2 - nu*(u[i] - u[i-1])/dx);
			
			}

			
			for (int i=1; i < (int)N - 1; i++) {
				
				u[i] = 0.5*(uhat[i] + u[i] + dt/dx*( -uhat[i+1]*uhat[i+1]/2 + nu*(uhat[i+1] - uhat[i])/dx
				     + uhat[i]*uhat[i]/2 - nu*(uhat[i] - uhat[i-1]) / dx ));	

			}	


		}
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
	free(u);
	free(uhat);	
	
	
	fclose(fileid);
        printf("Time elapsed: %g seconds\n", (double)(clock()-start)/CLOCKS_PER_SEC);

	return 0;
}
