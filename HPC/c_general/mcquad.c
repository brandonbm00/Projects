#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include "func.h"

//double func(double num) {
//	num = cos(num);
//	return num;
//}

int main(int argc, char* argv[]) {
	clock_t start = clock();
	
	// Preliminaries: Warnings and parameter input
	if (argc != 2) {
		printf("Incorrect usage: only enter the input data file name \n");
		return 0;
	}
	FILE* inputfile = fopen(argv[1], "r");
	if (!inputfile) {
		printf("Unable to open input file \n");
		return 0;
	}

	// start reading input data using function fscanf here
	int N;
	fscanf(inputfile , "%d", &N); // read an integer N from the input file
	fclose(inputfile);


	srand48(time(NULL));
	
	printf("Using %d Samples... \n", N);

	// Declare directly the array of log([ -----N Rands -----])

	double *logArr = (double *)malloc(N * sizeof(double));

	// Insert the values according to the importance sampling PDF
	int i;
	float avg = 0.0;
	for (i=0; i < N; i++) {
		double x = drand48();
		logArr[i] = -log(x);
		avg += -log(x);
	}
	avg = avg / N;	

	// Main loop


	double q	= 0.0; // Will become the result of the integration
	double var 	= 0.0; // Will become to sample variance	

	int j;
	for (j=0; j < N; j++) {
		q += func(logArr[j]);
		var += ((func(logArr[j]) - avg) * (func(logArr[j]) - avg));
		
	}
	q = q / (float)N; 		// The integral
	var = var / ((float)N - 1);	// The variance
	printf("q is equal to %f, and the sample variance is %f \n", q, var);

	printf("Time elapsed: %g seconds\n", (float)(clock()-start)/CLOCKS_PER_SEC);	
	return 0;
}

 
