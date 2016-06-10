#include <stdlib.h>
#include <stdio.h>
#include "trid.h"

void trid(double *udg, double *ldg, double *dia, double *vector, int dim) {
        double *retArr = (double *)calloc(dim, sizeof(double));


        for (int i=1; i < dim - 1; i++) {
                retArr[i] = ldg[i-1]*vector[i-1] + dia[i]*vector[i] + udg[i]*vector[i+1];
        }
        // Clean up the corner elements - can't think of a clean way to
        // do this
        retArr[0]   = vector[0]*dia[0] + vector[1]*udg[0];
        retArr[dim-1] = ldg[dim-2]*vector[dim-2] + dia[dim-1]*vector[dim-1];

        for (int i=0; i < dim; i++) {
                vector[i] = retArr[i];
        }
        free(retArr);
        return;
}

