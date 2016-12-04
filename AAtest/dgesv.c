/*******************************************************************************
* Copyright 2009-2015 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <time.h>

/* Auxiliary routines prototypes */
extern void print_matrix( char* desc, int m, int n, double* a, int lda );
extern void print_int_vector( char* desc, int n, int* a );

/* Parameters */
#define N 5000
#define NRHS 5000
#define LDA N
#define LDB N

/* Main program */
int main() {
	int i, j, k = 1;//times
	clock_t begin, end;
	MKL_INT n = N, nrhs = NRHS, lda = LDA, ldb = LDB, info;
	MKL_INT ipiv[N];

	double *a;
	double *a_tmp;
	a = (double *)malloc(sizeof(double)*LDA*N);
	a_tmp = (double *)malloc(sizeof(double)*LDA*N);
	unsigned int t = time(NULL);
	srand(t);
	for (i = 0; i < lda; i++) {
		for (j = 0; j < n; j++) {
				a[i*n + j] = rand()%100;
				a_tmp[i*n + j] = a[i*n + j];
		}
	}
	//print_matrix("Input matrix", lda, n, a, ldb);
	//system("pause");
	double *b;
	b = (double *)malloc(sizeof(double)*LDB*N);
	for (i = 0; i < ldb; i++) {
		for (j = 0; j < n; j++) {
			if (i == j)
				b[i*n + j] = 1;
			else
				b[i*n + j] = 0;
		}
	}

	

	begin = clock();
	for (i = 0; i < k; i++) {
		info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, nrhs, a_tmp, lda, ipiv);
		info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, n, a_tmp, lda, ipiv);
	}
	end = clock();
	if (info > 0) {
		printf("the solution could not be computed.\n");
		exit(1);
	}
	printf(" DGETRF/DGETRI Results\n");
	//print_matrix("dgesv_Solution", n, nrhs, a_tmp, ldb);
	printf("==dgetrf_dgetri_Elapsed_time %f ms==\n", (double)(end - begin) / CLOCKS_PER_SEC);


	begin = clock();
	for (i = 0; i < k; i++) {
	//info=dgesv(LAPACK_ROW_MAJOR, n, nrhs, a, lda, ipiv, b, ldb);
		dgesv(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
	}
	end = clock();
	if (info > 0) {
		printf("the solution could not be computed.\n");
		exit(1);
	}
	printf(" DGESV Results\n");
	//print_matrix( "dgesv_Solution", n, nrhs, b, ldb );
	printf("==dgesv_Elapsed_time %f ms==\n", (double)(end - begin) / CLOCKS_PER_SEC);



	for (i = 0; i < 10; i++) {
		printf("%f\n", a_tmp[i]);
		printf("%f\n", b[i]);

	}
	system("pause");
} 

/* Auxiliary routine: printing a matrix */
void print_matrix( char* desc, int m, int n, double* a, int lda ) {
	int i, j;
	printf( "\n %s\n", desc );
	for( i = 0; i < m; i++ ) {
		for( j = 0; j < n; j++ ) printf( " %6.4f", a[i+j*lda] );
		printf( "\n" );
	}
}

/* Auxiliary routine: printing a vector of integers */
void print_int_vector( char* desc, int n, int* a ) {
	int j;
	printf( "\n %s\n", desc );
	for( j = 0; j < n; j++ ) printf( " %6i", a[j] );
	printf( "\n" );
}
