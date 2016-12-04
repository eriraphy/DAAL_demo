/* file: m_dgesv.c */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation All Rights Reserved.
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

/*
!  Content:
!    Using Intel MKL on Matlab example of matrix inverse computation.
!
!******************************************************************************/
#include "mkl.h"
#include "stdlib.h"
#include "stdio.h"
#include "mex.h"
#include "matrix.h"



#define inputA prhs[0]
#define outputA plhs[0]

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
	
	double *ipmatrix;
	double *poutA;
	ipmatrix = mxGetPr(inputA);

	size_t N;
	N = mxGetM(inputA);

	MKL_INT n = N, m = N, lda = N, ldb = N, info;
	MKL_INT *ipiv;
	ipiv = (double*)malloc(sizeof(double)*N);


	double *b;
	b = (double *)malloc(sizeof(double)*ldb*n);
	for (int i = 0; i < ldb; i++) {
		for (int j = 0; j < n; j++) {
			if (i == j)
				b[i*n + j] = 1;
			else
				b[i*n + j] = 0;
		}
	}
	//dgesv(&n, &m, ipmatrix, &lda, ipiv, b, &ldb, &info);
	info=LAPACKE_dgesv(LAPACK_ROW_MAJOR,n,m,ipmatrix,lda,ipiv,b,ldb);

	if (info > 0) {
		mexPrintf("the solution could not be computed.\n");
		exit(1);
	}

	outputA = mxCreateDoubleMatrix(n, m, mxREAL);
	poutA = mxGetPr(outputA);
	int i;
	for (i = 0; i < n*m; i++) {
		poutA[i] = b[i];
	}
}


void main() {

	return;
}