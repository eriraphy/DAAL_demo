/* file: m_daal_abs.cpp */
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
!    Using Intel DAAL on Matlab example of abs algorithm.
!
!******************************************************************************/


#include "daal.h"
#include "mex.h"
#include "matrix.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::math;


#define inputA prhs[0]
#define outputA plhs[0]


/*Usage:
convert column-domain matrix to row-domain matrix(mexArray to C Array)
input_matrix: column-domain; output_matrix: row-domain
matrix_cr_conv(double *cdmatrix, int ncol, int nrow)

convert row-domain matrix to column-domain matrix(C Array to mexArray)
input_matrix: row-domain; output_matrix: column-domain
matrix_cr_conv(double *rdmatrix, int nrow, int ncol)
*/
void matrix_cr_conv(double *ipmatrix, int dx, int dy) {
	int i;
	int j;
	double *opmatrix;
	opmatrix = (double *)malloc(sizeof(double)*dx*dy);
	for (i = 0; i < dy; i++) {
		for (j = 0; j < dx; j++) {
			opmatrix[i*dx + j] = ipmatrix[i + j*dy];
		}
	}
	for (i = 0; i < dy*dx; i++) {
		ipmatrix[i] = opmatrix[i];
	}
	free(opmatrix);
	return;
}


/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	/*Input Arguments Check*/
	if (nrhs < 1) {
		mexErrMsgIdAndTxt("Sample:nrhs", "At least 1 input required");
	}
	if (nlhs < 1) {
		mexErrMsgIdAndTxt("Sample:nrhs", "At least 1 output required");
	}
	if (!mxIsDouble(inputA) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("Sample:prhs", "Input matrix must be double");
	}


	/*Defination*/
	mwSize nrow;
	mwSize ncol;
	double *pxtrain;
	double *poutA;

	/*Define the Size and Pointer of Input Matrix*/
	nrow = mxGetM(inputA);
	ncol = mxGetN(inputA);
	pxtrain = mxGetPr(inputA);

	/*Convert MexArray to C Array (Matlab to C++)*/
	matrix_cr_conv(pxtrain, ncol, nrow);

	/*Create an Intel DAAL NumericTable*/
	SharedPtr<NumericTable>inputdataA = SharedPtr<NumericTable>(new Matrix<double>(ncol, nrow, pxtrain));
	
	/* Create an algorithm */
	abs::Batch<double> abs;

	/* Set an input object for the algorithm */
	abs.input.set(abs::data, inputdataA);

	/* Compute Abs function */
	abs.compute();	
	
	/*Define Output Pointer*/
	SharedPtr<Matrix<double>>result = staticPointerCast<Matrix<double>, NumericTable>(abs.getResult()->get(abs::value));
	
	/*Create Output Matlab Matrix*/
	//outputA = mxCreateDoubleMatrix(nrow,ncol, mxREAL);

	/*Define Output Pointer*/
	poutA = mxGetPr(outputA);
	int i;
	for (i = 0; i < nrow*ncol; i++) {
		poutA[i] = (*result)[0][i];
	}

	/*Convert C Array to MexArray (C++ to Matlab)*/
	matrix_cr_conv(poutA, nrow, ncol);
	

}


/*Main Function is not Necessary*/
void main() {
	return;
}