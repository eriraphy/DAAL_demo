/*Raphael.2016.Aug  @Intel*/
#include "daal.h"
#include "mex.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::linear_regression;
using namespace daal::algorithms;

services::SharedPtr<prediction::Result> predictionResult;

#define inputA prhs[0]
#define inputB prhs[1]
#define inputC prhs[2]
#define outputA plhs[0]


/*Usage:
convert colomn-domain matrix to row-domain matrix(mexArray to C Array)
input_matrix: colomn-domain; output_matrix: row-domain
matrix_cr_conv(double *cdmatrix, int ncol, int nrow)

convert row-domain matrix to colomn-domain matrix(C Array to mexArray)
input_matrix: row-domain; output_matrix: colomn-domain
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
	if (nrhs <2) {
		mexErrMsgIdAndTxt("test:nrhs","At least 2 inputs required");
	}
	if (nlhs <1) {
		mexErrMsgIdAndTxt("test:nrhs","At least 1 outputs required");
	}
	if (!mxIsDouble(inputA) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs","Train matrix must be double");
	}
	if (!mxIsDouble(inputB) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs","Train label must be double");
	}
	if (!mxIsDouble(inputC) || mxIsComplex(inputC))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train label must be double");
	}


	
	/*Test process*/
	mwSize nrow, ncol, nweights;
	double *pxtest;
	double *pweight;
	double bias;
	double *poutA;
	nrow = mxGetM(inputA);//250
	ncol = mxGetN(inputA);//10
	pxtest = mxGetPr(inputA);
	pweight = mxGetPr(inputB);
	nweights = mxGetM(inputB);
	bias = mxGetScalar(inputC);
	matrix_cr_conv(pxtest, ncol, nrow);
	
	double *pmodel;
	pmodel = (double*)malloc(sizeof(double)*nweights);
	pmodel[0] = bias;
	for (int i = 0; i < nweights; i++) {
		pmodel[i + 1] = pweight[i];
	}
	/*Create NumericTable*/
	SharedPtr<NumericTable>inputdataA = SharedPtr<NumericTable>(new Matrix<double>(ncol, nrow, pxtest));
	SharedPtr<NumericTable>inputdataB = SharedPtr<NumericTable>(new Matrix<double>(1, nweights + 1, pmodel));
	SharedPtr<linear_regression::Parameter>para;
	////
	SharedPtr<linear_regression::Model>modelpara;

	prediction::Batch<> algorithm_test;
	algorithm_test.input.set(prediction::data, inputdataA);
	algorithm_test.input.set(prediction::model, modelpara);
	algorithm_test.compute();
	predictionResult = algorithm_test.getResult();
	SharedPtr<Matrix<double>>pred = staticPointerCast<Matrix<double>, NumericTable>(predictionResult->get(prediction::prediction));
	

	/*Test result output*/
	outputA = mxCreateDoubleMatrix(nrow, 1, mxREAL);
	poutA = mxGetPr(outputA);
	for (int i = 0; i < nrow; i++) {
		poutA[i] = (*pred)[0][i];
	}

	matrix_cr_conv(pxtest, nrow, ncol);
}


void main(){
	return;
}

