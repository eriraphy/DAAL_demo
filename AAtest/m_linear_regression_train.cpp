/*Raphael.2016.Aug  @Intel*/
#include "daal.h"
#include "mex.h"
#include "mxfunction_service.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms::linear_regression;
using namespace daal::algorithms;

services::SharedPtr<training::Result> trainingResult;

#define inputA prhs[0]
#define inputB prhs[1]
#define outputA plhs[0]
#define outputB plhs[1]

/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 2) {
		mexErrMsgIdAndTxt("test:nrhs","2 inputs required");
	}
	if (nlhs < 1) {
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


	if (mxGetM(inputB)!= mxGetM(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train label must be same length as train matrix");
	}


	
	/*Training Process*/
	mwSize nrow;
	mwSize ncol;
	mwSize ny;
	double *pxtrain;
	double *pytrain;
	nrow = mxGetM(inputA);
	ncol = mxGetN(inputA);
	ny = mxGetN(inputB);
	pxtrain = mxGetPr(inputA);
	pytrain = mxGetPr(inputB);
	matrix_cr_conv<double>(pxtrain, ncol, nrow );
	
	/*Create NumericTable*/
	SharedPtr<NumericTable>inputdataA = SharedPtr<NumericTable>(new Matrix<double>(ncol, nrow, pxtrain));
	SharedPtr<NumericTable>inputdataB = SharedPtr<NumericTable>(new Matrix<double>(ny, nrow, pytrain));
	training::Batch<> algorithm;
	algorithm.input.set(training::data, inputdataA);
	algorithm.input.set(training::dependentVariables, inputdataB);
	algorithm.compute();
	trainingResult = algorithm.getResult();
	SharedPtr<Matrix<double>>coeff = staticPointerCast<Matrix<double>, NumericTable>(trainingResult->get(training::model)->getBeta());
	
	/*Coefficient output*/
	outputA = mxCreateDoubleMatrix(ncol, 1, mxREAL);
	outputMatrix<double>(outputA, ncol, 1, coeff, 1);
	outputB = mxCreateDoubleScalar((*coeff)[0][0]);

	matrix_cr_conv<double>(pxtrain, nrow, ncol);
}


void main(){
	return;
}

