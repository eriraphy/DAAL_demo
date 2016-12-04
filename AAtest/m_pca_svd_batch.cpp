/*Raphael.2016.Aug  @Intel*/
#include "daal.h"
#include "mex.h"
#include "matrix.h"
#include "mxfunction_service.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::pca;


#define inputA prhs[0]
#define outputA plhs[0]
#define outputB plhs[1]


/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 1) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 1 inputs required");
	}
	if (nlhs < 2) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 2 outputs required");
	}
	if (!mxIsDouble(inputA) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train matrix must be double");
	}

	

	/*Training Process*/
	/*Defination*/
	mwSize nrow;
	mwSize ncol;
	double *pxtrain;
	nrow = mxGetM(inputA);
	ncol = mxGetN(inputA);

	pxtrain = mxGetPr(inputA);

	matrix_cr_conv<double>(pxtrain, ncol, nrow);

	/*Create NumericTable*/
	SharedPtr<NumericTable>inputdataA = SharedPtr<NumericTable>(new Matrix<double>(ncol, nrow, pxtrain));

	Batch<double, pca::svdDense> algorithm;
	algorithm.input.set(pca::data, inputdataA);
	algorithm.compute();



	SharedPtr<pca::Result> result = algorithm.getResult();
	SharedPtr<Matrix<double>>eval = staticPointerCast<Matrix<double>, NumericTable>(result->get(pca::eigenvalues));
	SharedPtr<Matrix<double>>evec = staticPointerCast<Matrix<double>, NumericTable>(result->get(pca::eigenvectors));


	/*Test result output*/
	outputA = mxCreateDoubleMatrix(ncol, 1, mxREAL);
	outputMatrix<double>(outputA, ncol, 1, eval);
	outputB = mxCreateDoubleMatrix(ncol, ncol, mxREAL);
	outputMatrix<double>(outputB, ncol, ncol, evec);

	matrix_cr_conv<double>(pxtrain, nrow, ncol);


}



void main() {
	return;
}

