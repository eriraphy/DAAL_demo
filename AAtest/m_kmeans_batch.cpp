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
using namespace daal::algorithms::kmeans;


#define inputA prhs[0]
#define inputB prhs[1]
#define inputC prhs[2]
#define inputD prhs[3]
#define inputE prhs[4]

#define outputA plhs[0]
#define outputB plhs[1]
#define outputC plhs[2]






/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 3) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 3 inputs required");
	}
	if (nlhs < 3) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 3 outputs required");
	}
	if (!mxIsDouble(inputA) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train matrix must be double");
	}
	if (!mxIsDouble(inputB) || mxIsComplex(inputB))
	{
		mexErrMsgIdAndTxt("test:prhs", "Number of Clusters must be double");
	}
	if (!mxIsDouble(inputC) || mxIsComplex(inputC))
	{
		mexErrMsgIdAndTxt("test:prhs", "Number of Iterations must be double");
	}

	/*Training Process*/
	/*Defination*/
	mwSize nrow;
	mwSize ncol;
	double *pxtrain;
	double *pcent;
	double acct;
	size_t nClusters;
	size_t nIterations;
	nrow = mxGetM(inputA);
	ncol = mxGetN(inputA);
	
	nClusters = mxGetScalar(inputB);
	nIterations = mxGetScalar(inputC);

	pxtrain = mxGetPr(inputA);

	matrix_cr_conv<double>(pxtrain, ncol, nrow);

	SharedPtr<NumericTable>inputdataA = SharedPtr<NumericTable>(new Matrix<double>(ncol, nrow, pxtrain));
	SharedPtr<NumericTable> centroids = SharedPtr<NumericTable>(new Matrix<double>(ncol, nClusters, NumericTable::notAllocate));


	//services::Environment::getInstance()->setNumberOfThreads(2);
	if (nrhs < 4) {
		acct = 0.01;
	}
	else {
		acct = mxGetScalar(inputD);
	}

	if (nrhs < 5) {
		kmeans::init::Batch<double, kmeans::init::randomDense> init(nClusters);
		init.input.set(kmeans::init::data, inputdataA);
		init.compute();
		centroids = init.getResult()->get(kmeans::init::centroids);
	}
	else {
		pcent = mxGetPr(inputE);
		matrix_cr_conv(pcent, ncol, nClusters);
		centroids = SharedPtr<NumericTable>(new Matrix<double>(ncol, nClusters, pcent));
	}






	Batch<> algorithm(nClusters, nIterations);
	algorithm.input.set(kmeans::data, inputdataA);
	algorithm.input.set(kmeans::inputCentroids, centroids);
	algorithm.parameter.accuracyThreshold = acct;

	algorithm.compute();

	SharedPtr<Matrix<int>>assign = staticPointerCast<Matrix<int>, NumericTable>(algorithm.getResult()->get(kmeans::assignments));
	SharedPtr<Matrix<double>>centro = staticPointerCast<Matrix<double>, NumericTable>(algorithm.getResult()->get(kmeans::centroids));
	SharedPtr<Matrix<int>>nit = staticPointerCast<Matrix<int>, NumericTable>(algorithm.getResult()->get(kmeans::nIterations));


	/*Test result output*/
	outputA = mxCreateDoubleMatrix(nrow, 1, mxREAL);
	outputMatrix<int>(outputA, nrow, 1, assign);


	outputB = mxCreateDoubleMatrix(nClusters, ncol, mxREAL);
	outputMatrix<double>(outputB, nClusters, ncol, centro);

	outputC = mxCreateDoubleScalar((*nit)[0][0]);

	matrix_cr_conv<double>(pxtrain, nrow, ncol);
	
	if (nrhs >= 5) {
		matrix_cr_conv(pcent, nClusters, ncol);
	}
}



void main() {
	return;
}