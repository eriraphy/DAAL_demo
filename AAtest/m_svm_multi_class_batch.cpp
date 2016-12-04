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
using namespace daal::algorithms::svm;


#define inputA prhs[0]
#define inputB prhs[1]
#define inputC prhs[2]
#define inputD prhs[3]
#define inputE prhs[4]
#define inputF prhs[5]
#define inputG prhs[6]
#define inputH prhs[7]
#define inputI prhs[8]
#define inputJ prhs[9]
#define outputA plhs[0]
//#define outputB plhs[1]


/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 4) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 4 inputs required");
	}
	if (nlhs < 1) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 2 outputs required");
	}
	if (!mxIsDouble(inputA) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train matrix must be double");
	}
	if (!mxIsDouble(inputB) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train label must be double");
	}
	if (!mxIsDouble(inputC) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Test matrix must be double");
	}
	if (!mxIsDouble(inputD) || mxIsComplex(inputD))
	{
		mexErrMsgIdAndTxt("test:prhs", "Test label must be double");
	}

	if (mxGetM(inputB) != mxGetM(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train label must be same length as train matrix");
	}
	if (mxGetM(inputC) != mxGetM(inputD))
	{
		mexErrMsgIdAndTxt("test:prhs", "Test label must be same length as test matrix");
	}
	if (mxGetN(inputC) != mxGetN(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Test matrix must be same width as train matrix");
	}
	if (mxGetN(inputB) != 1 && mxGetM(inputB) != 1)
	{
		mexErrMsgIdAndTxt("test:prhs", "Train label must have 1 column");
	}
	if (mxGetN(inputD) != 1 && mxGetM(inputD) != 1)
	{
		mexErrMsgIdAndTxt("test:prhs", "Test label must have 1 column");
	}

	//Environment::getInstance()->setNumberOfThreads(2);

	

	/*Training Process*/
	/*Defination*/
	mwSize ltr;
	mwSize nfeature;
	mwSize lte;
	mwSize ny;
	double *pxtrain;
	double *pytrain;
	double *pxtest;
	double *pytest;
	double *poutA;
	double iter;
	double boxC;
	double accu;
	double nClasses;
	char *buf;
	
	double sigma;
	ltr = mxGetM(inputA);
	lte = mxGetM(inputC);
	nfeature = mxGetN(inputA);
	ny = 1;

	pxtrain = mxGetPr(inputA);
	pytrain = mxGetPr(inputB);
	pxtest = mxGetPr(inputC);
	pytest = mxGetPr(inputD);

	matrix_cr_conv(pxtrain, nfeature, ltr);
	matrix_cr_conv(pxtest, nfeature, lte);

	nClasses = mxGetScalar(inputE);

	if (nrhs < 6)
		iter = 200;
	else
		iter = mxGetScalar(inputF);

	if (nrhs < 7)
		boxC = 1;
	else
		boxC = mxGetScalar(inputG);
	
	if (nrhs < 8)
		accu = 0.1;
	else {
		accu = mxGetScalar(inputH);
	}

	if (nrhs < 9)
		buf = "linear";
	else {
		int buflen = mxGetN(inputI) + 1;
		buf = (char *)mxCalloc(buflen, sizeof(char));
		int info = mxGetString(inputI, buf, buflen);
	}

	string kernelfunc(buf);
	

	if (nrhs < 10) 
		sigma = 1.0e3;
	else {
		sigma = mxGetScalar(inputJ);
	}
    /*Create NumericTable*/
	SharedPtr<NumericTable>trainData = SharedPtr<NumericTable>(new Matrix<double>(nfeature, ltr, pxtrain));
	SharedPtr<NumericTable>trainGroundTruth = SharedPtr<NumericTable>(new Matrix<double>(ny, ltr, pytrain));
	SharedPtr<NumericTable>testData = SharedPtr<NumericTable>(new Matrix<double>(nfeature, lte, pxtest));
	SharedPtr<NumericTable>testGroundTruth = SharedPtr<NumericTable>(new Matrix<double>(ny, lte, pytest));

	/* Define kernel function */
	SharedPtr<kernel_function::KernelIface> kernel;
	if (kernelfunc == "linear") {
		kernel = SharedPtr<kernel_function::KernelIface>(new kernel_function::linear::Batch<>());
	}
	else if (kernelfunc == "rbf") {
		kernel_function::rbf::Batch<> rbf;
		rbf.parameter.sigma = sigma;
		kernel = SharedPtr<kernel_function::KernelIface>(new kernel_function::rbf::Batch<>(rbf));
	}

	services::SharedPtr<svm::training::Batch<> > training(new svm::training::Batch<>());
	services::SharedPtr<svm::prediction::Batch<> > prediction(new svm::prediction::Batch<>());
	training->parameter.kernel = kernel;
	training->parameter.cacheSize = 160000000;
	training->parameter.maxIterations = iter;
	training->parameter.accuracyThreshold = accu;
	training->parameter.C = boxC;
	//algorithm.parameter.tau = 1.0e2;


	
	multi_class_classifier::training::Batch<> algorithm_tr;

	algorithm_tr.parameter.nClasses = nClasses;
	algorithm_tr.parameter.training = training;
	algorithm_tr.parameter.prediction = prediction;

	algorithm_tr.input.set(classifier::training::data, trainData);
	algorithm_tr.input.set(classifier::training::labels, trainGroundTruth);

	algorithm_tr.compute();

	services::SharedPtr<multi_class_classifier::training::Result> trainingResult = algorithm_tr.getResult();

	/*Test process*/
	multi_class_classifier::prediction::Batch<> algorithm_te;
	algorithm_te.parameter.nClasses = nClasses;
	algorithm_te.parameter.training = training;
	algorithm_te.parameter.prediction = prediction;

	algorithm_te.input.set(classifier::prediction::data, testData);
	algorithm_te.input.set(classifier::prediction::model,
		trainingResult->get(classifier::training::model));

	algorithm_te.compute();

	services::SharedPtr<classifier::prediction::Result>predictionResult = algorithm_te.getResult();


	SharedPtr<Matrix<double>>pred = staticPointerCast<Matrix<double>, NumericTable>(predictionResult->get(classifier::prediction::prediction));
	
	

	outputA = mxCreateDoubleMatrix(lte, 1, mxREAL);
	poutA = mxGetPr(outputA);
	for (int i = 0; i < lte; i++) {
		if ((*pred)[0][i] >= 0) {
			poutA[i] = 1;
		}
		else {
			poutA[i] = -1;
		}
	}

	matrix_cr_conv(pxtrain, ltr, nfeature);
	matrix_cr_conv(pxtest, lte, nfeature);
}



void main() {
	return;
}

