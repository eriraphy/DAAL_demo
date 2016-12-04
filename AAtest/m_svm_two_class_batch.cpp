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

	//Environment::getInstance()->setNumberOfThreads(4);

	/* Model object for SVM algorithm */
	SharedPtr<svm::training::Result> trainingResult;
	SharedPtr<classifier::prediction::Result> predictionResult;
	

	//Input format defination
	//#define inputA traindata
	//#define inputB trainlabel
	//#define inputC testdata
	//#define inputD testlabel
	//#define inputE number_of_classes
	//#define inputF max_number_of_iterartions
	//#define inputG C_boxconstrain
	//#define inputH accuracy_threshold
	//#define inputI kernel_function
	//#define inputJ RBF_kernel_sigma

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


	if (nrhs < 5)
		iter = 200;
	else
		iter = mxGetScalar(inputE);

	if (nrhs < 6)
		boxC = 1;
	else
		boxC = mxGetScalar(inputF);
	
	if (nrhs < 7)
		accu = 0.1;
	else {
		accu = mxGetScalar(inputG);
	}

	if (nrhs < 8)
		buf = "linear";
	else {
		int buflen = mxGetN(inputH) + 1;
		buf = (char *)mxCalloc(buflen, sizeof(char));
		int info = mxGetString(inputH, buf, buflen);
	}

	string kernelfunc(buf);
	

	if (nrhs < 9) 
		sigma = 1.0e3;
	else {
		sigma = mxGetScalar(inputI);
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

	training::Batch<> algorithm_tr;
	algorithm_tr.parameter.kernel = kernel;
	algorithm_tr.parameter.cacheSize = 160000000;
	algorithm_tr.parameter.maxIterations = iter;
	algorithm_tr.parameter.accuracyThreshold = accu;
	algorithm_tr.parameter.C = boxC;
	//algorithm.parameter.tau = 1.0e2;


	algorithm_tr.input.set(classifier::training::data, trainData);
	algorithm_tr.input.set(classifier::training::labels, trainGroundTruth);
	algorithm_tr.compute();
	trainingResult = algorithm_tr.getResult();

	

	/*Test process*/	
	prediction::Batch<> algorithm_te;
	algorithm_te.parameter.kernel = kernel;
	algorithm_te.input.set(classifier::prediction::data, testData);
	algorithm_te.input.set(classifier::prediction::model,
		trainingResult->get(classifier::training::model));
	algorithm_te.compute();
	predictionResult = algorithm_te.getResult();
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

