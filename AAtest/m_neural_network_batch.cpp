/*Raphael.2016.Aug  @Intel*/
#include "daal.h"
#include "mex.h"
#include "matrix.h"
#include "neural_network_service.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms; 
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;


#define inputA prhs[0]
#define inputB prhs[1]
#define inputC prhs[2]
#define inputD prhs[3]
#define inputE prhs[4]
#define outputA plhs[0]
#define outputB plhs[1]

services::SharedPtr<prediction::Model> predictionModel;
services::SharedPtr<prediction::Result> predictionResult;


/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 4) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 4 inputs required");
	}
	if (nlhs < 2) {
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
	int nclass;
	int isize = 5;
	int colsize = 1;
	double *poutA;
	double *poutB;	

	ltr = mxGetM(inputA);
	lte = mxGetM(inputC);
	nfeature = mxGetN(inputA);

	pxtrain = mxGetPr(inputA);
	pytrain = mxGetPr(inputB);
	pxtest = mxGetPr(inputC);
	pytest = mxGetPr(inputD);
	nclass = mxGetScalar(inputE);

	matrix_cr_conv(pxtrain, nfeature, ltr);
	matrix_cr_conv(pxtest, nfeature, lte);

	//services::Environment::getInstance()->setNumberOfThreads(2);

	/*Training process*/
	SharedPtr<Tensor> trainingData = Matrix_Tensor<double>(pxtrain, ltr, colsize, isize, isize);
	SharedPtr<Tensor> trainingGroundTruth = Matrix_Tensor<double>(pytrain, ltr);

	training::Batch<double> train_net;
	train_net.parameter.batchSize = 100;
	training::Topology layersConfiguration = configureNet(nclass);
	train_net.initialize(trainingData->getDimensions(), layersConfiguration);

	train_net.input.set(training::data, trainingData);
	train_net.input.set(training::groundTruth, trainingGroundTruth);

	SharedPtr<optimization_solver::sgd::Batch<double>>sgdAlgorithm(new optimization_solver::sgd::Batch<double>());

	double learningRate = 0.01;
	sgdAlgorithm->parameter.learningRateSequence = 
		SharedPtr<NumericTable>(new HomogenNumericTable<double>(1, 1, NumericTable::doAllocate, learningRate));

	train_net.parameter.optimizationSolver = sgdAlgorithm;
	train_net.compute();
	SharedPtr<training::Model> trainingModel = train_net.getResult()->get(training::model);
	predictionModel = trainingModel->getPredictionModel<double>();


	/*Test process*/
	SharedPtr<Tensor> predictionData = Matrix_Tensor<double>(pxtest, lte, colsize, isize, isize);
	SharedPtr<Tensor> predictionGroundTruth = Matrix_Tensor<double>(pytest, lte);

	prediction::Batch<double> test_net;
	test_net.input.set(prediction::model, predictionModel);
	test_net.input.set(prediction::data, predictionData);

	test_net.compute();
	predictionResult = test_net.getResult();


	int *pred = TensorPtr_alloc<int>(predictionResult->get(prediction::prediction), lte);

	/////////////////////////////////////////////////////////////////////////////////
	//Collection<size_t> dims = trainingData->getDimensions();

	/////////////////////////////////////////////////////////////////////////////////

	/*Test result output*/
	outputA = mxCreateDoubleMatrix(lte, nclass, mxREAL);
	poutA = mxGetPr(outputA);
	for (int i = 0; i < lte*nclass; i++) {
		poutA[i] = pred[i];
	}
	matrix_cr_conv(poutA, lte, nclass);

	
	outputB = mxCreateDoubleMatrix(lte, 1, mxREAL);
	poutB = mxGetPr(outputB);
	for (int i = 0; i < lte*1; i++) {
		poutB[i] = 0;
	}


	matrix_cr_conv(pxtrain, ltr, nfeature);
	matrix_cr_conv(pxtest, lte, nfeature);
}
