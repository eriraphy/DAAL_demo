/*Raphael.2016.Aug  @Intel*/

#include "daal.h"
#include "mex.h"
#include "matrix.h"
#include "mxfunction_service.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range2d.h"
#include <iostream>
//#include <stdio.h>
//#include <stdlib.h>

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::svm;
using namespace tbb;


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


class ovocomp {
	const int &nclass;
	const mwSize &nfeature;
	const mwSize &lte;
	int *&class_num;
	int *&class_ini;
	double *&pxtrain;
	const svm::training::Batch<> &algorithm_trmd;
	const svm::prediction::Batch<> &algorithm_temd;
public:
	double *&poutA;
	void operator() (const blocked_range2d<size_t>&r)const {
		for (size_t na = r.rows().begin(); na < r.rows().end(); na++) {
			for (size_t nb = r.cols().begin(); nb < r.cols().end(); nb++) {

				if (nb <= na) {
					continue;
				}
				int ltr_t = class_num[na] + class_num[nb];
				double *pxtr = (double*)malloc(sizeof(double)*ltr_t*nfeature);
				double *pytr = (double*)malloc(sizeof(double)*ltr_t);
				for (int i = 0; i < class_num[na] * nfeature; i++) {
					pxtr[i] = pxtrain[class_ini[na] * nfeature + i];
				}
				for (int i = class_num[na] * nfeature; i < ltr_t*nfeature; i++) {
					pxtr[i] = pxtrain[class_ini[nb] * nfeature + i - class_num[na] * nfeature];
				}
				for (int i = 0; i < class_num[na]; i++) {
					//pytr[i] = pytrain[class_ini[na] + i];
					pytr[i] = 1;
				}
				for (int i = class_num[na]; i < ltr_t; i++) {
					//pytr[i] = pytrain[class_ini[nb] + i - class_num[na]];
					pytr[i] = -1;
				}

				SharedPtr<NumericTable>trainData = SharedPtr<NumericTable>(new Matrix<double>(nfeature, ltr_t, pxtr));
				SharedPtr<NumericTable>trainGroundTruth = SharedPtr<NumericTable>(new Matrix<double>(1, ltr_t, pytr));



				svm::training::Batch<> algorithm_tr(algorithm_trmd);
				//algorithm.parameter.tau = 1.0e2;
				algorithm_tr.input.set(classifier::training::data, trainData);
				algorithm_tr.input.set(classifier::training::labels, trainGroundTruth);
				algorithm_tr.compute();
				SharedPtr<svm::training::Result>trainingResult = algorithm_tr.getResult();

				/*Test process*/

				svm::prediction::Batch<> algorithm_te(algorithm_temd);

				algorithm_te.input.set(classifier::prediction::model,
					trainingResult->get(classifier::training::model));
				algorithm_te.compute();
				SharedPtr<classifier::prediction::Result>predictionResult = algorithm_te.getResult();
				SharedPtr<Matrix<double>>pred = staticPointerCast<Matrix<double>, NumericTable>(predictionResult->get(classifier::prediction::prediction));

				for (int i = 0; i < lte; i++) {
					if ((*pred)[0][i] >= 0) {
						poutA[i*nclass + na] ++;
					}
					else {
						poutA[i*nclass + nb] ++;
					}
				}


				//std::cout << "Process " << na << " V " << nb << "done\n" << std::endl;
				
				printf("Process %i V %i done\n", na, nb);
				//mexEvalString("pause(0.0001);");
				//mexCallMATLAB(0, NULL, 0, NULL, "drawnow;");
				//mexEvalString("drawnow;");

				trainData->freeDataMemory();
				trainGroundTruth->freeDataMemory();
				pred->freeDataMemory();

				algorithm_tr.clean();
				algorithm_te.clean();

				free(pxtr);
				free(pytr);
			}
		}
	}
	ovocomp(const int &_nclass, const mwSize &_nfeature, const mwSize &_lte,
		int *&_class_num, int *&_class_ini, double *&_pxtrain, double *&_poutA,
		const svm::training::Batch<> &_algorithm_trmd, const svm::prediction::Batch<> &_algorithm_temd) :
		nclass(_nclass), nfeature(_nfeature), lte(_lte),
		class_num(_class_num), class_ini(_class_ini), pxtrain(_pxtrain), poutA(_poutA),
		algorithm_trmd(_algorithm_trmd), algorithm_temd(_algorithm_temd)
	{
	}
};


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

/*mex main function*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	if (nrhs < 4) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 4 inputs required");
	}
	if (nlhs < 1) {
		mexErrMsgIdAndTxt("test:nrhs", "At least 1 outputs required");
	}
	if (!mxIsDouble(inputA) || mxIsComplex(inputA))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train matrix must be double");
	}
	if (!mxIsDouble(inputB) || !mxIsSingle(inputB))
	{
		mexErrMsgIdAndTxt("test:prhs", "Train label must be double");
	}
	if (!mxIsDouble(inputC) || mxIsComplex(inputC))
	{
		mexErrMsgIdAndTxt("test:prhs", "Test matrix must be double");
	}
	if (!mxIsDouble(inputD) || !mxIsSingle(inputD))
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
	int iter;
	double boxC;
	double accu;
	char *buf;

	double sigma;
	mwSize ltr = mxGetM(inputA);
	mwSize lte = mxGetM(inputC);
	mwSize nfeature = mxGetN(inputA);
	mwSize ny = 1;
	
	int nclass = mxGetScalar(inputE);

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
	
	double *pxtrain = mxGetPr(inputA);
	double *pytrain = mxGetPr(inputB);
	double *pxtest = mxGetPr(inputC);
	double *pytest = mxGetPr(inputD);

	matrix_cr_conv(pxtrain, nfeature, ltr);

	matrix_cr_conv(pxtest, nfeature, lte);


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

	svm::training::Batch<> algorithm_trmd;
	algorithm_trmd.parameter.kernel = kernel;
	algorithm_trmd.parameter.cacheSize = 160000000;
	algorithm_trmd.parameter.maxIterations = iter;
	algorithm_trmd.parameter.accuracyThreshold = accu;
	algorithm_trmd.parameter.C = boxC;

	SharedPtr<NumericTable>testData = SharedPtr<NumericTable>(new Matrix<double>(nfeature, lte, pxtest));
	SharedPtr<NumericTable>testGroundTruth = SharedPtr<NumericTable>(new Matrix<double>(ny, lte, pytest));

	svm::prediction::Batch<> algorithm_temd;
	algorithm_temd.parameter.kernel = kernel;
	algorithm_temd.input.set(classifier::prediction::data, testData);
	
	

	int *class_ini = (int*)malloc(sizeof(int)*nclass);
	int *class_num = (int*)malloc(sizeof(int)*nclass);
	int inicount = 0;
	for (int i = 0; i < ltr; i++) {
		if (pytrain[i] == inicount) {
			class_ini[inicount] = i;
			if (inicount != 0) {
				class_num[inicount - 1] = class_ini[inicount] - class_ini[inicount - 1];
			}
			inicount++;
		}
	}
	class_num[nclass-1] = ltr - class_ini[nclass-1];
	

	outputA = mxCreateDoubleMatrix(lte, nclass, mxREAL);
	double *poutA = mxGetPr(outputA);
	////////////////////////////////////////////////////////////////////////

	parallel_for(blocked_range2d<size_t>(0, 3, 0, 3), 
		ovocomp(nclass, nfeature, lte, class_num, class_ini, pxtrain, poutA, algorithm_trmd, algorithm_temd));

	////////////////////////////////////////////////////////////////////////


	//for (int na = 0; na < nclass; na++) {
	//	for (int nb = 0; nb < nclass; nb++) {
	//		if (nb <= na) {
	//			continue;
	//		}
	////		int na = 0;
	////		int nb = 1;

	//		int ltr_t = class_num[na] + class_num[nb];
	//		double *pxtr = (double*)malloc(sizeof(double)*ltr_t*nfeature);
	//		double *pytr = (double*)malloc(sizeof(double)*ltr_t);
	//		for (int i = 0; i < class_num[na] * nfeature; i++) {
	//			pxtr[i] = pxtrain[class_ini[na] * nfeature + i];
	//		}
	//		for (int i = class_num[na] * nfeature; i < ltr_t*nfeature; i++) {
	//			pxtr[i] = pxtrain[class_ini[nb] * nfeature + i - class_num[na] * nfeature];
	//		}
	//		for (int i = 0; i < class_num[na]; i++) {
	//			//pytr[i] = pytrain[class_ini[na] + i];
	//			pytr[i] = 1;
	//		}
	//		for (int i = class_num[na]; i < ltr_t; i++) {
	//			//pytr[i] = pytrain[class_ini[nb] + i - class_num[na]];
	//			pytr[i] = -1;
	//		}

	//		SharedPtr<NumericTable>trainData = SharedPtr<NumericTable>(new Matrix<double>(nfeature, ltr_t, pxtr));
	//		SharedPtr<NumericTable>trainGroundTruth = SharedPtr<NumericTable>(new Matrix<double>(ny, ltr_t, pytr));

	//		svm::training::Batch<> algorithm_tr(algorithm_trmd);
	//		//algorithm.parameter.tau = 1.0e2;
	//		algorithm_tr.input.set(classifier::training::data, trainData);
	//		algorithm_tr.input.set(classifier::training::labels, trainGroundTruth);
	//		algorithm_tr.compute();
	//		SharedPtr<svm::training::Result> trainingResult = algorithm_tr.getResult();

	//		/*Test process*/
	//		svm::prediction::Batch<> algorithm_te(algorithm_temd);
	//		algorithm_te.input.set(classifier::prediction::data, testData);
	//		algorithm_te.input.set(classifier::prediction::model,
	//			trainingResult->get(classifier::training::model));
	//		algorithm_te.compute();
	//		SharedPtr<classifier::prediction::Result> predictionResult = algorithm_te.getResult();
	//		SharedPtr<Matrix<double>>pred = staticPointerCast<Matrix<double>, NumericTable>(predictionResult->get(classifier::prediction::prediction));

	//		for (int i = 0; i < lte; i++) {
	//			if ((*pred)[0][i] >= 0) {
	//				poutA[i*nclass + na] ++;
	//			}
	//			else {
	//				poutA[i*nclass + nb] ++;
	//			}
	//		}
	//		mexPrintf("Process %i V %i done\n", na, nb);
	//		mexEvalString("pause(0.0001)");
	//		mexCallMATLAB(0, NULL, 0, NULL, "drawnow");
	//		//mexEvalString("drawnow");

	//		trainData->freeDataMemory();
	//		trainGroundTruth->freeDataMemory();
	//		pred->freeDataMemory();
	//		
	//	}
	//}


	matrix_cr_conv(poutA, lte, nclass);
	
	matrix_cr_conv(pxtrain, ltr, nfeature);

	matrix_cr_conv(pxtest, lte, nfeature);
}


