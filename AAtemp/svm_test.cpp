#include "daal.h"
#include "fstream"
#include "sstream"
#include "iostream"
#include "time.h"
#include "mxfunction_service.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::svm;




/* Model object for SVM algorithm */
services::SharedPtr<svm::training::Result> trainingResult;
services::SharedPtr<classifier::prediction::Result> predictionResult;




void csvread(string location, double *pdata, const size_t nrow, const size_t ncol)
{
	ifstream file(location);
	for (int row = 0; row < nrow; row++) {
		string line;
		getline(file, line);
		if (!file.good())
			break;
		stringstream iss(line);
		for (int col = 0; col < ncol; col++) {
			string val;
			if (col == ncol-1) {
				getline(iss, val, ',');
			}
			else {
				getline(iss, val, ',');
				if (!iss.good())
					break;
			}

			stringstream convertor(val);
			convertor >> pdata[row*ncol + col];
		}
	}
}



//string trainDatasetFileName = "../data/temp/dtr.csv";
//string trainGroundTruthFileName = "../data/temp/ytr.csv";
//string testDatasetFileName = "../data/temp/dte.csv";
//string testGroundTruthFileName = "../data/temp/yte.csv";

string trainDatasetFileName = "xtr.csv";
string trainGroundTruthFileName = "ytr.csv";
string testDatasetFileName = "xte.csv";
string testGroundTruthFileName = "yte.csv";
const size_t ltr = 10000;
const size_t nfeature = 784;
const size_t lte = 2000;

//string trainDatasetFileName = "xtr_t.csv";
//string trainGroundTruthFileName = "ytr_t.csv";
//string testDatasetFileName = "xte_t.csv";
//string testGroundTruthFileName = "yte_t.csv";
//const size_t ltr = 1000;
//const size_t nfeature = 784;
//const size_t lte = 200;

void test(double *pxtrain, double *pytrain, double *pxtest, double *pytest, int maxite);

/*main function*/
int main(int argc, char *argv[])
{	
	/*Training Process*/
	/*Defination*/

	clock_t begin, end;
	double time;

	double *pxtrain;
	double *pytrain;
	double *pxtest;
	double *pytest;
	pxtrain = (double *)malloc(sizeof(double)*ltr*nfeature);
	pytrain = (double *)malloc(sizeof(double)*ltr);
	pxtest = (double *)malloc(sizeof(double)*lte*nfeature);
	pytest = (double *)malloc(sizeof(double)*lte);

	printf("Reading data from *.csv file...\n");
	begin = clock();
	csvread(trainDatasetFileName, pxtrain, ltr, nfeature);
	csvread(trainGroundTruthFileName, pytrain, ltr, 1);
	csvread(testDatasetFileName, pxtest, lte, nfeature);
	csvread(testGroundTruthFileName, pytest, lte, 1);

	end = clock();
	printf("Data loaded\n");
	printf("Time elapsed_%.4f\n", (double)(end - begin) / CLOCKS_PER_SEC);

	int maxite;

	//for (int i = 10; i < 40; i=i+5) {
		maxite = 1000;
		test(pxtrain, pytrain, pxtest, pytest, maxite);
	//}



	system("pause");

	return 0;

}



void test(double *pxtrain, double *pytrain, double *pxtest, double *pytest, int maxite) {

	clock_t begin, end;

	/*Create NumericTable*/
	SharedPtr<NumericTable>inputdataA = SharedPtr<NumericTable>(new Matrix<double>(nfeature, ltr, pxtrain));
	SharedPtr<NumericTable>inputdataB = SharedPtr<NumericTable>(new Matrix<double>(1, ltr, pytrain));
	SharedPtr<NumericTable>inputdataC = SharedPtr<NumericTable>(new Matrix<double>(nfeature, lte, pxtest));
	SharedPtr<NumericTable>inputdataD = SharedPtr<NumericTable>(new Matrix<double>(1, lte, pytest));

	services::SharedPtr<kernel_function::KernelIface> kernel_linear(new kernel_function::linear::Batch<>());

	training::Batch<> algorithm_train;
	algorithm_train.parameter.kernel = kernel_linear;
	algorithm_train.parameter.cacheSize = 160000000;
	algorithm_train.parameter.maxIterations = maxite;
	algorithm_train.parameter.accuracyThreshold = 1e-10;
	algorithm_train.parameter.C = 1;
	algorithm_train.input.set(classifier::training::data, inputdataA);
	algorithm_train.input.set(classifier::training::labels, inputdataB);

	printf("Training model...\n");
	begin = clock();

	algorithm_train.compute();

	end = clock();
	printf("Train process done\n");
	printf("Time elapsed_%.4f\n", (double)(end - begin) / CLOCKS_PER_SEC);

	trainingResult = algorithm_train.getResult();


	/*Test process*/
	prediction::Batch<> algorithm_test;
	algorithm_test.parameter.kernel = kernel_linear;
	algorithm_test.input.set(classifier::prediction::data, inputdataC);
	algorithm_test.input.set(classifier::prediction::model,
		trainingResult->get(classifier::training::model));


	predictionResult = algorithm_test.getResult();

	printf("Testing...\n");
	begin = clock();

	algorithm_test.compute();
	SharedPtr<Matrix<double>>pred = staticPointerCast<Matrix<double>, NumericTable>(predictionResult->get(classifier::prediction::prediction));

	end = clock();
	printf("Test process done\n");
	printf("Time elapsed_%.4f\n", (double)(end - begin) / CLOCKS_PER_SEC);

	/*Test result output*/

	double *poutA;
	double *poutB;
	poutB = (double *)malloc(sizeof(double)*lte * 1);
	poutA = (*pred)[0];

	double ccr = 0;
	for (int i = 0; i < lte; i++) {
		if (poutA[i] > 0) {
			poutB[i] = pytest[i];
		}
		else {
			poutB[i] = -pytest[i];
		}
		ccr = ccr + poutB[i];
	}


	//low_order_moments::Batch<> loworder;
	//SharedPtr<NumericTable>outputB = SharedPtr<NumericTable>(new Matrix<double>(nfeature, ltr, poutB));
	//loworder.input.set(low_order_moments::data, outputB);
	//services::SharedPtr<low_order_moments::Result> res = loworder.getResult();
	//SharedPtr<Matrix<double>> perr= staticPointerCast<Matrix<double>, NumericTable>(res->get(low_order_moments::sum));

	printf("CCR_%f\n", ccr / lte);

		

}