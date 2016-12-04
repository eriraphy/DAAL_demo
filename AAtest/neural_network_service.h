/*Raphael.2016.Aug  @Intel*/
#include "daal.h"
#include "time.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;


//enum LayerIndex
//{
//	cv1 = 0,
//	rl1 = 1,
//  po1 = 2,
//	fc1 = 3,
//	sm1 = 4
//};
//Collection<LayerDescriptor> configureNet(int nclass)
//{
//	/* Create layers of the neural network */
//	Collection<LayerDescriptor> configuration;
//
//	unsigned int t = time(NULL);
//	srand(t);
//	/* --------------------------------------------------------------------------------------------------- */
//	SharedPtr<convolution2d::Batch<double>>convolution2dLayer1(new convolution2d::Batch<double>());
//
//	convolution2dLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<double>>
//		(new initializers::uniform::Batch<double>(-0.001, 0.001, rand()));
//
//	convolution2dLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<double>>
//		(new initializers::uniform::Batch<double>(0, 0.05, rand()));
//
//	convolution2dLayer1->parameter.kernelSize = convolution2d::KernelSize(5, 5);
//
//	convolution2dLayer1->parameter.stride = convolution2d::Stride(1, 1);
//
//	convolution2dLayer1->parameter.padding = convolution2d::Padding(2, 2);
//
//	convolution2dLayer1->parameter.nKernels = 1;
//
//	configuration.push_back(LayerDescriptor(cv1, convolution2dLayer1, NextLayers(rl1)));
//
//	/* --------------------------------------------------------------------------------------------------- */
//	SharedPtr<relu::Batch<double>>reluLayer1(new relu::Batch<double>());
//	
//    configuration.push_back(LayerDescriptor(rl1, reluLayer1, NextLayers(po1)));
//
//	/* --------------------------------------------------------------------------------------------------- */
//	SharedPtr<maximum_pooling2d::Batch<double>>maximumpooling2dLayer1(new maximum_pooling2d::Batch<double>(3));
//
//	maximumpooling2dLayer1->parameter.padding = pooling2d::Padding(1, 1);
//
//	maximumpooling2dLayer1->parameter.kernelSize = pooling2d::KernelSize(2, 2);
//
//	maximumpooling2dLayer1->parameter.stride = pooling2d::Stride(2, 2);
//
//	maximumpooling2dLayer1->parameter.indices = pooling2d::SpatialDimensions(2, 3);
//
//	configuration.push_back(LayerDescriptor(po1, maximumpooling2dLayer1, NextLayers(fc1)));
//
//	/* --------------------------------------------------------------------------------------------------- */
//
//	SharedPtr<fullyconnected::Batch<double> > fullyConnectedLayer1(new fullyconnected::Batch<double>(nclass));
//
//	fullyConnectedLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<double> >
//		(new initializers::uniform::Batch<double>(0, 0.5, rand()));
//
//	fullyConnectedLayer1->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<double> >
//		(new initializers::uniform::Batch<double>(0, 0.5, rand()));
//
//	configuration.push_back(LayerDescriptor(fc1, fullyConnectedLayer1, NextLayers(sm1)));
//
//
//	/* --------------------------------------------------------------------------------------------------- */
//	SharedPtr<loss::softmax_cross::Batch<double> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<double>());
//
//	configuration.push_back(LayerDescriptor(sm1, softmaxCrossEntropyLayer, NextLayers()));
//
//	return configuration;
//}


enum LayerIndex
{
	fc1 = 0,
	fc2 = 1,
	sm1 = 2
};
training::Topology configureNet(int nclass)
{
	/* Create layers of the neural network */
	training::Topology configuration;
	unsigned int t = time(NULL);
	srand(t);

	/* --------------------------------------------------------------------------------------------------- */
	SharedPtr<fullyconnected::Batch<double> > fullyConnectedLayer1(new fullyconnected::Batch<double>(300));

	fullyConnectedLayer1->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<double> >(
		new initializers::uniform::Batch<double>(-0.1, 0.1, rand()));

	fullyConnectedLayer1->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<double> >(
		new initializers::uniform::Batch<double>(0, 0.1, rand()));

	configuration.push_back(LayerDescriptor(fc1, fullyConnectedLayer1, NextLayers(fc2)));
	/* --------------------------------------------------------------------------------------------------- */
	SharedPtr<fullyconnected::Batch<double> > fullyConnectedLayer2(new fullyconnected::Batch<double>(nclass));

	fullyConnectedLayer2->parameter.weightsInitializer = services::SharedPtr<initializers::uniform::Batch<double> >(
		new initializers::uniform::Batch<double>(0, 0.5, rand()));

	fullyConnectedLayer2->parameter.biasesInitializer = services::SharedPtr<initializers::uniform::Batch<double> >(
		new initializers::uniform::Batch<double>(0, 0.5, rand()));

	configuration.push_back(LayerDescriptor(fc2, fullyConnectedLayer2, NextLayers(sm1)));
	/* --------------------------------------------------------------------------------------------------- */
	SharedPtr<loss::softmax_cross::Batch<double> > softmaxCrossEntropyLayer(new loss::softmax_cross::Batch<double>());
	
	softmaxCrossEntropyLayer->parameter.accuracyThreshold = 0.1;

	configuration.push_back(LayerDescriptor(sm1, softmaxCrossEntropyLayer, NextLayers()));

	return configuration;
}


/*Usage:
convert colomn-domain matrix to row-domain matrix(mexArray to C Array)
input_matrix: colomn-domain; output_matrix: row-domain
matrix_cr_conv(double *cdmatrix, int ncol, int nrow)

convert row-domain matrix to colomn-domain matrix(C Array to mexArray)
input_matrix: row-domain; output_matrix: colomn-domain
matrix_cr_conv(double *rdmatrix, int nrow, int ncol)
*/
template<typename type>
void matrix_cr_conv(type *ipmatrix, size_t dx, size_t dy) {
	int i;
	int j;
	type *opmatrix;
	opmatrix = (type *)malloc(sizeof(type)*dx*dy);
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

template<typename type>
SharedPtr<Tensor> Matrix_Tensor(type *mxptr, mwSize na, mwSize nb, mwSize nc, mwSize nd) {
	Collection<size_t> dims;
	dims.push_back(na);
	dims.push_back(nb);
	dims.push_back(nc);
	dims.push_back(nd);
	HomogenTensor<type> *tensor = new HomogenTensor<type>(dims, mxptr);
	SharedPtr<Tensor> temp(tensor);
	return temp;
}

template<typename type>
SharedPtr<Tensor> Matrix_Tensor(type *mxptr, mwSize na, mwSize nb , mwSize nc) {
	Collection<size_t> dims;
	dims.push_back(na);
	dims.push_back(nb);
	dims.push_back(nc);
	HomogenTensor<type> *tensor = new HomogenTensor<type>(dims, mxptr);
	SharedPtr<Tensor> temp(tensor);
	return temp;
}

template<typename type>
SharedPtr<Tensor> Matrix_Tensor(type *mxptr, mwSize na, mwSize nb) {
	Collection<size_t> dims;
	dims.push_back(na);
	dims.push_back(nb);
	HomogenTensor<type> *tensor = new HomogenTensor<type>(dims, mxptr);
	SharedPtr<Tensor> temp(tensor);
	return temp;
}

template<typename type>
SharedPtr<Tensor> Matrix_Tensor(type *mxptr, mwSize na) {
	Collection<size_t> dims;
	dims.push_back(na);
	HomogenTensor<type> *tensor = new HomogenTensor<type>(dims, mxptr);
	SharedPtr<Tensor> temp(tensor);
	return temp;
}


template<typename type>
type *TensorPtr_alloc(SharedPtr<Tensor>input, mwSize length) {
	SubtensorDescriptor<type> block;
	input->getSubtensor(0, 0, 0, length, readOnly, block);
	type *temp = block.getPtr();
	return temp;
}