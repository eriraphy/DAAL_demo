/*Raphael.2016.Aug  @Intel*/
#include "daal.h"
#include "mex.h"
#include "matrix.h"

using namespace std;
using namespace daal;
using namespace daal::services;
using namespace daal::data_management;
using namespace daal::algorithms;

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
void outputMatrix(mxArray *output, mwSize nrow, mwSize ncol, SharedPtr<Matrix<type>> data, int pt_dev=0) {
	double *pout;
	pout = mxGetPr(output);
	for (int i = 0; i < ncol*nrow; i++) {
		pout[i] = (*data)[0][i+pt_dev];
	}
	if (nrow > 1 && ncol > 1) {
		matrix_cr_conv<double>(pout, nrow, ncol);
	}
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
type *TensorPtr_al(SharedPtr<Tensor>input, mwSize length) {
	SubtensorDescriptor<type> block;
	input->getSubtensor(0, 0, 0, length, readOnly, block);
	type *temp = block.getPtr();
	return temp;
}



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
			if (col == ncol - 1) {
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