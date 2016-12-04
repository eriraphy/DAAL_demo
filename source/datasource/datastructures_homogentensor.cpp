/* file: datastructures_homogentensor.cpp */
/*******************************************************************************
* Copyright 2014-2016 Intel Corporation All Rights Reserved.
*
* The source code,  information  and material  ("Material") contained  herein is
* owned by Intel Corporation or its  suppliers or licensors,  and  title to such
* Material remains with Intel  Corporation or its  suppliers or  licensors.  The
* Material  contains  proprietary  information  of  Intel or  its suppliers  and
* licensors.  The Material is protected by  worldwide copyright  laws and treaty
* provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
* modified, published,  uploaded, posted, transmitted,  distributed or disclosed
* in any way without Intel's prior express written permission.  No license under
* any patent,  copyright or other  intellectual property rights  in the Material
* is granted to  or  conferred  upon  you,  either   expressly,  by implication,
* inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
* property rights must be express and approved by Intel in writing.
*
* Unless otherwise agreed by Intel in writing,  you may not remove or alter this
* notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
* suppliers or licensors in any way.
*******************************************************************************/

/*
!  Content:
!    C++ example of using homogeneous tensor data structures
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_HOMOGENTENSOR">
 * \example datastructures_homogentensor.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;
using namespace data_management;

int main()
{
    float data[3][3][3] = {{{1,2,3},{4,5,6},{7,8,9}},{{11,12,13},{14,15,16},{17,18,19}},{{21,22,23},{24,25,26},{27,28,29}}};

    size_t nDim = 3, dims[] = {3,3,3};

    printf("Initial data:\n");
    for(size_t i= 0;i<dims[0]*dims[1]*dims[2];i++)
    {
        printf("% 5.1f ", ((float*)data)[i]);
    }
    printf("\n");

    HomogenTensor<float> hc(nDim, dims, (float*)data);

    SubtensorDescriptor<double> subtensor;
    size_t fDimN = 2, fDims[] = {0,1};
    hc.getSubtensor(fDimN, fDims, 1, 2, readWrite, subtensor);

    size_t d = subtensor.getNumberOfDims();
    printf("Subtensor dimensions: %i\n", (int)(d));
    size_t n = subtensor.getSize();
    printf("Subtensor size:       %i\n", (int)(n));
    double* p = subtensor.getPtr();
    printf("Subtensor data:\n");
    for(size_t i= 0;i<n;i++)
    {
        printf("% 5.1lf ", p[i]);
    }
    printf("\n");

    p[0]=-1;

    hc.releaseSubtensor(subtensor);

    printf("Data after modification:\n");
    for(size_t i= 0;i<dims[0]*dims[1]*dims[2];i++)
    {
        printf("% 5.1f ", ((float*)data)[i]);
    }
    printf("\n");

    return 0;
}
