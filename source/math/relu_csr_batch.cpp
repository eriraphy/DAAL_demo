/* file: relu_csr_batch.cpp */
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
!    C++ example of ReLU algorithm.
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-RELU_CSR_BATCH"></a>
 * \example relu_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;
using namespace daal::algorithms::math;

/* Input data set parameters */
string datasetName = "../data/batch/covcormoments_csr.csv";

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetName);

    /* Read datasetFileName from a file and create a numeric table to store input data */
    services::SharedPtr<CSRNumericTable> dataTable(createSparseTable<float>(datasetName));

    /* Create an algorithm */
    relu::Batch<float, relu::fastCSR> relu;

    /* Set an input object for the algorithm */
    relu.input.set(relu::data, dataTable);

    /* Compute ReLU function */
    relu.compute();

    /* Print the results of the algorithm */
    services::SharedPtr<relu::Result> res = relu.getResult();
    printNumericTable(res->get(relu::value), "ReLU result (first 5 rows):", 5);

    return 0;
}
