/* file: low_order_moms_csr_batch.cpp */
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
!    C++ example of computing low order moments in the batch processing mode.
!
!    Input matrix is stored in the compressed sparse row (CSR) format with
!    one-based indexing.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-LOW_ORDER_MOMENTS_CSR_BATCH">
 * \example low_order_moments_csr_batch.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace daal::algorithms;

typedef float  dataFPType;          /* Data floating-point type */
typedef double algorithmFPType;     /* Algorithm floating-point type */

/*
 * Input data set parameters
 * Input matrix is stored in the CSR format with one-based indexing
 */
const string datasetFileName = "../data/batch/covcormoments_csr.csv";

void printResults(const services::SharedPtr<low_order_moments::Result> &res);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read datasetFileName from a file and create numeric tables to store the input data */
    CSRNumericTable *dataTable = createSparseTable<dataFPType>(datasetFileName);

    /* Create an algorithm to compute low order moments in the batch processing mode using the default method */
    low_order_moments::Batch<algorithmFPType, low_order_moments::fastCSR> algorithm;

    /* Set input objects for the algorithm */
    algorithm.input.set(low_order_moments::data, services::SharedPtr<CSRNumericTable>(dataTable));

    /* Compute low order moments */
    algorithm.compute();


    /* Get the computed low order moments */
    services::SharedPtr<low_order_moments::Result> res = algorithm.getResult();

    printResults(res);

    return 0;
}

void printResults(const services::SharedPtr<low_order_moments::Result> &res)
{
    printNumericTable(res->get(low_order_moments::minimum),              "Minimum:");
    printNumericTable(res->get(low_order_moments::maximum),              "Maximum:");
    printNumericTable(res->get(low_order_moments::sum),                  "Sum:");
    printNumericTable(res->get(low_order_moments::sumSquares),           "Sum of squares:");
    printNumericTable(res->get(low_order_moments::sumSquaresCentered),   "Sum of squared difference from the means:");
    printNumericTable(res->get(low_order_moments::mean),                 "Mean:");
    printNumericTable(res->get(low_order_moments::secondOrderRawMoment), "Second order raw moment:");
    printNumericTable(res->get(low_order_moments::variance),             "Variance:");
    printNumericTable(res->get(low_order_moments::standardDeviation),    "Standard deviation:");
    printNumericTable(res->get(low_order_moments::variation),            "Variation:");
}
