/* file: datastructures_merged.cpp */
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
!    Merged data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_MERGED">
 * \example datastructures_merged.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Merged numeric table example" << std::endl << std::endl;

    const size_t nObservations = 5;
    const size_t nFeatures1 = 5;
    const size_t nFeatures2 = 6;
    const size_t firstReadRow = 3;
    const size_t nRead = 1;

    /*Example of using homogeneous numeric table*/
    double data1[nFeatures1 * nObservations] =
    {
        0.0, 0.1, 0.2, 0.3, 0.4,
        1.0, 1.1, 1.2, 1.3, 1.4,
        2.0, 2.1, 2.2, 2.3, 2.4,
        3.0, 3.1, 3.2, 3.3, 3.4,
        4.0, 4.1, 4.2, 4.3, 4.4,
    };
    double data2[nFeatures2 * nObservations] =
    {
        0.5, 0.6, 0.7, 0.8, 0.9, 1,
        1.5, 1.6, 1.7, 1.8, 1.9, 2,
        2.5, 2.6, 2.7, 2.8, 2.9, 3,
        3.5, 3.6, 3.7, 3.8, 3.9, 4,
        4.5, 4.6, 4.7, 4.8, 4.9, 5,
    };

    /* Create two homogen numeric tables from data arrays */
    services::SharedPtr<HomogenNumericTable<double> > dataTable1(new HomogenNumericTable<double>(data1, nFeatures1, nObservations));
    services::SharedPtr<HomogenNumericTable<double> > dataTable2(new HomogenNumericTable<double>(data2, nFeatures2, nObservations));

    /* Create merged numeric table consisting of two homogen numeric tables */
    MergedNumericTable dataTable;
    dataTable.addNumericTable(dataTable1);
    dataTable.addNumericTable(dataTable2);

    BlockDescriptor<double> block;

    /* Read one row from merged numeric table */
    dataTable.getBlockOfRows(firstReadRow, nRead, readWrite, block);
    printArray<double>(block.getBlockPtr(), nFeatures1 + nFeatures2, block.getNumberOfRows(), "Print 1 row from merged numeric table as double:");

    /* Modify row of the merged numeric table */
    double *row = block.getBlockPtr();
    for (size_t i = 0; i < nFeatures1 + nFeatures2; i++) row[i] *= row[i];
    dataTable.releaseBlockOfRows(block);

    /* Read the same row from homogen numeric tables */
    dataTable1->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    printArray<double>(block.getBlockPtr(), nFeatures1, block.getNumberOfRows(), "Print 1 row from first homogen numeric table as double:");
    dataTable1->releaseBlockOfRows(block);

    dataTable2->getBlockOfRows(firstReadRow, nRead, readOnly, block);
    printArray<double>(block.getBlockPtr(), nFeatures2, block.getNumberOfRows(), "Print 1 row from second homogen numeric table as double:");
    dataTable2->releaseBlockOfRows(block);

    return 0;
}
