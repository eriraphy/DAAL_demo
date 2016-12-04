/* file: datastructures_homogen.cpp */
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
!    C++ example of using homogeneous data structures
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_HOMOGEN">
 * \example datastructures_homogen.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Homogeneous numeric table example" << std::endl << std::endl;

    const size_t nObservations  = 10;
    const size_t nFeatures = 11;
    const size_t firstReadRow = 0;
    const size_t nRead = 3;
    size_t readFeatureIdx;

    /*Example of using a homogeneous numeric table*/
    double data[nFeatures * nObservations] =
    {
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1,
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2,
        2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3,
        3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4,
        4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5,
        5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 1,
        6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 2,
        7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 3,
        8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9, 4,
        9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9, 5
    };

    HomogenNumericTable<double> dataTable(data, nFeatures, nObservations);

    BlockDescriptor<double> block;

    /* Read a block of rows */
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl;
    printArray<double>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(), "Print 3 rows from homogeneous data array as double:");
    dataTable.releaseBlockOfRows(block);

    /* Read a feature(column) and write into it */
    readFeatureIdx = 2;
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nObservations, readOnly, block);
    printArray<double>(block.getBlockPtr(), 1, block.getNumberOfRows(), "Print the third feature of homogeneous data:");
    dataTable.releaseBlockOfColumnValues(block);

    /* Get a pointer to the inner array for HomogenNumericTable. This pointer is a pointer to the array data */
    data[0] = 999;
    double *dataFromNumericTable = dataTable.getArray();
    printArray<double>(dataFromNumericTable, nFeatures, nObservations, "Data from getArray:");

    const size_t nNewFeatures = 2;
    const size_t nNewVectors = 3;
    double newData[nNewFeatures * nNewVectors] =
    {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };

    /* Set new data to HomogenNumericTable. It mush have the same type as the numeric table. */
    dataTable.setArray(newData);

    /* Set a new number of columns and rows */
    dataTable.setNumberOfColumns(nNewFeatures);
    dataTable.setNumberOfRows(nNewVectors);

    /* Ensure the data has changed */
    readFeatureIdx = 1;
    dataTable.getBlockOfColumnValues(readFeatureIdx, firstReadRow, nNewVectors, readOnly, block);
    printArray<double>(block.getBlockPtr(), 1, block.getNumberOfRows(), "Print the second feature of new data:");
    dataTable.releaseBlockOfColumnValues(block);

    return 0;
}
