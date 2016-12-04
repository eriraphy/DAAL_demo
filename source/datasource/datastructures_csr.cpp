/* file: datastructures_csr.cpp */
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
!    Compressed sparse rows (CSR) data structures example.
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-DATASTRUCTURES_CSR">
 * \example datastructures_csr.cpp
 */

#include "daal.h"
#include "service.h"

using namespace daal;

int main()
{
    std::cout << "Compressed spares rows (CSR) numeric table example" << std::endl << std::endl;

    const size_t nObservations  = 5;
    const size_t nFeatures = 5;
    const size_t firstReadRow = 1;
    const size_t nRead = 3;

    /* Example of using CSR numeric table */
    double values[]     = {1, -1, -3, -2,  5,  4,  6,  4, -4,  2,  7,  8, -5};
    size_t colIndices[] = {1,  2,  4,  1,  2,  3,  4,  5,  1,  3,  4,  2,  5};
    size_t rowOffsets[] = {1,          4,      6,          9,         12,     14};

    CSRNumericTable dataTable(values, colIndices, rowOffsets, nFeatures, nObservations);

    /* Read block of rows in dense format */
    BlockDescriptor<double> block;
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl << std::endl;
    printArray<double>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(),
                       "Print 3 rows from CSR data array as dense double array:");
    dataTable.releaseBlockOfRows(block);

    /* Read block of rows in CSR format and write into it */
    CSRBlockDescriptor<float> csrBlock;
    dataTable.getSparseBlock(firstReadRow, nRead, readWrite, csrBlock);
    float *valuesBlock = csrBlock.getBlockValuesPtr();
    size_t nValuesInBlock = csrBlock.getDataSize();
    printArray<float>(valuesBlock, nValuesInBlock, 1,
                      "Values in 3 rows from CSR data array:");
    printArray<size_t>(csrBlock.getBlockColumnIndicesPtr(), nValuesInBlock, 1,
                      "Columns indices in 3 rows from CSR data array:");
    printArray<size_t>(csrBlock.getBlockRowIndicesPtr(), nRead + 1, 1,
                      "Rows offsets in 3 rows from CSR data array:");
    for (size_t i = 0; i < nValuesInBlock; i++)
    {
        valuesBlock[i] = -(1.0f + i);
    }
    dataTable.releaseSparseBlock(csrBlock);

    /* Read block of rows in dense format */
    dataTable.getBlockOfRows(firstReadRow, nRead, readOnly, block);
    std::cout << block.getNumberOfRows() << " rows are read" << std::endl << std::endl;
    printArray<double>(block.getBlockPtr(), nFeatures, block.getNumberOfRows(),
                       "Print 3 rows from CSR data array as dense double array:");
    dataTable.releaseBlockOfRows(block);

    return 0;
}
