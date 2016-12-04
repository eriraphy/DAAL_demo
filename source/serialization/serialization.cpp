/* file: serialization.cpp */
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
!    C++ example of numeric table serialization
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-SERIALIZATION"></a>
 * \example serialization.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;

typedef double  dataFPType;          /* Data floating-point type */

/* Input data set parameters */
const string datasetFileName = "../data/batch/serialization.csv";

void serializeNumericTable(NumericTablePtr dataTable, byte **buffer, size_t *length);
NumericTablePtr deserializeNumericTable(byte *buffer, size_t size);

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Initialize FileDataSource<CSVFeatureManager> to retrieve the input data from a .csv file */
    FileDataSource<CSVFeatureManager> dataSource(datasetFileName, DataSource::doAllocateNumericTable,
                                                 DataSource::doDictionaryFromContext);

    /* Retrieve the data from the input file */
    dataSource.loadDataBlock();

    /* Retrieve a numeric table */
    NumericTablePtr dataTable = dataSource.getNumericTable();

    /* Print the original data */
    printNumericTable(dataTable, "Data before serialization:");

    /* Serialize the numeric table into the memory buffer */
    byte *buffer;
    size_t length;
    serializeNumericTable(dataTable, &buffer, &length);

    /* Deserialize the numeric table from the memory buffer */
    NumericTablePtr restoredDataTable = deserializeNumericTable(buffer, length);

    /* Print the restored data */
    printNumericTable(restoredDataTable, "Data after deserialization:");

    delete [] buffer;
    return 0;
}

void serializeNumericTable(NumericTablePtr dataTable, byte **buffer, size_t *length)
{
    /* Create a data archive to serialize the numeric table */
    InputDataArchive dataArch;

    /* Serialize the numeric table into the data archive */
    dataTable->serialize(dataArch);

    /* Get the length of the serialized data in bytes */
    *length = dataArch.getSizeOfArchive();

    /* Store the serialized data in an array */
    *buffer = new byte[*length];
    dataArch.copyArchiveToArray(*buffer, *length);
}

NumericTablePtr deserializeNumericTable(byte *buffer, size_t length)
{
    /* Create a data archive to deserialize the numeric table */
    OutputDataArchive dataArch(buffer, length);

    /* Create a numeric table object */
    NumericTablePtr dataTable = NumericTablePtr( new HomogenNumericTable<dataFPType>() );

    /* Deserialize the numeric table from the data archive */
    dataTable->deserialize(dataArch);

    return dataTable;
}
