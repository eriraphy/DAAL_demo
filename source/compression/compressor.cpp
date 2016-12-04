/* file: compressor.cpp */
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
!    C++ example of a compressor
!
!******************************************************************************/

/**
 * <a name="DAAL-EXAMPLE-CPP-COMPRESSOR"></a>
 * \example compressor.cpp
 */

#include "daal.h"
#include "service.h"

using namespace std;
using namespace daal;
using namespace data_management;

string datasetFileName  = "../data/batch/logitboost_train.csv";

DataBlock sentDataStream;        /* Data stream to compress and send */
DataBlock uncompressedDataBlock; /* Current block of data stream to compress */
DataBlock compressedDataBlock;   /* Current compressed block of data */
DataBlock receivedDataStream;    /* Received uncompressed data stream */

queue<DataBlock> sendReceiveQueue; /* Queue for sending and receiving compressed data blocks */

const size_t maxDataBlockSize = 16384; /* Maximum size of a data block */

bool getUncompressedDataBlock(DataBlock &block);
void sendCompressedDataBlock(DataBlock &block);
bool receiveCompressedDataBlock(DataBlock &block);
void prepareMemory();
void releaseMemory();
void printCRC32();

int main(int argc, char *argv[])
{
    checkArguments(argc, argv, 1, &datasetFileName);

    /* Read data from a file and allocate memory */
    prepareMemory();

    /* Create a compressor */
    Compressor<zlib> compressor;

    /* Receive the next data block for compression */
    while(getUncompressedDataBlock(uncompressedDataBlock))
    {
        /* Associate data to compress with the compressor */
        compressor.setInputDataBlock(uncompressedDataBlock);

        /* Memory for a compressed block might not be enough to compress the input block at once */
        do
        {
            /* Compress uncompressedDataBlock to compressedDataBlock */
            compressor.run(compressedDataBlock.getPtr(), maxDataBlockSize, 0);

            /* Get the actual size of a compressed block */
            compressedDataBlock.setSize(compressor.getUsedOutputDataBlockSize());

            /* Send the current compressed block */
            sendCompressedDataBlock(compressedDataBlock);
        }
        /* Check if an additional data block is needed to complete compression */
        while (compressor.isOutputDataBlockFull());
    }

    /* Create a decompressor */
    Decompressor<zlib> decompressor;

    /* Receive compressed data by blocks */
    while(receiveCompressedDataBlock(compressedDataBlock))
    {
        /* Associate compressed data with the decompressor */
        decompressor.setInputDataBlock(compressedDataBlock);

        /* Decompress an incoming block to the end of receivedDataStream */
        decompressor.run(receivedDataStream.getPtr(), maxDataBlockSize, receivedDataStream.getSize());

        /* Update the size of actual data in receivedDataStream */
        receivedDataStream.setSize(receivedDataStream.getSize() + decompressor.getUsedOutputDataBlockSize());
    }

    /* Compute and print checksums for sentDataStream and receivedDataStream */
    printCRC32();

    releaseMemory();

    return 0;
}

void prepareMemory()
{
    /* Allocate sentDataStream and read an input file */
    byte *data;
    sentDataStream.setSize(readTextFile(datasetFileName, &data));
    sentDataStream.setPtr(data);

    byte *compressedData = (byte *)daal::services::daal_malloc(maxDataBlockSize);
    checkAllocation(compressedData);
    compressedDataBlock.setPtr(compressedData);
    compressedDataBlock.setSize(maxDataBlockSize);

    byte *receivedData = (byte *)daal::services::daal_malloc(sentDataStream.getSize());
    checkAllocation(receivedData);
    receivedDataStream.setPtr(receivedData);
}

bool getUncompressedDataBlock(DataBlock &block)
{
    static size_t availableDataSize = sentDataStream.getSize();

    /* Calculate the current block size and ptr */
    if(availableDataSize >= maxDataBlockSize)
    {
        block.setSize(maxDataBlockSize);
        block.setPtr(sentDataStream.getPtr() + sentDataStream.getSize() - availableDataSize);
        availableDataSize -= maxDataBlockSize;
    }
    else if((availableDataSize < maxDataBlockSize) && (availableDataSize > 0))
    {
        block.setSize(availableDataSize);
        block.setPtr(sentDataStream.getPtr() + sentDataStream.getSize() - availableDataSize);
        availableDataSize = 0;
    }
    else
    {
        return false;
    }

    return true;
}

void sendCompressedDataBlock(DataBlock &block)
{
    DataBlock currentBlock;

    /* Allocate memory for the current compressed block in the queue */
    byte *currentPtr = (byte *)daal::services::daal_malloc(block.getSize());
    checkAllocation(currentPtr);
    currentBlock.setPtr(currentPtr);

    /* Copy an incoming block to the current compressed block */
    currentBlock.setSize(block.getSize());
    copyBytes(currentBlock.getPtr(), block.getPtr(), currentBlock.getSize());

    /* Push the current compressed block to the queue */
    sendReceiveQueue.push(currentBlock);

    return;
}

bool receiveCompressedDataBlock(DataBlock &block)
{
    DataBlock currentBlock;

    /* Stop at the end of the queue */
    if(sendReceiveQueue.empty())
    {
        return false;
    }

    /* Receive the current compressed block from the queue */
    currentBlock = sendReceiveQueue.front();
    block.setSize(currentBlock.getSize());
    copyBytes(block.getPtr(), currentBlock.getPtr(), block.getSize());

    /* Release memory of the current compressed block in the queue */
    daal::services::daal_free(currentBlock.getPtr());

    sendReceiveQueue.pop();

    return true;
}

void printCRC32()
{
    unsigned int crcSentDataStream = 0;
    unsigned int crcReceivedDataStream = 0;

    /* Compute checksums for full input data and full received data */
    crcSentDataStream = getCRC32(sentDataStream.getPtr(), crcSentDataStream, sentDataStream.getSize());
    crcReceivedDataStream = getCRC32(receivedDataStream.getPtr(), crcReceivedDataStream, receivedDataStream.getSize());

    cout << endl << "Compression example program results:" << endl << endl;

    cout << "Input data checksum:    0x" << hex << crcSentDataStream << endl;
    cout << "Received data checksum: 0x" << hex << crcReceivedDataStream << endl;

    if (sentDataStream.getSize() != receivedDataStream.getSize())
    {
        cout << "ERROR: Received data size mismatches with the sent data size" << endl;
    }
    else if (crcSentDataStream != crcReceivedDataStream)
    {
        cout << "ERROR: Received data CRC mismatches with the sent data CRC" << endl;
    }
    else
    {
        cout << "OK: Received data CRC matches with the sent data CRC" << endl;
    }
}

void releaseMemory()
{
    if(compressedDataBlock.getPtr())
    {
        daal::services::daal_free(compressedDataBlock.getPtr());
    }
    if(receivedDataStream.getPtr())
    {
        daal::services::daal_free(receivedDataStream.getPtr());
    }
    if(sentDataStream.getPtr())
    {
        delete [] sentDataStream.getPtr();
    }
}
