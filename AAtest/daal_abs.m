%  file: daal_abs.m 
% *******************************************************************************
% * Copyright 2014-2016 Intel Corporation All Rights Reserved.
% *
% * The source code,  information  and material  ("Material") contained  herein is
% * owned by Intel Corporation or its  suppliers or licensors,  and  title to such
% * Material remains with Intel  Corporation or its  suppliers or  licensors.  The
% * Material  contains  proprietary  information  of  Intel or  its suppliers  and
% * licensors.  The Material is protected by  worldwide copyright  laws and treaty
% * provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
% * modified, published,  uploaded, posted, transmitted,  distributed or disclosed
% * in any way without Intel's prior express written permission.  No license under
% * any patent,  copyright or other  intellectual property rights  in the Material
% * is granted to  or  conferred  upon  you,  either   expressly,  by implication,
% * inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
% * property rights must be express and approved by Intel in writing.
% *
% * Unless otherwise agreed by Intel in writing,  you may not remove or alter this
% * notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
% * suppliers or licensors in any way.
% *******************************************************************************/
% 
% 
%  Content:
%      Using Intel DAAL on Matlab example of abs algorithm.
%
%******************************************************************************/

clc;clear;

% Mex cpp file
% Can also use daal_core_dll.lib daal_sequential.lib 
mex -v m_daal_abs.cpp daal_core.lib daal_thread.lib 

% Random Input
inputA=randn(10000,100);

% DAAL abs computation
outputA=m_daal_abs(inputA);

% Matlab build-in computation
outputB=abs(inputA);


