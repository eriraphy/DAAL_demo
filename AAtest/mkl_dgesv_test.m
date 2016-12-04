%  file: mkl_dgesv_test.m 
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
%      Using Intel MKL on Matlab example of matrix inverse computation.
%
%******************************************************************************/

clear;clc;

% Mex the *.c file in the current folder
mex -v m_dgetr.c mkl_rt.lib

% Following command can be use alternatively
% mex -v m_dgetr.c mkl_intel_lp64_dll.lib mkl_core_dll.lib mkl_intel_thread_dll.lib libiomp5md.lib

% m_dgetr.c calling the MKL dgetrf & dgetri

% Creating seperate random matrixs
% Matrix size
n=3000;
a=zeros(n,n);
b=zeros(n,n);
for i=1:n
    for j=1:n
        t=rand(1);
        a(i,j)=t;
        b(i,j)=t;
    end
end

% Matlab build-in inv function
tic
a=a^-1;
fprintf('Matlab_')
toc

% Using MKL dgetrf & dgetri as inverse funtion
tic
[~]=m_dgetr(b);
fprintf('MKL_')
toc
