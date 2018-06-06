#pragma once
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <numeric>
#include <vector>
#include <climits>
#include <algorithm>

#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cusp/csr_matrix.h>

#include "platform.h"
#include "utility.h"


typedef int index_t;
typedef double value_t;
typedef int attr_t;
typedef cusp::csr_matrix<index_t, value_t, cusp::host_memory> CSRHost;


struct Arg_t
{
	double alpha_coe;
};


struct csrPtr
{
public:
	int nnz;
	index_t *csrRowPtr, *csrColIdx;
	value_t *csrVal;

	csrPtr() : nnz(0), csrRowPtr(NULL), csrColIdx(NULL), csrVal(NULL) {}
	csrPtr(int _nnz, index_t *_csrRowPtr, index_t *_csrColIdx, value_t *_csrVal) :
		nnz(_nnz), csrRowPtr(_csrRowPtr), csrColIdx(_csrColIdx), csrVal(_csrVal) {}
};


#define HASHTABLESIZ 512
#define ALPHA_COEFFICIENT 1

#define INVALIDHASHIDX INT_MAX
#define SD_HT_INS_ERR_HTFULL (-1)

#define EMPTYATTR 0
#define CDATTR 1
#define SDATTR 2

#define APP_NAME "SpGEMM_SC"
#define MAJOR_VERSION 2
#define MINOR_VERSION 7
#define REVISION 2
#define CPRT_YEAR 2018
#define AUTHOR "BUPT GPU Architecture Study Group"
#define CODER "Kxzuir"

// Uncomment this to disable stdout
#define NO_OUTPUT



#endif

/*
Release note

v0.1beta
Initial version
Can only handle tiny matrixes (probably with crashes)

v1.0
First release version
Bugfix

v2.0
Optimize CDGroup speed by replacing 2D-pool into flat array
Optimize SDGroup speed by inline bitonic sort

v2.1
Optimize SDGroup memory by pre-allocate destination array

v2.2
Skipped version, due to template code issue

v2.3
Import templated SDGroup function
Improve CDGroup performance by replacing thrust library into hand-written code

v2.4
Optimize CDGroup speed by import similar-size boxing
Merge CDGroup kernels

v2.5
Optimize CDGroup speed by rewritten bitonic sort kernel
Split CDGroup kernels again (practice proves that v2.5 is a bad try)

v2.6
Add dynamic-size hash table support for SDGroup
Add falldown mechanism for SDGroup rows (fall into CDGroup)
Add memory management
Add warmup function
Code struct reconstruct

v2.7
Add support for 32-bit floating-point computation
Bugfix

v2.7.1
Support alpha coefficient adjustment

v2.7.2
Add struct code for analysis
Cross-platform work
*/
