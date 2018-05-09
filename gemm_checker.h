#pragma once
#ifndef GEMM_CHECKER_H
#define GEMM_CHECKER_H

#include "common.h"



#define GEMMCHECKER_NNZMISMARTCH 1
#define GEMMCHECKER_CSRROWMISMATCH 2
#define GEMMCHECKER_CSRCOLIDXMISMATCH 3
#define GEMMCHECKER_CSRCVALMISMATCH 4
#define GEMMCHECKER_EPS (1e-4)


class GeMMChecker
{
public:
	static int checkCSR(CSRHost A, CSRHost B, csrPtr Cptr, bool printInfo = true);

	template<class I, class T>
	static void csr_sort_indices(const I n_row, const I Ap[], I Aj[], T Ax[]);
private:
};

#endif