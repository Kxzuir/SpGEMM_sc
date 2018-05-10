#include "gemm_checker.h"
#include <cusp/multiply.h>
#include <cusp/system/detail/sequential/reference/csr.h>


int GeMMChecker::checkCSR(CSRHost A, CSRHost B, csrPtr Cptr, bool printInfo)
{
	if(printInfo) printf("[GeMMChecker] Info: Checking results using CUSP Library.\n");
	CSRHost C;
	if (printInfo) printf("[GeMMChecker] Performing CUSP_multiply...");
	StopWatchInterface *checkerTimer = NULL;
	sdkCreateTimer(&checkerTimer);
	sdkStartTimer(&checkerTimer);
	cusp::multiply(A, B, C);
	csr_sort_indices<index_t, value_t>(C.num_rows, &C.row_offsets[0], &C.column_indices[0], &C.values[0]);
	sdkStopTimer(&checkerTimer);
	if (printInfo) printf("done, in %.2f ms.\n", sdkGetTimerValue(&checkerTimer));
	
	if (printInfo) printf("[GeMMChecker] Comparing...");
	if (C.num_entries != Cptr.nnz)
	{
		fprintf(stderr, "Check FAILED: NNZ mismatch. (sample:%d, std:%d)\n", Cptr.nnz, (int)C.num_entries);
		return GEMMCHECKER_NNZMISMARTCH;
	}

	for (int i = 0; i <= C.num_rows; i++)
	{
		if (C.row_offsets[i] != Cptr.csrRowPtr[i])
		{
			fprintf(stderr, "Check FAILED: csrRowPtr mismatch at row %d.\n", i);
			return GEMMCHECKER_CSRROWMISMATCH;
		}
	}
	for (int i = 0; i < C.num_entries; i++)
	{
		if (C.column_indices[i] != Cptr.csrColIdx[i])
		{
			fprintf(stderr, "Check FAILED: csrColIdx mismatch at offset %d.\n", i);
			return GEMMCHECKER_CSRCOLIDXMISMATCH;
		}
		if (fabs(C.values[i] - Cptr.csrVal[i]) > GEMMCHECKER_EPS)
		{
			fprintf(stderr, "Check FAILED: csrVal mismatch at offset %d.\n", i);
			return GEMMCHECKER_CSRCOLIDXMISMATCH;
		}
	}

	if (printInfo) printf("PASS!\n");
	return 0;
}

template<class I, class T>
void GeMMChecker::csr_sort_indices(const I n_row, const I Ap[], I Aj[], T Ax[])
{
	std::vector< std::pair<I, T> > temp;
	for (I i = 0; i < n_row; i++)
	{
		I row_start = Ap[i];
		I row_end = Ap[i + 1];
		temp.clear();
		for (I j = row_start; j < row_end; j++)
			temp.push_back(std::make_pair(Aj[j], Ax[j]));

		std::sort(temp.begin(), temp.end(), kv_pair_less<I, T>);

		for (I j = row_start, n = 0; j < row_end; j++, n++)
		{
			Aj[j] = temp[n].first;
			Ax[j] = temp[n].second;
		}
	}
}
