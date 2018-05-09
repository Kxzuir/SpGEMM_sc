#include "common.h"
#include "scsparse_host.h"
#include "gemm_checker.h"
#include <cusp/io/matrix_market.h>

int benchmark_spgemm(const char *datasetA, const char *datasetB) 
{
	int err = 0;
	CSRHost A, B;
	printf("[I/O Interface] Loading matrix data from disk...");
	cusp::io::read_matrix_market_file(A, datasetA);
	if (strlen(datasetB) == 0) B = A;
	else cusp::io::read_matrix_market_file(B, datasetB);
	GeMMChecker::csr_sort_indices<index_t, value_t>(A.num_rows, &A.row_offsets[0], &A.column_indices[0], &A.values[0]);
	GeMMChecker::csr_sort_indices<index_t, value_t>(B.num_rows, &B.row_offsets[0], &B.column_indices[0], &B.values[0]);
	for (int i = 0; i < A.num_entries; i++) A.values[i] = i % 10 + 1;
	for (int i = 0; i < B.num_entries; i++) B.values[i] = i % 10 + 1;
	printf("done.\n");
	int n = A.num_rows, m = A.num_cols, p = B.num_cols;
	scsparse *sc_sparse = new scsparse();
	sc_sparse->initDevice();
	sc_sparse->initData(n, m, p,
		csrPtr(A.num_entries, &A.row_offsets[0], &A.column_indices[0], &A.values[0]),
		csrPtr(B.num_entries, &B.row_offsets[0], &B.column_indices[0], &B.values[0]));
	sc_sparse->warmup();
	sc_sparse->spgemm();

	csrPtr Cptr;
	sc_sparse->getCptr(Cptr);

	int checkRes = GeMMChecker::checkCSR(A, B, Cptr, true);
	if (checkRes != 0) fprintf(stderr, "Result mismatch!\n");
	
	sc_sparse->freeMem();
	return err;
}


int main(int argc, char *argv[])
{
	int err = 0;
	if (argc != 2)
	{
		printHelp(argv[0], "file", "Perform SpGeMM benchmark on matrix market file");
		return 255;
	}
	printHeader(APP_NAME, MAJOR_VERSION, MINOR_VERSION, CPRT_YEAR, AUTHOR);
	printf("[AppFramework] Using arg = \"%s\"\n", argv[1]);
	char *fileName = argv[1];
	err = benchmark_spgemm(fileName, "");
	
	if (err) fprintf(stderr, "[AppFramework] App exited with code %d\n", err);
	else printf("[AppFramework] App exited normally.\n");
	return err;
}
