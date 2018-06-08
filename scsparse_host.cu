#include "scsparse_host.h"
#include "scsparse_device.cuh"


scsparse::scsparse()
{
	_hasPFInited = _hasCalced = _hasDataInited = false;
}

int scsparse::initDevice(int deviceId, bool printInfo)
{
	if (_hasPFInited)
	{
		fprintf(stderr, "[InitDevice] Warning: Initialized device [%d] detected and will be replaced.\n", deviceId);
		return 1;
	}
	_hasPFInited = false;
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
	cudaError_t cuErr = cudaGetDeviceProperties(&deviceProp, deviceId);
	if (cuErr != cudaSuccess)
	{
		fprintf(stderr, "[InitDevice] Error: Can not get CUDA info for Device [%d].\n", deviceId);
		return 1;
	}
	_deviceId = deviceId;
	_num_smx = deviceProp.multiProcessorCount;
	_max_blocks_per_smx = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
	_num_cuda_cores_per_smx = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	_num_cuda_cores = _num_cuda_cores_per_smx * _num_smx;

	if (printInfo)
	{
		printf("[InitDevice] Info: Using device [%d] %s @ %.2f MHz (%d.%d).\n",
			deviceId, deviceProp.name, deviceProp.clockRate * 1e-3f, deviceProp.major, deviceProp.minor);
		printf("[InitDevice] Info: %d SMXs, %d CUDA cores per SMX.\n", _num_smx, _num_cuda_cores_per_smx);
	}
	_hasPFInited = true;
	return 0;
}


int scsparse::initData(int n, int m, int p, csrPtr Aptr, csrPtr Bptr, bool printInfo)
{
	int err = 0;
	if (!_hasPFInited)
	{
		fprintf(stderr, "[InitData] Error: Device not initialized.\n");
		return err = 1;
	}
	if (_hasDataInited)
	{
		fprintf(stderr, "[InitData] Error: Data already loaded.\n");
		return err = 2;
	}

	sdkCreateTimer(&_spgemm_timer);
	sdkCreateTimer(&_stage1_timer);
	sdkCreateTimer(&_stage1_1_timer);
	sdkCreateTimer(&_stage1_2_timer);
	sdkCreateTimer(&_stage2_timer);
	sdkCreateTimer(&_stage2_1_timer);
	sdkCreateTimer(&_stage3_timer);
	sdkCreateTimer(&_stage3_1_timer);
	sdkCreateTimer(&_stage3_2_timer);
	sdkCreateTimer(&_stage4_timer);
	sdkCreateTimer(&_stage4_1_timer);

	_n = n, _m = m, _p = p;
	_h_A = Aptr, _h_B = Bptr;
	_d_A.nnz = _h_A.nnz, _d_B.nnz = _h_B.nnz;

	checkCudaErrors(cudaMalloc((void **)&_d_A.csrRowPtr, (_n + 1) * sizeof(index_t)));
	checkCudaErrors(cudaMalloc((void **)&_d_A.csrColIdx, _d_A.nnz * sizeof(index_t)));
	checkCudaErrors(cudaMalloc((void **)&_d_A.csrVal, _d_A.nnz * sizeof(value_t)));

	checkCudaErrors(cudaMalloc((void **)&_d_B.csrRowPtr, (_m + 1) * sizeof(index_t)));
	checkCudaErrors(cudaMalloc((void **)&_d_B.csrColIdx, _d_B.nnz * sizeof(index_t)));
	checkCudaErrors(cudaMalloc((void **)&_d_B.csrVal, _d_B.nnz * sizeof(value_t)));

	checkCudaErrors(cudaMemcpy(_d_A.csrRowPtr, _h_A.csrRowPtr, (_n + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_A.csrColIdx, _h_A.csrColIdx, _d_A.nnz * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_A.csrVal, _h_A.csrVal, _d_A.nnz * sizeof(value_t), cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(_d_B.csrRowPtr, _h_B.csrRowPtr, (_m + 1) * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_B.csrColIdx, _h_B.csrColIdx, _d_B.nnz * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_B.csrVal, _h_B.csrVal, _d_B.nnz * sizeof(value_t), cudaMemcpyHostToDevice));

	_hasDataInited = true;
	if (printInfo) printf("[InitData] Info: n = %d, m = %d, p = %d, %d elements in total.\n", _n, _m, _p, _h_A.nnz);
	printf("[InitData] Matrix successfully loaded.\n");
	return err;
}

int scsparse::initConfig(Arg_t args)
{
	int err = 0;
	_alpha_coefficient = args.alpha_coe;
	_matName = args.matName;
	return err;
}

int scsparse::warmup(int times)
{
	int err = 0;
	printf("[Warmup] Warming up");
	for (int i = 0; i < times; i++)
	{
		compute_nnzRowCt_eigenRow(true);
		printf(".");
	}
	printf("done.\n");
	return err;
}

int scsparse::spgemm()
{
	int err = 0;
	if (!_hasDataInited)
	{
		fprintf(stderr, "[SpGeMM] Error: Data not loaded.\n");
		return err = 1;
	}
	if (_hasCalced)
	{
		fprintf(stderr, "[SpGeMM] Warning: Duplicated calculation detected.\n");
		freeMem(false);
		_hasCalced = false;
	}

	printf("[SpGeMM] Benchmark started.\n");
	sdkStartTimer(&_spgemm_timer);
	printLine();
	
	//STAGE 1: Analyse
	printf("[SpGeMM] Stage 1/4: Analyse\n");
	sdkStartTimer(&_stage1_timer);
	
	//STAGE 1 - STEP 1: Compute nnzCt and eigenV
	printf("compute_nnzRowCt_eigenRow()...");
	sdkStartTimer(&_stage1_1_timer);
	err = compute_nnzRowCt_eigenRow();	// func
	sdkStopTimer(&_stage1_1_timer);
	if (err) fprintf(stderr, "failed, error code = %d\n", err);
	else printf("done, in %.2f ms\n", sdkGetTimerValue(&_stage1_1_timer));

	//STAGE 1 - STEP 1: Compute nnzCt and eigenV
	printf("computeAlpha()...");
	sdkStartTimer(&_stage1_2_timer);
	err = computeAlpha();	// func
	sdkStopTimer(&_stage1_2_timer);
	if (err) fprintf(stderr, "failed, error code = %d\n", err);
	else printf("done, in %.2f ms\n", sdkGetTimerValue(&_stage1_2_timer));

	sdkStopTimer(&_stage1_timer);
	printf("[SpGeMM] Alpha = %.2f (%.2fx)\n", _alpha, _alpha_coefficient);
	printf("[SpGeMM] Stage 1/4 finished in %.2f ms.\n", sdkGetTimerValue(&_stage1_timer));
	printLine();

	// STAGE 2: Group tasks
	printf("[SpGeMM] Stage 2/4: Group tasks\n");
	sdkStartTimer(&_stage2_timer);

	printf("taskClassify()...");
	sdkStartTimer(&_stage2_1_timer);
	err = taskClassify();	// func
	sdkStopTimer(&_stage2_1_timer);
	if (err) fprintf(stderr, "failed, error code = %d\n", err);
	else printf("done, in %.2f ms\n", sdkGetTimerValue(&_stage2_1_timer));

	sdkStopTimer(&_stage2_timer);
	printf("[SpGeMM] EmptyRow = %d, SDGSiz = %d, CDGSiz = %d\n", _emptyRowCnt, _SDRowCnt, _CDRowCnt);
	printf("[SpGeMM] Stage 2/4 finished in %.2f ms.\n", sdkGetTimerValue(&_stage2_timer));
	printLine();

	// STAGE 3: Compute
	printf("[SpGeMM] Stage 3/4: Compute\n");
	sdkStartTimer(&_stage3_timer);

	// STAGE 3 - STEP 1 : compute SD group
	printf("computeSDGroup() [hash_table_size = %d]...", HASHTABLESIZ);
	sdkStartTimer(&_stage3_2_timer);
	err = computeSDGroup();	// func
	err |= kernelBarrier();
	sdkStopTimer(&_stage3_2_timer);
	if (err) fprintf(stderr, "failed, error code = %d\n", err);
	else printf("done (%d rows dropped), in %.2f ms\n", _SDRowFailedCnt, sdkGetTimerValue(&_stage3_2_timer));

	// STAGE 3 - STEP 2 : compute CD group
	printf("computeCDGroup()...");
	sdkStartTimer(&_stage3_1_timer);
	err = computeCDGroup();	// func
	err |= kernelBarrier();
	sdkStopTimer(&_stage3_1_timer);
	if (err) fprintf(stderr, "failed, error code = %d\n", err);
	else printf("done, in %.2f ms\n", sdkGetTimerValue(&_stage3_1_timer));

	sdkStopTimer(&_stage3_timer);
	printf("[SpGeMM] Stage 3/4 finished in %.2f ms.\n", sdkGetTimerValue(&_stage3_timer));
	printLine();

	// STAGE 4: Conclude
	printf("[SpGeMM] Stage 4/4: Conclude\n");
	sdkStartTimer(&_stage4_timer);

	printf("postProcess()...");
	sdkStartTimer(&_stage4_1_timer);
	err = postProcess();	// func
	sdkStopTimer(&_stage4_1_timer);
	if (err) fprintf(stderr, "failed, error code = %d\n", err);
	else printf("done, in %.2f ms\n", sdkGetTimerValue(&_stage4_1_timer));

	sdkStopTimer(&_stage4_timer);
	printf("[SpGeMM] Stage 4/4 finished in %.2f ms.\n", sdkGetTimerValue(&_stage4_timer));
	printLine();

	sdkStopTimer(&_spgemm_timer);
	printf("[SpGeMM] Benchmark finished in %.2f ms.\n", sdkGetTimerValue(&_spgemm_timer));
	_hasCalced = true;
	return err;
}

int scsparse::kernelBarrier()
{
	return cudaDeviceSynchronize();
}

int scsparse::compute_nnzRowCt_eigenRow(bool isWarmup)
{
	int err = 0;

	int num_threads = 256;
	//int num_blocks = ceil((double)_n / (double)num_threads);
	int num_blocks = (_n + num_threads - 1) / num_threads;

	checkCudaErrors(cudaMalloc((void **)&_d_nnzRowCt, _n * sizeof(int)));
	checkCudaErrors(cudaMalloc((void **)&_d_eigenRow, _n * sizeof(double)));
	checkCudaErrors(cudaMemset(_d_nnzRowCt, 0, _n * sizeof(int)));

	compute_row_eigen <<< num_blocks, num_threads >>> (_n,
		_d_A.csrRowPtr, _d_A.csrColIdx,
		_d_B.csrRowPtr,
		_d_nnzRowCt,
		_d_eigenRow);

	kernelBarrier();

	cudaError_t cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		fprintf(stderr, "err = %s\n", cudaGetErrorString(cuErr));
		return err = -1;
	}

	_h_nnzRowCt = (int *)malloc(sizeof(int)*_n);
	_h_eigenRow = (double *)malloc(sizeof(double)*_n);

	checkCudaErrors(cudaMemcpy(_h_nnzRowCt, _d_nnzRowCt, _n * sizeof(int), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(_h_eigenRow, _d_eigenRow, _n * sizeof(double), cudaMemcpyDeviceToHost));

	if (isWarmup)
	{
		checkCudaErrors(cudaFree(_d_nnzRowCt));
		checkCudaErrors(cudaFree(_d_eigenRow));
		free(_h_nnzRowCt);
		free(_h_eigenRow);
	}

	return err;
} 

int scsparse::computeAlpha()
{
	//_nnzCt = std::accumulate(_h_nnzRowCt, _h_nnzRowCt + _n, 0);
	_nnzCt = thrust::reduce(tIntPtr_t(_d_nnzRowCt), tIntPtr_t(_d_nnzRowCt) + _n);
	_alpha = (double)_h_A.nnz / _nnzCt;
	_alpha *= _alpha_coefficient;
	//_alpha = 1e100;
	//_alpha = -1;
	return 0;
}

int scsparse::taskClassify()
{
	for (int i = 0; i < _n; i++)
	{
		if (_h_nnzRowCt[i] == 0) _emptyRowIdx.push_back(i);
		else if (_h_eigenRow[i] <= _alpha) _lesRowIdx.push_back(i);
		else _grtRowIdx.push_back(i);
	}
	_emptyRowCnt = _emptyRowIdx.size();
	_SDRowCnt = _lesRowIdx.size();
	_CDRowCnt = _grtRowIdx.size();
	_SDRowFailedCnt = 0;
	return 0;
}


int scsparse::computeSDGroup()
{
	int err = 0;
	_SDEleTotal = 0;
	if (_SDRowCnt == 0) return err;
	_h_SDRowIdx = (index_t*)malloc(sizeof(index_t)*_SDRowCnt);
	//_h_SDRowNNZ = (index_t*)malloc(sizeof(index_t)*_SDRowCnt);
	memcpy(_h_SDRowIdx, &_lesRowIdx[0], _SDRowCnt * sizeof(index_t));
	//for (int i = 0; i < _SDRowCnt; i++) _h_SDRowNNZ[i] = _h_nnzRowCt[_h_SDRowIdx[i]];
	checkCudaErrors(cudaMalloc((void **)&_d_SDRowIdx, _SDRowCnt * sizeof(index_t)));
	//checkCudaErrors(cudaMalloc((void **)&_d_SDRowNNZ, _SDRowCnt * sizeof(int)));
	checkCudaErrors(cudaMemcpy(_d_SDRowIdx, _h_SDRowIdx, _SDRowCnt * sizeof(index_t), cudaMemcpyHostToDevice));
	//checkCudaErrors(cudaMemcpy(_d_SDRowNNZ, _h_SDRowNNZ, _SDRowCnt * sizeof(int), cudaMemcpyHostToDevice));
	//thrust::sort_by_key(tIntPtr_t(_d_SDRowNNZ), tIntPtr_t(_d_SDRowNNZ)+_SDRowCnt, tIntPtr_t(_d_SDRowIdx));
	//checkCudaErrors(cudaMemcpy(_h_SDRowIdx, _d_SDRowIdx, _SDRowCnt * sizeof(index_t), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaMemcpy(_h_SDRowNNZ, _d_SDRowNNZ, _SDRowCnt * sizeof(int), cudaMemcpyDeviceToHost));

// Group into different bins
// In this version we simply alloc one pool of pre-defined size
	_binNum = 1;
	_h_binOfs = (int*)malloc(sizeof(int)*(_binNum + 1));
	_h_binOfs[0] = 0, _h_binOfs[_binNum] = _SDRowCnt;
// -----------------------------------------------------------

	_h_htValidLen = (int*)malloc(sizeof(int) * _SDRowCnt);
	_h_htOffset = (int*)malloc(sizeof(int) * _SDRowCnt);
	checkCudaErrors(cudaMalloc((void**)&_d_htValidLen, sizeof(int) * _SDRowCnt));
	checkCudaErrors(cudaMalloc((void**)&_d_htOffset, sizeof(int) * _SDRowCnt));

	int htEndLoc = 0;
	for (int i = 0; i < _SDRowCnt; i++)
	{
		_h_htOffset[i] = htEndLoc;
		int absIdx = _h_SDRowIdx[i];
		htEndLoc += _h_nnzRowCt[absIdx];
	}
	_htSize = htEndLoc;

	checkCudaErrors(cudaMemcpy(_d_htOffset, _h_htOffset, sizeof(int)*_SDRowCnt, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void**)&_d_htIdxPool, sizeof(index_t) * _htSize));
	checkCudaErrors(cudaMalloc((void**)&_d_htValPool, sizeof(value_t) * _htSize));

	int groupID, num_blocks, num_threads, beginLoc, endLoc;
// Kernel begin
	// Group 0: size $(HASHTABLESIZ)
	groupID = 0;
	num_blocks = 100;
	num_threads = 256;
	beginLoc = _h_binOfs[groupID], endLoc = _h_binOfs[groupID + 1];
	compute_sd_group <HASHTABLESIZ><<<num_blocks, num_threads >>> (
		_d_A.csrRowPtr, _d_A.csrColIdx, _d_A.csrVal,
		_d_B.csrRowPtr, _d_B.csrColIdx, _d_B.csrVal,
		_d_SDRowIdx + beginLoc, endLoc - beginLoc,
		_d_htIdxPool + beginLoc, _d_htValPool + beginLoc, 
		_d_htOffset + beginLoc, _d_htValidLen + beginLoc);
	// You can add other groups here

// Kernel end
	kernelBarrier();

	cudaError_t cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		fprintf(stderr, "errStr = %s", cudaGetErrorString(cuErr));
		return err = -1;
	}

	checkCudaErrors(cudaMemcpy(_h_htValidLen, _d_htValidLen, sizeof(int)*_SDRowCnt, cudaMemcpyDeviceToHost));

	_SDEleTotal = 0;
	_SDRowFailedCnt = 0;
	for (int i = 0; i < _SDRowCnt; i++)
	{
		if (_h_htValidLen[i] == 0)
		{
			_grtRowIdx.push_back(_h_SDRowIdx[i]);
			_SDRowFailedCnt++;
		}
		_SDEleTotal += _h_htValidLen[i];
	}

	return err;
}

int scsparse::computeCDGroup()
{
	int err = 0;
	_CDEleTotal = 0;
	_CDRowCnt = _grtRowIdx.size();
	if (_CDRowCnt == 0) return err;
	
// Group into boxes of similar size 
// In this version we use a simple method
	_CDNumBlocks = 80;
	_CDNumThreads = 32;
	_CDBoxCnt = _CDNumBlocks * _CDNumThreads;
	int nnzCDTotal = 0;
	for (int i = 0; i < _CDRowCnt; i++) nnzCDTotal += _h_nnzRowCt[_grtRowIdx[i]];
	int nnzAvg = (nnzCDTotal + _CDBoxCnt - 1) / _CDBoxCnt;
	_h_CDRowIdxOfs = (int*)malloc((_CDBoxCnt + 1) * sizeof(int));
	for (int i = 0, curCDLoc = 0; i < _CDBoxCnt; i++)
	{
		_h_CDRowIdxOfs[i] = curCDLoc;
		int curBoxLoad = 0;
		while (curCDLoc < _CDRowCnt)
		{
			int preAdd = _h_nnzRowCt[_grtRowIdx[curCDLoc]];
			if (curBoxLoad + preAdd <= nnzAvg)
			{
				curBoxLoad += preAdd;
				curCDLoc++;
			}
			else
			{
				if ((_CDRowCnt - curCDLoc) == (_CDBoxCnt - i) ||
					(nnzAvg - curBoxLoad) >(curBoxLoad + preAdd - nnzAvg))
				{
					curBoxLoad += preAdd;
					curCDLoc++;
				}
				break;
			}
		}
	}
	_h_CDRowIdxOfs[_CDBoxCnt] = _CDRowCnt;
	
// -----------------------------------------------------------
	_h_CDRowIdx = (index_t*)malloc(sizeof(index_t)*_CDRowCnt);
	memcpy(_h_CDRowIdx, &_grtRowIdx[0], _CDRowCnt * sizeof(index_t));
	checkCudaErrors(cudaMalloc((void **)&_d_CDRowIdx, _CDRowCnt * sizeof(index_t)));
	checkCudaErrors(cudaMemcpy(_d_CDRowIdx, _h_CDRowIdx, _CDRowCnt * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void **)&_d_CDRowIdxOfs, (_CDBoxCnt + 1) * sizeof(int)));
	checkCudaErrors(cudaMemcpy(_d_CDRowIdxOfs, _h_CDRowIdxOfs, (_CDBoxCnt + 1) * sizeof(index_t), cudaMemcpyHostToDevice));

	_h_poolOffset = (int*)malloc(sizeof(int)*_CDRowCnt);
	_h_poolLen = (int*)malloc(sizeof(int)*_CDRowCnt);
	checkCudaErrors(cudaMalloc((void **)&_d_poolOffset, sizeof(int)*_CDRowCnt));
	checkCudaErrors(cudaMalloc((void**)&_d_poolLen, sizeof(int) * _CDRowCnt));

	int poolEndLoc = 0;
	for (int i = 0; i < _CDRowCnt; i++)
	{
		_h_poolOffset[i] = poolEndLoc;
		int absIdx = _h_CDRowIdx[i];
		poolEndLoc += upper2bound(_h_nnzRowCt[absIdx]);
	}
	_poolSize = poolEndLoc;
	checkCudaErrors(cudaMemcpy(_d_poolOffset, _h_poolOffset, sizeof(int)*_CDRowCnt, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMalloc((void **)&_d_poolIdx, sizeof(index_t) * _poolSize));
	checkCudaErrors(cudaMalloc((void **)&_d_poolVal, sizeof(value_t) * _poolSize));

	compute_cd_group <<<_CDNumBlocks, _CDNumThreads >>> (
		_d_A.csrRowPtr, _d_A.csrColIdx, _d_A.csrVal,
		_d_B.csrRowPtr, _d_B.csrColIdx, _d_B.csrVal,
		_d_CDRowIdx, _d_CDRowIdxOfs,
		_d_poolIdx, _d_poolVal, _d_poolOffset, _d_poolLen);

	kernelBarrier();

	bitonicSort_by_key_block <<<10, 128 >>> (
		_CDRowCnt, _d_poolIdx, _d_poolVal, _d_poolOffset, _d_poolLen);

	kernelBarrier();
	compute_cd_group_merge <<<_CDNumBlocks, _CDNumThreads >>> (
		_d_A.csrRowPtr, _d_A.csrColIdx, _d_A.csrVal,
		_d_B.csrRowPtr, _d_B.csrColIdx, _d_B.csrVal,
		_d_CDRowIdx, _d_CDRowIdxOfs,
		_d_poolIdx, _d_poolVal, _d_poolOffset, _d_poolLen);


	checkCudaErrors(cudaMemcpy(_h_poolLen, _d_poolLen, sizeof(int) * _CDRowCnt, cudaMemcpyDeviceToHost));
	_CDEleTotal = thrust::reduce(tIntPtr_t(_d_poolLen), tIntPtr_t(_d_poolLen) + _CDRowCnt);

	cudaError_t cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		fprintf(stderr, "errStr = %s", cudaGetErrorString(cuErr));
		return err = -1;
	}
	return err;
}


int scsparse::postProcess()
{
	int err = 0;
	_h_C.nnz = _SDEleTotal + _CDEleTotal;
	_isCached = false;
	_h_rowAttr = (attr_t*)malloc(sizeof(attr_t) * _n);
	_h_row2PoolIdx = (index_t*)malloc(sizeof(index_t) * _n);
	for (int i = 0; i < _emptyRowCnt; i++)
	{
		int rowId = _emptyRowIdx[i];
		_h_rowAttr[rowId] = EMPTYATTR;
		_h_row2PoolIdx[rowId] = i;
	}
	for (int i = 0; i < _SDRowCnt; i++)
	{
		int rowId = _h_SDRowIdx[i];
		if (_h_htValidLen[i] != 0)
		{
			_h_rowAttr[rowId] = SDATTR;
			_h_row2PoolIdx[rowId] = i;
		}
	}
	for (int i = 0; i < _CDRowCnt; i++)
	{
		int rowId = _h_CDRowIdx[i];
		_h_rowAttr[rowId] = CDATTR;
		_h_row2PoolIdx[rowId] = i;
	}
	//_h_C.csrRowPtr = (index_t*)malloc(sizeof(index_t)*(_n + 1));
	//_h_C.csrColIdx = (index_t*)malloc(sizeof(index_t)*(_h_C.nnz));
	//_h_C.csrVal = (value_t*)malloc(sizeof(value_t)*(_h_C.nnz));
	return err;
}
	
int scsparse::getCptr(csrPtr &Cptr, bool printInfo)
{
	if (!_hasCalced)
	{
		fprintf(stderr, "[I/O Interface] Error: C has not been calculated yet.\n");
		return -1;
	}
	if (!_isCached)
	{
		printf("[I/O Interface] No cached data found. Gathering data...");
		_h_C.csrRowPtr = (index_t*)malloc(sizeof(index_t)*(_n + 1));
		_h_C.csrColIdx = (index_t*)malloc(sizeof(index_t)*(_h_C.nnz));
		_h_C.csrVal = (value_t*)malloc(sizeof(value_t)*(_h_C.nnz));
		for (int i = 0, pos = 0; i < _n; i++)
		{
			_h_C.csrRowPtr[i] = pos;
			if (_h_rowAttr[i] == SDATTR)
			{
				int sdIdx = _h_row2PoolIdx[i];
				int posInc = _h_htValidLen[sdIdx];
				int offset = _h_htOffset[sdIdx];
				checkCudaErrors(cudaMemcpy(_h_C.csrColIdx + pos, _d_htIdxPool + offset,
					sizeof(index_t)*posInc, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(_h_C.csrVal + pos, _d_htValPool + offset,
					sizeof(value_t)*posInc, cudaMemcpyDeviceToHost));
				pos += posInc;
			}
			else if (_h_rowAttr[i] == CDATTR)
			{
				int cdIdx = _h_row2PoolIdx[i];
				int posInc = _h_poolLen[cdIdx];
				int offset = _h_poolOffset[cdIdx];
				checkCudaErrors(cudaMemcpy(_h_C.csrColIdx + pos, _d_poolIdx + offset,
					sizeof(index_t)*posInc, cudaMemcpyDeviceToHost));
				checkCudaErrors(cudaMemcpy(_h_C.csrVal + pos, _d_poolVal + offset,
					sizeof(value_t)*posInc, cudaMemcpyDeviceToHost));
				pos += posInc;
			}
		}
		_h_C.csrRowPtr[_n] = _h_C.nnz;
		_isCached = true;
		printf("done.\n");
	}
	else printf("[I/O Interface] Using cached data.\n");
	Cptr = _h_C;
	return 0;
}

int scsparse::analyse()
{
	int *a_nnzRowA = (int*)malloc(_n* sizeof(int));
	for (int i = 0; i < _n; i++)
		a_nnzRowA[i] = _h_A.csrRowPtr[i + 1] - _h_A.csrRowPtr[i];
	double a_avgRowNnzA = _h_A.nnz * 1.0 / _n;

	int *a_nnzRowCt = _h_nnzRowCt;
	double a_avgRowNnzCt = _nnzCt * 1.0 / _n;
	int a_nnzCt = _nnzCt;

	int *a_nnzRowC = (int*)malloc(_n * sizeof(int));
	for (int i = 0; i < _n; i++)
		a_nnzRowC[i] = _h_C.csrRowPtr[i + 1] - _h_C.csrRowPtr[i];
	double a_avgRowNnzC = _h_C.nnz * 1.0 / _n;
	int a_nnzC = _h_C.nnz;

	//------------ANA_BEGIN-------------------



	//-------------ANA_END--------------------
	

	free(a_nnzRowA);
	free(a_nnzRowC);
	return 0;
}

int scsparse::freeMem(bool freeData, bool freeSpgemm, bool freeCache)
{
	int err = 0;

	if (_hasDataInited && freeData)
	{
		sdkDeleteTimer(&_spgemm_timer);
		sdkDeleteTimer(&_stage1_timer);
		sdkDeleteTimer(&_stage1_1_timer);
		sdkDeleteTimer(&_stage1_2_timer);
		sdkDeleteTimer(&_stage2_timer);
		sdkDeleteTimer(&_stage2_1_timer);
		sdkDeleteTimer(&_stage3_timer);
		sdkDeleteTimer(&_stage3_1_timer);
		sdkDeleteTimer(&_stage3_2_timer);
		sdkDeleteTimer(&_stage4_timer);
		sdkDeleteTimer(&_stage4_1_timer);

		checkCudaErrors(cudaFree(_d_A.csrRowPtr));
		checkCudaErrors(cudaFree(_d_A.csrColIdx));
		checkCudaErrors(cudaFree(_d_A.csrVal));

		checkCudaErrors(cudaFree(_d_B.csrRowPtr));
		checkCudaErrors(cudaFree(_d_B.csrColIdx));
		checkCudaErrors(cudaFree(_d_B.csrVal));
		_hasDataInited = false;
	}
	if (_hasCalced && freeSpgemm)
	{
		checkCudaErrors(cudaFree(_d_nnzRowCt));
		checkCudaErrors(cudaFree(_d_eigenRow));
		free(_h_nnzRowCt);
		free(_h_eigenRow);

		_emptyRowIdx.clear();
		_lesRowIdx.clear();
		_grtRowIdx.clear();

		if (_SDRowCnt != 0)
		{
			free(_h_SDRowIdx);
			checkCudaErrors(cudaFree(_d_SDRowIdx));
			free(_h_binOfs);
			free(_h_htValidLen);
			free(_h_htOffset);
			checkCudaErrors(cudaFree(_d_htValidLen));
			checkCudaErrors(cudaFree(_d_htOffset));
			checkCudaErrors(cudaFree(_d_htIdxPool));
			checkCudaErrors(cudaFree(_d_htValPool));
		}

		if (_CDRowCnt != 0)
		{
			free(_h_CDRowIdxOfs);
			free(_h_CDRowIdx);
			checkCudaErrors(cudaFree(_d_CDRowIdxOfs));
			checkCudaErrors(cudaFree(_d_CDRowIdx));
			free(_h_poolOffset);
			free(_h_poolLen);
			checkCudaErrors(cudaFree(_d_poolOffset));
			checkCudaErrors(cudaFree(_d_poolLen));
			checkCudaErrors(cudaFree(_d_poolIdx));
			checkCudaErrors(cudaFree(_d_poolVal));
		}

		free(_h_rowAttr);
		free(_h_row2PoolIdx);
		_hasCalced = false;
	}
	if (_isCached && freeCache)
	{
		free(_h_C.csrRowPtr);
		free(_h_C.csrColIdx);
		free(_h_C.csrVal);
	}

	return err;
}

scsparse::~scsparse() 
{
	freeMem();
}
