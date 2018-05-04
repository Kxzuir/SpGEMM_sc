#include "scsparse_host.h"
#include "scsparse_device.cuh"


scsparse::scsparse()
{
	_hasPFInited = _hasCalced = _hasDataInited = false;
}

int scsparse::initDevice(int deviceId, bool printInfo)
{
	cudaSetDevice(deviceId);
	cudaDeviceProp deviceProp;
	cudaError_t cuErr = cudaGetDeviceProperties(&deviceProp, deviceId);
	if (cuErr != cudaSuccess)
	{
		fprintf(stderr, "[initDevice] Error: Can not get CUDA info for Device [%d].\n", deviceId);
		return 1;
	}
	_deviceId = deviceId;
	_num_smx = deviceProp.multiProcessorCount;
	_max_blocks_per_smx = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
	_num_cuda_cores_per_smx = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	_num_cuda_cores = _num_cuda_cores_per_smx * _num_smx;

	if (printInfo)
	{
		printf("[initDevice] Info: Using device [%d] %s @ %.2f MHz (%d.%d).\n",
			deviceId, deviceProp.name, deviceProp.clockRate * 1e-3f, deviceProp.major, deviceProp.minor);
		printf("[initDevice] Info: %d SMXs, %d CUDA cores per SMX.\n", _num_smx, _num_cuda_cores_per_smx);
	}
	_hasPFInited = true;
	return 0;
}


int scsparse::initData(int n, int m, int p, csrPtr Aptr, csrPtr Bptr, bool printInfo)
{
	int err = 0;
	if (_hasPFInited == false) fprintf(stderr, "[initData] Warning: Device not initialized.\n");

	_spgemm_timer = NULL;
	_stage1_timer = NULL;
	_stage1_1_timer = NULL;
	_stage1_2_timer = NULL;
	_stage2_timer = NULL;
	_stage2_1_timer = NULL;
	_stage3_timer = NULL;
	_stage3_1_timer = NULL;
	_stage3_2_timer = NULL;
	_stage4_timer = NULL;
	_stage4_1_timer = NULL;
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
	if (printInfo) printf("[initData] Info: n = %d, m = %d, p = %d, %d elements in total.\n", _n, _m, _p, _h_A.nnz);
	printf("[initData] Matrix successfully loaded.\n");
	return err;
}


int scsparse::spgemm()
{
	int err = 0;
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
	printf("[SpGeMM] Alpha = %.2f\n", _alpha);
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
	printf("[SpGeMM] CDGSiz = %d, SDGSiz = %d\n", _CDRowCnt, _SDRowCnt);
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
	else printf("done, in %.2f ms\n", sdkGetTimerValue(&_stage3_2_timer));

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

	// STAGE 4: Merge Data
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
	return err;
}

int scsparse::kernelBarrier()
{
	return cudaDeviceSynchronize();
}

int scsparse::compute_nnzRowCt_eigenRow()
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

	return err;
} 

int scsparse::computeAlpha()
{
	//_nnzCt = std::accumulate(_h_nnzRowCt, _h_nnzRowCt + _n, 0);
	_nnzCt = thrust::reduce(tIntPtr_t(_d_nnzRowCt), tIntPtr_t(_d_nnzRowCt) + _n);
	_alpha = (double)_h_A.nnz / _nnzCt;
	//_alpha *= 10;
	//_alpha = 1e100;
	return 0;
}

int scsparse::taskClassify()
{
	_h_rowAttr = (attr_t*)malloc(sizeof(attr_t) * _n);
	_h_row2PoolIdx = (index_t*)malloc(sizeof(index_t) * _n);
	std::vector<index_t> grtRowIdx, lesRowIdx;
	_SDRowCnt = _CDRowCnt = 0;
	int nnzCDTotal = 0;
	for (int i = 0; i < _n; i++)
	{
		if (_h_eigenRow[i] <= _alpha)		
		{
			lesRowIdx.push_back(i);
			_h_rowAttr[i] = SDATTR;
			_h_row2PoolIdx[i] = _SDRowCnt++;
		}
		else
		{
			grtRowIdx.push_back(i);
			_h_rowAttr[i] = CDATTR;
			_h_row2PoolIdx[i] = _CDRowCnt++;
			nnzCDTotal += _h_nnzRowCt[i];
		}
	}
	//---------rearrange grtRowIdx------------------
	_CDNumBlocks = 80;
	_CDNumThreads = 32;
	_CDBoxCnt = _CDNumBlocks * _CDNumThreads;
	int nnzAvg = (nnzCDTotal + _CDBoxCnt - 1) / _CDBoxCnt;
	_h_CDRowIdxOfs = (int*)malloc((_CDBoxCnt + 1) * sizeof(int));
	for (int i = 0, curCDLoc = 0; i < _CDBoxCnt; i++)
	{
		_h_CDRowIdxOfs[i] = curCDLoc;
		int curBoxLoad = 0;
		while (curCDLoc < _CDRowCnt)
		{
			int preAdd = _h_nnzRowCt[grtRowIdx[curCDLoc]];
			if (curBoxLoad + preAdd <= nnzAvg)
			{
				curBoxLoad += preAdd;
				curCDLoc++;
			}
			else
			{
				if ((nnzAvg - curBoxLoad) > (curBoxLoad + preAdd - nnzAvg))
				{
					curBoxLoad += preAdd;
					curCDLoc++;
				}
				break;
			}
		}
	}
	_h_CDRowIdxOfs[_CDBoxCnt] = _CDRowCnt;
	//----------------------------------------------


	//_SDRowCnt = lesRowIdx.size();
	//_CDRowCnt = grtRowIdx.size();
	_h_SDRowIdx = (index_t*)malloc(sizeof(index_t)*_SDRowCnt);
	_h_CDRowIdx = (index_t*)malloc(sizeof(index_t)*_CDRowCnt);
	if (_SDRowCnt > 0) memcpy(_h_SDRowIdx, &lesRowIdx[0], _SDRowCnt * sizeof(index_t));
	if (_CDRowCnt > 0) memcpy(_h_CDRowIdx, &grtRowIdx[0], _CDRowCnt * sizeof(index_t));

	checkCudaErrors(cudaMalloc((void **)&_d_SDRowIdx, _SDRowCnt * sizeof(index_t)));
	checkCudaErrors(cudaMalloc((void **)&_d_CDRowIdx, _CDRowCnt * sizeof(index_t)));
	checkCudaErrors(cudaMalloc((void **)&_d_CDRowIdxOfs, (_CDBoxCnt+1) * sizeof(int)));
	checkCudaErrors(cudaMemcpy(_d_SDRowIdx, _h_SDRowIdx, _SDRowCnt * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_CDRowIdx, _h_CDRowIdx, _CDRowCnt * sizeof(index_t), cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(_d_CDRowIdxOfs, _h_CDRowIdxOfs, (_CDBoxCnt+1) * sizeof(index_t), cudaMemcpyHostToDevice));

	return 0;
}

int scsparse::computeCDGroup()
{
	int err = 0;
	if (_CDRowCnt == 0)
	{
		_CDEleTotal = 0;
		return 0;
	}
	
	//int *rPos = NULL;
	//checkCudaErrors(cudaMalloc((void**)&rPos, sizeof(int)));
	//checkCudaErrors(cudaMemset(rPos, 0, sizeof(int)));

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

int scsparse::computeSDGroup()
{
	int err = 0;
	
	_SDEleTotal = 0;
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

	int num_blocks = 100;
	int num_threads = 256;
	compute_sd_group <512><<<num_blocks, num_threads >>> (
		_d_A.csrRowPtr, _d_A.csrColIdx, _d_A.csrVal,
		_d_B.csrRowPtr, _d_B.csrColIdx, _d_B.csrVal,
		_d_SDRowIdx, _SDRowCnt,
		_d_htIdxPool, _d_htValPool, _d_htOffset, _d_htValidLen
		);
	kernelBarrier();

	cudaError_t cuErr = cudaGetLastError();
	if (cuErr != cudaSuccess)
	{
		fprintf(stderr, "errStr = %s", cudaGetErrorString(cuErr));
		return err = -1;
	}

	checkCudaErrors(cudaMemcpy(_h_htValidLen, _d_htValidLen, sizeof(int)*_SDRowCnt, cudaMemcpyDeviceToHost));
/*
	for (int i = 0; i < _SDRowCnt; i++)
	{
	//	if(i%10==0)
	//		printf("\b\b\b\b\b\b\b\b\b\b\b\b%d/%d", i, _SDRowCnt);
		int offset = HASHTABLESIZ * i;
		thrust::sort_by_key(tIntPtr_t(_d_htIdxPool) + offset, tIntPtr_t(_d_htIdxPool) + offset + HASHTABLESIZ, tDoublePtr_t(_d_htValPool) + offset);
		_h_htValidLen[i] = HASHTABLESIZ - 
			thrust::count(tIntPtr_t(_d_htIdxPool) + offset, tIntPtr_t(_d_htIdxPool) + offset + HASHTABLESIZ, INVALIDHASHIDX);
		_SDEleTotal += _h_htValidLen[i];
	}
*/
	_SDEleTotal = thrust::reduce(tIntPtr_t(_d_htValidLen), tIntPtr_t(_d_htValidLen) + _SDRowCnt);
	/*
	for (int i = 0; i < _SDRowCnt; i++)
	{
		int offset = HASHTABLESIZ * i;
		_SDEleTotal += _h_htValidLen[i];
		thrust::sort_by_key(tIntPtr_t(_d_htIdxPool) + offset, 
			tIntPtr_t(_d_htIdxPool) + offset + _h_htValidLen[i], tDoublePtr_t(_d_htValPool) + offset);
	}
	*/

	//int invalidCntTot = thrust::count(_d_htIdxPool, _d_htIdxPool + _SDRowCnt * HASHTABLESIZ, INVALIDHASHIDX);
	//_SDEleTotal = _SDRowCnt * HASHTABLESIZ - invalidCntTot;
	return err;
}

int scsparse::postProcess()
{
	int err = 0;
	_h_C.nnz = _SDEleTotal + _CDEleTotal;
	_isCached = false;
	//_h_C.csrRowPtr = (index_t*)malloc(sizeof(index_t)*(_n + 1));
	//_h_C.csrColIdx = (index_t*)malloc(sizeof(index_t)*(_h_C.nnz));
	//_h_C.csrVal = (value_t*)malloc(sizeof(value_t)*(_h_C.nnz));
	return err;
}
	
int scsparse::getCptr(csrPtr &Cptr, bool printInfo)
{
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



int scsparse::freeMem() {
	int err = 0;
	//TODO
	return err;
}

scsparse::~scsparse() {
	freeMem();
}
