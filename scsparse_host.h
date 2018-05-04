#pragma once
#ifndef SCSPARSE_HOST_H
#define SCSPARSE_HOST_H


#include "common.h"
#include <thrust/sort.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

typedef thrust::device_ptr<int> tIntPtr_t;
typedef thrust::device_ptr<double> tDoublePtr_t;


class scsparse
{
public:
	scsparse();
	int initDevice(int deviceId = 0, bool printInfo = true);
	int initData(int n, int m, int p, csrPtr Aptr, csrPtr Bptr, 
		bool printInfo = true);
	int spgemm();
	int getCptr(csrPtr &Cptr, bool printInfo = true);
	int freeMem();
	~scsparse();
private:
	bool _hasPFInited, _hasDataInited, _hasCalced;
	int _deviceId, _num_smx, _num_cuda_cores_per_smx, _max_blocks_per_smx;
	int _num_cuda_cores;

	int _n, _m, _p;
	csrPtr _h_A, _h_B, _h_C;
	csrPtr _d_A, _d_B, _d_C;

	StopWatchInterface *_spgemm_timer;
	StopWatchInterface *_stage1_timer, *_stage1_1_timer, *_stage1_2_timer;
	StopWatchInterface *_stage2_timer, *_stage2_1_timer;
	StopWatchInterface *_stage3_timer, *_stage3_1_timer, *_stage3_2_timer;
	StopWatchInterface *_stage4_timer, *_stage4_1_timer;

	int kernelBarrier();

	int *_d_nnzRowCt, *_h_nnzRowCt;
	double *_d_eigenRow, *_h_eigenRow;
	int compute_nnzRowCt_eigenRow();

	int _nnzCt;
	double _alpha;
	int computeAlpha();

	int _SDRowCnt, _CDRowCnt, _CDBoxCnt;
	index_t *_h_CDRowIdx, *_h_SDRowIdx;
	index_t *_d_CDRowIdx, *_d_SDRowIdx;
	int *_h_CDRowIdxOfs, *_d_CDRowIdxOfs;
	index_t *_h_rowAttr;
	index_t *_h_row2PoolIdx;
	int _CDNumBlocks, _CDNumThreads;
	
	int taskClassify();


	int *_h_poolLen, *_d_poolLen;	//real len
	index_t *_d_poolIdx;
	value_t *_d_poolVal;
	int *_h_poolOffset, *_d_poolOffset;
	int _poolSize;
	int _CDEleTotal;
	int computeCDGroup();

	index_t * _d_htIdxPool;
	value_t * _d_htValPool;
	int *_h_htOffset, *_d_htOffset;
	int * _h_htValidLen, *_d_htValidLen;
	int _htSize;
	int _SDEleTotal;
	int computeSDGroup();
	
	bool _isCached;
	int postProcess();
};

#endif

