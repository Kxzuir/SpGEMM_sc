#pragma once
#ifndef SCSPARSE_DEVICE_H
#define SCSPARSE_DEVICE_H
#include "common.h"

__global__ void
compute_row_eigen(const int n,
	const int *d_csrRowPtrA, const int *d_csrColIndA,
	const int *d_csrRowPtrB,
	int *d_nnzRowCt,
	double *d_EigenRow)
{
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int start, stop, index, strideB, row_size_Ct = 0;

	if (global_id < n)
	{
		start = d_csrRowPtrA[global_id];
		stop = d_csrRowPtrA[global_id + 1];

		for (int i = start; i < stop; i++)
		{
			index = d_csrColIndA[i];
			strideB = d_csrRowPtrB[index + 1] - d_csrRowPtrB[index];
			row_size_Ct += strideB;
		}

		d_nnzRowCt[global_id] = row_size_Ct;
		if (row_size_Ct == 0) d_EigenRow[global_id] = 0.0;
		else d_EigenRow[global_id] = (double)(stop - start) / (double)row_size_Ct;
	}
}


__device__
void swap(int &a, int &b)
{
	int t = a;
	a = b;
	b = t;
}

__device__
void swap(double &a, double &b)
{
	double t = a;
	a = b;
	b = t;
}

__device__ int
sd_hash(int idx, int idxMax)
{
	return idx % idxMax;
}

__device__ unsigned int
upper2bound_cuda(unsigned int v)
{
	v--;
	v |= v >> 1;
	v |= v >> 2;
	v |= v >> 4;
	v |= v >> 8;
	v |= v >> 16;
	v++;
	return v;
}


__device__ void
heapAdjust(index_t *key, value_t *val, int length, int k)
{
	index_t tmp_key = key[k];
	value_t tmp_val = val[k];
	int i = 2 * k + 1;
	while (i < length)
	{
		if (i + 1 < length && key[i] < key[i + 1]) i++;
		if (tmp_key > key[i]) break;
		key[k] = key[i];
		val[k] = val[i];
		k = i;
		i = 2 * k + 1;
	}
	key[k] = tmp_key;
	val[k] = tmp_val;
}

__device__ void
heapSort(index_t *key, value_t *val, int length)
{
	if (key == NULL || length <= 0) return;
	for (int i = length / 2 - 1; i >= 0; --i)
		heapAdjust(key, val, length, i);

	for (int i = length - 1; i >= 0; --i)
	{
		swap(key[0], key[i]);
		swap(val[0], val[i]);
		heapAdjust(key, val, i, 0);
	}
	return;
}

__global__ void
compute_cd_group(
	index_t *d_csrRowPtrA, index_t *d_csrColIdxA, value_t *d_csrValA,
	index_t *d_csrRowPtrB, index_t *d_csrColIdxB, value_t *d_csrValB,
	index_t *d_CDRowIdx, int *d_CDRowIdxOfs,
	index_t *d_poolIdx, value_t *d_poolVal, int *d_poolOffset,
	int *d_poolRowLen)
{
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int wkLstStart = d_CDRowIdxOfs[global_id], wkLstEnd = d_CDRowIdxOfs[global_id + 1];
	for (int CDIdx = wkLstStart; CDIdx < wkLstEnd; CDIdx++)
	{
		int offset = d_poolOffset[CDIdx];
		index_t *poolIdxLoc = d_poolIdx + offset;
		value_t *poolValLoc = d_poolVal + offset;
		int r_cnt = 0;
		index_t curRowIdx = d_CDRowIdx[CDIdx];
		index_t start = d_csrRowPtrA[curRowIdx], end = d_csrRowPtrA[curRowIdx + 1];
		for (index_t i = start; i < end; i++)
		{
			index_t j = d_csrColIdxA[i];
			value_t v = d_csrValA[i];
			index_t bStart = d_csrRowPtrB[j], bEnd = d_csrRowPtrB[j + 1];
			for (index_t k = bStart; k < bEnd; k++)
			{
				index_t loc = d_csrColIdxB[k];
				value_t u = d_csrValB[k];
				poolValLoc[r_cnt] = u * v;
				poolIdxLoc[r_cnt] = loc;
				r_cnt++;
			}
		}
		d_poolRowLen[CDIdx] = r_cnt;
	}
}

template <int sharedMemSize = 4096>
__global__ void
bitonicSort_by_key_block(int CDRowCnt,
	index_t *d_poolIdx, value_t *d_poolVal, int *d_poolOffset, int *d_poolRowLen)
{
	__shared__ index_t keyBuf[sharedMemSize];
	__shared__ value_t valBuf[sharedMemSize];
	int num_blocks = gridDim.x;
	int num_threads = blockDim.x;
	int thread_id = threadIdx.x;
	for (int curCDItemId = blockIdx.x; curCDItemId < CDRowCnt; curCDItemId += num_blocks)
	{
		int validLen = d_poolRowLen[curCDItemId];
//		if (validLen == 0) continue;
		int offset = d_poolOffset[curCDItemId];
		index_t *key = d_poolIdx + offset;
		value_t *val = d_poolVal + offset;
		
		
		int len = upper2bound_cuda(validLen);
		index_t *hashIdx = key;
		value_t *hashVal = val;
		if (len <= sharedMemSize)
		{
			for (int i = thread_id; i < validLen; i += num_threads)
			{
				keyBuf[i] = key[i];
				valBuf[i] = val[i];
			}
			for (int i = validLen + thread_id; i < len; i += num_threads)
			{
				keyBuf[i] = INT_MAX;
			}
			hashIdx = keyBuf;
			hashVal = valBuf;
		}
		__syncthreads();
		int num_iter = len / num_threads;
		if (num_iter == 0) num_iter = 1;

		for (int i = 2; i <= len; i <<= 1)
		{
			for (int j = i >> 1; j > 0; j >>= 1)
			{
				for (int iter = 0; iter < num_iter; iter++)
				{
					int tid = num_threads * iter + thread_id;
					if (tid > len) break;
					int tid_comp = tid ^ j;
					if (tid_comp > tid)
					{
						if ((tid & i) == 0)
						{
							if (hashIdx[tid] > hashIdx[tid_comp])
							{
								swap(hashIdx[tid], hashIdx[tid_comp]);
								swap(hashVal[tid], hashVal[tid_comp]);
							}
						}
						else
						{
							if (hashIdx[tid] < hashIdx[tid_comp])
							{
								swap(hashIdx[tid], hashIdx[tid_comp]);
								swap(hashVal[tid], hashVal[tid_comp]);
							}
						}
					}
				}
				__syncthreads();
			}
		}
		__syncthreads();
		if (len <= sharedMemSize)
		{
			for (int i = thread_id; i < validLen; i += num_threads)
			{
				key[i] = keyBuf[i];
				val[i] = valBuf[i];
			}
		}
		__syncthreads();
	}
}

__global__ void
compute_cd_group_merge(index_t *d_csrRowPtrA, index_t *d_csrColIdxA, value_t *d_csrValA,
	index_t *d_csrRowPtrB, index_t *d_csrColIdxB, value_t *d_csrValB,
	index_t *d_CDRowIdx, int *d_CDRowIdxOfs, 
	index_t *d_poolIdx, value_t *d_poolVal, int *d_poolOffset,
	int *d_poolRowLen)
{
	int global_id = blockIdx.x * blockDim.x + threadIdx.x;
	int wkLstStart = d_CDRowIdxOfs[global_id], wkLstEnd = d_CDRowIdxOfs[global_id + 1];
	for (int CDIdx = wkLstStart; CDIdx < wkLstEnd; CDIdx++)
	{
		int curPoolLen = d_poolRowLen[CDIdx];
//		if (curPoolLen == 0) continue;
		int offset = d_poolOffset[CDIdx];
		index_t *poolIdxLoc = d_poolIdx + offset;
		value_t *poolValLoc = d_poolVal + offset;
		
		//heapSort(poolIdxLoc, poolValLoc, curPoolLen);
		int wPos = 0;
		index_t wIdx = poolIdxLoc[0];
		value_t wVal = poolValLoc[0];
		index_t pos = 1;
		while (pos < curPoolLen)
		{
			if (poolIdxLoc[pos] == wIdx) wVal += poolValLoc[pos];
			else
			{
				poolIdxLoc[wPos] = wIdx;
				poolValLoc[wPos] = wVal;
				wPos++;
				wIdx = poolIdxLoc[pos];
				wVal = poolValLoc[pos];
			}
			pos++;
		}
		poolIdxLoc[wPos] = wIdx;
		poolValLoc[wPos] = wVal;
		d_poolRowLen[CDIdx] = wPos + 1;
	}
}




__device__  int
sd_hashtable_insert(
	index_t *hashIdx, value_t *hashVal, int htSize,
	index_t idx, value_t val)
{
	int tarLoc = sd_hash(idx, htSize);
	bool foundLoc = false;
	for (int trial = 0, alterLoc = tarLoc; trial < htSize; trial++)
	{
		index_t oldIdx = atomicCAS(&hashIdx[alterLoc], INVALIDHASHIDX, idx);
		if (oldIdx == INVALIDHASHIDX) {
			foundLoc = true;
			atomicAdd(&hashVal[alterLoc], val);
			break;
		}
		else if (oldIdx == idx)
		{
			foundLoc = true;
			atomicAdd(&hashVal[alterLoc], val);
			break;
		}
		alterLoc = (alterLoc + 1) % htSize;
	}
	if (!foundLoc) return SD_HT_INS_ERR_HTFULL;
	return 0;
}

//hashMax = 4096
template <int htAllocSize = 1024>
__global__ void
compute_sd_group(
	index_t * d_csrRowPtrA, index_t * d_csrColIdxA, value_t * d_csrValA,
	index_t * d_csrRowPtrB, index_t * d_csrColIdxB, value_t * d_csrValB,
	index_t *d_SDRowIdx, int SDRowCnt,
	index_t * d_htIdx, value_t * d_htVal, int * d_htOffset, int *d_htLen)
{
	int num_blocks = gridDim.x;
	int num_threads = blockDim.x;
	int thread_id = threadIdx.x;
	for (int curSDItemId = blockIdx.x; curSDItemId < SDRowCnt; curSDItemId += num_blocks)
	{
		int curRowIdx = d_SDRowIdx[curSDItemId];
		int curRowEleCnt = d_csrRowPtrA[curRowIdx + 1] - d_csrRowPtrA[curRowIdx];
		index_t *curRowColPtr = d_csrColIdxA + d_csrRowPtrA[curRowIdx];
		value_t *curRowValPtr = d_csrValA + d_csrRowPtrA[curRowIdx];
		index_t idxUpBound = INVALIDHASHIDX;
		value_t valInit = 0;
		__shared__ index_t hashIdx[htAllocSize];
		__shared__ value_t hashVal[htAllocSize];
		for (int i = thread_id; i < htAllocSize; i += num_threads) hashIdx[i] = idxUpBound;
		for (int i = thread_id; i < htAllocSize; i += num_threads) hashVal[i] = valInit;
		__syncthreads();
		for (int jNo = thread_id; jNo < curRowEleCnt; jNo += num_threads)
		{
			index_t j = curRowColPtr[jNo];
			value_t v = curRowValPtr[jNo];
			index_t bStart = d_csrRowPtrB[j], bEnd = d_csrRowPtrB[j + 1];
			for (index_t k = bStart; k < bEnd; k++)
			{
				index_t loc = d_csrColIdxB[k];
				value_t u = d_csrValB[k];
				int err = sd_hashtable_insert(hashIdx, hashVal, htAllocSize, loc, u*v);
				//Todo: Full hash table
				//if (err == SD_HT_INS_ERR_HTFULL)
			}
		}
		__syncthreads();
		//Getvalid len
		//Todo : optimize speed
		if (thread_id == 0)
		{
			int wPos = 0;
			for (int i = 0; i < htAllocSize; i++)
			{
				if (hashIdx[i] != idxUpBound)
				{
					hashIdx[wPos] = hashIdx[i];
					hashVal[wPos] = hashVal[i];
					wPos++;
				}
			}
			d_htLen[curSDItemId] = wPos;
			int bitLim = upper2bound_cuda(wPos);
			for (int i = wPos; i < bitLim; i++)
			{
				hashIdx[i] = INVALIDHASHIDX;
				hashVal[i] = valInit;
			}
		}
		__syncthreads();
		//Sort within block
		int validLen = d_htLen[curSDItemId];
		int len = upper2bound_cuda(validLen);
		int num_iter = len / num_threads;
		if (num_iter == 0) num_iter = 1;

		for (int i = 2; i <= len; i <<= 1)
		{
			for (int j = i >> 1; j > 0; j >>= 1)
			{
				for (int iter = 0; iter < num_iter; iter++)
				{
					int tid = num_threads * iter + thread_id;
					if (tid > len) break;
					int tid_comp = tid ^ j;
					if (tid_comp > tid)
					{
						if ((tid & i) == 0)
						{
							if (hashIdx[tid] > hashIdx[tid_comp])
							{
								swap(hashIdx[tid], hashIdx[tid_comp]);
								swap(hashVal[tid], hashVal[tid_comp]);
							}
						}
						else
						{
							if (hashIdx[tid] < hashIdx[tid_comp])
							{
								swap(hashIdx[tid], hashIdx[tid_comp]);
								swap(hashVal[tid], hashVal[tid_comp]);
							}
						}
					}
				}
				__syncthreads();
			}
		}
		__syncthreads();
		//copy back to global mem
		int offset = d_htOffset[curSDItemId];
		index_t *d_htIdxLoc = d_htIdx + offset;
		value_t *d_htValLoc = d_htVal + offset;
		for (int i = thread_id; i < validLen; i += num_threads) d_htIdxLoc[i] = hashIdx[i];
		for (int i = thread_id; i < validLen; i += num_threads) d_htValLoc[i] = hashVal[i];
	}
}

#endif 