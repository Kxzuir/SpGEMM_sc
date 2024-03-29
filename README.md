# SpGEMM_sc

This is the source code for SpGEMM computation in CUDA implementation.

You can either use Visual Studio 2017 under Windows, or use Makefile under Linux to build the project.

Before building, you need to put [cusplibrary](https://github.com/cusplibrary/cusplibrary) into folder `/cusp `. For Linux users, change `CUDA_INSTALL_PATH ` in `Makefile` is also required.

The program has been tested on NVIDIA GeForce GTX 1060 6G with CUDA SDK v9.2, CUSP v5.1 and major operating systems (Ubuntu 16.04.3 and Windows 10 Pro, 1803).

Usage example: `./spgemm_sc cant.max 1.0 ` 

**Note**: The 32-bit floating-point version is only supported by devices of compute capability **2.x** and higher. The 64-bit floating-point version is only supported by devices of compute capability **6.x** and higher. If your device won't meet the requirement, change the definition of `value_t `in `common.h`. 