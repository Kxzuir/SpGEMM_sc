# SpGEMM_sc

This is the source code for SpGEMM computation in CUDA implementation.

You can either use Visual Studio 2017 under Windows, or use Makefile under Linux to build the project.

Before building, you need to put [cusplibrary](https://github.com/cusplibrary/cusplibrary) into folder `/cusp `. For Linux users, change `CUDA_INSTALL_PATH ` in `Makefile` is also required.

The program has been tested on nVIDIA GeForce GTX 1060 6G with CUDA SDK v9.1, CUSP v5.1 and major operating systems (Ubuntu 16.04.3 and Windows 10 Pro, 1803).

Usage example: `./spgemm_sc cant.max`