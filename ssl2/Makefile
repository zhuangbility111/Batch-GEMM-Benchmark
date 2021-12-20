all:
#	frt -Kfast,openmp -SSL2BLAMP -KSVE main.f 
#	frt -KSVE -Kopenmp main_multithread.f -SSL2
	FCC -std=c++11 -Kfast,openmp -SSL2BLAMP test_batchgemm_single-core_gemm.cpp -o test_batchgemm_single-core_gemm
	FCC -std=c++11 -Kfast,openmp -SSL2BLAMP test_batchgemm_multi-cores_gemm.cpp -o test_batchgemm_multi-cores_gemm
