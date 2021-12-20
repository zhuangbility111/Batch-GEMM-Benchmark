echo "Testing Batch GEMM: BERT dimensions"

batch_array=(1   1    1    1 128 128    1    1    1    1    1    1)
m_array=(4096 1024 2048 3072 512 512 4096 4096 4096 4096 4096 4096)
n_array=(1024 1024 1024 1024 512  64 4096 1024 1024 1024 2048 3072)
k_array=(4096 4096 4096 4096 64  512 1024 1024 2048 3072 1024 1024)
epoch=100

echo "single-core_gemm version begin....."
echo "******************************************************************"
echo "1 CMG......"
for((i=0;i<=11;i++));  
do   
	echo "--------------------------------------------------------------"
    OMP_NUM_THREADS=12 numactl -N 4 -m 4 ./test_batchgemm_single-core_gemm FP32 ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]} ${epoch}
	echo "--------------------------------------------------------------"
done 

echo "4 CMGs......"
for((i=0;i<=11;i++));  
do   
	echo "--------------------------------------------------------------"
    OMP_NUM_THREADS=48 numactl -N 4-7 -m 4-7 ./test_batchgemm_single-core_gemm FP32 ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]} ${epoch}
	echo "--------------------------------------------------------------"
done 

echo "multi-cores_gemm version begin....."
echo "******************************************************************"
echo "1 CMG......"
for((i=0;i<=11;i++));  
do   
	echo "--------------------------------------------------------------"
    OMP_NUM_THREADS=12 numactl -N 4 -m 4 ./test_batchgemm_multi-cores_gemm FP32 ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]} ${epoch}
	echo "--------------------------------------------------------------"
done 

echo "4 CMGs......"
for((i=0;i<=11;i++));  
do   
	echo "--------------------------------------------------------------"
    OMP_NUM_THREADS=48 numactl -N 4-7 -m 4-7 ./test_batchgemm_multi-cores_gemm FP32 ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]} ${epoch}
	echo "--------------------------------------------------------------"
done 


