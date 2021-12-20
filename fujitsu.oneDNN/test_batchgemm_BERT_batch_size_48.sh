echo "Testing Batch GEMM: BERT dimensions"

batch_array=(48 48   48   48  96  96   48   48   48   48   48   48)
m_array=(4096 1024 2048 3072 512 512 4096 4096 4096 4096 4096 4096)
n_array=(1024 1024 1024 1024 512  64 4096 1024 1024 1024 2048 3072)
k_array=(4096 4096 4096 4096 64  512 1024 1024 2048 3072 1024 1024)
epoch=100

echo "1 CMG......"
for((i=0;i<=11;i++));  
do   
	echo "--------------------------------------------------------------"
    OMP_NUM_THREADS=12 numactl -C 12-23 -m 4 ./primitives-matmul-cpp FP32 ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]} ${epoch}
	echo "--------------------------------------------------------------"
done 

echo "4 CMGs......"
for((i=0;i<=11;i++));  
do   
	echo "--------------------------------------------------------------"
    OMP_NUM_THREADS=48 numactl -C 12-59 -m 4-7 ./primitives-matmul-cpp FP32 ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]} ${epoch}
	echo "--------------------------------------------------------------"
done 
