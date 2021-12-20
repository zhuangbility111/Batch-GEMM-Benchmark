echo "Testing Batch GEMM: BERT dimensions"

# export PATH=/home/apps/oss/PyTorch-1.7.0/bin:$PATH
# export LD_LIBRARY_PATH=/home/apps/oss/PyTorch-1.7.0/lib:$LD_LIBRARY_PATH
. /vol0004/ra000012/a04453/pytorch/scripts/fujitsu/env.src
. ${PREFIX}/${VENV_NAME}/bin/activate

batch_array=(48 48   48   48  96  96   48   48   48   48   48   48)
m_array=(4096 1024 2048 3072 512 512 4096 4096 4096 4096 4096 4096)
n_array=(1024 1024 1024 1024 512  64 4096 1024 1024 1024 2048 3072)
k_array=(4096 4096 4096 4096 64  512 1024 1024 2048 3072 1024 1024)
epoch=100

echo "1 CMG......"
for((i=0;i<=11;i++));  
do   
    echo "--------------------------------------------------------------"
#    LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=12 numactl -N 4 -m 4 python3 -m benchmarker --framework=torch --problem=batchmatmul \
#                        --problem_size=${m_array[i]},${n_array[i]},${k_array[i]} --batch_size=${batch_array[i]} --nb_epoch=100
	LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=12 numactl -C 12-23 -m 4 python3 test_bmm.py ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]}
    echo "--------------------------------------------------------------"
done

echo "4 CMGs......"
for((i=0;i<=11;i++));  
do   
    echo "--------------------------------------------------------------"
#    LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=48 numactl -N 4-7 -m 4-7 python3 -m benchmarker --framework=torch --problem=batchmatmul \
#                        --problem_size=${m_array[i]},${n_array[i]},${k_array[i]} --batch_size=${batch_array[i]} --nb_epoch=100
	LD_PRELOAD=libtcmalloc.so OMP_NUM_THREADS=48 numactl -N 4-7 -m 4-7 python3 test_bmm.py ${batch_array[i]} ${m_array[i]} ${n_array[i]} ${k_array[i]}
    echo "--------------------------------------------------------------"
done
