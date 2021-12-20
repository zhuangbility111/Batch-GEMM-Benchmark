#include <iostream>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include "cssl.h" /* standard C-SSL header file */
#include <cblas.h>
#include "./args.hpp"

using namespace std::chrono; 


int main(int argc, char * argv[]) {
    size_t m, n, k;
    size_t batch_size = 100;  // todo(bai): a parameter should go here instead a constant
    float **A, **B, **C;
    double dtime;
    Options options = parse_args(argc, argv);
    batch_size = options.batch_size;
    m = options.cnt_rows_A_rows_C;
    n = options.cnt_cols_A_rows_B;
    k = options.cnt_cols_B_cols_C;
    get_batched_matrices<float>(m, k, n, A, B, C, batch_size);
    const float alpha = 1;
    const float beta = 0;
    int64_t M = (int64_t)m;
    int64_t N = (int64_t)n;
    int64_t K = (int64_t)k;
    const int64_t lda=(int64_t)k; // k for row major;
    const int64_t ldb=(int64_t)n; //n; 
    const int64_t ldc=(int64_t)n; //n;

    // pre-heat
    if (batch_size == 1) {
        for(size_t j=0;j<batch_size;j++){
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        M,
                        N,
                        K,
                        alpha,
                        A[j],
                        K,
                        B[j],
                        N, 
                        beta,
                        C[j],
                        N);
        }
    } else {
        #pragma omp parallel for
        for(size_t j=0;j<batch_size;j++){
            cblas_sgemm(CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        M,
                        N,
                        K,
                        alpha,
                        A[j],
                        K,
                        B[j],
                        N, 
                        beta,
                        C[j],
                        N);
            // dnnl_sgemm('N',
            //            'N',
            //            M,
            //            N,
            //            K,
            //            alpha,
            //            A[j], lda,
            //            B[j], ldb,
            //            beta,
            //            C[j], ldc);
        }
    
    }

    auto start = high_resolution_clock::now(); 
    for (size_t i=0; i<options.nb_epoch; i++)
    {
        if (options.precision == "FP32")
        {
			if (batch_size == 1) {
                for(size_t j=0;j<batch_size;j++){
                    cblas_sgemm(CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                M,
                                N,
                                K,
                                alpha,
                                A[j],
                                K,
                                B[j],
                                N, 
                                beta,
                                C[j],
                                N);
                }
			} else {
                #pragma omp parallel for
                for(size_t j=0;j<batch_size;j++){
                    cblas_sgemm(CblasRowMajor,
                                CblasNoTrans,
                                CblasNoTrans,
                                M,
                                N,
                                K,
                                alpha,
                                A[j],
                                K,
                                B[j],
                                N, 
                                beta,
                                C[j],
                                N);
                    // dnnl_sgemm('N',
                    //            'N',
                    //            M,
                    //            N,
                    //            K,
                    //            alpha,
                    //            A[j], lda,
                    //            B[j], ldb,
                    //            beta,
                    //            C[j], ldc);
			    }
            
            }
        }
        else
        {
            // TODO  (Alex): implement FP16
            // ugly throw here to make sure benchmarker chrashes alright
            fprintf(stderr, "not implemented yet");
            throw "madamada";
        }
    }
    auto stop = high_resolution_clock::now();
    std::chrono::duration<double> seconds = (stop - start); 
    std::cerr << "MNK " << m << " " << n << " " << k << std::endl;
    std::cerr << "Batch size " << batch_size << std::endl;
    dtime = seconds.count();
    double gflop = (2.0 * m * n * k * batch_size) / (1000 * 1000 * 1000);
    gflop *= static_cast<double>(options.nb_epoch);
    double gflops = gflop / dtime;
    printf("%f\n", dtime);
    fprintf(stderr, "gflops: \t%f\n", gflop);
    fprintf(stderr, "time: \t%f\n", dtime);
    fprintf(stderr, "ips: \t%f\n", 1 / dtime);
    fprintf(stderr, "gflops/s: \t%f\n", gflops);
    return 0;
}
