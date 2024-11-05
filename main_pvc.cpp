
//////////////////////////////////////////////////////////////////////
// A simple example to show how CUDA WMMA API works with Tensor Cores
//    Created by Zong-Sheng Wang @ 2018/11/25
// Performance Tips:
//    To minimize bank conflicts, you should try to shift row or
// column of matrics in shared memory
// cmd:
//    $ nvcc -o main main.cu -arch sm_75

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 16

// MMA matrix tile dimensions.
#define M 8
#define N 16
#define K 16

// GEMM configuration.
#define M_TILES 2
#define N_TILES 1
#define K_TILES 1

#define M_TOTAL (M * M_TILES)
#define N_TOTAL (N * N_TILES)
#define K_TOTAL (K * K_TILES)


//__global__ void WMMAINT8()

void InitMatrix(sycl::half *A, sycl::half *B, float *C)
{
/*      for (int i = 0; i < M_TOTAL*K_TOTAL; i++)
        A[i] = sycl::vec<float, 1>(rand() % 1000 / 1000.0f)
                           .convert<sycl::half,
                                    sycl::rounding_mode::automatic>()[0];
        for (int i = 0; i < K_TOTAL*N_TOTAL; i++)
                B[i] = sycl::vec<float, 1>(rand() % 1000 / 1000.0f)
                           .convert<sycl::half,
                                    sycl::rounding_mode::automatic>()[0];
        for (int i = 0; i < M_TOTAL*N_TOTAL; i++)
                C[i] = rand() % 1000 / 1000.0f;*/
        for (int i = 0; i < M_TOTAL; i++) {
    for (int j = 0; j < K_TOTAL; j++) {
            if(i==j)
      A[i * K_TOTAL + j] =1;// (sycl::half)(rand() % 3);
        else
                A[i * K_TOTAL + j] =0;
    }
  }

  for (int i = 0; i < N_TOTAL; i++) {
    for (int j = 0; j < K_TOTAL; j++) {
      B[i * K_TOTAL + j] =5;// (sycl::half)(rand() % 3);
    }
  }

  for (int t = 0; t < M_TOTAL * N_TOTAL; t++) {
    C[t] = 2;//static_cast<float>(rand() % 3);
  }
}
template <typename T>
void matrix_transpose(unsigned int rows, unsigned int cols, T* src, T* dest)
{
    for (unsigned int i = 0; i < rows; i++) {
        for (unsigned int j = 0; j < cols; j++) {

            dest[j * rows + i] = src[i * cols + j];
        }
    }

}


void WMMAF16TensorCore(sycl::half *A, sycl::half *B, float *C, float *D,
                       const sycl::nd_item<3> &item_ct1)
{
        int ix = (item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                  item_ct1.get_local_id(2)) /
                 WARP_SIZE;
        int iy = (item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                  item_ct1.get_local_id(1));

        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::a, M, N, K, sycl::half,
            dpct::experimental::matrix::row_major>
            a_frag;
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::b, M, N, K, sycl::half,
            dpct::experimental::matrix::row_major>
            b_frag;
//        wmma::fragment<wmma::accumulator, M, N, K, float> ab_frag;
//      wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
      dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::accumulator, M, N, K, float>ab_frag;
        dpct::experimental::matrix::joint_matrix<
            dpct::experimental::matrix::accumulator, M, N, K, float>c_frag;
        sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(
            item_ct1.get_sub_group(), (ab_frag).get(), 0.0f);

        // AB = A*B
        int a_col, a_row, b_col, b_row, c_col, c_row;
        a_row = ix * M;
        b_row = iy * N;
        for (int k=0; k<K_TOTAL; k+=K) {
                a_col = b_col = k;

                if (a_row < M_TOTAL && a_col < K_TOTAL && b_row < K_TOTAL && b_col < N_TOTAL) {
                        // Load the inputs
                        sycl::ext::oneapi::experimental::matrix::
                            joint_matrix_load(
                                item_ct1.get_sub_group(), a_frag.get(),
                                sycl::address_space_cast<
                                    sycl::access::address_space::generic_space,
                                    sycl::access::decorated::no,
                                    const sycl::half>(A + a_col +
                                                      a_row * M_TOTAL),
                                M_TOTAL);
                        sycl::ext::oneapi::experimental::matrix::
                            joint_matrix_load(
                                item_ct1.get_sub_group(), b_frag.get(),
                                sycl::address_space_cast<
                                    sycl::access::address_space::generic_space,
                                    sycl::access::decorated::no,
                                    const sycl::half>(B + b_col +
                                                      b_col * K_TOTAL),
                                K_TOTAL);

                        // Perform the matrix multiplication
                        //wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
                        sycl::ext::oneapi::experimental::matrix::joint_matrix_mad(
          item_ct1.get_sub_group(), ab_frag.get(), a_frag.get(), b_frag.get(),
          ab_frag.get());
                }
        }

        // D = AB + C
        c_col = b_row;
        c_row = a_row;
        if (c_row < M_TOTAL && c_col < N_TOTAL) {
                sycl::ext::oneapi::experimental::matrix::joint_matrix_load(
                    item_ct1.get_sub_group(), (c_frag).get(),
                    sycl::address_space_cast<
                        sycl::access::address_space::generic_space,
                        sycl::access::decorated::no, float>(C + c_col +
                                                            c_row * N_TOTAL),
                    N_TOTAL,
                    sycl::ext::oneapi::experimental::matrix::layout::row_major);

                for (int i = 0; i < c_frag.num_elements; i++) {
                        c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
                }

                // Store the output
  /*              wmma::store_matrix_sync(
                    D + c_col + c_row * N_TOTAL, c_frag, N_TOTAL,
*/                  sycl::ext::oneapi::experimental::matrix::layout::row_major);
        sycl::ext::oneapi::experimental::matrix::joint_matrix_store(
        item_ct1.get_sub_group(), c_frag.get(),
        sycl::address_space_cast<sycl::access::address_space::generic_space,
                                 sycl::access::decorated::no, float>(D + c_col + c_row * N_TOTAL),
        N_TOTAL, sycl::ext::oneapi::experimental::matrix::layout::row_major);
        }
}

dpct::err0 CalcWMMA(sycl::half *A, sycl::half *B, float *C, float *D) try {
        dpct::err0 cuda_status;
        dpct::dim3 gridDim, blockDim;
        // 16 warps in one block
        blockDim.x = 4 * WARP_SIZE;
        blockDim.y = 4;

        gridDim.x = (M_TOTAL + (M * blockDim.x / WARP_SIZE - 1)) / (M * blockDim.x / WARP_SIZE);
        gridDim.y = (N_TOTAL + N * blockDim.y - 1) / (N * blockDim.y);

        // for Performance Metrics
        dpct::event_ptr start, stop;
        start = new sycl::event();
        stop = new sycl::event();
        dpct::sync_barrier(start);

        /*
        DPCT1049:0: The work-group size passed to the SYCL kernel may
         * exceed the limit. To get the device limit, query
         * info::device::max_work_group_size. Adjust the work-group size if
         * needed.
        */
        {
                dpct::get_device(dpct::get_device_id(
                                     dpct::get_in_order_queue().get_device()))
                    .has_capability_or_fail({sycl::aspect::fp16});

                dpct::get_in_order_queue().parallel_for(
                    sycl::nd_range<3>(gridDim * blockDim, blockDim),
                    [=](sycl::nd_item<3> item_ct1) {
                            WMMAF16TensorCore(A, B, C, D, item_ct1);
                    });
        }
        cuda_status = DPCT_CHECK_ERROR(
            dpct::get_current_device().queues_wait_and_throw());

        dpct::sync_barrier(stop);
        stop->wait_and_throw();

        float milliseconds = 0;
        milliseconds = (stop->get_profiling_info<
                            sycl::info::event_profiling::command_end>() -
                        start->get_profiling_info<
                            sycl::info::event_profiling::command_start>()) /
                       1000000.0f;

        // for Performance Metrics
        printf("[+] GPU(with Tensor Cores) Elapsed Time: %f ms\n", milliseconds);
        // references from https://devblogs.nvidia.com/how-implement-performance-metrics-cuda-cc/
        printf("[+] TFLOPS: %.2f\n", ((double)M_TOTAL * N_TOTAL* K_TOTAL * 2) / milliseconds / 1e9);
        dpct::destroy_event(start);
        dpct::destroy_event(stop);

        return cuda_status;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main() try {
        dpct::err0 cuda_status;
        /*
        DPCT1093:1: The "0" device may be not the one intended for use.
         * Adjust the selected device if needed.
        */
        cuda_status = DPCT_CHECK_ERROR(dpct::select_device(0));

        // Matrix on device
        sycl::half *A;
        sycl::half *B;
        sycl::half *B_t;
        float *C;
        float *D;

        // CUDA Unified Memory
        A = (sycl::half *)sycl::malloc_shared(
            sizeof(sycl::half) * M_TOTAL * K_TOTAL, dpct::get_in_order_queue());
        B = (sycl::half *)sycl::malloc_shared(
            sizeof(sycl::half) * K_TOTAL * N_TOTAL, dpct::get_in_order_queue());
        B_t = (sycl::half *)sycl::malloc_shared(
            sizeof(sycl::half) * K_TOTAL * N_TOTAL, dpct::get_in_order_queue());
        C = (float *)sycl::malloc_shared(sizeof(float) * M_TOTAL * N_TOTAL,
                                         dpct::get_in_order_queue());
        D = (float *)sycl::malloc_shared(sizeof(float) * M_TOTAL * N_TOTAL,
                                         dpct::get_in_order_queue());

        // Init matrix A B C on host
        //InitHostMatrix(host_A, host_B, host_C);
        printf("[*] Initializing Matrix...\n");
        InitMatrix(A, B, C);
        matrix_transpose<sycl::half>(K_TOTAL, N_TOTAL, B, B_t);
        printf("[+]   A: %d x %d\n", M_TOTAL, K_TOTAL);
        printf("[+]   B: %d x %d\n", K_TOTAL, N_TOTAL);
        printf("[+]   C: %d x %d\n", M_TOTAL, N_TOTAL);

        // computing gemm using tensor core
        printf("[*] Computing D = A * B +C with Tensor Cores...\n");
        // D = A * B +C, D holds the result after ret
        cuda_status = CalcWMMA(A, B_t, C, D);
        for (int i = 0; i < N_TOTAL * M_TOTAL; i++)
        {
              printf("  D[%d]=%f \n", i, D[i]);
        }
        cuda_status = DPCT_CHECK_ERROR(dpct::get_current_device().reset());

        // Todo: Add a function to verify the result by using the result of CPU version implementation.

        dpct::dpct_free(A, dpct::get_in_order_queue());
        dpct::dpct_free(B, dpct::get_in_order_queue());
        dpct::dpct_free(C, dpct::get_in_order_queue());
        dpct::dpct_free(D, dpct::get_in_order_queue());

        return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
