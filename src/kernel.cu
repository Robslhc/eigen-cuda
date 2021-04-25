#ifdef USE_CUDA

#include <kernel.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <stdio.h>
#include <assert.h>


static void HandleError( cudaError_t err, const char *file, int line )
{
	// CUDA error handeling from the "CUDA by example" book
	if (err != cudaSuccess)
    {
		printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file, line );
		exit( EXIT_FAILURE );
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

// CUDA Version
namespace Kernel
{
    __global__ void cu_dot(Eigen::Vector3d *v1, Eigen::Vector3d *v2, double *out, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
        {
            out[idx] = v1[idx].dot(v2[idx]);
        }
        return;
    }

    // The wrapper for the calling of the actual kernel
    double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2)
    {        
        int n = v1.size();
        double *ret = new double[n];

        Eigen::Vector3d *host_v2 = new Eigen::Vector3d[n];

        // Allocate device arrays
        Eigen::Vector3d *dev_v1, *dev_v2;
        HANDLE_ERROR(cudaMalloc((void **)&dev_v1, sizeof(Eigen::Vector3d)*n));
        HANDLE_ERROR(cudaMalloc((void **)&dev_v2, sizeof(Eigen::Vector3d)*n));
        double* dev_ret;
        HANDLE_ERROR(cudaMalloc((void **)&dev_ret, sizeof(double)*n));

        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_v1, v1.data(), sizeof(Eigen::Vector3d)*n, cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(dev_v2, v2.data(), sizeof(Eigen::Vector3d)*n, cudaMemcpyHostToDevice));

        // Dot product
        cu_dot<<<(n+1023)/1024, 1024>>>(dev_v1, dev_v2, dev_ret, n);
        
        // Copy to host
        HANDLE_ERROR(cudaMemcpy(host_v2, dev_v2, sizeof(Eigen::Vector3d)*n, cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(ret, dev_ret, sizeof(double)*n, cudaMemcpyDeviceToHost));

        // Reduction of the array
        for (int i=1; i<n; ++i)
        {
            ret[0] += ret[i];
        }

        // Return
        return ret[0];
    }

    __global__ void kernel_eigensolve(Eigen::Matrix3f *m, Eigen::Vector3f *eigen_values, Eigen::Matrix3f *eigen_vectors, size_t N)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx < N)
        {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(m[idx]);
            eigen_values[idx] = es.eigenvalues();
            eigen_vectors[idx] = es.eigenvectors();
        }
    }

    void run_eigen_solver(const std::vector<Eigen::Matrix3f> &m)
    {
        int n = m.size();

        // Host arrays
        Eigen::Vector3f *eigen_values = new Eigen::Vector3f[n];
        Eigen::Matrix3f *eigen_vectors = new Eigen::Matrix3f[n];

        // Device arrays
        Eigen::Matrix3f *dev_m;
        Eigen::Vector3f *dev_eigen_values;
        Eigen::Matrix3f *dev_eigen_vectors;

        HANDLE_ERROR(cudaMalloc((void **)&dev_m, n*sizeof(Eigen::Matrix3f)));
        HANDLE_ERROR(cudaMalloc((void **)&dev_eigen_values, n*sizeof(Eigen::Vector3f)));
        HANDLE_ERROR(cudaMalloc((void **)&dev_eigen_vectors, n*sizeof(Eigen::Matrix3f)));

        // Copy to device
        HANDLE_ERROR(cudaMemcpy(dev_m, m.data(), n*sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));

        // EigenSolver
        kernel_eigensolve<<<(n+63/64), 64>>>(dev_m, dev_eigen_values, dev_eigen_vectors, n);

        // Copy to host
        HANDLE_ERROR(cudaMemcpy(eigen_values, dev_eigen_values, n*sizeof(Eigen::Vector3f), cudaMemcpyDeviceToHost));
        HANDLE_ERROR(cudaMemcpy(eigen_vectors, dev_eigen_vectors, n*sizeof(Eigen::Matrix3f), cudaMemcpyDeviceToHost));

        // show the result
        for(int i = 0; i < n; ++i)
        {
            std::cout << "Matrix " << i << ":" << std::endl << m[i] << std::endl;
            std::cout << "The eigenvalues :" << std::endl << eigen_values[i] << std::endl;
            std::cout << "The eigenvectors :" << std::endl << eigen_vectors[i] << std::endl;
            std::cout << "==================================================================" << std::endl;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es_cpu(m[i]);
            assert(eigen_values[i].isApprox(es_cpu.eigenvalues()));
            assert(eigen_vectors[i].isApprox(es_cpu.eigenvectors()));
        }
    }
}

#endif