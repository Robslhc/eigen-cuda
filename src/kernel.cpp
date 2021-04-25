#ifndef USE_CUDA

#include <iostream>

#include <kernel.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

// C++ Version
namespace Kernel
{
    double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2)
    {
        double x=0;
        for (int i=0; i<v1.size(); ++i)
        {
            x += v1[i].dot(v2[i]);
        }
        return x;
    }

    void run_eigen_solver(const std::vector<Eigen::Matrix3f> &m)
    {
        for (int i = 0; i < m.size(); ++i){
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(m[i]);
            std::cout << "Matrix " << i << ":" << std::endl << m[i] << std::endl;
            std::cout << "The eigenvalues :" << std::endl << es.eigenvalues() << std::endl;
            std::cout << "The eigenvectors :" << std::endl << es.eigenvectors() << std::endl;
            std::cout << "==================================================================" << std::endl;
        }
    }
}

#endif