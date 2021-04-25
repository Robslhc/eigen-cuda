#include <test.hpp>
#include <kernel.hpp>
#include <Eigen/Core>

namespace Test
{
    double testfunction()
    {
        std::vector<Eigen::Vector3d> v1(10, Eigen::Vector3d{ 1.0, 1.0, 1.0 });
        std::vector<Eigen::Vector3d> v2(10, Eigen::Vector3d{ -1.0, 1.0, 1.0 });

        double x = Kernel::dot(v1,v2);

        return x;
    }

    void testEigenValueSolver()
    {
        Eigen::Matrix3f a = Eigen::Matrix3f::Random();

        std::vector<Eigen::Matrix3f> m(10, a + a.transpose());

        Kernel::run_eigen_solver(m);
    } 
}