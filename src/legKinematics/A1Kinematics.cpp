//
// Created by shuoy on 8/10/21.
//

#include "A1Kinematics.h"
// #include <iostream>
// #include <chrono>

A1Kinematics::A1Kinematics()
{
    fk_gen = casadi::external("fk", "/home/EstimationUser/estimation_ws/src/Cerberus/src/codegen/shared/fk.so");
    J_gen = casadi::external("J", "/home/EstimationUser/estimation_ws/src/Cerberus/src/codegen/shared/J.so");
    dfk_drho_gen = casadi::external("dfk_drho", "/home/EstimationUser/estimation_ws/src/Cerberus/src/codegen/shared/dfk_drho.so");
    dJ_dq_gen = casadi::external("dJ_dq", "/home/EstimationUser/estimation_ws/src/Cerberus/src/codegen/shared/dJ_dq.so");
    dJ_drho_gen = casadi::external("dJ_drho", "/home/EstimationUser/estimation_ws/src/Cerberus/src/codegen/shared/dJ_drho.so");
}

std::vector<casadi::DM> A1Kinematics::cookArgs(const Vector_dof &q, const Vector_rho &rho)
{
    std::vector<double> q_vec;
    q_vec.resize(q.size());
    Eigen::Matrix<double, NUM_OF_DOF, 1>::Map(&q_vec[0], q.size()) = q;

    std::vector<double> rho_vec;
    rho_vec.resize(rho.size());
    Eigen::Matrix<double, NUM_OF_LEG, 1>::Map(&rho_vec[0], rho.size()) = rho;

    std::vector<casadi::DM> arg = {casadi::DM(q_vec), casadi::DM(rho_vec)};

    return arg;
}

void A1Kinematics::cookArgs(const Vector_dof &q, const Vector_rho &rho, std::vector<casadi::DM>& args)
{
    std::vector<double> q_vec;
    q_vec.resize(q.size());
    Eigen::Matrix<double, NUM_OF_DOF, 1>::Map(&q_vec[0], q.size()) = q;

    std::vector<double> rho_vec;
    rho_vec.resize(rho.size());
    Eigen::Matrix<double, NUM_OF_LEG, 1>::Map(&rho_vec[0], rho.size()) = rho;

    args = {casadi::DM(q_vec), casadi::DM(rho_vec)};
    return;
}

template <int num_legs, int rows, int cols>
void A1Kinematics::casadiDMToEigen(const std::vector<casadi::DM> & src, std::vector<Eigen::Matrix<double, rows, cols>> & dst)
{
    size_t const m = src.at(0).size1();
    size_t const n = src.at(0).size2();

    dst.resize(num_legs);

    for (size_t k = 0; k < num_legs; ++k)
    {        
        for (size_t i = 0; i < m; ++i)
        {
            for (size_t j = 0; j < n; ++j)
            {
                dst.at(k)(i, j) = static_cast<double>(src.at(k)(i, j));
            }
        }
    }
}


std::vector<Eigen::Vector3d> A1Kinematics::fk(const Vector_dof &q, const Vector_rho &rho)
{
    std::vector<casadi::DM> args(2);
    cookArgs(q, rho, args);
    casadi::DMVector res_casadi = fk_gen(args);
    std::vector<Eigen::Vector3d> res_eigen(NUM_OF_LEG);
    casadiDMToEigen<4, 3, 1>(res_casadi, res_eigen);
    return res_eigen;
}

std::vector<Eigen::Matrix3d> A1Kinematics::jac(const Vector_dof &q, const Vector_rho &rho)
{
    std::vector<casadi::DM> args(2);
    cookArgs(q, rho, args);
    casadi::DMVector res_casadi = J_gen(args);
    std::vector<Eigen::Matrix3d> res_eigen(NUM_OF_LEG);
    casadiDMToEigen<4, 3, 3>(res_casadi, res_eigen);
    return res_eigen;
}

std::vector<Eigen::Matrix<double, 3, RHO_OPT_SIZE>> A1Kinematics::dfk_drho(const Vector_dof &q, const Vector_rho &rho)
{
    std::vector<casadi::DM> args(2);
    cookArgs(q, rho, args);
    casadi::DMVector res_casadi = dfk_drho_gen(args);
    std::vector<Eigen::Matrix<double, 3, RHO_OPT_SIZE>> res_eigen(NUM_OF_LEG);
    casadiDMToEigen<4, 3, RHO_OPT_SIZE>(res_casadi, res_eigen);
    return res_eigen;
}

std::vector<Eigen::Matrix<double, 9, 3>> A1Kinematics::dJ_dq(const Vector_dof &q, const Vector_rho &rho)
{
    std::vector<casadi::DM> args(2);
    cookArgs(q, rho, args);
    casadi::DMVector res_casadi = dJ_dq_gen(args);
    std::vector<Eigen::Matrix<double, 9, 3>> res_eigen(NUM_OF_LEG);
    casadiDMToEigen<4, 9, 3>(res_casadi, res_eigen);
    return res_eigen;
}

std::vector<Eigen::Matrix<double, 9, RHO_OPT_SIZE>> A1Kinematics::dJ_drho(const Vector_dof &q, const Vector_rho &rho)
{
    std::vector<casadi::DM> args(2);
    cookArgs(q, rho, args);
    casadi::DMVector res_casadi = dJ_drho_gen(args);
    std::vector<Eigen::Matrix<double, 9, RHO_OPT_SIZE>> res_eigen(NUM_OF_LEG);
    casadiDMToEigen<4, 9, RHO_OPT_SIZE>(res_casadi, res_eigen);
    return res_eigen;
}


Eigen::Vector3d A1Kinematics::fk(Eigen::Vector3d q, Eigen::VectorXd rho_opt, Eigen::VectorXd rho_fix)
{
    Eigen::Vector3d out;
    autoFunc_fk_pf_pos(q.data(), rho_opt.data(), rho_fix.data(), out.data());
    return out;
}

Eigen::Matrix3d A1Kinematics::jac(Eigen::Vector3d q, Eigen::VectorXd rho_opt, Eigen::VectorXd rho_fix)
{
    Eigen::Matrix3d mtx;
    autoFunc_d_fk_dt(q.data(), rho_opt.data(), rho_fix.data(), mtx.data());
    return mtx;
}

Eigen::Matrix<double, 3, RHO_OPT_SIZE> A1Kinematics::dfk_drho(Eigen::Vector3d q, Eigen::VectorXd rho_opt, Eigen::VectorXd rho_fix)
{
    Eigen::Matrix<double, 3, RHO_OPT_SIZE> mtx;
    autoFunc_d_fk_drho(q.data(), rho_opt.data(), rho_fix.data(), mtx.data());
    return mtx;
}

Eigen::Matrix<double, 9, 3> A1Kinematics::dJ_dq(Eigen::Vector3d q, Eigen::VectorXd rho_opt, Eigen::VectorXd rho_fix)
{
    Eigen::Matrix<double, 9, 3> mtx;
    autoFunc_dJ_dt(q.data(), rho_opt.data(), rho_fix.data(), mtx.data());
    return mtx;
}

Eigen::Matrix<double, 9, RHO_OPT_SIZE> A1Kinematics::dJ_drho(Eigen::Vector3d q, Eigen::VectorXd rho_opt, Eigen::VectorXd rho_fix)
{
    Eigen::Matrix<double, 9, RHO_OPT_SIZE> mtx;
    autoFunc_dJ_drho(q.data(), rho_opt.data(), rho_fix.data(), mtx.data());
    return mtx;
}

// functions generated by matlab
void A1Kinematics::autoFunc_fk_pf_pos(const double in1[3], const double in2[RHO_OPT_SIZE], const double in3[RHO_FIX_SIZE], double p_bf[3])
{
    double lc = in2[0];
    double p_bf_tmp;
    double t10;
    double t5;
    double t6;
    double t7;
    double t8;
    double t9;
    //     This function was generated by the Symbolic Math Toolbox version 8.7.
    //     19-Jan-2022 15:23:15
    t5 = std::cos(in1[0]);
    t6 = std::cos(in1[1]);
    t7 = std::cos(in1[2]);
    t8 = std::sin(in1[0]);
    t9 = std::sin(in1[1]);
    t10 = std::sin(in1[2]);
    p_bf[0] = (in3[0] - in3[3] * t9) - lc * std::sin(in1[1] + in1[2]);
    p_bf[1] = (((in3[1] + in3[2] * t5) + in3[3] * t6 * t8) + lc * t6 * t7 * t8) -
              lc * t8 * t9 * t10;
    p_bf_tmp = lc * t5;
    p_bf[2] = ((in3[2] * t8 - in3[3] * t5 * t6) - p_bf_tmp * t6 * t7) +
              p_bf_tmp * t9 * t10;
}

void A1Kinematics::autoFunc_d_fk_dt(const double in1[3], const double in2[RHO_OPT_SIZE], const double in3[RHO_FIX_SIZE], double jacobian[9])
{
    double lc = in2[0];
    double jacobian_tmp;
    double t10;
    double t11;
    double t16;
    double t18;
    double t5;
    double t6;
    double t7;
    double t8;
    double t9;
    //     This function was generated by the Symbolic Math Toolbox version 8.7.
    //     19-Jan-2022 15:23:15
    t5 = std::cos(in1[0]);
    t6 = std::cos(in1[1]);
    t7 = std::cos(in1[2]);
    t8 = std::sin(in1[0]);
    t9 = std::sin(in1[1]);
    t10 = std::sin(in1[2]);
    t11 = in1[1] + in1[2];
    t16 = lc * std::sin(t11);
    t11 = -(lc * std::cos(t11));
    t18 = in3[3] * t9 + t16;
    jacobian[0] = 0.0;
    jacobian_tmp = lc * t5;
    jacobian[1] = ((-in3[2] * t8 + in3[3] * t5 * t6) + jacobian_tmp * t6 * t7) -
                  jacobian_tmp * t9 * t10;
    jacobian_tmp = in3[3] * t6;
    jacobian[2] = ((in3[2] * t5 + jacobian_tmp * t8) + lc * t6 * t7 * t8) -
                  lc * t8 * t9 * t10;
    jacobian[3] = t11 - jacobian_tmp;
    jacobian[4] = -t8 * t18;
    jacobian[5] = t5 * t18;
    jacobian[6] = t11;
    jacobian[7] = -t8 * t16;
    jacobian[8] = t5 * t16;
}

void A1Kinematics::autoFunc_d_fk_drho(const double in1[3], const double in2[RHO_OPT_SIZE], const double in3[RHO_FIX_SIZE], double d_fk_drho[D_FK_DRHO_SIZE])
{
    double t5;
    double t6;
    //     This function was generated by the Symbolic Math Toolbox version 8.7.
    //     19-Jan-2022 15:23:15
    t5 = in1[1] + in1[2];
    t6 = std::cos(t5);
    d_fk_drho[0] = -std::sin(t5);
    d_fk_drho[1] = t6 * std::sin(in1[0]);
    d_fk_drho[2] = -t6 * std::cos(in1[0]);
}

void A1Kinematics::autoFunc_dJ_dt(const double in1[3], const double in2[RHO_OPT_SIZE], const double in3[RHO_FIX_SIZE],
                                  double dJ_dq[27])
{
    double lc = in2[0];
    double dJ_dq_tmp;
    double t10;
    double t11;
    double t12;
    double t16;
    double t17;
    double t18;
    double t24;
    double t25;
    double t26;
    double t29;
    double t30;
    double t5;
    double t6;
    double t7;
    double t8;
    double t9;
    //     This function was generated by the Symbolic Math Toolbox version 8.7.
    //     19-Jan-2022 15:23:15
    t5 = std::cos(in1[0]);
    t6 = std::cos(in1[1]);
    t7 = std::cos(in1[2]);
    t8 = std::sin(in1[0]);
    t9 = std::sin(in1[1]);
    t10 = std::sin(in1[2]);
    t11 = in1[1] + in1[2];
    t12 = in3[3] * t6;
    t16 = lc * std::cos(t11);
    t17 = lc * std::sin(t11);
    t18 = t5 * t16;
    t25 = t12 + t16;
    t26 = in3[3] * t9 + t17;
    t11 = -(t8 * t16);
    t16 = -(t5 * t17);
    t24 = -(t8 * t17);
    t29 = -(t5 * t26);
    t30 = -(t8 * t26);
    dJ_dq[0] = 0.0;
    dJ_dq[1] =
        ((-in3[2] * t5 - t8 * t12) - lc * t6 * t7 * t8) + lc * t8 * t9 * t10;
    dJ_dq_tmp = lc * t5;
    dJ_dq[2] =
        ((-in3[2] * t8 + t5 * t12) + dJ_dq_tmp * t6 * t7) - dJ_dq_tmp * t9 * t10;
    dJ_dq[3] = 0.0;
    dJ_dq[4] = t29;
    dJ_dq[5] = t30;
    dJ_dq[6] = 0.0;
    dJ_dq[7] = t16;
    dJ_dq[8] = t24;
    dJ_dq[9] = 0.0;
    dJ_dq[10] = t29;
    dJ_dq[11] = t30;
    dJ_dq[12] = t26;
    dJ_dq[13] = -t8 * t25;
    dJ_dq[14] = t5 * t25;
    dJ_dq[15] = t17;
    dJ_dq[16] = t11;
    dJ_dq[17] = t18;
    dJ_dq[18] = 0.0;
    dJ_dq[19] = t16;
    dJ_dq[20] = t24;
    dJ_dq[21] = t17;
    dJ_dq[22] = t11;
    dJ_dq[23] = t18;
    dJ_dq[24] = t17;
    dJ_dq[25] = t11;
    dJ_dq[26] = t18;
}

void A1Kinematics::autoFunc_dJ_drho(const double in1[3], const double in2[RHO_OPT_SIZE], const double in3[RHO_FIX_SIZE],
                                    double dJ_drho[D_J_DRHO_SIZE])
{
    double t10;
    double t5;
    double t6;
    double t7;
    double t8;
    //     This function was generated by the Symbolic Math Toolbox version 8.7.
    //     19-Jan-2022 15:23:16
    t5 = std::cos(in1[0]);
    t6 = std::sin(in1[0]);
    t7 = in1[1] + in1[2];
    t8 = std::cos(t7);
    t7 = std::sin(t7);
    t10 = t6 * t7;
    t7 *= t5;
    dJ_drho[0] = 0.0;
    dJ_drho[1] = t5 * t8;
    dJ_drho[2] = t6 * t8;
    dJ_drho[3] = -t8;
    dJ_drho[4] = -t10;
    dJ_drho[5] = t7;
    dJ_drho[6] = -t8;
    dJ_drho[7] = -t10;
    dJ_drho[8] = t7;
}