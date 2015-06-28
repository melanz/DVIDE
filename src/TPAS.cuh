/*
 * TPAS.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef TPAS_CUH_
#define TPAS_CUH_

#include "include.cuh"
#include "System.cuh"
#include "Solver.cuh"

#include <spike/solver.h>
#include <spike/spmv.h>

typedef double PREC_REAL;

// use array1d_view to wrap the individual arrays
typedef typename cusp::array1d_view< thrust::device_ptr<int> > DeviceIndexArrayView;
typedef typename cusp::array1d_view< thrust::device_ptr<double> > DeviceValueArrayView;

//combine the three array1d_views into a coo_matrix_view
typedef typename cusp::coo_matrix_view<DeviceIndexArrayView, DeviceIndexArrayView, DeviceValueArrayView> DeviceView;

typedef typename cusp::coo_matrix<int, double, cusp::device_memory> DeviceMatrix;

typedef typename spike::Solver<DeviceValueArrayView, PREC_REAL> SpikeSolver;
typedef typename cusp::array1d<int, cusp::device_memory> DeviceIndexArray;
typedef typename cusp::array1d<double, cusp::device_memory> DeviceValueArray;

class MySpmv : public cusp::linear_operator<double, cusp::device_memory>{
public:
  typedef cusp::linear_operator<double, cusp::device_memory> super;
  //MySpmv(DeviceView& lhs_mass, DeviceView& A, DeviceValueArrayView& A) : m_A(A) {}
  MySpmv(DeviceView& Minv, DeviceView& D, DeviceView& DT, DeviceView& phi_gamma, DeviceView& phi_gammaT, DeviceValueArrayView& temp_vel, DeviceValueArrayView& temp_gamma, DeviceValueArrayView& temp_delta) : mMinv(Minv), mD(D), mDT(DT), mphi_gamma(phi_gamma), mphi_gammaT(phi_gammaT), mtemp_vel(temp_vel), mtemp_gamma(temp_gamma), mtemp_delta(temp_delta), super(temp_delta.size(), temp_delta.size()) {}
  void operator()(const DeviceValueArray& v, DeviceValueArray& Av) {
    thrust::device_ptr<double> wrapped_device_v(CASTD1(v.data()));
    DeviceValueArrayView v_gamma(wrapped_device_v, wrapped_device_v + mtemp_gamma.size());
    DeviceValueArrayView v_lambda(wrapped_device_v + mtemp_gamma.size(), wrapped_device_v + mtemp_delta.size());
    thrust::device_ptr<double> wrapped_device_Av(CASTD1(Av.data()));
    DeviceValueArrayView Av_gamma(wrapped_device_Av, wrapped_device_Av + mtemp_gamma.size());
    DeviceValueArrayView Av_lambda(wrapped_device_Av + mtemp_gamma.size(), wrapped_device_Av + mtemp_delta.size());

    // b_gamma = D*Minv*D'*x_gamma
    cusp::multiply(mDT, v_gamma, mtemp_vel);
    cusp::multiply(mMinv, mtemp_vel, mtemp_gamma);
    cusp::multiply(mD, mtemp_gamma, Av_gamma);

    // b_gamma = b_gamma + phi_gamma'*x_lambda
    cusp::multiply(mphi_gammaT, v_lambda, mtemp_gamma);
    cusp::blas::axpy(mtemp_gamma, Av_gamma, 1.0);

    // b_lambda = phi_gamma*x_gamma
    cusp::multiply(mphi_gamma, v_gamma, Av_lambda);
  }
private:
  DeviceView& mMinv;
  DeviceView& mD;
  DeviceView& mDT;
  DeviceView& mphi_gamma;
  DeviceView& mphi_gammaT;
  DeviceValueArrayView& mtemp_vel;
  DeviceValueArrayView& mtemp_gamma;
  DeviceValueArrayView& mtemp_delta;
};

class System;
class TPAS :public Solver {
  friend class System;
private:
  System* system;

  // spike stuff
  int partitions;
  SpikeSolver* mySolver;
  MySpmv* m_spmv;
  spike::Options  solverOptions;
  int preconditionerUpdateModulus;
  float preconditionerMaxKrylovIterations;
  vector<float> spikeSolveTime;
  vector<float> spikeNumIter;
  bool  precUpdated;
  float stepKrylovIterations;
  // end spike stuff

  uint numActiveNormalConstraints;
  uint numActiveTangentConstraints;

  // APGD Solver
  DeviceValueArrayView gammaHat;
  DeviceValueArrayView gammaNew;
  DeviceValueArrayView g;
  DeviceValueArrayView y;
  DeviceValueArrayView yNew;
  // End APGD Solver

  DeviceValueArrayView f;
  DeviceValueArrayView lambda;
  DeviceValueArrayView lambdaTmp;
  DeviceValueArrayView ones;
  DeviceValueArrayView r_d;
  DeviceValueArrayView r_g;
  DeviceValueArrayView delta;
  DeviceValueArrayView delta_gamma;
  DeviceValueArrayView delta_lambda;
  DeviceValueArrayView gammaTmp;
  DeviceValueArrayView rhs;
  DeviceValueArrayView res;
  DeviceValueArrayView res_gamma;
  DeviceValueArrayView res_lambda;
  DeviceView grad_f;
  DeviceView grad_f_T;
  DeviceView diagLambda;
  DeviceView Dinv;
  DeviceView M_hat;
  DeviceMatrix A;

  thrust::host_vector<double> f_h;
  thrust::host_vector<double> lambda_h;
  thrust::host_vector<double> lambdaTmp_h;
  thrust::host_vector<double> ones_h;
  thrust::host_vector<double> r_d_h;
  thrust::host_vector<double> r_g_h;
  thrust::host_vector<double> delta_h;
  thrust::host_vector<double> gammaTmp_h;
  thrust::host_vector<double> res_h;

  thrust::host_vector<int> lambdaI_h;
  thrust::host_vector<int> lambdaJ_h;

  thrust::host_vector<int> DinvI_h;
  thrust::host_vector<int> DinvJ_h;
  thrust::host_vector<double> Dinv_h;

  thrust::host_vector<int> MhatI_h;
  thrust::host_vector<int> MhatJ_h;
  thrust::host_vector<double> Mhat_h;

  thrust::host_vector<int> grad_fI_h;
  thrust::host_vector<int> grad_fJ_h;
  thrust::host_vector<double> grad_f_h;

  thrust::host_vector<int> grad_fI_T_h;
  thrust::host_vector<int> grad_fJ_T_h;
  thrust::host_vector<double> grad_f_T_h;

  // APGD Solver
  thrust::device_vector<double> gammaHat_d;
  thrust::device_vector<double> gammaNew_d;
  thrust::device_vector<double> g_d;
  thrust::device_vector<double> y_d;
  thrust::device_vector<double> yNew_d;
  // End APGD Solver

  thrust::device_vector<int> activeSetNormal_d;
  thrust::device_vector<int> activeSetNormalNew_d;
  thrust::device_vector<int> activeSetTangent_d;
  thrust::device_vector<int> activeSetTangentNew_d;

  thrust::host_vector<int> activeSet_h; //TODO: GET RID OF THIS

  thrust::device_vector<double> f_d;
  thrust::device_vector<double> lambda_d;
  thrust::device_vector<double> lambdaTmp_d;
  thrust::device_vector<double> ones_d;
  thrust::device_vector<double> r_d_d;
  thrust::device_vector<double> r_g_d;
  thrust::device_vector<double> delta_d;
  thrust::device_vector<double> gammaTmp_d;
  thrust::device_vector<double> rhs_d;
  thrust::device_vector<double> res_d;

  thrust::device_vector<int> lambdaI_d;
  thrust::device_vector<int> lambdaJ_d;

  thrust::device_vector<int> DinvI_d;
  thrust::device_vector<int> DinvJ_d;
  thrust::device_vector<double> Dinv_d;

  thrust::device_vector<int> MhatI_d;
  thrust::device_vector<int> MhatJ_d;
  thrust::device_vector<double> Mhat_d;

  thrust::device_vector<int> grad_fI_d;
  thrust::device_vector<int> grad_fJ_d;
  thrust::device_vector<double> grad_f_d;

  thrust::device_vector<int> grad_fI_T_d;
  thrust::device_vector<int> grad_fJ_T_d;
  thrust::device_vector<double> grad_f_T_d;

  //int performSchurComplementProduct(DeviceValueArrayView src);
  int initializeActiveConstraintGradient();
  int initializeConstraintGradientTranspose();
  int updateResidualVector();
  //int initializeM_hat();
  //int initializeDinv();
  //int initializeDiagLambda();
  int updateNewtonStepVector(DeviceValueArrayView gamma, DeviceValueArrayView lambda, DeviceValueArrayView f, double t);
  int buildAMatrix();

  int performSchurComplementProduct(DeviceValueArrayView src);
  double getResidual(DeviceValueArrayView src);

public:
  double mu_pdip;
  double alpha;
  double beta;
  double totalKrylovIterations;

  TPAS(System* sys);
	int setup();
	int solve();

  void    setNumPartitions(int num_partitions) {partitions = num_partitions;}
  void    setMaxSpikeIterations(int max_it)   {solverOptions.maxNumIterations = max_it;}
  void    setSolverType(int solverType);
  void    setPrecondType(int useSpike);
  void    printSolverParams();
};

#endif /* PDIP_CUH_ */
