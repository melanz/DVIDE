/*
 * PDIP.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef PDIP_CUH_
#define PDIP_CUH_

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
typedef typename cusp::array1d<double, cusp::device_memory> DeviceValueArray;

class System;
class PDIP :public Solver {
  friend class System;
private:
  System* system;

  // spike stuff
  int partitions;
  SpikeSolver* mySolver;
  //MySpmv* m_spmv;
  spike::Options  solverOptions;
  int preconditionerUpdateModulus;
  float preconditionerMaxKrylovIterations;
  vector<float> spikeSolveTime;
  vector<float> spikeNumIter;
  bool  precUpdated;
  float stepKrylovIterations;
  // end spike stuff

  DeviceValueArrayView f;
  DeviceValueArrayView lambda;
  DeviceValueArrayView lambdaTmp;
  DeviceValueArrayView ones;
  DeviceValueArrayView r_d;
  DeviceValueArrayView r_g;
  DeviceValueArrayView delta_gamma;
  DeviceValueArrayView delta_lambda;
  DeviceValueArrayView gammaTmp;
  DeviceValueArrayView rhs;
  DeviceView grad_f;
  DeviceView grad_f_global;
  DeviceView grad_f_T;
  DeviceView diagLambda;
  DeviceView Dinv;
  DeviceView M_hat;
  DeviceMatrix A;
  DeviceView D;
  DeviceView DT;
  DeviceView B;
  DeviceMatrix N;
  DeviceMatrix MinvDT;
  DeviceMatrix C;

  thrust::host_vector<double> f_h;
  thrust::host_vector<double> lambda_h;
  thrust::host_vector<double> lambdaTmp_h;
  thrust::host_vector<double> ones_h;
  thrust::host_vector<double> r_d_h;
  thrust::host_vector<double> r_g_h;
  thrust::host_vector<double> delta_gamma_h;
  thrust::host_vector<double> delta_lambda_h;
  thrust::host_vector<double> gammaTmp_h;

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

  thrust::device_vector<double> f_d;
  thrust::device_vector<double> lambda_d;
  thrust::device_vector<double> lambdaTmp_d;
  thrust::device_vector<double> ones_d;
  thrust::device_vector<double> r_d_d;
  thrust::device_vector<double> r_g_d;
  thrust::device_vector<double> delta_gamma_d;
  thrust::device_vector<double> delta_lambda_d;
  thrust::device_vector<double> gammaTmp_d;
  thrust::device_vector<double> rhs_d;

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

  thrust::device_vector<int> BI_d;
  thrust::device_vector<int> BJ_d;

  //int performSchurComplementProduct(DeviceValueArrayView src);
  int initializeConstraintGradient();
  int initializeConstraintGradientTranspose();
  int initializeM_hat();
  int initializeDinv();
  int initializeDiagLambda();
  int updateNewtonStepVector(DeviceValueArrayView gamma, DeviceValueArrayView lambda, DeviceValueArrayView f, double t);
  int buildAMatrix();

  int buildSchurComplementMatrix();
  int initializeB();

public:
  double mu_pdip;
  double alpha;
  double beta;

	PDIP(System* sys);
	int setup();
	int solve();

  void    setNumPartitions(int num_partitions) {partitions = num_partitions;}
  void    setMaxSpikeIterations(int max_it)   {solverOptions.maxNumIterations = max_it;}
  void    setSolverType(int solverType);
  void    setPrecondType(int useSpike);
  void    printSolverParams();
};

#endif /* PDIP_CUH_ */
