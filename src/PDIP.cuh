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

#include <spike/solver.h>
#include <spike/spmv.h>

typedef double PREC_REAL;

// use array1d_view to wrap the individual arrays
typedef typename cusp::array1d_view<thrust::device_ptr<int> > DeviceIndexArrayView;
typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

//combine the three array1d_views into a coo_matrix_view
typedef typename cusp::coo_matrix_view<DeviceIndexArrayView, DeviceIndexArrayView, DeviceValueArrayView> DeviceView;

typedef typename spike::Solver<DeviceValueArrayView, PREC_REAL> SpikeSolver;
typedef typename cusp::array1d<double, cusp::device_memory> DeviceValueArray;


class MySpmv : public cusp::linear_operator<double, cusp::device_memory>{
public:
  typedef cusp::linear_operator<double, cusp::device_memory> super;

  MySpmv(DeviceView& grad_f,
      DeviceView& grad_f_T,
      DeviceView& D,
      DeviceView& D_T,
      DeviceView& Minv,
      DeviceValueArrayView& diagLambda,
      DeviceValueArrayView& lambdaTmp,
      DeviceValueArrayView& Dinv,
      DeviceValueArrayView& Mhat,
      DeviceValueArrayView& gammaTmp,
      DeviceValueArrayView& f_contact,
      DeviceValueArrayView& tmp
      ) : mgrad_f(grad_f), mgrad_f_T(grad_f_T), mD(D), mD_T(D_T), mMinv(Minv),
          mdiagLambda(diagLambda), mlambdaTmp(lambdaTmp), mDinv(Dinv),
          mMhat(Mhat), mgammaTmp(gammaTmp), mf_contact(f_contact), mtmp(tmp), super(gammaTmp.size(), gammaTmp.size()) {}
  void operator()(const DeviceValueArray& v, DeviceValueArray& Av) {
    // Step 1
    cusp::multiply(mgrad_f, v, mlambdaTmp);
    cusp::blas::xmy(mdiagLambda,mlambdaTmp,mlambdaTmp);
    cusp::blas::xmy(mDinv,mlambdaTmp,mlambdaTmp);
    cusp::multiply(mgrad_f_T, mlambdaTmp, mgammaTmp);

    // Step 2
    cusp::blas::xmy(mMhat,v,Av);
    cusp::blas::axpy(mgammaTmp,Av,1.0);

    // Step 3
    cusp::multiply(mD_T, v, mf_contact);
    cusp::multiply(mMinv, mf_contact, mtmp);
    cusp::multiply(mD, mtmp, mgammaTmp);
    cusp::blas::axpy(mgammaTmp,Av,1.0);
  }

private:
  DeviceView& mgrad_f;
  DeviceView& mgrad_f_T;
  DeviceView& mD;
  DeviceView& mD_T;
  DeviceView& mMinv;
  DeviceValueArrayView& mdiagLambda;
  DeviceValueArrayView& mlambdaTmp;
  DeviceValueArrayView& mDinv;
  DeviceValueArrayView& mMhat;
  DeviceValueArrayView& mgammaTmp;
  DeviceValueArrayView& mf_contact;
  DeviceValueArrayView& mtmp;
};

class System;
class PDIP {
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

  DeviceValueArrayView f;
  DeviceValueArrayView lambda;
  DeviceValueArrayView lambdaTmp;
  DeviceValueArrayView ones;
  DeviceValueArrayView r_d;
  DeviceValueArrayView r_g;
  DeviceValueArrayView delta_gamma;
  DeviceValueArrayView delta_lambda;
  DeviceValueArrayView gammaTmp;
  DeviceValueArrayView Dinv;
  DeviceValueArrayView M_hat;
  DeviceValueArrayView rhs;
  DeviceView grad_f;
  DeviceView grad_f_T;

  thrust::host_vector<double> f_h;
  thrust::host_vector<double> lambda_h;
  thrust::host_vector<double> lambdaTmp_h;
  thrust::host_vector<double> ones_h;
  thrust::host_vector<double> r_d_h;
  thrust::host_vector<double> r_g_h;
  thrust::host_vector<double> delta_gamma_h;
  thrust::host_vector<double> delta_lambda_h;
  thrust::host_vector<double> gammaTmp_h;
  thrust::host_vector<double> Dinv_h;
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
  thrust::device_vector<double> Dinv_d;
  thrust::device_vector<double> Mhat_d;
  thrust::device_vector<double> rhs_d;

  thrust::device_vector<int> grad_fI_d;
  thrust::device_vector<int> grad_fJ_d;
  thrust::device_vector<double> grad_f_d;

  thrust::device_vector<int> grad_fI_T_d;
  thrust::device_vector<int> grad_fJ_T_d;
  thrust::device_vector<double> grad_f_T_d;

  int performSchurComplementProduct(DeviceValueArrayView src);
  int initializeConstraintGradient();
  int initializeConstraintGradientTranspose();
  int updateNewtonStepVector(DeviceValueArrayView gamma, DeviceValueArrayView lambda, DeviceValueArrayView f, double t);

public:
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
