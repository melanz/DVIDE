/*
 * JKIP.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef JKIP_CUH_
#define JKIP_CUH_

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

class MySpmvJKIP : public cusp::linear_operator<double, cusp::device_memory>{
public:
  typedef cusp::linear_operator<double, cusp::device_memory> super;
  MySpmvJKIP(DeviceView& Minv, DeviceView& D, DeviceView& DT, DeviceView& Pw, DeviceView& Ty, DeviceView& invTx, DeviceValueArrayView& temp_vel, DeviceValueArrayView& temp_vel2, DeviceValueArrayView& temp_gamma) : mMinv(Minv), mD(D), mDT(DT), mPw(Pw), mTy(Ty), minvTx(invTx), mtemp_vel(temp_vel), mtemp_vel2(temp_vel2), mtemp_gamma(temp_gamma), super(temp_gamma.size(), temp_gamma.size()) {}
  void operator()(const DeviceValueArray& v, DeviceValueArray& Av) {
    // Av = D*Minv*D'*v
    cusp::multiply(minvTx,v,mtemp_gamma);
    cusp::multiply(mDT, mtemp_gamma, mtemp_vel);
    cusp::multiply(mMinv, mtemp_vel, mtemp_vel2);
    cusp::multiply(mD, mtemp_vel2, mtemp_gamma);
    cusp::multiply(mTy,mtemp_gamma,Av);

    // tmp = P(w)*v
    cusp::multiply(mPw, v, mtemp_gamma);

    // Av = tmp + Av
    cusp::blas::axpy(mtemp_gamma, Av, 1.0);
  }
private:
  DeviceView& mMinv;
  DeviceView& mD;
  DeviceView& mDT;
  DeviceView& mPw;
  DeviceView& mTy;
  DeviceView& minvTx;

  DeviceValueArrayView& mtemp_vel;
  DeviceValueArrayView& mtemp_vel2;
  DeviceValueArrayView& mtemp_gamma;
};

class System;
class JKIP :public Solver {
  friend class System;
private:
  System* system;

  // spike stuff
  int partitions;
  SpikeSolver* mySolver;
  MySpmvJKIP* m_spmv;
  spike::Options  solverOptions;
  int preconditionerUpdateModulus;
  float preconditionerMaxKrylovIterations;
  vector<float> spikeSolveTime;
  vector<float> spikeNumIter;
  bool  precUpdated;
  float stepKrylovIterations;
  // end spike stuff

  // vectors:  x, y, dx, dy, d, b, r, tmp
  // matrices: invTx, Ty, Pw, R(?)

  DeviceValueArrayView x;
  DeviceValueArrayView y;
  DeviceValueArrayView dx;
  DeviceValueArrayView dy;
  DeviceValueArrayView d;
  DeviceValueArrayView b;
  DeviceValueArrayView r;
  DeviceValueArrayView tmp;
  DeviceView invTx;
  DeviceView Ty;
  DeviceView Pw;

  thrust::host_vector<double> x_h;
  thrust::host_vector<double> y_h;
  thrust::host_vector<double> dx_h;
  thrust::host_vector<double> dy_h;
  thrust::host_vector<double> d_h;
  thrust::host_vector<double> b_h;
  thrust::host_vector<double> r_h;
  thrust::host_vector<double> tmp_h;

  thrust::host_vector<int> invTxI_h;
  thrust::host_vector<int> invTxJ_h;
  thrust::host_vector<double> invTx_h;

  thrust::host_vector<int> TyI_h;
  thrust::host_vector<int> TyJ_h;
  thrust::host_vector<double> Ty_h;

  thrust::host_vector<int> PwI_h;
  thrust::host_vector<int> PwJ_h;
  thrust::host_vector<double> Pw_h;

  thrust::device_vector<double> x_d;
  thrust::device_vector<double> y_d;
  thrust::device_vector<double> dx_d;
  thrust::device_vector<double> dy_d;
  thrust::device_vector<double> d_d;
  thrust::device_vector<double> b_d;
  thrust::device_vector<double> r_d;
  thrust::device_vector<double> tmp_d;

  thrust::device_vector<int> invTxI_d;
  thrust::device_vector<int> invTxJ_d;
  thrust::device_vector<double> invTx_d;

  thrust::device_vector<int> TyI_d;
  thrust::device_vector<int> TyJ_d;
  thrust::device_vector<double> Ty_d;

  thrust::device_vector<int> PwI_d;
  thrust::device_vector<int> PwJ_d;
  thrust::device_vector<double> Pw_d;

  int initializeT();
  int initializePw();
  int performSchurComplementProduct(DeviceValueArrayView src, DeviceValueArrayView tmp2);
  double updateAlpha(double s);

public:
  bool careful;
  double totalKrylovIterations;

	JKIP(System* sys);
	int setup();
	int solve();

  void    setNumPartitions(int num_partitions) {partitions = num_partitions;}
  void    setMaxSpikeIterations(int max_it)   {solverOptions.maxNumIterations = max_it;}
  void    setSolverType(int solverType);
  void    setPrecondType(int useSpike);
  void    printSolverParams();
};

#endif /* JKIP_CUH_ */
