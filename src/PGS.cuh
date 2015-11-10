/*
 * PGS.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef PGS_CUH_
#define PGS_CUH_

#include "include.cuh"
#include "System.cuh"
#include "Solver.cuh"

typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

class System;
class PGS :public Solver {
  friend class System;
private:

  // PGS Solver
  DeviceValueArrayView gammaTmp;
  DeviceValueArrayView B;
  // End PGS Solver

  // PGS Solver
  thrust::host_vector<double> gammaTmp_h;
  thrust::host_vector<double> B_h;
  // End PGS Solver

  // PGS Solver
  thrust::device_vector<double> gammaTmp_d;
  thrust::device_vector<double> B_d;
  // End PGS Solver

  thrust::host_vector<uint> bodyIdentifierA_h;
  thrust::host_vector<uint> bodyIdentifierB_h;

  int performSchurComplementProduct(DeviceValueArrayView src);
  double getResidual(DeviceValueArrayView src);
  int updateImpulseVector_CPU();

public:

  double omega;
  double lambda;
  PGS(System* sys);
	int setup();
	int solve();
};

#endif /* PGS_CUH_ */
