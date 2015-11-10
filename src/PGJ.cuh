/*
 * PGJ.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef PGJ_CUH_
#define PGJ_CUH_

#include "include.cuh"
#include "System.cuh"
#include "Solver.cuh"

typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

class System;
class PGJ :public Solver {
  friend class System;
private:

  // PGJ Solver
  DeviceValueArrayView gammaHat;
  DeviceValueArrayView gammaTmp;
  DeviceValueArrayView B;
  // End PGJ Solver

  // PGJ Solver
  thrust::host_vector<double> gammaHat_h;
  thrust::host_vector<double> gammaTmp_h;
  thrust::host_vector<double> B_h;
  // End PGJ Solver

  // PGJ Solver
  thrust::device_vector<double> gammaHat_d;
  thrust::device_vector<double> gammaTmp_d;
  thrust::device_vector<double> B_d;
  // End PGJ Solver

  int performSchurComplementProduct(DeviceValueArrayView src);
  double getResidual(DeviceValueArrayView src);

public:

  double omega;
  double lambda;
  PGJ(System* sys);
	int setup();
	int solve();
};

#endif /* PGJ_CUH_ */
