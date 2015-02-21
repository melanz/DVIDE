/*
 * Solver.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef SOLVER_CUH_
#define SOLVER_CUH_

#include "include.cuh"
#include "System.cuh"

typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

class System;
class Solver {
  friend class System;
private:
  System* system;

  // APGD Solver
  DeviceValueArrayView gammaHat;
  DeviceValueArrayView gammaNew;
  DeviceValueArrayView g;
  DeviceValueArrayView y;
  DeviceValueArrayView yNew;
  DeviceValueArrayView gammaTmp;
  // End APGD Solver

  // APGD Solver
  thrust::host_vector<double> gammaHat_h;
  thrust::host_vector<double> gammaNew_h;
  thrust::host_vector<double> g_h;
  thrust::host_vector<double> y_h;
  thrust::host_vector<double> yNew_h;
  thrust::host_vector<double> gammaTmp_h;
  // End APGD Solver

  // APGD Solver
  thrust::device_vector<double> gammaHat_d;
  thrust::device_vector<double> gammaNew_d;
  thrust::device_vector<double> g_d;
  thrust::device_vector<double> y_d;
  thrust::device_vector<double> yNew_d;
  thrust::device_vector<double> gammaTmp_d;
  // End APGD Solver

  int performSchurComplementProduct(DeviceValueArrayView src);
  double getResidual(DeviceValueArrayView src);

public:
	Solver(System* sys);
	int setup();
	int solve();
};

#endif /* SOLVER_CUH_ */
