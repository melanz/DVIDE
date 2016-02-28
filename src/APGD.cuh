/*
 * APGD.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef APGD_CUH_
#define APGD_CUH_

#include "include.cuh"
#include "System.cuh"
#include "Solver.cuh"

typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

class System;
class APGD :public Solver {
  friend class System;
private:

  // APGD Solver
  DeviceValueArrayView gammaHat;
  DeviceValueArrayView gammaNew;
  DeviceValueArrayView g;
  DeviceValueArrayView y;
  DeviceValueArrayView yNew;
  DeviceValueArrayView gammaTmp;
  DeviceValueArrayView antiRelaxation;
  //DeviceValueArrayView pOld;
  // End APGD Solver

  // APGD Solver
  thrust::host_vector<double> gammaHat_h;
  thrust::host_vector<double> gammaNew_h;
  thrust::host_vector<double> g_h;
  thrust::host_vector<double> y_h;
  thrust::host_vector<double> yNew_h;
  thrust::host_vector<double> gammaTmp_h;
  thrust::host_vector<double> antiRelaxation_h;
  //thrust::host_vector<double> pOld_h;
  // End APGD Solver

  // APGD Solver
  thrust::device_vector<double> gammaHat_d;
  thrust::device_vector<double> gammaNew_d;
  thrust::device_vector<double> g_d;
  thrust::device_vector<double> y_d;
  thrust::device_vector<double> yNew_d;
  thrust::device_vector<double> gammaTmp_d;
  thrust::device_vector<double> antiRelaxation_d;
  //thrust::device_vector<double> pOld_d;
  // End APGD Solver

  bool useWarmStarting;
  bool useAntiRelaxation;

  int performSchurComplementProduct(DeviceValueArrayView src);
  double getResidual(DeviceValueArrayView src);

public:
  void setWarmStarting(bool useWarmStarting) {this->useWarmStarting = useWarmStarting;}
  void setAntiRelaxation(bool useAntiRelaxation) {this->useAntiRelaxation = useAntiRelaxation;}

	APGD(System* sys);
	int setup();
	int solve();
};

#endif /* APGD_CUH_ */
