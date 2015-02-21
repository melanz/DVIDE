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

typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

class System;
class PDIP {
  friend class System;
private:
  System* system;

  DeviceValueArrayView f;
  DeviceValueArrayView lambda;
  DeviceValueArrayView lambdaTmp;
  DeviceValueArrayView ones;
  DeviceValueArrayView r_d;
  DeviceValueArrayView r_g;
  DeviceValueArrayView delta_gamma;
  DeviceValueArrayView delta_lambda;
  DeviceValueArrayView gammaTmp;

  thrust::host_vector<double> f_h;
  thrust::host_vector<double> lambda_h;
  thrust::host_vector<double> lambdaTmp_h;
  thrust::host_vector<double> ones_h;
  thrust::host_vector<double> r_d_h;
  thrust::host_vector<double> r_g_h;
  thrust::host_vector<double> delta_gamma_h;
  thrust::host_vector<double> delta_lambda_h;
  thrust::host_vector<double> gammaTmp_h;

  thrust::device_vector<double> f_d;
  thrust::device_vector<double> lambda_d;
  thrust::device_vector<double> lambdaTmp_d;
  thrust::device_vector<double> ones_d;
  thrust::device_vector<double> r_d_d;
  thrust::device_vector<double> r_g_d;
  thrust::device_vector<double> delta_gamma_d;
  thrust::device_vector<double> delta_lambda_d;
  thrust::device_vector<double> gammaTmp_d;

  double getSupremum(DeviceValueArrayView src);

public:
	PDIP(System* sys);
	int setup();
	int solve();
};

#endif /* PDIP_CUH_ */
