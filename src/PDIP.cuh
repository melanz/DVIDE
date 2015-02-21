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

// use array1d_view to wrap the individual arrays
typedef typename cusp::array1d_view<thrust::device_ptr<int> > DeviceIndexArrayView;
typedef typename cusp::array1d_view<thrust::device_ptr<double> > DeviceValueArrayView;

//combine the three array1d_views into a coo_matrix_view
typedef typename cusp::coo_matrix_view<DeviceIndexArrayView, DeviceIndexArrayView, DeviceValueArrayView> DeviceView;

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
};

#endif /* PDIP_CUH_ */
