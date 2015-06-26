/*
 * System.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef SYSTEM_CUH_
#define SYSTEM_CUH_

#include "include.cuh"
#include "Body.cuh"
#include "CollisionDetector.cuh"
#include "Solver.cuh"

#include <stdio.h>

// use array1d_view to wrap the individual arrays
typedef typename cusp::array1d_view< thrust::device_ptr<int> > DeviceIndexArrayView;
typedef typename cusp::array1d_view< thrust::device_ptr<double> > DeviceValueArrayView;

//combine the three array1d_views into a coo_matrix_view
typedef typename cusp::coo_matrix_view<DeviceIndexArrayView, DeviceIndexArrayView, DeviceValueArrayView> DeviceView;

typedef typename cusp::coo_matrix<int, double, cusp::device_memory> DeviceMatrix;

class CollisionDetector;
class Solver;
class System {
public:
  // variables
  int timeIndex;
  double time; //current time
  double h; //time step
  double tol;
  double3 gravity;
  double elapsedTime;
  double totalGPUMemoryUsed;

	// cusp
	DeviceValueArrayView p;
	DeviceValueArrayView v;
	DeviceValueArrayView a;
	DeviceValueArrayView f;
  DeviceValueArrayView f_contact;
  DeviceValueArrayView tmp;
  DeviceValueArrayView r;
  DeviceValueArrayView k;
  DeviceValueArrayView gamma;
	DeviceView mass;
	DeviceView D;
	DeviceView DT;
	DeviceMatrix MinvDT;
	DeviceMatrix N;

	// host vectors
	thrust::host_vector<double> p_h;
	thrust::host_vector<double> v_h;
	thrust::host_vector<double> a_h;
	thrust::host_vector<double> f_h;
  thrust::host_vector<double> f_contact_h;
  thrust::host_vector<double> tmp_h;
  thrust::host_vector<double> r_h;
  thrust::host_vector<double> k_h;
  thrust::host_vector<double> gamma_h;
  thrust::host_vector<double> friction_h;

	thrust::host_vector<int> massI_h;
	thrust::host_vector<int> massJ_h;
	thrust::host_vector<double> mass_h;

  thrust::host_vector<int> DI_h;
  thrust::host_vector<int> DJ_h;
  thrust::host_vector<double> D_h;

	// device vectors
	thrust::device_vector<double> p_d;
	thrust::device_vector<double> v_d;
	thrust::device_vector<double> a_d;
	thrust::device_vector<double> f_d;
  thrust::device_vector<double> f_contact_d;
  thrust::device_vector<double> tmp_d;
  thrust::device_vector<double> r_d;
  thrust::device_vector<double> k_d;
  thrust::device_vector<double> gamma_d;
  thrust::device_vector<double> friction_d;

	thrust::device_vector<int> massI_d;
	thrust::device_vector<int> massJ_d;
	thrust::device_vector<double> mass_d;

  thrust::device_vector<int> DI_d;
  thrust::device_vector<int> DJ_d;
  thrust::device_vector<double> D_d;

  thrust::device_vector<int> DTI_d;
  thrust::device_vector<int> DTJ_d;
  thrust::device_vector<double> DT_d;

	// library of indices for bodies
  thrust::host_vector<int> indices_h;
  thrust::device_vector<int> indices_d;

  thrust::host_vector<int> fixedBodies_h;
  thrust::device_vector<int> fixedBodies_d;

  // library of contact geometry
  thrust::host_vector<double3> contactGeometry_h;
  thrust::device_vector<double3> contactGeometry_d;

  CollisionDetector* collisionDetector;


public:
	System();
	System(int solverType);
	vector<Body*> bodies;
	Solver* solver;
	double  getCurrentTime() const    {return time;}
	double  getTimeStep() const       {return h;}
	double  getTolerance() const      {return tol;}
	int     getTimeIndex() const      {return timeIndex;}
	void    setTimeStep(double step_size);
	int     add(Body* body);
	int     DoTimeStep();
	int     initializeDevice();
	int     initializeSystem();
	int     applyContactForces_CPU();
	int     buildContactJacobian();
	int     buildContactJacobianTranspose();
	int     performSchurComplementProduct(DeviceValueArrayView src);
	int     buildAppliedImpulseVector();
	int     buildSchurVector();
	int     buildSchurMatrix();
	int     exportSystem(string filename);
	int     importSystem(string filename);
};

#endif /* SYSTEM_CUH_ */
