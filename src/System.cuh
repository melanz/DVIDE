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

#include <spike/solver.h>
#include <spike/spmv.h>

#include <stdio.h>

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


	MySpmv(DeviceView& mass) : A(mass) {}

	void operator()(const DeviceValueArray& v, DeviceValueArray& Av) {
		cusp::multiply(A, v, Av);
	}

private:
	DeviceView&      A;
};

class CollisionDetector;
class System {
public:
  // variables
  int timeIndex;
  double time; //current time
  double h; //time step
  double tol;
  double3 gravity;

	// spike stuff
	int partitions;
//	SpikeSolver* mySolver;
//	MySpmv* m_spmv;
	spike::Options  solverOptions;
	int preconditionerUpdateModulus;
	float preconditionerMaxKrylovIterations;
	vector<float> spikeSolveTime;
	vector<float> spikeNumIter;
	bool  precUpdated;
  float stepKrylovIterations;
	// end spike stuff

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
  DeviceValueArrayView gammaHat;
  DeviceValueArrayView gammaNew;
  DeviceValueArrayView g;
  DeviceValueArrayView y;
  DeviceValueArrayView yNew;
  DeviceValueArrayView gammaTmp;
	DeviceView mass;
	DeviceView D;
	DeviceView DT;

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
  thrust::host_vector<double> gammaHat_h;
  thrust::host_vector<double> gammaNew_h;
  thrust::host_vector<double> g_h;
  thrust::host_vector<double> y_h;
  thrust::host_vector<double> yNew_h;
  thrust::host_vector<double> gammaTmp_h;

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
  thrust::device_vector<double> gammaHat_d;
  thrust::device_vector<double> gammaNew_d;
  thrust::device_vector<double> g_d;
  thrust::device_vector<double> y_d;
  thrust::device_vector<double> yNew_d;
  thrust::device_vector<double> gammaTmp_d;

	thrust::device_vector<int> massI_d;
	thrust::device_vector<int> massJ_d;
	thrust::device_vector<double> mass_d;

  thrust::device_vector<int> DI_d;
  thrust::device_vector<int> DJ_d;
  thrust::device_vector<double> D_d;

  thrust::device_vector<int> DTI_d;
  thrust::device_vector<int> DTJ_d;
  thrust::device_vector<double> DT_d;

//	dim3 dimBlockConstraint;
//	dim3 dimGridConstraint;
//
//	dim3 dimBlockElement;
//	dim3 dimGridElement;
//
//	dim3 dimBlockParticles;
//	dim3 dimGridParticles;
//
//	dim3 dimBlockCollision;
//	dim3 dimGridCollision;

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
	vector<Body*> bodies;

	double  getCurrentTime() const    {return time;}
	double  getTimeStep() const       {return h;}
	double  getTolerance() const      {return tol;}
	int     getTimeIndex() const      {return timeIndex;}
	void    setTimeStep(double step_size, double precision = 1e-10);
	void    setNumPartitions(int num_partitions) {partitions = num_partitions;}
	void    setMaxSpikeIterations(int max_it)   {solverOptions.maxNumIterations = max_it;}
	void    setSolverType(int solverType);
	void    setPrecondType(int useSpike);
	void    printSolverParams();
	int     add(Body* body);
	int     DoTimeStep();
	int     initializeDevice();
	int     initializeSystem();
	int     applyContactForces();
	int     applyContactForces_CPU();
	int     buildContactJacobian();
  int     buildContactJacobian_CPU();
	int     buildContactJacobianTranspose();
	int     performSchurComplementProduct(DeviceValueArrayView src);
	int     multiplyByMass(thrust::device_vector<double> src, thrust::device_vector<double> dst);
	int     buildAppliedImpulseVector();
	int     buildRightHandSideVector();
	int     solve_APGD();
	int     project(thrust::device_vector<double> src);
  int     project_CPU(thrust::device_vector<double> src);
	double  getResidual(DeviceValueArrayView src);
};

#endif /* SYSTEM_CUH_ */
