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
#include "Beam.cuh"
#include "Plate.cuh"
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
class Beam;
class Plate;
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
	DeviceValueArrayView fElastic;
	DeviceValueArrayView fApplied;
  DeviceValueArrayView f_contact;
  DeviceValueArrayView tmp;
  DeviceValueArrayView r;
  DeviceValueArrayView b;
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
	thrust::host_vector<double> fElastic_h;
	thrust::host_vector<double> fApplied_h;
  thrust::host_vector<double> f_contact_h;
  thrust::host_vector<double> tmp_h;
  thrust::host_vector<double> r_h;
  thrust::host_vector<double> b_h;
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
	thrust::device_vector<double> fElastic_d;
	thrust::device_vector<double> fApplied_d;
  thrust::device_vector<double> f_contact_d;
  thrust::device_vector<double> tmp_d;
  thrust::device_vector<double> r_d;
  thrust::device_vector<double> b_d;
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

  // library of contact geometry TODO: this is duplicate information, needed for now
  thrust::host_vector<double3> contactGeometry_h;
  thrust::device_vector<double3> contactGeometry_d;

  thrust::host_vector<double3> collisionGeometry_h;
  thrust::device_vector<double3> collisionGeometry_d;

  thrust::host_vector<int4> collisionMap_h;
  thrust::device_vector<int4> collisionMap_d;

  // list of constraints
  thrust::host_vector<int2> constraintsBilateralDOF_h;
  thrust::device_vector<int2> constraintsBilateralDOF_d;

  // library of material information (beams)
  thrust::host_vector<double3> materialsBeam_h;
  thrust::device_vector<double3> materialsBeam_d;

  // library of material information (plates)
  thrust::host_vector<double4> materialsPlate_h;
  thrust::device_vector<double4> materialsPlate_d;

  thrust::host_vector<double> strainDerivative_h;
  thrust::device_vector<double> strainDerivative_d;

  thrust::host_vector<double> strain_h;
  thrust::device_vector<double> strain_d;

  thrust::host_vector<double> Sx_h;
  thrust::device_vector<double> Sx_d;

  thrust::host_vector<double> Sxx_h;
  thrust::device_vector<double> Sxx_d;

  thrust::host_vector<double> wt5;
  thrust::host_vector<double> pt5;
  thrust::host_vector<double> wt3;
  thrust::host_vector<double> pt3;

  CollisionDetector* collisionDetector;
  thrust::device_vector<int> nonzerosPerContact_d;

public:
	System();
	System(int solverType);
	vector<Body*> bodies;
	vector<Beam*> beams;
	vector<Plate*> plates;
	Solver* solver;
	double  getCurrentTime() const    {return time;}
	double  getTimeStep() const       {return h;}
	double  getTolerance() const      {return tol;}
	int     getTimeIndex() const      {return timeIndex;}
	void    setTimeStep(double step_size);
	int     add(Body* body);
	int     add(Beam* beam);
	int     add(Plate* plate);
	int     DoTimeStep();
	int     initializeDevice();
	int     initializeSystem();
	int     applyForce(Body* body, double3 force);
	int     clearAppliedForces();
	int     buildContactJacobian();
	int     buildContactJacobianTranspose();
	int     performSchurComplementProduct(DeviceValueArrayView src);
	int     buildAppliedImpulseVector();
	int     buildSchurVector();
	int     buildSchurMatrix();
	int     exportSystem(string filename);
	int     importSystem(string filename);
	int     exportMatrices(string directory);
	double4 getCCPViolation();
	int     updateElasticForces();
	int     addBilateralConstraintDOF(int DOFA, int DOFB);
};

#endif /* SYSTEM_CUH_ */
