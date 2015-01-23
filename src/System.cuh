/*
 * System.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef SYSTEM_CUH_
#define SYSTEM_CUH_

#include "include.cuh"
#include "Element.cuh"

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

class System {
public:
  double3 gravity;
	double stepTime;
	int stepNewtonIterations;
	float stepKrylovIterations;
	int maxNewtonIterations;

	// spike stuff
	int partitions;
	SpikeSolver* mySolver;
	MySpmv* m_spmv;
	spike::Options  solverOptions;
	int preconditionerUpdateModulus;
	float preconditionerMaxKrylovIterations;

	vector<float> spikeSolveTime;
	vector<float> spikeNumIter;

	bool  precUpdated;
	// end spike stuff

	// variables
	int timeIndex;
	double time; //current time
	double simTime; //time to end simulation
	double h; //time step
	double tol;

	// cusp
	DeviceValueArrayView p;
	DeviceValueArrayView v;
	DeviceValueArrayView a;
	DeviceValueArrayView f;
	DeviceView mass;

	// host vectors
	thrust::host_vector<double> p_h;
	thrust::host_vector<double> v_h;
	thrust::host_vector<double> a_h;
	thrust::host_vector<double> f_h;

	thrust::host_vector<int> massI_h;
	thrust::host_vector<int> massJ_h;
	thrust::host_vector<double> mass_h;

	// device vectors
	thrust::device_vector<double> p_d;
	thrust::device_vector<double> v_d;
	thrust::device_vector<double> a_d;
	thrust::device_vector<double> f_d;

	thrust::device_vector<int> massI_d;
	thrust::device_vector<int> massJ_d;
	thrust::device_vector<double> mass_d;

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

public:
	System();
	vector<Element> elements;

	double getCurrentTime() const    {return time;}
	double getSimulationTime() const {return simTime;}
	double getTimeStep() const       {return h;}
	double getTolerance() const      {return tol;}
	int    getTimeIndex() const      {return timeIndex;}
	void setTimeStep(double step_size, double precision = 1e-10);
	void setSimulationTime(double sim_time)   {simTime = sim_time;}
	void setNumPartitions(int num_partitions) {partitions = num_partitions;}
	void setMaxKrylovIterations(int max_it)   {solverOptions.maxNumIterations = max_it;}
	void setSolverType(int solverType);
	void setPrecondType(int useSpike);
	void printSolverParams();
	int add(Element* element);
	int DoTimeStep();
	int initializeDevice();
	int initializeSystem();
};

#endif /* SYSTEM_CUH_ */
