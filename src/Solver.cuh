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

public:
  double tolerance;
  int maxIterations;

  Solver() {
    system = 0;
    tolerance = 0;
    maxIterations = 1000000;
  }
	Solver(System* sys) {
	  system = sys;
	  tolerance = 0;
	  maxIterations = 1000000;
	}
	virtual int setup() = 0;
	virtual int solve() = 0;
};

#endif /* SOLVER_CUH_ */
