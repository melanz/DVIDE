/*
 * Beam.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef BEAM_CUH_
#define BEAM_CUH_

#include "include.cuh"
#include "System.cuh"

class System;
class Beam {
  friend class System;
private:
  uint identifier;
	uint index;
	int numDOF;

  double3 p_n0;
  double3 p_dn0;
  double3 p_n1;
  double3 p_dn1;

	double3 v_n0;
	double3 v_dn0;
	double3 v_n1;
	double3 v_dn1;

  double3 a_n0;
  double3 a_dn0;
  double3 a_n1;
  double3 a_dn1;

	double density;
	double elasticModulus;

	double3 contactGeometry;

	System* sys;

public:
	Beam() {
	  numDOF = 12;
	  identifier = 0;
	  index = 0;
	  sys = 0;

		// create test element!
	  p_n0 = make_double3(0, 0, 0);
	  p_dn0 = make_double3(1.0, 0, 0);
	  p_n1 = make_double3(1.0, 0, 0);
	  p_dn1 = make_double3(1.0, 0, 0);

    v_n0 = make_double3(0, 0, 0);
    v_dn0 = make_double3(0, 0, 0);
    v_n1 = make_double3(0, 0, 0);
    v_dn1 = make_double3(0, 0, 0);

    a_n0 = make_double3(0, 0, 0);
    a_dn0 = make_double3(0, 0, 0);
    a_n1 = make_double3(0, 0, 0);
    a_dn1 = make_double3(0, 0, 0);

		density = 7200.0;
		elasticModulus = 2.0e7;

		contactGeometry = make_double3(0.02,1.0,10);
	}

	Beam(double3 node0, double3 node1) {
    numDOF = 3;
    identifier = 0;
    index = 0;
    sys = 0;

    // create test element!
    double l = length(node0-node1);
    double3 dir = (node1-node0)/l;
    p_n0 = node0;
    p_dn0 = dir;
    p_n1 = node1;
    p_dn1 = dir;

    v_n0 = make_double3(0, 0, 0);
    v_dn0 = make_double3(0, 0, 0);
    v_n1 = make_double3(0, 0, 0);
    v_dn1 = make_double3(0, 0, 0);

    a_n0 = make_double3(0, 0, 0);
    a_dn0 = make_double3(0, 0, 0);
    a_n1 = make_double3(0, 0, 0);
    a_dn1 = make_double3(0, 0, 0);

    density = 7200.0;
    elasticModulus = 2.0e7;

    contactGeometry = make_double3(0.02,l,10);
  }

	double3 getPosition_node0()
	{
	  return p_n0;
	}
  double3 getPosition_node1()
  {
    return p_n1;
  }

  double3 getVelocity_node0()
  {
    return v_n0;
  }

  double3 getVelocity_node1()
  {
    return v_n1;
  }

  double getDensity()
  {
    return density;
  }
  double getElasticModulus()
  {
    return elasticModulus;
  }
  void setDensity(double density)
  {
    this->density = density;
  }
  void setElasticModulus(double elasticModulus)
  {
    this->elasticModulus = elasticModulus;
  }

  uint getIndex()
  {
    return index;
  }

	void setIndex(uint index) {
		this->index = index;
	}

	void setIdentifier(uint identifier) {
		this->identifier = identifier;
	}

  double3 getGeometry()
  {
    return contactGeometry;
  }
  void setGeometry(double3 geometry)
  {
    this->contactGeometry = geometry;
  }

  double3 transformNodalToCartesian(double xi);
  int addBeam(int j);
};

#endif /* BEAM_CUH_ */
