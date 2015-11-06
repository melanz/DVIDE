/*
 * Plate.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef PLATE_CUH_
#define PLATE_CUH_

#include "include.cuh"
#include "System.cuh"

class System;
class Plate {
  friend class System;
private:
  uint identifier;
	uint index;
	int numDOF;
	bool isCurved;

  double3 p_n0;
  double3 p_dxi0;
  double3 p_deta0;

  double3 p_n1;
  double3 p_dxi1;
  double3 p_deta1;

  double3 p_n2;
  double3 p_dxi2;
  double3 p_deta2;

  double3 p_n3;
  double3 p_dxi3;
  double3 p_deta3;

  double3 v_n0;
  double3 v_dxi0;
  double3 v_deta0;

  double3 v_n1;
  double3 v_dxi1;
  double3 v_deta1;

  double3 v_n2;
  double3 v_dxi2;
  double3 v_deta2;

  double3 v_n3;
  double3 v_dxi3;
  double3 v_deta3;

  double3 a_n0;
  double3 a_dxi0;
  double3 a_deta0;

  double3 a_n1;
  double3 a_dxi1;
  double3 a_deta1;

  double3 a_n2;
  double3 a_dxi2;
  double3 a_deta2;

  double3 a_n3;
  double3 a_dxi3;
  double3 a_deta3;

	double density;
	double elasticModulus;
	double nu;
	double thickness;

	double3 contactGeometry;
	int collisionFamily;
	System* sys;

public:
	Plate() {
	  numDOF = 36;
	  identifier = 0;
	  index = 0;
	  sys = 0;
	  collisionFamily = -1;
	  isCurved = false;

		// create test element!
	  p_n0 = make_double3(0,0,0);
	  p_dxi0 = make_double3(1.0,0,0);
	  p_deta0 = make_double3(0,0,1.0);

	  p_n1 = make_double3(1.0,0,0);
	  p_dxi1 = make_double3(1.0,0,0);
	  p_deta1 = make_double3(0,0,1.0);

	  p_n2 = make_double3(1.0,0,1.0);
	  p_dxi2 = make_double3(1.0,0,0);
	  p_deta2 = make_double3(0,0,1.0);

	  p_n3 = make_double3(0,0,1.0);
	  p_dxi3 = make_double3(1.0,0,0);
	  p_deta3 = make_double3(0,0,1.0);

	  v_n0 = make_double3(0,0,0);
	  v_dxi0 = make_double3(0,0,0);
	  v_deta0 = make_double3(0,0,0);

	  v_n1 = make_double3(0,0,0);
	  v_dxi1 = make_double3(0,0,0);
	  v_deta1 = make_double3(0,0,0);

	  v_n2 = make_double3(0,0,0);
	  v_dxi2 = make_double3(0,0,0);
	  v_deta2 = make_double3(0,0,0);

	  v_n3 = make_double3(0,0,0);
	  v_dxi3 = make_double3(0,0,0);
	  v_deta3 = make_double3(0,0,0);

	  a_n0 = make_double3(0,0,0);
	  a_dxi0 = make_double3(0,0,0);
	  a_deta0 = make_double3(0,0,0);

	  a_n1 = make_double3(0,0,0);
	  a_dxi1 = make_double3(0,0,0);
	  a_deta1 = make_double3(0,0,0);

	  a_n2 = make_double3(0,0,0);
	  a_dxi2 = make_double3(0,0,0);
	  a_deta2 = make_double3(0,0,0);

	  a_n3 = make_double3(0,0,0);
	  a_dxi3 = make_double3(0,0,0);
	  a_deta3 = make_double3(0,0,0);

		density = 7200.0;
		elasticModulus = 2.0e7;
		nu = 0.25;
		thickness = 0.02;

		// geometry = length (xi-direction), width (eta-direction), number of contact points in each direction
		contactGeometry = make_double3(1.0,1.0,10);
	}

  Plate(double a, double b, double3 p_n0, double3 p_dxi0, double3 p_deta0,
      double3 p_n1, double3 p_dxi1, double3 p_deta1,
      double3 p_n2, double3 p_dxi2, double3 p_deta2,
      double3 p_n3, double3 p_dxi3, double3 p_deta3) {
    numDOF = 36;
    identifier = 0;
    index = 0;
    sys = 0;
    collisionFamily = -1;
    isCurved = false;

    // create test element!
    this->p_n0 = p_n0;
    this->p_dxi0 = p_dxi0;
    this->p_deta0 = p_deta0;

    this->p_n1 = p_n1;
    this->p_dxi1 = p_dxi1;
    this->p_deta1 = p_deta1;

    this->p_n2 = p_n2;
    this->p_dxi2 = p_dxi2;
    this->p_deta2 = p_deta2;

    this->p_n3 = p_n3;
    this->p_dxi3 = p_dxi3;
    this->p_deta3 = p_deta3;

    v_n0 = make_double3(0,0,0);
    v_dxi0 = make_double3(0,0,0);
    v_deta0 = make_double3(0,0,0);

    v_n1 = make_double3(0,0,0);
    v_dxi1 = make_double3(0,0,0);
    v_deta1 = make_double3(0,0,0);

    v_n2 = make_double3(0,0,0);
    v_dxi2 = make_double3(0,0,0);
    v_deta2 = make_double3(0,0,0);

    v_n3 = make_double3(0,0,0);
    v_dxi3 = make_double3(0,0,0);
    v_deta3 = make_double3(0,0,0);

    a_n0 = make_double3(0,0,0);
    a_dxi0 = make_double3(0,0,0);
    a_deta0 = make_double3(0,0,0);

    a_n1 = make_double3(0,0,0);
    a_dxi1 = make_double3(0,0,0);
    a_deta1 = make_double3(0,0,0);

    a_n2 = make_double3(0,0,0);
    a_dxi2 = make_double3(0,0,0);
    a_deta2 = make_double3(0,0,0);

    a_n3 = make_double3(0,0,0);
    a_dxi3 = make_double3(0,0,0);
    a_deta3 = make_double3(0,0,0);

    density = 7200.0;
    elasticModulus = 2.0e7;
    nu = 0.25;
    thickness = 0.02;

    // geometry = length (xi-direction), width (eta-direction), number of contact points in each direction
    contactGeometry = make_double3(a,b,10);
  }

	double3 getPosition_node0()
	{
	  return p_n0;
	}
  double3 getPosition_node1()
  {
    return p_n1;
  }
  double3 getPosition_node2()
  {
    return p_n2;
  }
  double3 getPosition_node3()
  {
    return p_n3;
  }

  double3 getVelocity_node0()
  {
    return v_n0;
  }
  double3 getVelocity_node1()
  {
    return v_n1;
  }
  double3 getVelocity_node2()
  {
    return v_n2;
  }
  double3 getVelocity_node3()
  {
    return v_n3;
  }

  double getDensity()
  {
    return density;
  }
  double getElasticModulus()
  {
    return elasticModulus;
  }
  double getPoissonRatio()
  {
    return nu;
  }
  double getThickness()
  {
    return thickness;
  }
  void setDensity(double density)
  {
    this->density = density;
  }
  void setElasticModulus(double elasticModulus)
  {
    this->elasticModulus = elasticModulus;
  }
  void setPoissonRatio(double nu)
  {
    this->nu = nu;
  }
  void setThickness(double thickness)
  {
    this->thickness = thickness;
  }

  uint getIndex()
  {
    return index;
  }

	void setIndex(uint index) {
		this->index = index;
	}

  bool getCurved() {
    return isCurved;
  }

  void setCurved(bool isCurved) {
    this->isCurved = isCurved;
  }

	void setIdentifier(uint identifier) {
		this->identifier = identifier;
	}

  double3 getGeometry()
  {
    return contactGeometry;
  }
  void setLength(double length)
  {
    this->contactGeometry.x = length;
  }
  void setWidth(double width)
  {
    this->contactGeometry.y = width;
  }
  void setNumContactPoints(int numPoints)
  {
    this->contactGeometry.z = (double)numPoints;
  }

  void setCollisionFamily(int collisionFamily)
  {
    this->collisionFamily = collisionFamily;
  }

  int getCollisionFamily()
  {
    return collisionFamily;
  }

  double3 transformNodalToCartesian(double xi, double eta);
  int addPlate(int j);
};

#endif /* PLATE_CUH_ */
