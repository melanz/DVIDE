/*
 * Body.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef BODY_CUH_
#define BODY_CUH_

#include "include.cuh"

class Body {
  friend class System;
private:
  uint identifier;
	uint index;
	int numDOF;

	double3 pos;
	double3 vel;
	double3 acc;

	double mass;

	bool fixed;
	double3 contactGeometry;

public:
	Body() {
	  numDOF = 3;
	  identifier = 0;
	  index = 0;

		// create test element!
		pos = make_double3(0, 0, 0);
		vel = make_double3(0, 0, 0);
		acc = make_double3(0, 0, 0);

		mass = 1.0;

		fixed = false;
		contactGeometry = make_double3(1.0,0,0);
	}

  Body(double3 position) {
    numDOF = 3;
    identifier = 0;
    index = 0;

    // create test element!
    pos = position;
    vel = make_double3(0, 0, 0);
    acc = make_double3(0, 0, 0);

    mass = 1.0;

    fixed = false;
    contactGeometry = make_double3(1.0,0,0);
  }

  bool isFixed()
  {
    return fixed;
  }

	double3 getPosition()
	{
	  return pos;
	}

  double3 getVelocity()
  {
    return vel;
  }

  double getMass()
  {
    return mass;
  }
  void setMass(double mass)
  {
    this->mass = mass;
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

  void setBodyFixed(bool fixed)
  {
    this->fixed = fixed;
  }
  double3 getGeometry()
  {
    return contactGeometry;
  }
  void setGeometry(double3 geometry)
  {
    this->contactGeometry = geometry;
  }
};

#endif /* ELEMENT_CUH_ */
