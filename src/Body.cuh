/*
 * Body.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef BODY_CUH_
#define BODY_CUH_

#include "include.cuh"
#include "PhysicsItem.cuh"

class Body : public PhysicsItem {
  friend class System;
private:

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
	  collisionFamily = -1;

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
    collisionFamily = -1;

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
  void setVelocity(double3 velocity)
  {
    this->vel = velocity;
  }
};

#endif /* BODY_CUH_ */
