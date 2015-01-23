/*
 * Element.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef ELEMENT_CUH_
#define ELEMENT_CUH_

#include "include.cuh"
#include "Node.cuh"

class Element {
  friend class System;
private:
  uint identifier;
	uint index;
	int numDOF;

	double3 pos;
	double3 vel;
	double3 acc;

	double mass;

public:
	Element() {
	  this->numDOF = 3;
	  this->identifier = 0;
	  this->index = 0;

		// create test element!
		this->pos = make_double3(0, 0, 0);
		this->vel = make_double3(0, 0, 0);
		this->acc = make_double3(0, 0, 0);

		this->mass = 1.0;
	}

  Element(double3 position) {
    this->numDOF = 3;
    this->identifier = 0;
    this->index = 0;

    // create test element!
    this->pos = position;
    this->vel = make_double3(0, 0, 0);
    this->acc = make_double3(0, 0, 0);

    this->mass = 1.0;
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
};

#endif /* ELEMENT_CUH_ */
