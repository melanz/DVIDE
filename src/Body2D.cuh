/*
 * Body2D.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef BODY2D_CUH_
#define BODY2D_CUH_

#include "include.cuh"
#include "System.cuh"
#include "PhysicsItem.cuh"

class System;
class Body2D : public PhysicsItem {
	friend class System;
private:

	double3 p;
	double3 v;
	double3 a;

	double mass;
	double inertia;

	System* sys;

public:
	Body2D() {
		numDOF = 3;
		identifier = 0;
		index = 0;
		sys = 0;
		collisionFamily = -1;

		// create test element!
		p = make_double3(0, 0, 0);
		v = make_double3(0, 0, 0);
		a = make_double3(0, 0, 0);

		mass = 1.0;
		inertia = 1.0;
	}

	Body2D(double3 pos, double3 vel, double mass, double inertia) {
		numDOF = 3;
		identifier = 0;
		index = 0;
		sys = 0;
		collisionFamily = -1;

		// create test element!
		p = pos;
		v = vel;
		a = make_double3(0, 0, 0);

		this->mass = mass;
		this->inertia = inertia;
	}

	double3 getPosition()
	{
		return p;
	}
	double3 getVelocity()
	{
		return v;
	}
	double getMass()
	{
		return mass;
	}
	double getInertia()
	{
		return inertia;
	}
	void setMass(double mass)
	{
		this->mass = mass;
	}
	void setInertia(double inertia)
	{
		this->inertia = inertia;
	}

	int addBody2D(int j);
};

#endif /* BODY2D_CUH_ */
