/*
 * PhysicsItem.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef PHYSICSITEM_CUH_
#define PHYSICSITEM_CUH_

#include "include.cuh"

class PhysicsItem {
  friend class System;
protected:
  uint identifier;
	uint index;
	int numDOF;
	int collisionFamily;
public:
	PhysicsItem() {
	  identifier = 0;
	  index = 0;
	  numDOF = 0;
	  collisionFamily = -1;
  }
  uint getIndex() {return index;}
	void setIndex(uint index) {this->index = index;}
  uint getIdentifier() {return identifier;}
  void setIdentifier(uint identifier) {this->identifier = identifier;}
  int getCollisionFamily() {return collisionFamily;}
  void setCollisionFamily(int collisionFamily) {this->collisionFamily = collisionFamily;}
};

#endif /* PHYSICSITEM_CUH_ */
