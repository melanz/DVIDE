/*
 * CollisionDetector.cuh
 *
 *  Created on: Sep 20, 2013
 *      Author: melanz
 */

#ifndef COLLISIONDETECTOR_CUH_
#define COLLISIONDETECTOR_CUH_

#include "include.cuh"
#include "System.cuh"

typedef thrust::pair<double3, double3> AABB;

// collision detection structures
struct AABBstruct {
  double3 min, max;
};

// collision detection structures
struct collision {
  double3 normal;
  double penetration;
  uint bodyB; // What is the other body that this is contacting?
};

// reduce a pair of bounding boxes (a,b) to a bounding box containing a and b
struct AABB_reduction: public thrust::binary_function<AABB, AABB, AABB> {
  AABB __host__ __device__ operator()(
      AABB a,
      AABB b) {
    double3 ll = make_double3(fmin(a.first.x, b.first.x), fmin(a.first.y, b.first.y), fmin(a.first.z, b.first.z)); // lower left corner
    double3 ur = make_double3(fmax(a.second.x, b.second.x), fmax(a.second.y, b.second.y), fmax(a.second.z, b.second.z)); // upper right corner
    return AABB(ll, ur);
  }
};

// convert a point to a AABB containing that point, (point) -> (point, point)
struct AABB_transformation: public thrust::unary_function<double3, AABB> {
  AABB __host__ __device__ operator()(
      double3 point) {
    return AABB(point, point);
  }
};

class System;
class CollisionDetector {
  friend class System;
private:
  System* system;

  // Data for spatial subdivision
  uint numAABB;
  thrust::device_vector<double3> aabbData_d;
  thrust::device_vector<uint> numBinsIntersected_d;
  double3 minBoundingPoint;
  double3 maxBoundingPoint;
  double3 globalOrigin;
  double3 binSizeInverse;
  uint3 binsPerAxis;
  uint totalBinIntersections;
  thrust::device_vector<uint> binIdentifier_d;
  thrust::device_vector<uint> aabbIdentifier_d;
  thrust::device_vector<uint> binStartIndex_d;
  uint lastActiveBin;
  thrust::device_vector<uint> numAabbCollisionsPerBin_d;
  uint numPossibleCollisions;
  thrust::device_vector<long long> potentialCollisions_d;
  thrust::device_vector<uint2> possibleCollisionPairs_d;
  // End spatial subdivision data

  // Data for parallel collision detection
  thrust::device_vector<uint> numCollisionsPerPair_d;
  thrust::device_vector<double4> normalsAndPenetrations_d;
  thrust::device_vector<uint> collisionStartIndex_d;
  uint lastActiveCollision;
  // End parallel collision detection data

  // TODO: Get rid of these vectors (needed for serial collision detection)
  thrust::host_vector<uint> bodyIdentifierA_h;
  thrust::host_vector<uint> bodyIdentifierB_h;
  thrust::host_vector<double4> normalsAndPenetrations_h;
  // TODO: Get rid of these vectors (needed for serial collision detection)

public:
  uint numCollisions;
  thrust::device_vector<uint> bodyIdentifierA_d;
  thrust::device_vector<uint> bodyIdentifierB_d;

	CollisionDetector(System* sys);
	int detectPossibleCollisions_spatialSubdivision();
	int detectPossibleCollisions_nSquared();
	int detectCollisions();
  int generateAxisAlignedBoundingBoxes();
  void setBinsPerAxis(uint3 binsPerAxis);
  int detectCollisions_CPU(); // TODO: Get rid of this function
  int exportSystem(string filename);

};

#endif /* COLLISIONDETECTOR_CUH_ */
