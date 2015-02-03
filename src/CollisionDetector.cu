#include <algorithm>
#include <vector>
#include "include.cuh"
#include "CollisionDetector.cuh"

void CollisionDetector::setBinsPerAxis(uint3 binsPerAxis) {
  this->binsPerAxis = binsPerAxis;
}

inline uint3 __device__ getHashMin(const double3 &A, const double3 & binSizeInverse) {
  uint3 temp;
  temp.x = A.x * binSizeInverse.x;
  temp.y = A.y * binSizeInverse.y;
  temp.z = A.z * binSizeInverse.z;

  return temp;
}

inline uint3 __device__ getHashMax(const double3 &A, const double3 & binSizeInverse) {
  uint3 temp;
  temp.x = A.x * binSizeInverse.x;
  temp.y = A.y * binSizeInverse.y;
  temp.z = A.z * binSizeInverse.z;

  return temp;
}

inline uint __device__ getHashIndex(const uint3 &A, const uint3 &binsPerAxis) {
  //return ((A.x * 73856093) ^ (A.y * 19349663) ^ (A.z * 83492791));
  return A.x+A.y*binsPerAxis.x+A.z*binsPerAxis.x*binsPerAxis.y;
}

__global__ void countAabbBinIntersections(double3* aabbData, uint* numBinsIntersected, double3 binSizeInverse, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numAABB);

  uint3 gmin = getHashMin(aabbData[index], binSizeInverse);
  uint3 gmax = getHashMax(aabbData[index + numAABB], binSizeInverse);
  //uint3 check = getHashMin(aabbData[index + numAABB]-aabbData[index],binSizeInverse);
  //check += make_uint3(1,1,1);
  //printf("AABB #[%d]: (%d-%d+1)*(%d-%d+1)*(%d-%d+1) or (%d*%d*%d)\n",index,gmax.x,gmin.x,gmax.y,gmin.y,gmax.z,gmin.z,check.x,check.y,check.z);
  numBinsIntersected[index] = (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
  //numBinsIntersected[index] = check.x*check.y*check.z;
}

__global__ void storeAabbBinIntersections(double3* aabbData, uint* numBinsIntersected, uint * binIdentifier, uint * aabbIdentifier, double3 binSizeInverse, uint3 binsPerAxis, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numAABB);

  uint count = 0, i, j, k;
  uint3 gmin = getHashMin(aabbData[index], binSizeInverse);
  uint3 gmax = getHashMax(aabbData[index + numAABB], binSizeInverse);
  uint mInd = (index == 0) ? 0 : numBinsIntersected[index - 1];

  for (i = gmin.x; i <= gmax.x; i++) {
    for (j = gmin.y; j <= gmax.y; j++) {
      for (k = gmin.z; k <= gmax.z; k++) {
        binIdentifier[mInd + count] = getHashIndex(make_uint3(i, j, k),binsPerAxis);
        aabbIdentifier[mInd + count] = index;
        count++;
      }
    }
  }
}

__global__ void countAabbAabbIntersections(double3* aabbData, uint * binIdentifier, uint * aabbIdentifier, uint * binStartIndex, uint* numAabbCollisionsPerBin, uint lastActiveBin, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, lastActiveBin);

  uint end = binStartIndex[index], count = 0, i = (!index) ? 0 : binStartIndex[index - 1];
  uint tempa, tempb;
  AABBstruct A, B;
  for (; i < end; i++) {
    tempa = aabbIdentifier[i];
    A.min = aabbData[tempa];
    A.max = aabbData[tempa + numAABB];
    for (int k = i + 1; k < end; k++) {
      tempb = aabbIdentifier[k];
      B.min = aabbData[tempb];
      B.max = aabbData[tempb + numAABB];
      bool inContact = (A.min.x <= B.max.x && B.min.x <= A.max.x) && (A.min.y <= B.max.y && B.min.y <= A.max.y) && (A.min.z <= B.max.z && B.min.z <= A.max.z);
      if (inContact) count++;
    }
  }
  numAabbCollisionsPerBin[index] = count;
}

__global__ void storeAabbAabbIntersections(double3* aabbData, uint * binIdentifier, uint * aabbIdentifier, uint * binStartIndex, uint* Num_ContactD, long long* potentialCollisions, uint lastActiveBin, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, lastActiveBin);

  uint end = binStartIndex[index], count = 0, i = (!index) ? 0 : binStartIndex[index - 1], Bin = binIdentifier[index];
  uint offset = (!index) ? 0 : Num_ContactD[index - 1];
  if (end - i == 1) {
    return;
  }
  uint tempa, tempb;
  AABBstruct A, B;
  for (; i < end; i++) {
    ;
    tempa = aabbIdentifier[i];
    A.min = aabbData[tempa];
    A.max = aabbData[tempa + numAABB];
    for (int k = i + 1; k < end; k++) {
      tempb = aabbIdentifier[k];
      B.min = aabbData[tempb];
      B.max = aabbData[tempb + numAABB];
      bool inContact = (A.min.x <= B.max.x && B.min.x <= A.max.x) && (A.min.y <= B.max.y && B.min.y <= A.max.y) && (A.min.z <= B.max.z && B.min.z <= A.max.z);
      if (inContact) {
        int a = tempa;
        int b = tempb;
        if (b < a) {
          int t = a;
          a = b;
          b = t;
        }
        potentialCollisions[offset + count] = ((long long) a << 32 | (long long) b); //the two indices of the objects that make up the contact
        count++;
      }
    }
  }
}

CollisionDetector::CollisionDetector(System* sys)
{
  system = sys;
  numAABB = 0;
  binsPerAxis = make_uint3(20,20,20);
  numPossibleCollisions = 0;
  totalBinIntersections = 0;
  lastActiveBin = 0;
  possibleCollisionPairs_d.clear();
  collisionPairs_d.clear();

  cudaFuncSetCacheConfig(countAabbBinIntersections, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(storeAabbBinIntersections, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(countAabbAabbIntersections, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(storeAabbAabbIntersections, cudaFuncCachePreferL1);
}

int CollisionDetector::detectPossibleCollisions_nSquared()
{
  // Perform n-squared collision detection, only needs to be called once!
  for(int i=0; i<system->bodies.size(); i++) {
    for(int j=i+1; j<system->bodies.size(); j++) {
      possibleCollisionPairs_h.push_back(make_uint2(i,j));
    }
  }
  possibleCollisionPairs_d = possibleCollisionPairs_h;

  return 0;
}

__global__ void generateAabbData(double3* aabbData, int* indices, double* position, double3* geometries, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numAABB);

  double3 pos = make_double3(position[indices[index]],position[indices[index]+1],position[indices[index]+2]);
  double3 geometry = geometries[index];
  if(geometry.y == 0) {
    // sphere case
    geometry = make_double3(geometry.x,geometry.x,geometry.x);
  }
  aabbData[index] = pos-geometry;
  aabbData[index + numAABB] = pos+geometry;
}

int CollisionDetector::generateAxisAlignedBoundingBoxes()
{
  aabbData_d.resize(2*system->bodies.size());
  generateAabbData<<<BLOCKS(system->bodies.size()),THREADS>>>(CASTD3(aabbData_d), CASTI1(system->indices_d), CASTD1(system->p_d), CASTD3(system->contactGeometry_d), system->bodies.size());

  return 0;
}

__global__ void convertLongsToInts(long long* potentialCollisions, uint2 * possibleCollisionPairs, uint numPossibleCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numPossibleCollisions);

  possibleCollisionPairs[index].x = int(potentialCollisions[index] >> 32);
  possibleCollisionPairs[index].y = int(potentialCollisions[index] & 0xffffffff);
}

int CollisionDetector::detectPossibleCollisions_spatialSubdivision()
{
  // Step 1: Initialize
  numAABB = aabbData_d.size()*0.5;
  possibleCollisionPairs_d.clear();
  // End Step 1

  // Step 2: Determine the bounds on the total space and subdivide based on the bins per axis
  double3 first = aabbData_d[0];//make_double3(0,0,0);
  AABB init = AABB(first, first); // create a zero volume AABB
  AABB_transformation unary_op;
  AABB_reduction binary_op;
  AABB result = thrust::transform_reduce(aabbData_d.begin(), aabbData_d.end(), unary_op, init, binary_op);
  minBoundingPoint = result.first-make_double3(0.01,0.01,0.01);
  maxBoundingPoint = result.second+make_double3(0.01,0.01,0.01);
  globalOrigin = minBoundingPoint;

  binSizeInverse.x = ((double)binsPerAxis.x)/fabs(maxBoundingPoint.x - minBoundingPoint.x);
  binSizeInverse.y = ((double)binsPerAxis.y)/fabs(maxBoundingPoint.y - minBoundingPoint.y);
  binSizeInverse.z = ((double)binsPerAxis.z)/fabs(maxBoundingPoint.z - minBoundingPoint.z);

  thrust::transform(aabbData_d.begin(), aabbData_d.end(), thrust::constant_iterator<double3>(globalOrigin), aabbData_d.begin(), thrust::minus<double3>());
  // End Step 2

  // Step 3: Count the number of AABB's that lie in each bin, allocate space for each AABB
  numBinsIntersected_d.resize(numAABB);

  // need to figure out how many bins each AABB intersects
  countAabbBinIntersections<<<BLOCKS(numAABB),THREADS>>>(CASTD3(aabbData_d), CASTU1(numBinsIntersected_d), binSizeInverse, numAABB);

  // need to use an inclusive scan to figure out where each thread should start entering the bin that each AABB is in (also counts total bin intersections)
  Thrust_Inclusive_Scan_Sum(numBinsIntersected_d, totalBinIntersections);

  binIdentifier_d.resize(totalBinIntersections);
  aabbIdentifier_d.resize(totalBinIntersections);
  binStartIndex_d.resize(totalBinIntersections);
  // End Step 3

  // Step 4: Indicate what bin each AABB belongs to, then sort based on bin number
  storeAabbBinIntersections<<<BLOCKS(numAABB),THREADS>>>(CASTD3(aabbData_d), CASTU1(numBinsIntersected_d), CASTU1(binIdentifier_d), CASTU1(aabbIdentifier_d), binSizeInverse, binsPerAxis, numAABB);

  // After figuring out which bin each AABB belongs to, sort the AABB's based on bin number
  Thrust_Sort_By_Key(binIdentifier_d, aabbIdentifier_d);

  // Next, count the number of AABB's that each bin has (this destroys the information in binIdentifier and puts it into aabbIdentifier)
  Thrust_Reduce_By_KeyA(lastActiveBin, binIdentifier_d, binStartIndex_d);

  binStartIndex_d.resize(lastActiveBin);

  // reduce the # of AABB's per bin to create a library so a thread knows where each bin starts and ends
  Thrust_Inclusive_Scan(binStartIndex_d);

  numAabbCollisionsPerBin_d.resize(lastActiveBin);
  // End Step 4

  // Step 5: Count the number of AABB collisions
  // At this point, binIdentifier has the bin number for each thread, binStartIndex tells the thread where to start and stop, and aabbIdentifier has the AABB that is in the bin
  countAabbAabbIntersections<<<BLOCKS(lastActiveBin),THREADS>>>(CASTD3(aabbData_d), CASTU1(binIdentifier_d), CASTU1(aabbIdentifier_d), CASTU1(binStartIndex_d), CASTU1(numAabbCollisionsPerBin_d), lastActiveBin, numAABB);

  Thrust_Inclusive_Scan_Sum(numAabbCollisionsPerBin_d, numPossibleCollisions);
  potentialCollisions_d.resize(numPossibleCollisions);
  // End Step 5

  // Step 6: Store the possible AABB collision pairs
  storeAabbAabbIntersections<<<BLOCKS(lastActiveBin),THREADS>>>(CASTD3(aabbData_d), CASTU1(binIdentifier_d), CASTU1(aabbIdentifier_d), CASTU1(binStartIndex_d), CASTU1(numAabbCollisionsPerBin_d), CASTLL(potentialCollisions_d), lastActiveBin, numAABB);
  thrust::sort(potentialCollisions_d.begin(), potentialCollisions_d.end());
  numPossibleCollisions = thrust::unique(potentialCollisions_d.begin(), potentialCollisions_d.end()) - potentialCollisions_d.begin();
  // End Step 6

  // Step 7: Convert long long potentialCollisions_d to int2 possibleCollisionPairs_d
  possibleCollisionPairs_d.resize(numPossibleCollisions);
  convertLongsToInts<<<BLOCKS(numPossibleCollisions),THREADS>>>(CASTLL(potentialCollisions_d), CASTU2(possibleCollisionPairs_d), numPossibleCollisions);
  // End Step 7

  return 0;
}

int CollisionDetector::detectCollisions()
{
  //TODO: Perform in parallel
  possibleCollisionPairs_h = possibleCollisionPairs_d; // need to do this in case we use spatial subdivision
  collisionPairs_h.clear();
  normals_h.clear();
  penetrations_h.clear();

  for(int i=0; i<numPossibleCollisions; i++) {
    int bodyA = possibleCollisionPairs_h[i].x;
    int bodyB = possibleCollisionPairs_h[i].y;

    // Both spheres
    if(system->contactGeometry_h[bodyA].y == 0 && system->contactGeometry_h[bodyB].y == 0) {
      double3 posA = make_double3(system->p_h[system->indices_h[bodyA]],system->p_h[system->indices_h[bodyA]+1],system->p_h[system->indices_h[bodyA]+2]);
      double3 posB = make_double3(system->p_h[system->indices_h[bodyB]],system->p_h[system->indices_h[bodyB]+1],system->p_h[system->indices_h[bodyB]+2]);
      double3 normal = normalize(posB-posA); // from A to B!
      double penetration = (system->contactGeometry_h[bodyA].x+system->contactGeometry_h[bodyB].x) - length(posB-posA);
      if(penetration>0) {
        collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
        normals_h.push_back(normal);
        penetrations_h.push_back(penetration);
      }
    }

    // A = Sphere, B = Box
    else if(system->contactGeometry_h[bodyA].y == 0 && system->contactGeometry_h[bodyB].y != 0) {
      double3 posA = make_double3(system->p_h[system->indices_h[bodyA]],system->p_h[system->indices_h[bodyA]+1],system->p_h[system->indices_h[bodyA]+2]);
      double3 posB = make_double3(system->p_h[system->indices_h[bodyB]],system->p_h[system->indices_h[bodyB]+1],system->p_h[system->indices_h[bodyB]+2]);

      // check x-face
      if((posA.y>=(posB.y-system->contactGeometry_h[bodyB].y) && posA.y<=(posB.y+system->contactGeometry_h[bodyB].y)) && (posA.z>=(posB.z-system->contactGeometry_h[bodyB].z) && posA.z<=(posB.z+system->contactGeometry_h[bodyB].z)))
      {
        double3 normal = make_double3(posB.x-posA.x,0,0);
        double penetration = (system->contactGeometry_h[bodyB].x + system->contactGeometry_h[bodyA].x) - fabs(posB.x-posA.x);
        if(penetration>0) {
          collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
          normals_h.push_back(normalize(normal));
          penetrations_h.push_back(penetration);
        }
      }

      // check y
      else if((posA.x>=(posB.x-system->contactGeometry_h[bodyB].x) && posA.x<=(posB.x+system->contactGeometry_h[bodyB].x)) && (posA.z>=(posB.z-system->contactGeometry_h[bodyB].z) && posA.z<=(posB.z+system->contactGeometry_h[bodyB].z)))
      {
        double3 normal = make_double3(0,posB.y-posA.y,0);
        double penetration = (system->contactGeometry_h[bodyB].y + system->contactGeometry_h[bodyA].x) - fabs(posB.y-posA.y);
        if(penetration>0) {
          collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
          normals_h.push_back(normalize(normal));
          penetrations_h.push_back(penetration);
        }
      }

      // check z
      else if((posA.x>=(posB.x-system->contactGeometry_h[bodyB].x) && posA.x<=(posB.x+system->contactGeometry_h[bodyB].x)) && (posA.y>=(posB.y-system->contactGeometry_h[bodyB].y) && posA.y<=(posB.y+system->contactGeometry_h[bodyB].y)))
      {
        double3 normal = make_double3(0,0,posB.z-posA.z);
        double penetration = (system->contactGeometry_h[bodyB].z + system->contactGeometry_h[bodyA].x) - fabs(posB.z-posA.z);
        if(penetration>0) {
          collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
          normals_h.push_back(normalize(normal));
          penetrations_h.push_back(penetration);
        }
      }

    }

    // A = Box, B = Sphere
    else if(system->contactGeometry_h[bodyA].y != 0 && system->contactGeometry_h[bodyB].y == 0) {
      double3 posA = make_double3(system->p_h[system->indices_h[bodyA]],system->p_h[system->indices_h[bodyA]+1],system->p_h[system->indices_h[bodyA]+2]);
      double3 posB = make_double3(system->p_h[system->indices_h[bodyB]],system->p_h[system->indices_h[bodyB]+1],system->p_h[system->indices_h[bodyB]+2]);

      // check x-face
      if((posB.y>=(posA.y-system->contactGeometry_h[bodyA].y) && posB.y<=(posA.y+system->contactGeometry_h[bodyA].y)) && (posB.z>=(posA.z-system->contactGeometry_h[bodyA].z) && posB.z<=(posA.z+system->contactGeometry_h[bodyA].z)))
      {
        double3 normal = make_double3(posB.x-posA.x,0,0);
        double penetration = (system->contactGeometry_h[bodyB].x + system->contactGeometry_h[bodyA].x) - fabs(posB.x-posA.x);
        if(penetration>0) {
          collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
          normals_h.push_back(normalize(normal));
          penetrations_h.push_back(penetration);
        }
      }

      // check y
      else if((posB.x>=(posA.x-system->contactGeometry_h[bodyA].x) && posB.x<=(posA.x+system->contactGeometry_h[bodyA].x)) && (posB.z>=(posA.z-system->contactGeometry_h[bodyA].z) && posB.z<=(posA.z+system->contactGeometry_h[bodyA].z)))
      {
        double3 normal = make_double3(0,posB.y-posA.y,0);
        double penetration = (system->contactGeometry_h[bodyB].x + system->contactGeometry_h[bodyA].y) - fabs(posB.y-posA.y);
        if(penetration>0) {
          collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
          normals_h.push_back(normalize(normal));
          penetrations_h.push_back(penetration);
        }
      }

      // check z
      else if((posB.x>=(posA.x-system->contactGeometry_h[bodyA].x) && posB.x<=(posA.x+system->contactGeometry_h[bodyA].x)) && (posB.y>=(posA.y-system->contactGeometry_h[bodyA].y) && posB.y<=(posA.y+system->contactGeometry_h[bodyA].y)))
      {
        double3 normal = make_double3(0,0,posB.z-posA.z);
        double penetration = (system->contactGeometry_h[bodyB].x + system->contactGeometry_h[bodyA].z) - fabs(posB.z-posA.z);
        if(penetration>0) {
          collisionPairs_h.push_back(make_uint2(bodyA,bodyB));
          normals_h.push_back(normalize(normal));
          penetrations_h.push_back(penetration);
        }
      }
    }
  }
  collisionPairs_d = collisionPairs_h;
  normals_d = normals_h;
  penetrations_d = penetrations_h;

  return 0;
}
