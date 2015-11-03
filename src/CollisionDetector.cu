#include <algorithm>
#include <vector>
#include "include.cuh"
#include "CollisionDetector.cuh"

void CollisionDetector::setBinsPerAxis(uint3 binsPerAxis) {
  this->binsPerAxis = binsPerAxis;
}

inline uint3 __device__ getHash(const double3 &A, const double3 & binSizeInverse) {
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

__global__ void generateAabbData(double3* aabbData, int* indices, double* pos, double3* geometries, double3* collisionGeometries, int4* map, double envelope, int numBodies, int numBeams, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numAABB);

  int identifier = map[index].x;
  double3 position; // the position of the collision geometry must be calculated differently for different physics items
  if(identifier<numBodies) {
    position = make_double3(pos[indices[identifier]],pos[indices[identifier]+1],pos[indices[identifier]+2]);
  }
  else if (identifier < (numBeams+numBodies)) {
    double xi = static_cast<double>(map[index].y)/(static_cast<double>(geometries[identifier].z-1));
    double l = geometries[identifier].y;
    int offset = indices[identifier];
    position.x = pos[offset]*(2*xi*xi*xi - 3*xi*xi + 1) + pos[6+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*pos[3+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*pos[9+offset]*(- xi*xi*xi + xi*xi);
    position.y = pos[1+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + pos[7+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*pos[4+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*pos[10+offset]*(- xi*xi*xi + xi*xi);
    position.z = pos[2+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + pos[8+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*pos[5+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*pos[11+offset]*(- xi*xi*xi + xi*xi);
  }
  else {
    double xi = static_cast<double>(map[index].y)/(static_cast<double>(geometries[identifier].z-1));
    double eta = static_cast<double>(map[index].z)/(static_cast<double>(geometries[identifier].z-1));
    double a = geometries[identifier].x;
    double b = geometries[identifier].y;
    int offset = indices[identifier];
    position.x = -eta*pos[18+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-pos[9+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*pos[27+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+pos[0+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*pos[15+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*pos[24+offset]*xi*(eta-1.0)+a*eta*pos[21+offset]*(xi*xi)*(xi-1.0)+a*eta*pos[30+offset]*xi*pow(xi-1.0,2.0)-b*eta*pos[6+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*pos[33+offset]*(eta-1.0)*(xi-1.0)-a*pos[3+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*pos[12+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    position.y = -eta*pos[19+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-pos[10+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*pos[28+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+pos[1+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*pos[16+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*pos[25+offset]*xi*(eta-1.0)+a*eta*pos[22+offset]*(xi*xi)*(xi-1.0)+a*eta*pos[31+offset]*xi*pow(xi-1.0,2.0)-b*eta*pos[7+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*pos[34+offset]*(eta-1.0)*(xi-1.0)-a*pos[4+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*pos[13+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    position.z = -eta*pos[20+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-pos[11+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*pos[29+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+pos[2+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*pos[17+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*pos[26+offset]*xi*(eta-1.0)+a*eta*pos[23+offset]*(xi*xi)*(xi-1.0)+a*eta*pos[32+offset]*xi*pow(xi-1.0,2.0)-b*eta*pos[8+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*pos[35+offset]*(eta-1.0)*(xi-1.0)-a*pos[5+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*pos[14+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
  }

  double3 geometry = collisionGeometries[index];
  if(geometry.y == 0) {
    // sphere case
    geometry = make_double3(geometry.x,geometry.x,geometry.x);
  }
  geometry += make_double3(envelope,envelope,envelope);
  aabbData[index] = position-geometry;
  aabbData[index + numAABB] = position+geometry;
}

__global__ void countAabbBinIntersections(double3* aabbData, uint* numBinsIntersected, double3 binSizeInverse, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numAABB);

  uint3 gmin = getHash(aabbData[index], binSizeInverse);
  uint3 gmax = getHash(aabbData[index + numAABB], binSizeInverse);
  //uint3 check = getHashMin(aabbData[index + numAABB]-aabbData[index],binSizeInverse);
  //check += make_uint3(1,1,1);
  //printf("AABB #[%d]: (%d-%d+1)*(%d-%d+1)*(%d-%d+1) or (%d*%d*%d)\n",index,gmax.x,gmin.x,gmax.y,gmin.y,gmax.z,gmin.z,check.x,check.y,check.z);
  numBinsIntersected[index] = (gmax.x - gmin.x + 1) * (gmax.y - gmin.y + 1) * (gmax.z - gmin.z + 1);
  //numBinsIntersected[index] = check.x*check.y*check.z;
}

__global__ void storeAabbBinIntersections(double3* aabbData, uint* numBinsIntersected, uint * binIdentifier, uint * aabbIdentifier, double3 binSizeInverse, uint3 binsPerAxis, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numAABB);

  uint count = 0, i, j, k;
  uint3 gmin = getHash(aabbData[index], binSizeInverse);
  uint3 gmax = getHash(aabbData[index + numAABB], binSizeInverse);
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

__global__ void countAabbAabbIntersections(int4* collisionMap, double3* aabbData, uint * binIdentifier, uint * aabbIdentifier, uint * binStartIndex, uint* numAabbCollisionsPerBin, uint lastActiveBin, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, lastActiveBin);

  uint end = binStartIndex[index], count = 0, i = (!index) ? 0 : binStartIndex[index - 1];
  uint tempa, tempb;
  AABBstruct A, B;
  for (; i < end; i++) {
    tempa = aabbIdentifier[i];
    A.min = aabbData[tempa];
    A.max = aabbData[tempa + numAABB];
    int identifierA = collisionMap[tempa].x;
    int collisionFamilyA = collisionMap[tempa].w;
    for (int k = i + 1; k < end; k++) {
      tempb = aabbIdentifier[k];
      B.min = aabbData[tempb];
      B.max = aabbData[tempb + numAABB];
      bool inContact = (A.min.x <= B.max.x && B.min.x <= A.max.x) && (A.min.y <= B.max.y && B.min.y <= A.max.y) && (A.min.z <= B.max.z && B.min.z <= A.max.z);
      int identifierB = collisionMap[tempb].x;
      int collisionFamilyB = collisionMap[tempb].w;
      if(identifierA==identifierB) inContact = false;
      if(collisionFamilyA==collisionFamilyB && collisionFamilyA!=-1) inContact = false;
      if (inContact) count++;
    }
  }
  numAabbCollisionsPerBin[index] = count;
}

__global__ void storeAabbAabbIntersections(int4* collisionMap, double3* aabbData, uint * binIdentifier, uint * aabbIdentifier, uint * binStartIndex, uint* numAabbCollisionsPerBin, long long* potentialCollisions, uint lastActiveBin, uint numAABB) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, lastActiveBin);

  uint end = binStartIndex[index], count = 0, i = (!index) ? 0 : binStartIndex[index - 1], Bin = binIdentifier[index];
  uint offset = (!index) ? 0 : numAabbCollisionsPerBin[index - 1];
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
    int identifierA = collisionMap[tempa].x;
    int collisionFamilyA = collisionMap[tempa].w;
    for (int k = i + 1; k < end; k++) {
      tempb = aabbIdentifier[k];
      B.min = aabbData[tempb];
      B.max = aabbData[tempb + numAABB];
      bool inContact = (A.min.x <= B.max.x && B.min.x <= A.max.x) && (A.min.y <= B.max.y && B.min.y <= A.max.y) && (A.min.z <= B.max.z && B.min.z <= A.max.z);
      int identifierB = collisionMap[tempb].x;
      int collisionFamilyB = collisionMap[tempb].w;
      if(identifierA==identifierB) inContact = false;
      if(collisionFamilyA==collisionFamilyB && collisionFamilyA!=-1) inContact = false;
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

__global__ void convertLongsToInts(long long* potentialCollisions, uint2 * possibleCollisionPairs, uint numPossibleCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numPossibleCollisions);

  possibleCollisionPairs[index].x = int(potentialCollisions[index] >> 32);
  possibleCollisionPairs[index].y = int(potentialCollisions[index] & 0xffffffff);
}

__global__ void countActualCollisions(uint* numCollisionsPerPair, uint2* possibleCollisionPairs, double* p, int* indices, double3* geometries, double3* collisionGeometries, int4* map, double envelope, int numBodies, int numBeams, uint numPossibleCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numPossibleCollisions);

  double penetration = 0;
  int numCollisions = 0;

  int collGeomA = possibleCollisionPairs[index].x;
  int collGeomB = possibleCollisionPairs[index].y;

  int identifierA = map[collGeomA].x;
  double3 posA; // the position of the collision geometry must be calculated differently for different physics items
  if(identifierA<numBodies) {
    posA = make_double3(p[indices[identifierA]],p[indices[identifierA]+1],p[indices[identifierA]+2]);
  }
  else if(identifierA<numBodies+numBeams) {
    double xi = static_cast<double>(map[collGeomA].y)/(static_cast<double>(geometries[identifierA].z-1));
    double l = geometries[identifierA].y;
    int offset = indices[identifierA];
    posA.x = p[offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[6+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[3+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[9+offset]*(- xi*xi*xi + xi*xi);
    posA.y = p[1+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[7+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[4+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[10+offset]*(- xi*xi*xi + xi*xi);
    posA.z = p[2+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[8+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[5+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[11+offset]*(- xi*xi*xi + xi*xi);
  }
  else {
    double xi = static_cast<double>(map[collGeomA].y)/(static_cast<double>(geometries[identifierA].z-1));
    double eta = static_cast<double>(map[collGeomA].z)/(static_cast<double>(geometries[identifierA].z-1));
    double a = geometries[identifierA].x;
    double b = geometries[identifierA].y;
    int offset = indices[identifierA];
    posA.x = -eta*p[18+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[9+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[27+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[0+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[15+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[24+offset]*xi*(eta-1.0)+a*eta*p[21+offset]*(xi*xi)*(xi-1.0)+a*eta*p[30+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[6+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[33+offset]*(eta-1.0)*(xi-1.0)-a*p[3+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[12+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    posA.y = -eta*p[19+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[10+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[28+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[1+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[16+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[25+offset]*xi*(eta-1.0)+a*eta*p[22+offset]*(xi*xi)*(xi-1.0)+a*eta*p[31+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[7+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[34+offset]*(eta-1.0)*(xi-1.0)-a*p[4+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[13+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    posA.z = -eta*p[20+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[11+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[29+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[2+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[17+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[26+offset]*xi*(eta-1.0)+a*eta*p[23+offset]*(xi*xi)*(xi-1.0)+a*eta*p[32+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[8+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[35+offset]*(eta-1.0)*(xi-1.0)-a*p[5+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[14+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
  }

  int identifierB = map[collGeomB].x;
  double3 posB; // the position of the collision geometry must be calculated differently for different physics items
  if(identifierB<numBodies) {
    posB = make_double3(p[indices[identifierB]],p[indices[identifierB]+1],p[indices[identifierB]+2]);
  }
  else if(identifierB<numBodies+numBeams) {
    double xi = static_cast<double>(map[collGeomB].y)/(static_cast<double>(geometries[identifierB].z-1));
    double l = geometries[identifierB].y;
    int offset = indices[identifierB];
    posB.x = p[offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[6+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[3+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[9+offset]*(- xi*xi*xi + xi*xi);
    posB.y = p[1+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[7+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[4+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[10+offset]*(- xi*xi*xi + xi*xi);
    posB.z = p[2+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[8+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[5+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[11+offset]*(- xi*xi*xi + xi*xi);
  }
  else {
    double xi = static_cast<double>(map[collGeomB].y)/(static_cast<double>(geometries[identifierB].z-1));
    double eta = static_cast<double>(map[collGeomB].z)/(static_cast<double>(geometries[identifierB].z-1));
    double a = geometries[identifierB].x;
    double b = geometries[identifierB].y;
    int offset = indices[identifierB];
    posB.x = -eta*p[18+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[9+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[27+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[0+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[15+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[24+offset]*xi*(eta-1.0)+a*eta*p[21+offset]*(xi*xi)*(xi-1.0)+a*eta*p[30+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[6+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[33+offset]*(eta-1.0)*(xi-1.0)-a*p[3+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[12+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    posB.y = -eta*p[19+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[10+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[28+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[1+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[16+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[25+offset]*xi*(eta-1.0)+a*eta*p[22+offset]*(xi*xi)*(xi-1.0)+a*eta*p[31+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[7+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[34+offset]*(eta-1.0)*(xi-1.0)-a*p[4+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[13+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    posB.z = -eta*p[20+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[11+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[29+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[2+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[17+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[26+offset]*xi*(eta-1.0)+a*eta*p[23+offset]*(xi*xi)*(xi-1.0)+a*eta*p[32+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[8+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[35+offset]*(eta-1.0)*(xi-1.0)-a*p[5+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[14+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
  }

  double3 geometryA = collisionGeometries[collGeomA];
  double3 geometryB = collisionGeometries[collGeomB];

  if(geometryA.y == 0 && geometryB.y == 0) {
    // sphere-sphere case
    penetration = (geometryA.x+geometryB.x) - length(posB-posA);
    if(penetration>=-envelope) numCollisions++;
  }

  else if((geometryA.y != 0 && geometryB.y == 0) || (geometryA.y == 0 && geometryB.y != 0)) {
    // box-sphere case
    double dmin = 0;
    double r = geometryB.x;
    double3 center = posB;
    double3 bmin = posA-geometryA;
    double3 bmax = posA+geometryA;
    if (geometryA.y == 0 && geometryB.y != 0) {
      // sphere-box case
      r = geometryA.x;
      center = posA;
      bmin = posB-geometryB;
      bmax = posB+geometryB;
    }

    if (center.x < bmin.x) {
        dmin += pow(center.x - bmin.x, 2.0);
    } else if (center.x > bmax.x) {
        dmin += pow(center.x - bmax.x, 2.0);
    }

    if (center.y < bmin.y) {
        dmin += pow(center.y - bmin.y, 2.0);
    } else if (center.y > bmax.y) {
        dmin += pow(center.y - bmax.y, 2.0);
    }

    if (center.z < bmin.z) {
        dmin += pow(center.z - bmin.z, 2.0);
    } else if (center.z > bmax.z) {
        dmin += pow(center.z - bmax.z, 2.0);
    }

    penetration = r-sqrt(dmin);
    if(penetration>=-envelope) numCollisions++;
  }

  numCollisionsPerPair[index] = numCollisions;
}

__global__ void storeActualCollisions(uint* numCollisionsPerPair, uint2* possibleCollisionPairs, double* p, int* indices, double3* geometries, double3* collisionGeometries, int4* map, double4* normalsAndPenetrations, uint* collisionIdentifiersA, uint* collisionIdentifiersB, uint numPossibleCollisions, int numBodies, int numBeams, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numPossibleCollisions);

  uint startIndex = (index == 0) ? 0 : numCollisionsPerPair[index - 1];
  uint endIndex = numCollisionsPerPair[index];

  int count = 0;
  for (int i = startIndex; i < endIndex; i++) {
    int collGeomA = possibleCollisionPairs[index].x;
    int collGeomB = possibleCollisionPairs[index].y;

    int identifierA = map[collGeomA].x;
    double3 posA; // the position of the collision geometry must be calculated differently for different physics items
    if(identifierA<numBodies) {
      posA = make_double3(p[indices[identifierA]],p[indices[identifierA]+1],p[indices[identifierA]+2]);
    }
    else if(identifierA<numBodies+numBeams) {
      double xi = static_cast<double>(map[collGeomA].y)/(static_cast<double>(geometries[identifierA].z-1));
      double l = geometries[identifierA].y;
      int offset = indices[identifierA];
      posA.x = p[offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[6+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[3+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[9+offset]*(- xi*xi*xi + xi*xi);
      posA.y = p[1+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[7+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[4+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[10+offset]*(- xi*xi*xi + xi*xi);
      posA.z = p[2+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[8+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[5+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[11+offset]*(- xi*xi*xi + xi*xi);
    }
    else {
      double xi = static_cast<double>(map[collGeomA].y)/(static_cast<double>(geometries[identifierA].z-1));
      double eta = static_cast<double>(map[collGeomA].z)/(static_cast<double>(geometries[identifierA].z-1));
      double a = geometries[identifierA].x;
      double b = geometries[identifierA].y;
      int offset = indices[identifierA];
      posA.x = -eta*p[18+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[9+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[27+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[0+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[15+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[24+offset]*xi*(eta-1.0)+a*eta*p[21+offset]*(xi*xi)*(xi-1.0)+a*eta*p[30+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[6+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[33+offset]*(eta-1.0)*(xi-1.0)-a*p[3+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[12+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
      posA.y = -eta*p[19+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[10+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[28+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[1+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[16+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[25+offset]*xi*(eta-1.0)+a*eta*p[22+offset]*(xi*xi)*(xi-1.0)+a*eta*p[31+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[7+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[34+offset]*(eta-1.0)*(xi-1.0)-a*p[4+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[13+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
      posA.z = -eta*p[20+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[11+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[29+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[2+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[17+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[26+offset]*xi*(eta-1.0)+a*eta*p[23+offset]*(xi*xi)*(xi-1.0)+a*eta*p[32+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[8+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[35+offset]*(eta-1.0)*(xi-1.0)-a*p[5+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[14+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    }

    int identifierB = map[collGeomB].x;
    double3 posB; // the position of the collision geometry must be calculated differently for different physics items
    if(identifierB<numBodies) {
      posB = make_double3(p[indices[identifierB]],p[indices[identifierB]+1],p[indices[identifierB]+2]);
    }
    else if(identifierB<numBodies+numBeams) {
      double xi = static_cast<double>(map[collGeomB].y)/(static_cast<double>(geometries[identifierB].z-1));
      double l = geometries[identifierB].y;
      int offset = indices[identifierB];
      posB.x = p[offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[6+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[3+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[9+offset]*(- xi*xi*xi + xi*xi);
      posB.y = p[1+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[7+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[4+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[10+offset]*(- xi*xi*xi + xi*xi);
      posB.z = p[2+offset]*(2*xi*xi*xi - 3*xi*xi + 1) + p[8+offset]*(-2*xi*xi*xi + 3*xi*xi) + l*p[5+offset]*(xi*xi*xi - 2*xi*xi + xi) - l*p[11+offset]*(- xi*xi*xi + xi*xi);
    }
    else {
      double xi = static_cast<double>(map[collGeomB].y)/(static_cast<double>(geometries[identifierB].z-1));
      double eta = static_cast<double>(map[collGeomB].z)/(static_cast<double>(geometries[identifierB].z-1));
      double a = geometries[identifierB].x;
      double b = geometries[identifierB].y;
      int offset = indices[identifierB];
      posB.x = -eta*p[18+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[9+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[27+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[0+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[15+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[24+offset]*xi*(eta-1.0)+a*eta*p[21+offset]*(xi*xi)*(xi-1.0)+a*eta*p[30+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[6+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[33+offset]*(eta-1.0)*(xi-1.0)-a*p[3+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[12+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
      posB.y = -eta*p[19+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[10+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[28+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[1+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[16+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[25+offset]*xi*(eta-1.0)+a*eta*p[22+offset]*(xi*xi)*(xi-1.0)+a*eta*p[31+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[7+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[34+offset]*(eta-1.0)*(xi-1.0)-a*p[4+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[13+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
      posB.z = -eta*p[20+offset]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p[11+offset]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p[29+offset]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p[2+offset]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p[17+offset]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p[26+offset]*xi*(eta-1.0)+a*eta*p[23+offset]*(xi*xi)*(xi-1.0)+a*eta*p[32+offset]*xi*pow(xi-1.0,2.0)-b*eta*p[8+offset]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p[35+offset]*(eta-1.0)*(xi-1.0)-a*p[5+offset]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p[14+offset]*(xi*xi)*(eta-1.0)*(xi-1.0);
    }

    double3 geometryA = collisionGeometries[collGeomA];
    double3 geometryB = collisionGeometries[collGeomB];

    double3 normal;
    normal.x = 1;
    normal.y = 0;
    normal.z = 0;
    double penetration = 0;

    if(geometryA.y == 0 && geometryB.y == 0) {
      // sphere-sphere case
      penetration = (geometryA.x+geometryB.x) - length(posB-posA);
      normal = normalize(posB-posA); // from A to B!
    }

    else if((geometryA.y != 0 && geometryB.y == 0) || (geometryA.y == 0 && geometryB.y != 0)) {
      // box-sphere case
      double dmin = 0;
      double r = geometryB.x;
      double3 center = posB;
      double3 bmin = posA-geometryA;
      double3 bmax = posA+geometryA;
      if (geometryA.y == 0 && geometryB.y != 0) {
        // sphere-box case
        r = geometryA.x;
        center = posA;
        bmin = posB-geometryB;
        bmax = posB+geometryB;
      }
      normal = make_double3(0,0,0);

      if (center.x < bmin.x) {
          dmin += pow(center.x - bmin.x, 2.0);
          normal+=make_double3(center.x - bmin.x,0,0);
      } else if (center.x > bmax.x) {
          dmin += pow(center.x - bmax.x, 2.0);
          normal+=make_double3(center.x - bmax.x,0,0);
      }

      if (center.y < bmin.y) {
          dmin += pow(center.y - bmin.y, 2.0);
          normal+=make_double3(0,center.y - bmin.y,0);
      } else if (center.y > bmax.y) {
          dmin += pow(center.y - bmax.y, 2.0);
          normal+=make_double3(0,center.y - bmax.y,0);
      }

      if (center.z < bmin.z) {
          dmin += pow(center.z - bmin.z, 2.0);
          normal+=make_double3(0,0,center.z - bmin.z);
      } else if (center.z > bmax.z) {
          dmin += pow(center.z - bmax.z, 2.0);
          normal+=make_double3(0,0,center.z - bmax.z);
      }

      normal = normalize(normal);
      if (geometryA.y == 0 && geometryB.y != 0) normal = -normal;
      penetration = r-sqrt(dmin);
    }

    collisionIdentifiersA[i] = collGeomA;
    collisionIdentifiersB[i] = collGeomB;
    normalsAndPenetrations[i] = make_double4(-normal.x,-normal.y,-normal.z,-penetration); // from B to A!
    count++;
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
  numCollisions = 0;
  lastActiveCollision = 0;
  envelope = 0.001;

  cudaFuncSetCacheConfig(countAabbBinIntersections, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(storeAabbBinIntersections, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(countAabbAabbIntersections, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(storeAabbAabbIntersections, cudaFuncCachePreferL1);

  cudaFuncSetCacheConfig(countActualCollisions, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig(storeActualCollisions, cudaFuncCachePreferL1);
}

int CollisionDetector::setEnvelope(double envelope)
{
  this->envelope = envelope;

  return 0;
}

int CollisionDetector::detectPossibleCollisions_nSquared()
{
  thrust::host_vector<uint2> possibleCollisionPairs_h;
  // Perform n-squared collision detection, only needs to be called once!
  for(int i=0; i<system->bodies.size(); i++) {
    for(int j=i+1; j<system->bodies.size(); j++) {
      possibleCollisionPairs_h.push_back(make_uint2(i,j));
    }
  }
  numPossibleCollisions = possibleCollisionPairs_h.size();
  possibleCollisionPairs_d = possibleCollisionPairs_h;

  return 0;
}

int CollisionDetector::generateAxisAlignedBoundingBoxes()
{
  aabbData_d.resize(2*system->collisionGeometry_d.size());
  generateAabbData<<<BLOCKS(system->collisionGeometry_d.size()),THREADS>>>(CASTD3(aabbData_d), CASTI1(system->indices_d), CASTD1(system->p_d), CASTD3(system->contactGeometry_d), CASTD3(system->collisionGeometry_d), CASTI4(system->collisionMap_d), envelope, system->bodies.size(), system->beams.size(), system->collisionGeometry_d.size());

  return 0;
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
  minBoundingPoint = result.first -make_double3(0.01,0.01,0.01);
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
  countAabbAabbIntersections<<<BLOCKS(lastActiveBin),THREADS>>>(CASTI4(system->collisionMap_d), CASTD3(aabbData_d), CASTU1(binIdentifier_d), CASTU1(aabbIdentifier_d), CASTU1(binStartIndex_d), CASTU1(numAabbCollisionsPerBin_d), lastActiveBin, numAABB);

  Thrust_Inclusive_Scan_Sum(numAabbCollisionsPerBin_d, numPossibleCollisions);
  potentialCollisions_d.resize(numPossibleCollisions);
  // End Step 5

  // Step 6: Store the possible AABB collision pairs
  storeAabbAabbIntersections<<<BLOCKS(lastActiveBin),THREADS>>>(CASTI4(system->collisionMap_d), CASTD3(aabbData_d), CASTU1(binIdentifier_d), CASTU1(aabbIdentifier_d), CASTU1(binStartIndex_d), CASTU1(numAabbCollisionsPerBin_d), CASTLL(potentialCollisions_d), lastActiveBin, numAABB);
  //thrust::sort(potentialCollisions_d.begin(), potentialCollisions_d.end());
  thrust::stable_sort(potentialCollisions_d.begin(), potentialCollisions_d.end());
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
  numCollisions = 0;
  numCollisionsPerPair_d.clear();
  collisionIdentifierA_d.clear();
  collisionIdentifierB_d.clear();
  normalsAndPenetrations_d.clear();
  collisionStartIndex_d.clear();

  if(numPossibleCollisions) {
    // Step 1: Detect how many collisions actually occur between each pair
    numCollisionsPerPair_d.resize(numPossibleCollisions);
    countActualCollisions<<<BLOCKS(numPossibleCollisions),THREADS>>>(CASTU1(numCollisionsPerPair_d), CASTU2(possibleCollisionPairs_d), CASTD1(system->p_d), CASTI1(system->indices_d), CASTD3(system->contactGeometry_d), CASTD3(system->collisionGeometry_d), CASTI4(system->collisionMap_d), envelope, system->bodies.size(), system->beams.size(), numPossibleCollisions);
    // End Step 1

    // Step 2: Figure out where each thread needs to start and end for each collision
    Thrust_Inclusive_Scan_Sum(numCollisionsPerPair_d, numCollisions);
    normalsAndPenetrations_d.resize(numCollisions);
    collisionIdentifierA_d.resize(numCollisions);
    collisionIdentifierB_d.resize(numCollisions);
    // End Step 2

    if(numCollisions) {
      // Step 3: Store the actual collisions
      storeActualCollisions<<<BLOCKS(numPossibleCollisions),THREADS>>>(CASTU1(numCollisionsPerPair_d), CASTU2(possibleCollisionPairs_d), CASTD1(system->p_d), CASTI1(system->indices_d), CASTD3(system->contactGeometry_d), CASTD3(system->collisionGeometry_d), CASTI4(system->collisionMap_d), CASTD4(normalsAndPenetrations_d), CASTU1(collisionIdentifierA_d), CASTU1(collisionIdentifierB_d), numPossibleCollisions, system->bodies.size(), system->beams.size(), numCollisions);
      // End Step 3
    }
  }

  return 0;
}

int CollisionDetector::exportSystem(string filename) {

  ofstream filestream;
  filestream.open(filename.c_str());

  //Print the collision information
  // collisionIndex bodyIdentifierA bodyIdentifierB normalX normalY normalZ penetration posA_x posA_y posA_z velA_x velA_y velA_z geomA_x geomA_y geomA_z posB_x posB_y posB_z velB_x velB_y velB_z geomB_x geomB_y geomB_z
  collisionIdentifierA_h = collisionIdentifierA_d;
  collisionIdentifierB_h = collisionIdentifierB_d;
  normalsAndPenetrations_h = normalsAndPenetrations_d;
  system->p_h = system->p_d;
  system->v_h = system->v_d;
  for(int i=0; i<collisionIdentifierA_h.size();i++) {
    filestream
    << i << ", "
    << collisionIdentifierA_h[i] << ", "
    << collisionIdentifierB_h[i] << ", "
    << normalsAndPenetrations_h[i].x << ", "
    << normalsAndPenetrations_h[i].y << ", "
    << normalsAndPenetrations_h[i].z << ", "
    << normalsAndPenetrations_h[i].w << ", "
    << system->p_h[3*collisionIdentifierA_h[i]] << ", "
    << system->p_h[3*collisionIdentifierA_h[i]+1] << ", "
    << system->p_h[3*collisionIdentifierA_h[i]+2] << ", "
    << system->v_h[3*collisionIdentifierA_h[i]] << ", "
    << system->v_h[3*collisionIdentifierA_h[i]+1] << ", "
    << system->v_h[3*collisionIdentifierA_h[i]+2] << ", "
    << system->contactGeometry_h[collisionIdentifierA_h[i]].x << ", "
    << system->contactGeometry_h[collisionIdentifierA_h[i]].y << ", "
    << system->contactGeometry_h[collisionIdentifierA_h[i]].z << ", "
    << system->p_h[3*collisionIdentifierB_h[i]] << ", "
    << system->p_h[3*collisionIdentifierB_h[i]+1] << ", "
    << system->p_h[3*collisionIdentifierB_h[i]+2] << ", "
    << system->v_h[3*collisionIdentifierB_h[i]] << ", "
    << system->v_h[3*collisionIdentifierB_h[i]+1] << ", "
    << system->v_h[3*collisionIdentifierB_h[i]+2] << ", "
    << system->contactGeometry_h[collisionIdentifierB_h[i]].x << ", "
    << system->contactGeometry_h[collisionIdentifierB_h[i]].y << ", "
    << system->contactGeometry_h[collisionIdentifierB_h[i]].z << ", "
    << "\n";
  }
  filestream.close();

  return 0;
}


