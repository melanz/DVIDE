#include <algorithm>
#include <vector>
#include "include.cuh"
#include "PGS.cuh"

PGS::PGS(System* sys)
{
  system = sys;

  tolerance = 1e-4;
  maxIterations = 1000000;
  iterations = 0;

  omega = 0.3;
  lambda = 2.0/3.0;
}

int PGS::setup()
{
  gammaTmp_d = system->a_h;
  B_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_B(CASTD1(B_d));

  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  B = DeviceValueArrayView(wrapped_device_B, wrapped_device_B + B_d.size());

  return 0;
}

__global__ void project_PGS(double* src, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing
  double3 gamma = make_double3(src[3*index],src[3*index+1],src[3*index+2]);
  double gamma_n = gamma.x;
  double gamma_t = sqrt(pow(gamma.y,2.0)+pow(gamma.z,2.0));

  if(mu == 0) {
    gamma = make_double3(gamma_n,0,0);
    if (gamma_n < 0) gamma = make_double3(0,0,0);
  }
  else if(gamma_t < mu * gamma_n) {
    // Don't touch gamma!
  }
  else if((gamma_t < -(1.0/mu)*gamma_n) || (abs(gamma_n) < 10e-15)) {
    gamma = make_double3(0,0,0);
  }
  else {
    double gamma_n_proj = (gamma_t * mu + gamma_n)/(pow(mu,2.0)+1.0);
    double gamma_t_proj = gamma_n_proj * mu;
    double tproj_div_t = gamma_t_proj/gamma_t;
    double gamma_u_proj = tproj_div_t * gamma.y;
    double gamma_v_proj = tproj_div_t * gamma.z;
    gamma = make_double3(gamma_n_proj, gamma_u_proj, gamma_v_proj);
  }

  src[3*index  ] = gamma.x;
  src[3*index+1] = gamma.y;
  src[3*index+2] = gamma.z;
}

__global__ void updateImpulseVector(double* src, double* b, double* B, double* D, double* mass, double* v, double* friction, uint* bodyIdentifiersA, uint* bodyIdentifiersB, double omega, double lambda, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing
  //printf("mu = %f\n",mu);
  double3 gamma = make_double3(src[3*index],src[3*index+1],src[3*index+2]);
  double3 gamma_old = gamma;
  int indexA = bodyIdentifiersA[index];
  int indexB = bodyIdentifiersB[index];

  // TODO: This would change with different physics items (different DOF's)
  double3 vA = make_double3(v[3*indexA],v[3*indexA+1],v[3*indexA+2]);
  double3 vB = make_double3(v[3*indexB],v[3*indexB+1],v[3*indexB+2]);
  //printf("vA: (%f, %f %f)\n", vA.x, vA.y, vA.z);
  //printf("vB: (%f, %f %f)\n", vB.x, vB.y, vB.z);

  double3 D_n = make_double3(D[18*index+0], D[18*index+1], D[18*index+2]);
  double3 D_u = make_double3(D[18*index+6], D[18*index+7], D[18*index+8]);
  double3 D_v = make_double3(D[18*index+12],D[18*index+13],D[18*index+14]);
  //printf("D_n: (%f, %f %f)\n", D_n.x, D_n.y, D_n.z);
  //printf("D_u: (%f, %f %f)\n", D_u.x, D_u.y, D_u.z);
  //printf("D_v: (%f, %f %f)\n", D_v.x, D_v.y, D_v.z);

  // TODO: This might cause problems (there are 0's for fixed bodies)
  double MinvA = mass[3*indexA];
  double MinvB = mass[3*indexB];
  //printf("MinvA = %f, MinvB = %f\n",MinvA,MinvB);

  // Get delta
  //printf("omega = %f, lambda = %f\n",omega,lambda);
  //printf("gamma: (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);
  //printf("B: (%f, %f %f)\n", B[3*index+0], B[3*index+1], B[3*index+2]);
  //printf("D'*v: (%f, %f %f)\n", D_n.x*vA.x + D_n.y*vA.y + D_n.z*vA.z - D_n.x*vB.x - D_n.y*vB.y - D_n.z*vB.z, D_u.x*vA.x + D_u.y*vA.y + D_u.z*vA.z - D_u.x*vB.x - D_u.y*vB.y - D_u.z*vB.z, D_v.x*vA.x + D_v.y*vA.y + D_v.z*vA.z - D_v.x*vB.x - D_v.y*vB.y - D_v.z*vB.z);
  gamma.x = gamma.x - omega*B[3*index+0]*(D_n.x*vA.x + D_n.y*vA.y + D_n.z*vA.z - D_n.x*vB.x - D_n.y*vB.y - D_n.z*vB.z + b[3*index]);
  gamma.y = gamma.y - omega*B[3*index+1]*(D_u.x*vA.x + D_u.y*vA.y + D_u.z*vA.z - D_u.x*vB.x - D_u.y*vB.y - D_u.z*vB.z);
  gamma.z = gamma.z - omega*B[3*index+2]*(D_v.x*vA.x + D_v.y*vA.y + D_v.z*vA.z - D_v.x*vB.x - D_v.y*vB.y - D_v.z*vB.z);
  //printf("b = %f\n",b[3*index]);
  //printf("delta: (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);

  // Project!
  double gamma_n = gamma.x;
  double gamma_t = sqrt(pow(gamma.y,2.0)+pow(gamma.z,2.0));
  if(mu == 0) {
    gamma = make_double3(gamma_n,0,0);
    if (gamma_n < 0) gamma = make_double3(0,0,0);
  }
  else if(gamma_t < mu * gamma_n) {
    // Don't touch gamma!
  }
  else if((gamma_t < -(1.0/mu)*gamma_n) || (abs(gamma_n) < 10e-15)) {
    gamma = make_double3(0,0,0);
  }
  else {
    double gamma_n_proj = (gamma_t * mu + gamma_n)/(pow(mu,2.0)+1.0);
    double gamma_t_proj = gamma_n_proj * mu;
    double tproj_div_t = gamma_t_proj/gamma_t;
    double gamma_u_proj = tproj_div_t * gamma.y;
    double gamma_v_proj = tproj_div_t * gamma.z;
    gamma = make_double3(gamma_n_proj, gamma_u_proj, gamma_v_proj);
  }
  //printf("Proj(delta): (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);

  // update gamma
  src[3*index  ] = lambda*gamma.x + (1.0-lambda)*gamma_old.x;
  src[3*index+1] = lambda*gamma.y + (1.0-lambda)*gamma_old.y;
  src[3*index+2] = lambda*gamma.z + (1.0-lambda)*gamma_old.z;
  //printf("gamma: (%f, %f %f)\n", lambda*gamma.x+(1.0-lambda)*gamma_old.x, lambda*gamma.y+(1.0-lambda)*gamma_old.y, lambda*gamma.z+(1.0-lambda)*gamma_old.z);

  // get delta_gamma
  gamma.x = lambda * (gamma.x - gamma_old.x);
  gamma.y = lambda * (gamma.y - gamma_old.y);
  gamma.z = lambda * (gamma.z - gamma_old.z);

  // TODO: This would change with different physics items (different DOF's)
  v[3*indexA+0] = vA.x + MinvA * (D_n.x*gamma.x + D_u.x*gamma.y + D_v.x*gamma.z);
  v[3*indexA+1] = vA.y + MinvA * (D_n.y*gamma.x + D_u.y*gamma.y + D_v.y*gamma.z);
  v[3*indexA+2] = vA.z + MinvA * (D_n.z*gamma.x + D_u.z*gamma.y + D_v.z*gamma.z);

  v[3*indexB+0] = vB.x - MinvB * (D_n.x*gamma.x + D_u.x*gamma.y + D_v.x*gamma.z);
  v[3*indexB+1] = vB.y - MinvB * (D_n.y*gamma.x + D_u.y*gamma.y + D_v.y*gamma.z);
  v[3*indexB+2] = vB.z - MinvB * (D_n.z*gamma.x + D_u.z*gamma.y + D_v.z*gamma.z);
  //printf("vA: (%f, %f %f)\n", vA.x + MinvA*(D_n.x*gamma.x + D_u.x*gamma.y + D_v.x*gamma.z), vA.y + MinvA*(D_n.y*gamma.x + D_u.y*gamma.y + D_v.y*gamma.z), vA.z + MinvA*(D_n.z*gamma.x + D_u.z*gamma.y + D_v.z*gamma.z));
  //printf("vB: (%f, %f %f)\n", vB.x - MinvB*(D_n.x*gamma.x + D_u.x*gamma.y + D_v.x*gamma.z), vB.y - MinvB*(D_n.y*gamma.x + D_u.y*gamma.y + D_v.y*gamma.z), vB.z - MinvB*(D_n.z*gamma.x + D_u.z*gamma.y + D_v.z*gamma.z));
}

int PGS::updateImpulseVector_CPU() {
  for(int index=0; index<system->collisionDetector->numCollisions; index++) {
    double mu = system->friction_h[index]; // TODO: Keep an eye on friction indexing
    //printf("mu = %f\n",mu);
    double3 gamma = make_double3(system->gamma_h[3*index],system->gamma_h[3*index+1],system->gamma_h[3*index+2]);
    double3 gamma_old = gamma;
    int indexA = bodyIdentifierA_h[index];
    int indexB = bodyIdentifierB_h[index];

    // TODO: This would change with different physics items (different DOF's)
    double3 vA = make_double3(system->v_h[3*indexA],system->v_h[3*indexA+1],system->v_h[3*indexA+2]);
    double3 vB = make_double3(system->v_h[3*indexB],system->v_h[3*indexB+1],system->v_h[3*indexB+2]);
    //printf("vA: (%f, %f %f)\n", vA.x, vA.y, vA.z);
    //printf("vB: (%f, %f %f)\n", vB.x, vB.y, vB.z);

    double3 D_n = make_double3(system->D_h[18*index+0], system->D_h[18*index+1], system->D_h[18*index+2]);
    double3 D_u = make_double3(system->D_h[18*index+6], system->D_h[18*index+7], system->D_h[18*index+8]);
    double3 D_v = make_double3(system->D_h[18*index+12],system->D_h[18*index+13],system->D_h[18*index+14]);
    //printf("D_n: (%f, %f %f)\n", D_n.x, D_n.y, D_n.z);
    //printf("D_u: (%f, %f %f)\n", D_u.x, D_u.y, D_u.z);
    //printf("D_v: (%f, %f %f)\n", D_v.x, D_v.y, D_v.z);

    // TODO: This might cause problems (there are 0's for fixed bodies)
    double MinvA = system->mass_h[3*indexA];
    double MinvB = system->mass_h[3*indexB];
    //printf("MinvA = %f, MinvB = %f\n",MinvA,MinvB);

    // Get delta
    //printf("omega = %f, lambda = %f\n",omega,lambda);
    //printf("gamma: (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);
    //printf("B: (%f, %f %f)\n", B_h[3*index+0], B_h[3*index+1], B_h[3*index+2]);
    //printf("D'*v: (%f, %f %f)\n", D_n.x*vA.x + D_n.y*vA.y + D_n.z*vA.z - D_n.x*vB.x - D_n.y*vB.y - D_n.z*vB.z, D_u.x*vA.x + D_u.y*vA.y + D_u.z*vA.z - D_u.x*vB.x - D_u.y*vB.y - D_u.z*vB.z, D_v.x*vA.x + D_v.y*vA.y + D_v.z*vA.z - D_v.x*vB.x - D_v.y*vB.y - D_v.z*vB.z);
    gamma.x = gamma.x - omega*B_h[3*index+0]*(D_n.x*vA.x + D_n.y*vA.y + D_n.z*vA.z - D_n.x*vB.x - D_n.y*vB.y - D_n.z*vB.z + system->b_h[3*index]);
    gamma.y = gamma.y - omega*B_h[3*index+1]*(D_u.x*vA.x + D_u.y*vA.y + D_u.z*vA.z - D_u.x*vB.x - D_u.y*vB.y - D_u.z*vB.z);
    gamma.z = gamma.z - omega*B_h[3*index+2]*(D_v.x*vA.x + D_v.y*vA.y + D_v.z*vA.z - D_v.x*vB.x - D_v.y*vB.y - D_v.z*vB.z);
    //printf("b = %f\n",system->b_h[3*index]);
    //printf("delta: (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);

    // Project!
    double gamma_n = gamma.x;
    double gamma_t = sqrt(pow(gamma.y,2.0)+pow(gamma.z,2.0));
    if(mu == 0) {
      gamma = make_double3(gamma_n,0,0);
      if (gamma_n < 0) gamma = make_double3(0,0,0);
    }
    else if(gamma_t < mu * gamma_n) {
      // Don't touch gamma!
    }
    else if((gamma_t < -(1.0/mu)*gamma_n) || (abs(gamma_n) < 10e-15)) {
      gamma = make_double3(0,0,0);
    }
    else {
      double gamma_n_proj = (gamma_t * mu + gamma_n)/(pow(mu,2.0)+1.0);
      double gamma_t_proj = gamma_n_proj * mu;
      double tproj_div_t = gamma_t_proj/gamma_t;
      double gamma_u_proj = tproj_div_t * gamma.y;
      double gamma_v_proj = tproj_div_t * gamma.z;
      gamma = make_double3(gamma_n_proj, gamma_u_proj, gamma_v_proj);
    }
    //printf("Proj(delta): (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);

    // update gamma
    system->gamma_h[3*index  ] = lambda*gamma.x + (1.0-lambda)*gamma_old.x;
    system->gamma_h[3*index+1] = lambda*gamma.y + (1.0-lambda)*gamma_old.y;
    system->gamma_h[3*index+2] = lambda*gamma.z + (1.0-lambda)*gamma_old.z;
    //printf("gamma: (%f, %f %f)\n", lambda*gamma.x+(1.0-lambda)*gamma_old.x, lambda*gamma.y+(1.0-lambda)*gamma_old.y, lambda*gamma.z+(1.0-lambda)*gamma_old.z);

    // get delta_gamma
    gamma.x = lambda * (gamma.x - gamma_old.x);
    gamma.y = lambda * (gamma.y - gamma_old.y);
    gamma.z = lambda * (gamma.z - gamma_old.z);

    // TODO: This would change with different physics items (different DOF's)
    system->v_h[3*indexA+0] = vA.x + MinvA * (D_n.x*gamma.x + D_u.x*gamma.y + D_v.x*gamma.z);
    system->v_h[3*indexA+1] = vA.y + MinvA * (D_n.y*gamma.x + D_u.y*gamma.y + D_v.y*gamma.z);
    system->v_h[3*indexA+2] = vA.z + MinvA * (D_n.z*gamma.x + D_u.z*gamma.y + D_v.z*gamma.z);

    system->v_h[3*indexB+0] = vB.x - MinvB * (D_n.x*gamma.x + D_u.x*gamma.y + D_v.x*gamma.z);
    system->v_h[3*indexB+1] = vB.y - MinvB * (D_n.y*gamma.x + D_u.y*gamma.y + D_v.y*gamma.z);
    system->v_h[3*indexB+2] = vB.z - MinvB * (D_n.z*gamma.x + D_u.z*gamma.y + D_v.z*gamma.z);
  }
  system->gamma_d = system->gamma_h;

  return 0;
}

//int PGS::updateImpulseVector_CPU() {
//  for(int index=0; index<system->collisionDetector->numCollisions; index++) {
//    double mu = system->friction_h[index]; // TODO: Keep an eye on friction indexing
//    //printf("mu = %f\n",mu);
//    double3 gamma = make_double3(system->gamma_h[3*index],system->gamma_h[3*index+1],system->gamma_h[3*index+2]);
//    double3 gamma_old = gamma;
//
//    //performSchurComplementProduct(system->gamma);
//    //cusp::blas::axpy(system->r,gammaTmp,1.0);
//    cusp::multiply(system->D,system->v,gammaTmp);
//    cusp::blas::axpy(system->b,gammaTmp,1.0);
//    gammaTmp_h = gammaTmp_d;
//
//    gamma.x = gamma.x-omega*B_h[3*index+0]*gammaTmp_h[3*index+0];
//    gamma.y = gamma.y-omega*B_h[3*index+1]*gammaTmp_h[3*index+1];
//    gamma.z = gamma.z-omega*B_h[3*index+2]*gammaTmp_h[3*index+2];
//
//    // Project!
//    double gamma_n = gamma.x;
//    double gamma_t = sqrt(pow(gamma.y,2.0)+pow(gamma.z,2.0));
//    if(mu == 0) {
//      gamma = make_double3(gamma_n,0,0);
//      if (gamma_n < 0) gamma = make_double3(0,0,0);
//    }
//    else if(gamma_t < mu * gamma_n) {
//      // Don't touch gamma!
//    }
//    else if((gamma_t < -(1.0/mu)*gamma_n) || (abs(gamma_n) < 10e-15)) {
//      gamma = make_double3(0,0,0);
//    }
//    else {
//      double gamma_n_proj = (gamma_t * mu + gamma_n)/(pow(mu,2.0)+1.0);
//      double gamma_t_proj = gamma_n_proj * mu;
//      double tproj_div_t = gamma_t_proj/gamma_t;
//      double gamma_u_proj = tproj_div_t * gamma.y;
//      double gamma_v_proj = tproj_div_t * gamma.z;
//      gamma = make_double3(gamma_n_proj, gamma_u_proj, gamma_v_proj);
//    }
//    //printf("Proj(delta): (%f, %f %f)\n", gamma.x, gamma.y, gamma.z);
//
//    // update gamma
//    cusp::blas::copy(system->gamma,gammaTmp);
//    system->gamma_h[3*index  ] = lambda*gamma.x + (1.0-lambda)*gamma_old.x;
//    system->gamma_h[3*index+1] = lambda*gamma.y + (1.0-lambda)*gamma_old.y;
//    system->gamma_h[3*index+2] = lambda*gamma.z + (1.0-lambda)*gamma_old.z;
//    system->gamma_d = system->gamma_h;
//
//    // update velocity
//    cusp::blas::axpby(system->gamma,gammaTmp,gammaTmp,1.0,-1.0);
//    cusp::multiply(system->DT,gammaTmp,system->f_contact);
//    cusp::multiply(system->mass,system->f_contact,system->tmp);
//    cusp::blas::axpy(system->tmp,system->v,1.0);
//    //printf("gamma: (%f, %f %f)\n", lambda*gamma.x+(1.0-lambda)*gamma_old.x, lambda*gamma.y+(1.0-lambda)*gamma_old.y, lambda*gamma.z+(1.0-lambda)*gamma_old.z);
//  }
//
//  return 0;
//}

__global__ void buildB_PGS(double* B, double* D, double* mass, uint* bodyIdentifiersA, uint* bodyIdentifiersB, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  int indexA = bodyIdentifiersA[index];
  int indexB = bodyIdentifiersB[index];

  // TODO: This might cause problems (there are 0's for fixed bodies)
  double MinvA = mass[3*indexA];
  double MinvB = mass[3*indexB];

  double g = (MinvA+MinvB)*(pow(D[18*index],2.0)+pow(D[18*index+1],2.0)+pow(D[18*index+2],2.0)+pow(D[18*index+6],2.0)+pow(D[18*index+7],2.0)+pow(D[18*index+8],2.0)+pow(D[18*index+12],2.0)+pow(D[18*index+13],2.0)+pow(D[18*index+14],2.0));
  g = 3.0/g;

  B[3*index  ] = g;
  B[3*index+1] = g;
  B[3*index+2] = g;
}

int PGS::performSchurComplementProduct(DeviceValueArrayView src) {
  cusp::multiply(system->DT,src,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,gammaTmp);

  return 0;
}

double PGS::getResidual(DeviceValueArrayView src) {
  double gdiff = 1.0 / pow(system->collisionDetector->numCollisions,2.0);
  performSchurComplementProduct(src); //cusp::multiply(system->N,src,gammaTmp); //
  cusp::blas::axpy(system->r,gammaTmp,1.0);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0,-gdiff);
  project_PGS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0/gdiff,-1.0/gdiff);

  return cusp::blas::nrmmax(gammaTmp);
}

__global__ void getResidual_PGS(double* src, double* gamma, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index] = src[3*index]*gamma[3*index]+src[3*index+1]*gamma[3*index+1]+src[3*index+2]*gamma[3*index+2];
  src[3*index+1] = 0;
  src[3*index+2] = 0;
}

__global__ void getFeasibleX_PGS(double* src, double* dst, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing

  double xn = src[3*index];
  double xt1 = src[3*index+1];
  double xt2 = src[3*index+2];

  xn = mu*xn-sqrt(pow(xt1,2.0)+pow(xt2,2.0));
  if(xn!=xn) xn = 0.0;
  dst[3*index] = -fmin(0.0,xn);
  dst[3*index+1] = -10e30;
  dst[3*index+2] = -10e30;
}

__global__ void getFeasibleY_PGS(double* src, double* dst, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing

  double xn = src[3*index];
  double xt1 = src[3*index+1];
  double xt2 = src[3*index+2];

  xn = (1.0/mu)*xn-sqrt(pow(xt1,2.0)+pow(xt2,2.0));
  if(xn!=xn) xn = 0.0;
  dst[3*index] = -fmin(0.0,xn);
  dst[3*index+1] = -10e30;
  dst[3*index+2] = -10e30;
}

__global__ void initializeImpulseVector_PGS(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index  ] = 1.0;
  src[3*index+1] = 0.0;
  src[3*index+2] = 0.0;
}

int PGS::solve() {
  system->gamma_d.resize(3*system->collisionDetector->numCollisions);
  gammaTmp_d.resize(3*system->collisionDetector->numCollisions);
  B_d.resize(3*system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_B(CASTD1(B_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  B = DeviceValueArrayView(wrapped_device_B, wrapped_device_B + B_d.size());

  // Provide an initial guess for gamma
  initializeImpulseVector_PGS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), system->collisionDetector->numCollisions);

  // initialize speeds
  cusp::multiply(system->DT,system->gamma,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->v);
  cusp::multiply(system->mass,system->k,system->tmp);
  cusp::blas::axpy(system->tmp,system->v,1.0);

  // Initialize B matrix (vector in this case, since it's diagonal)
  buildB_PGS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(B_d), CASTD1(system->D_d), CASTD1(system->mass_d), CASTU1(system->collisionDetector->collisionIdentifierA_d), CASTU1(system->collisionDetector->collisionIdentifierB_d), system->collisionDetector->numCollisions);
  //cusp::print(B);

  // copy everything to host (ONLY FOR SEQUENTIAL VERSION)
  system->gamma_h = system->gamma_d;
  system->b_h = system->b_d;
  B_h = B_d;
  system->D_h = system->D_d;
  system->mass_h = system->mass_d;
  system->v_h = system->v_d;
  system->friction_h = system->friction_d;
  bodyIdentifierA_h = system->collisionDetector->collisionIdentifierA_d;
  bodyIdentifierB_h = system->collisionDetector->collisionIdentifierB_d;

  // (1) for k := 0 to N_max
  double residual;
  int k;
  for (k=0; k < maxIterations; k++) {
    // (2) gamma_hat = ProjectionOperator(gamma - omega * B * (N * gamma + r))
    //cusp::print(system->gamma);
    //updateImpulseVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), CASTD1(system->b_d), CASTD1(B_d), CASTD1(system->D_d), CASTD1(system->mass_d), CASTD1(system->v_d), CASTD1(system->friction_d), CASTU1(system->collisionDetector->bodyIdentifierA_d), CASTU1(system->collisionDetector->bodyIdentifierB_d), omega, lambda, system->collisionDetector->numCollisions);
    updateImpulseVector_CPU();
    //cin.get();
    //cusp::print(system->gamma);

    // (4) r = r(gamma)
    getFeasibleX_PGS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma), CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
    double feasibleX = Thrust_Max(gammaTmp_d);

    //residual = getResidual(system->gamma);
    performSchurComplementProduct(system->gamma);
    cusp::blas::axpy(system->r,gammaTmp,1.0);
    getResidual_PGS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(system->gamma), system->collisionDetector->numCollisions);
    double res3 = cusp::blas::nrmmax(gammaTmp);

    performSchurComplementProduct(system->gamma);
    cusp::blas::axpy(system->r,gammaTmp,1.0);
    getFeasibleY_PGS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
    double feasibleY = Thrust_Max(gammaTmp_d);

    residual = fmax(feasibleX,feasibleY);
    residual = fmax(residual,res3);

    // (5) if r < Tau
    if (residual < tolerance) {
      // (6) break
      break;
    }

    // (7) endfor
    //cout << "  Iterations: " << k << " Residual: " << residual << endl;
  }
  cout << "  Iterations: " << k << " Residual: " << residual << endl;
  //cin.get();

  // (8) return Value at time step t_(l+1), gamma_(l+1) := gamma
  iterations = k;

  return 0;
}
