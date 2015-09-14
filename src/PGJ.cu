#include <algorithm>
#include <vector>
#include "include.cuh"
#include "PGJ.cuh"

PGJ::PGJ(System* sys)
{
  system = sys;

  tolerance = 1e-4;
  maxIterations = 10000;
  iterations = 0;

  omega = 0.3;
  lambda = 2.0/3.0;
}

int PGJ::setup()
{
  gammaHat_d = system->a_h;
  gammaTmp_d = system->a_h;
  B_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_gammaHat(CASTD1(gammaHat_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_B(CASTD1(B_d));

  gammaHat = DeviceValueArrayView(wrapped_device_gammaHat, wrapped_device_gammaHat + gammaHat_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  B = DeviceValueArrayView(wrapped_device_B, wrapped_device_B + B_d.size());

  return 0;
}

__global__ void project_PGJ(double* src, double* friction, uint numCollisions) {
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

__global__ void buildB(double* B, double* D, double* mass, uint* bodyIdentifiersA, uint* bodyIdentifiersB, uint numCollisions) {
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

int PGJ::performSchurComplementProduct(DeviceValueArrayView src) {
  cusp::multiply(system->DT,src,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,gammaTmp);

  return 0;
}


double PGJ::getResidual(DeviceValueArrayView src) {
  double gdiff = 1.0 / pow(system->collisionDetector->numCollisions,2.0);
  performSchurComplementProduct(src); //cusp::multiply(system->N,src,gammaTmp); //
  cusp::blas::axpy(system->r,gammaTmp,1.0);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0,-gdiff);
  project_PGJ<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0/gdiff,-1.0/gdiff);

  return cusp::blas::nrmmax(gammaTmp);
}


int PGJ::solve() {

  system->gamma_d.resize(3*system->collisionDetector->numCollisions);
  gammaHat_d.resize(3*system->collisionDetector->numCollisions);
  gammaTmp_d.resize(3*system->collisionDetector->numCollisions);
  B_d.resize(3*system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_gammaHat(CASTD1(gammaHat_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_B(CASTD1(B_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  gammaHat = DeviceValueArrayView(wrapped_device_gammaHat, wrapped_device_gammaHat + gammaHat_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  B = DeviceValueArrayView(wrapped_device_B, wrapped_device_B + B_d.size());

  // Initialize B matrix (vector in this case, since it's diagonal)
  buildB<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(B_d), CASTD1(system->D_d), CASTD1(system->mass_d), CASTU1(system->collisionDetector->bodyIdentifierA_d), CASTU1(system->collisionDetector->bodyIdentifierB_d), system->collisionDetector->numCollisions);

  // (1) for k := 0 to N_max
  double residual;
  int k;
  for (k=0; k < maxIterations; k++) {
//    cusp::print(system->gamma);
    // (2) gamma_hat = ProjectionOperator(gamma - omega * B * (N * gamma + r))
    performSchurComplementProduct(system->gamma);                    // gammaTmp = N * gamma
    cusp::blas::axpby(gammaTmp,system->r,gammaHat,1.0,1.0);          // gammaHat = gammaTmp+r
    cusp::blas::xmy(B,gammaHat,gammaTmp);                            // gammaTmp = B*gammaHat
    cusp::blas::axpby(system->gamma,gammaTmp,gammaHat,1.0,-omega);   // gammaHat = gamma - omega * gammaTmp
    project_PGJ<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaHat_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);

    // (3) gamma = lambda * gammaHat + (1-lambda) * gamma
    cusp::blas::axpby(gammaHat,system->gamma,system->gamma,lambda,(1.0-lambda));
//    cin.get();
//    cusp::print(system->gamma);

    // (4) r = r(gamma)
    residual = getResidual(system->gamma);

    // (5) if r < Tau
    if (residual < tolerance) {
      // (6) break
      break;
    }

    // (7) endfor
    //cout << "  Iterations: " << k << " Residual: " << residual << endl;
  }
  cout << "  Iterations: " << k << " Residual: " << residual << endl;

  // (8) return Value at time step t_(l+1), gamma_(l+1) := gamma
  iterations = k;

  return 0;
}
