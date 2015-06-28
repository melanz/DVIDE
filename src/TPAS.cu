#include <algorithm>
#include <vector>
#include "include.cuh"
#include "TPAS.cuh"
#include <thrust/remove.h>

struct is_zero
{
__host__ __device__
bool operator()(const uint x)
{
return !x;
}
};

TPAS::TPAS(System* sys)
{
  system = sys;

  mu_pdip = 50.0;
  alpha = 0.01; // should be [0.01, 0.1]
  beta = 0.8; // should be [0.3, 0.8]
  tolerance = 1e-4;
  maxIterations = 1000;
  iterations = 0;

  // spike stuff
  partitions = 1;
  solverOptions.safeFactorization = true;
  solverOptions.trackReordering = true;
  solverOptions.maxNumIterations = 5000;
  preconditionerUpdateModulus = -1; // the preconditioner updates every ___ time steps
  preconditionerMaxKrylovIterations = -1; // the preconditioner updates if Krylov iterations are greater than ____ iterations
  mySolver = new SpikeSolver(partitions, solverOptions);
  //m_spmv = new MySpmv(grad_f, grad_f_T, system->D, system->DT, system->mass, lambda, lambdaTmp, Dinv, M_hat, gammaTmp, system->f_contact, system->tmp);
  stepKrylovIterations = 0;
  precUpdated = 0;
  // end spike stuff
}

int TPAS::setup()
{
  f_d = system->a_h;
  lambda_d = system->a_h;
  lambdaTmp_d = system->a_h;
  ones_d = system->a_h;
  r_d_d = system->a_h;
  r_g_d = system->a_h;
  delta_d = system->a_h;
  gammaTmp_d = system->a_h;
  rhs_d = system->a_h;
  res_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
  thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
  thrust::device_ptr<double> wrapped_device_lambdaTmp(CASTD1(lambdaTmp_d));
  thrust::device_ptr<double> wrapped_device_ones(CASTD1(ones_d));
  thrust::device_ptr<double> wrapped_device_r_d(CASTD1(r_d_d));
  thrust::device_ptr<double> wrapped_device_r_g(CASTD1(r_g_d));
  thrust::device_ptr<double> wrapped_device_delta(CASTD1(delta_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_rhs(CASTD1(rhs_d));
  thrust::device_ptr<double> wrapped_device_res(CASTD1(res_d));

  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());
  lambdaTmp = DeviceValueArrayView(wrapped_device_lambdaTmp, wrapped_device_lambdaTmp + lambdaTmp_d.size());
  ones = DeviceValueArrayView(wrapped_device_ones, wrapped_device_ones + ones_d.size());
  r_d = DeviceValueArrayView(wrapped_device_r_d, wrapped_device_r_d + r_d_d.size());
  r_g = DeviceValueArrayView(wrapped_device_r_g, wrapped_device_r_g + r_g_d.size());
  delta = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  rhs = DeviceValueArrayView(wrapped_device_rhs, wrapped_device_rhs + rhs_d.size());
  res = DeviceValueArrayView(wrapped_device_res, wrapped_device_res + res_d.size());

  return 0;
}

void TPAS::setSolverType(int solverType)
{
  switch(solverType) {
  case 0:
    solverOptions.solverType = spike::BiCGStab;
    break;
  case 1:
    solverOptions.solverType = spike::BiCGStab1;
    break;
  case 2:
    solverOptions.solverType = spike::BiCGStab2;
    break;
  case 3:
    solverOptions.solverType = spike::MINRES;
    break;
  case 4:
    solverOptions.solverType = spike::CG_C;
    break;
  case 5:
    solverOptions.solverType = spike::CR_C;
    break;
  }
}

void TPAS::setPrecondType(int useSpike)
{
  solverOptions.precondType = useSpike ? spike::Spike : spike::None;
}

void TPAS::printSolverParams()
{
  //  printf("Step size: %e\n", h);
  //  printf("Newton tolerance: %e\n", tol);
  printf("Krylov relTol: %e  abdTol: %e\n", solverOptions.relTol, solverOptions.absTol);
  printf("Max. Krylov iterations: %d\n", solverOptions.maxNumIterations);
  printf("----------------------------\n");
}

__global__ void initializeImpulseVector2(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index  ] = 1.0;
  src[3*index+1] = 0.0;
  src[3*index+2] = 0.0;
}



__global__ void initializeLambda2(double* src, double* dst, uint numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  dst[index] = -1.0/src[index];
}

__global__ void getSupremum2(double* lambdaTmp, double* lambda, double* delta_lambda, uint numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  double dLambda = delta_lambda[index];
  double tmp = -lambda[index]/dLambda;
  if(dLambda > 0) tmp = 1.0;

  lambdaTmp[index] = tmp;
}

__global__ void constructActiveConstraintGradientTangent(int* grad_fI, int* grad_fJ, double* grad_f, double* gamma, double* friction, int* activeTangentConstraints, uint numTangentConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numTangentConstraints);

  double mu = friction[index];
  int constraintIndex = activeTangentConstraints[index]-1;

  grad_fI[3*index  ] = index;
  grad_fI[3*index+1] = index;
  grad_fI[3*index+2] = index;

  grad_fJ[3*index  ] = 3*constraintIndex;
  grad_fJ[3*index+1] = 3*constraintIndex+1;
  grad_fJ[3*index+2] = 3*constraintIndex+2;

  grad_f[3*index  ] = -pow(mu,2.0)*gamma[3*constraintIndex];
  grad_f[3*index+1] = gamma[3*constraintIndex+1];
  grad_f[3*index+2] = gamma[3*constraintIndex+2];
}

__global__ void constructActiveConstraintGradientNormal(int* grad_fI, int* grad_fJ, double* grad_f, int* activeNormalConstraints, uint numTangentConstraints, uint numNormalConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numNormalConstraints);

  int constraintIndex = activeNormalConstraints[index]-1;

  grad_fI[3*numTangentConstraints+index] = index+numTangentConstraints;
  grad_fJ[3*numTangentConstraints+index] = 3*constraintIndex;
  grad_f[3*numTangentConstraints+index]  = -1.0;
}

__global__ void constructActiveConstraintGradientTransposeTangent(int* grad_fI, int* grad_fJ, double* grad_f, double* gamma, double* friction, int* activeTangentConstraints, uint numTangentConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numTangentConstraints);

  double mu = friction[index];
  int constraintIndex = activeTangentConstraints[index]-1;

  grad_fJ[3*index  ] = index;
  grad_fJ[3*index+1] = index;
  grad_fJ[3*index+2] = index;

  grad_fI[3*index  ] = 3*constraintIndex;
  grad_fI[3*index+1] = 3*constraintIndex+1;
  grad_fI[3*index+2] = 3*constraintIndex+2;

  grad_f[3*index  ] = -pow(mu,2.0)*gamma[3*constraintIndex];
  grad_f[3*index+1] = gamma[3*constraintIndex+1];
  grad_f[3*index+2] = gamma[3*constraintIndex+2];
}

__global__ void constructActiveConstraintGradientTransposeNormal(int* grad_fI, int* grad_fJ, double* grad_f, int* activeNormalConstraints, uint numTangentConstraints, uint numNormalConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numNormalConstraints);

  int constraintIndex = activeNormalConstraints[index]-1;

  grad_fI[3*numTangentConstraints+index] = 3*constraintIndex;
  grad_fJ[3*numTangentConstraints+index] = index+numTangentConstraints;
  grad_f[3*numTangentConstraints+index]  = -1.0;
}

// TODO: GET RID OF THIS FUNCTION
//__global__ void constructConstraintGradientTranspose2(int* grad_fI, int* grad_fJ, double* grad_f, double* gamma, double* friction, uint numCollisions) {
//  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);
//
//  double mu = friction[index];
//
//  grad_fI[4*index  ] = 3*index;
//  grad_fI[4*index+1] = 3*index;
//  grad_fI[4*index+2] = 3*index+1;
//  grad_fI[4*index+3] = 3*index+2;
//
//  grad_fJ[4*index  ] = index;
//  grad_fJ[4*index+1] = index+numCollisions;
//  grad_fJ[4*index+2] = index;
//  grad_fJ[4*index+3] = index;
//
//  grad_f[4*index  ] = -pow(mu,2.0)*gamma[3*index];
//  grad_f[4*index+1] = -1.0;
//  grad_f[4*index+2] = gamma[3*index+1];
//  grad_f[4*index+3] = gamma[3*index+2];
//}

__global__ void updateConstraintVectorTangent(double* gamma, double* friction, double* res, int* activeTangentConstraints, uint numTangentConstraints, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numTangentConstraints);

  double mu = friction[index];
  int constraintIndex = activeTangentConstraints[index]-1;

  res[3*numCollisions+index] = 0.5 * (pow(gamma[3*constraintIndex+1], 2.0) + pow(gamma[3*constraintIndex+2], 2.0) - pow(mu, 2.0) * pow(gamma[3*constraintIndex], 2.0));
}

__global__ void updateConstraintVectorNormal(double* gamma, double* res, int* activeNormalConstraints, uint numTangentConstraints, uint numNormalConstraints, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numNormalConstraints);

  int constraintIndex = activeNormalConstraints[index]-1;

  res[3*numCollisions+numTangentConstraints+index] = -gamma[3*constraintIndex];
}

int TPAS::updateResidualVector() {
  res_d.resize(3*system->collisionDetector->numCollisions+numActiveTangentConstraints+numActiveNormalConstraints);
  thrust::device_ptr<double> wrapped_device_res(CASTD1(res_d));
  res = DeviceValueArrayView(wrapped_device_res, wrapped_device_res + res_d.size());
  res_gamma = DeviceValueArrayView(wrapped_device_res, wrapped_device_res + system->gamma_d.size());
  res_lambda = DeviceValueArrayView(wrapped_device_res + system->gamma_d.size(), wrapped_device_res + res_d.size());

  // Update residual vector associated with gammas
  performSchurComplementProduct(system->gamma); //cusp::multiply(system->N,system->gamma,gammaTmp);
  cusp::multiply(grad_f_T,lambda,res_gamma);
  cusp::blas::axpbypcz(gammaTmp, system->r, res_gamma, res_gamma, 1.0, 1.0, 1.0);

  // Update residual vector associated with lambdas
  updateConstraintVectorTangent<<<BLOCKS(numActiveTangentConstraints),THREADS>>>(CASTD1(system->gamma_d), CASTD1(system->friction_d), CASTD1(res_d), CASTI1(activeSetTangentNew_d), numActiveTangentConstraints, system->collisionDetector->numCollisions);
  updateConstraintVectorNormal<<<BLOCKS(numActiveNormalConstraints),THREADS>>>(CASTD1(system->gamma_d), CASTD1(res_d), CASTI1(activeSetNormalNew_d), numActiveTangentConstraints, numActiveNormalConstraints, system->collisionDetector->numCollisions);

  return 0;
}

int TPAS::initializeActiveConstraintGradient() {

  grad_fI_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_fJ_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_f_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);

  constructActiveConstraintGradientTangent<<<BLOCKS(numActiveTangentConstraints),THREADS>>>(CASTI1(grad_fI_d), CASTI1(grad_fJ_d), CASTD1(grad_f_d), CASTD1(system->gamma_d), CASTD1(system->friction_d), CASTI1(activeSetTangentNew_d), numActiveTangentConstraints);
  constructActiveConstraintGradientNormal<<<BLOCKS(numActiveNormalConstraints),THREADS>>>(CASTI1(grad_fI_d), CASTI1(grad_fJ_d), CASTD1(grad_f_d), CASTI1(activeSetNormalNew_d), numActiveTangentConstraints, numActiveNormalConstraints);

  // create constraint gradient using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(grad_fI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + grad_fI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(grad_fJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + grad_fJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(grad_f_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + grad_f_d.size());

  grad_f = DeviceView(numActiveTangentConstraints+numActiveNormalConstraints, 3*system->collisionDetector->numCollisions, grad_f_d.size(), row_indices, column_indices, values);
  // end create constraint gradient

  initializeConstraintGradientTranspose();

  return 0;
}

int TPAS::initializeConstraintGradientTranspose() {
  grad_fI_T_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_fJ_T_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_f_T_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);

  constructActiveConstraintGradientTransposeTangent<<<BLOCKS(numActiveTangentConstraints),THREADS>>>(CASTI1(grad_fI_T_d), CASTI1(grad_fJ_T_d), CASTD1(grad_f_T_d), CASTD1(system->gamma_d), CASTD1(system->friction_d), CASTI1(activeSetTangentNew_d), numActiveTangentConstraints);
  constructActiveConstraintGradientTransposeNormal<<<BLOCKS(numActiveNormalConstraints),THREADS>>>(CASTI1(grad_fI_T_d), CASTI1(grad_fJ_T_d), CASTD1(grad_f_T_d), CASTI1(activeSetNormalNew_d), numActiveTangentConstraints, numActiveNormalConstraints);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(grad_fI_T_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + grad_fI_T_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(grad_fJ_T_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + grad_fJ_T_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(grad_f_T_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + grad_f_T_d.size());

  grad_f_T = DeviceView(3*system->collisionDetector->numCollisions, numActiveTangentConstraints+numActiveNormalConstraints, grad_f_T_d.size(), row_indices, column_indices, values);
  // end create constraint gradient transpose

  grad_f_T.sort_by_row();

  return 0;
}



//__global__ void constructM_hat2(int* i_indices, int* j_indices, double* values, double* lambda, double* friction, uint numCollisions) {
//  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);
//
//  double mu = friction[index];
//
//  i_indices[3*index  ] = 3*index;
//  i_indices[3*index+1] = 3*index+1;
//  i_indices[3*index+2] = 3*index+2;
//
//  j_indices[3*index  ] = 3*index;
//  j_indices[3*index+1] = 3*index+1;
//  j_indices[3*index+2] = 3*index+2;
//
//  double l = lambda[index];
//  values[3*index  ] = -pow(mu,2.0)*l;
//  values[3*index+1] = l;
//  values[3*index+2] = l;
//}
//
//int TPAS::initializeM_hat() {
//  constructM_hat2<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(MhatI_d), CASTI1(MhatJ_d), CASTD1(Mhat_d), CASTD1(lambda_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
//
//  // create constraint gradient transpose using cusp library
//  thrust::device_ptr<int> wrapped_device_I(CASTI1(MhatI_d));
//  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + MhatI_d.size());
//
//  thrust::device_ptr<int> wrapped_device_J(CASTI1(MhatJ_d));
//  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + MhatJ_d.size());
//
//  thrust::device_ptr<double> wrapped_device_V(CASTD1(Mhat_d));
//  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Mhat_d.size());
//
//  M_hat = DeviceView(3*system->collisionDetector->numCollisions, 3*system->collisionDetector->numCollisions, Mhat_d.size(), row_indices, column_indices, values);
//  // end create constraint gradient transpose
//
//  return 0;
//}

//__global__ void constructDinv2(int* i_indices, int* j_indices, double* values, double* f, uint numConstraints) {
//  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);
//
//  i_indices[index] = index;
//  j_indices[index] = index;
//
//  values[index] = -1.0/f[index];
//}
//
//int TPAS::initializeDinv() {
//  constructDinv2<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTI1(DinvI_d), CASTI1(DinvJ_d), CASTD1(Dinv_d), CASTD1(f_d), 2*system->collisionDetector->numCollisions);
//
//  // create constraint gradient transpose using cusp library
//  thrust::device_ptr<int> wrapped_device_I(CASTI1(DinvI_d));
//  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + DinvI_d.size());
//
//  thrust::device_ptr<int> wrapped_device_J(CASTI1(DinvJ_d));
//  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + DinvJ_d.size());
//
//  thrust::device_ptr<double> wrapped_device_V(CASTD1(Dinv_d));
//  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Dinv_d.size());
//
//  Dinv = DeviceView(2*system->collisionDetector->numCollisions, 2*system->collisionDetector->numCollisions, Dinv_d.size(), row_indices, column_indices, values);
//  // end create constraint gradient transpose
//
//  return 0;
//}
//
//__global__ void constructDiagLambda2(int* i_indices, int* j_indices, uint numConstraints) {
//  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);
//
//  i_indices[index] = index;
//  j_indices[index] = index;
//}
//
//int TPAS::initializeDiagLambda() {
//  constructDiagLambda2<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTI1(lambdaI_d), CASTI1(lambdaJ_d), 2*system->collisionDetector->numCollisions);
//
//  // create constraint gradient transpose using cusp library
//  thrust::device_ptr<int> wrapped_device_I(CASTI1(lambdaI_d));
//  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + lambdaI_d.size());
//
//  thrust::device_ptr<int> wrapped_device_J(CASTI1(lambdaJ_d));
//  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + lambdaJ_d.size());
//
//  thrust::device_ptr<double> wrapped_device_V(CASTD1(lambda_d));
//  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Dinv_d.size());
//
//  diagLambda = DeviceView(2*system->collisionDetector->numCollisions, 2*system->collisionDetector->numCollisions, lambda_d.size(), row_indices, column_indices, values);
//  // end create constraint gradient transpose
//
//  return 0;
//}

// TODO: GET RID OF THIS FUNCTION
//__global__ void updateConstraintGradient2(double* grad_f, double* grad_f_T, double* gamma, double* friction, uint numCollisions) {
//  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);
//
//  double mu = friction[index];
//
//  grad_f[3*index  ] = -pow(mu,2.0)*gamma[3*index];
//  grad_f[3*index+1] = gamma[3*index+1];
//  grad_f[3*index+2] = gamma[3*index+2];
//  grad_f[3*numCollisions+index] = -1.0;
//
//  grad_f_T[4*index  ] = -pow(mu,2.0)*gamma[3*index];
//  grad_f_T[4*index+1] = -1.0;
//  grad_f_T[4*index+2] = gamma[3*index+1];
//  grad_f_T[4*index+3] = gamma[3*index+2];
//}
//
//int PDIP::performSchurComplementProduct(DeviceValueArrayView src) {
//  cusp::multiply(system->DT,src,system->f_contact);
//  cusp::multiply(system->mass,system->f_contact,system->tmp);
//  cusp::multiply(system->D,system->tmp,gammaTmp);
//
//  return 0;
//}

int TPAS::updateNewtonStepVector(DeviceValueArrayView gamma, DeviceValueArrayView lambda, DeviceValueArrayView f, double t) {
  //performSchurComplementProduct(gamma); // gammaTmp = N*gamma NOTE: rhs is being used as temporary variable
  cusp::multiply(system->N,gamma,rhs);

  cusp::multiply(grad_f_T,lambda,r_d);
  cusp::blas::axpbypcz(rhs, system->r, r_d, r_d, 1.0, 1.0, 1.0);
  cusp::blas::xmy(lambda,f,r_g);
  cusp::blas::axpby(ones,r_g,r_g,-1.0/t,-1.0);

  return 0;
}

__global__ void updateM_hat2(double* M_hat, double* lambda, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  double l = lambda[index];
  M_hat[3*index  ] = -pow(mu,2.0)*l;
  M_hat[3*index+1] = l;
  M_hat[3*index+2] = l;
}

int TPAS::buildAMatrix() {
  // Step 1: A = grad_fT*Dinv*(diagLambda)*grad_f
  DeviceMatrix tmp;
  cusp::multiply(grad_f_T,Dinv,A);
  cusp::multiply(A,diagLambda,tmp);
  cusp::multiply(tmp,grad_f,A);

  // Step 2: tmp = M_hat - A
  cusp::add(M_hat,A,tmp);

  // Step 3: A = N + tmp
  cusp::add(system->N,tmp,A);

  return 0;
}

__global__ void project2(double* src, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index];
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

__global__ void updateActiveSet(double* src, double* friction, int* activeSetNormal, int* activeSetTangent, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double gamma_n = src[3*index];
  double gamma_t = pow(src[3*index+1],2.0)+pow(src[3*index+2],2.0);

  if(gamma_n <= 0) {
    activeSetNormal[index] = index+1;
  }
  else {
    activeSetNormal[index] = 0.0;
  }

  if(gamma_t >= pow(friction[index] * gamma_n,2.0) && gamma_n > 0) { //TODO: What kind of tolerance should this have?
    activeSetTangent[index] = index+1;
  }
  else {
    activeSetTangent[index] = 0.0;
  }
}

int TPAS::performSchurComplementProduct(DeviceValueArrayView src) {
  cusp::multiply(system->DT,src,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,gammaTmp);

  return 0;
}


double TPAS::getResidual(DeviceValueArrayView src) {
  double gdiff = 1.0 / pow(system->collisionDetector->numCollisions,2.0);
  performSchurComplementProduct(src); //cusp::multiply(system->N,src,gammaTmp); //
  cusp::blas::axpy(system->r,gammaTmp,1.0);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0,-gdiff);
  project2<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0/gdiff,-1.0/gdiff);

  return cusp::blas::nrmmax(gammaTmp);
}

int TPAS::solve() {

  system->gamma_d.resize(3*system->collisionDetector->numCollisions);
  gammaHat_d.resize(3*system->collisionDetector->numCollisions);
  gammaNew_d.resize(3*system->collisionDetector->numCollisions);
  g_d.resize(3*system->collisionDetector->numCollisions);
  y_d.resize(3*system->collisionDetector->numCollisions);
  yNew_d.resize(3*system->collisionDetector->numCollisions);
  gammaTmp_d.resize(3*system->collisionDetector->numCollisions);

  // Initialize active set data structures
  activeSetNormal_d.resize(system->collisionDetector->numCollisions);
  thrust::fill(activeSetNormal_d.begin(),activeSetNormal_d.end(),0.0);
  activeSetNormalNew_d.resize(system->collisionDetector->numCollisions);
  activeSetTangent_d.resize(system->collisionDetector->numCollisions);
  thrust::fill(activeSetTangent_d.begin(),activeSetTangent_d.end(),0.0);
  activeSetTangentNew_d.resize(system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_gammaHat(CASTD1(gammaHat_d));
  thrust::device_ptr<double> wrapped_device_gammaNew(CASTD1(gammaNew_d));
  thrust::device_ptr<double> wrapped_device_g(CASTD1(g_d));
  thrust::device_ptr<double> wrapped_device_y(CASTD1(y_d));
  thrust::device_ptr<double> wrapped_device_yNew(CASTD1(yNew_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  gammaHat = DeviceValueArrayView(wrapped_device_gammaHat, wrapped_device_gammaHat + gammaHat_d.size());
  gammaNew = DeviceValueArrayView(wrapped_device_gammaNew, wrapped_device_gammaNew + gammaNew_d.size());
  g = DeviceValueArrayView(wrapped_device_g, wrapped_device_g + g_d.size());
  y = DeviceValueArrayView(wrapped_device_y, wrapped_device_y + y_d.size());
  yNew = DeviceValueArrayView(wrapped_device_yNew, wrapped_device_yNew + yNew_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());

  // (1) gamma_0 = zeros(nc,1)
  cusp::blas::fill(system->gamma,0);

  // (2) gamma_hat_0 = ones(nc,1)
  cusp::blas::fill(gammaHat,1.0);

  // (3) y_0 = gamma_0
  cusp::blas::copy(system->gamma,y);

  // (4) theta_0 = 1
  double theta = 1.0;
  double thetaNew = theta;
  double Beta = 0.0;
  double obj1 = 0.0;
  double obj2 = 0.0;
  double residual = 10e30;
  bool sameActiveSetNormal = false;
  bool sameActiveSetTangent = false;

  // (5) L_k = norm(N * (gamma_0 - gamma_hat_0)) / norm(gamma_0 - gamma_hat_0)
  cusp::blas::axpby(system->gamma,gammaHat,gammaTmp,1.0,-1.0);
  double L = cusp::blas::nrm2(gammaTmp);
  performSchurComplementProduct(gammaTmp); //cusp::multiply(system->N,gammaTmp,g); //
  L = cusp::blas::nrm2(gammaTmp)/L;

  // (6) t_k = 1 / L_k
  double t = 1.0/L;

  // (7) for k := 0 to N_max
  int k;
  for (k=0; k < maxIterations; k++) {
    // (8) g = N * y_k - r
    performSchurComplementProduct(y); //cusp::multiply(system->N,y,gammaTmp); //
    cusp::blas::axpby(gammaTmp,system->r,g,1.0,1.0);

    // (9) gamma_(k+1) = ProjectionOperator(y_k - t_k * g)
    cusp::blas::axpby(y,g,gammaNew,1.0,-t);
    project2<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);

    // (10) while 0.5 * gamma_(k+1)' * N * gamma_(k+1) - gamma_(k+1)' * r >= 0.5 * y_k' * N * y_k - y_k' * r + g' * (gamma_(k+1) - y_k) + 0.5 * L_k * norm(gamma_(k+1) - y_k)^2
    performSchurComplementProduct(gammaNew); //cusp::multiply(system->N,gammaNew,gammaTmp); //
    obj1 = 0.5 * cusp::blas::dot(gammaNew,gammaTmp) + cusp::blas::dot(gammaNew,system->r);
    performSchurComplementProduct(y); //cusp::multiply(system->N,y,gammaTmp); //
    obj2 = 0.5 * cusp::blas::dot(y,gammaTmp) + cusp::blas::dot(y,system->r);
    cusp::blas::axpby(gammaNew,y,gammaTmp,1.0,-1.0);
    obj2 += cusp::blas::dot(g,gammaTmp) + 0.5 * L * pow(cusp::blas::nrm2(gammaTmp),2.0);

    while (obj1 >= obj2) {
      // (11) L_k = 2 * L_k
      L = 2.0 * L;

      // (12) t_k = 1 / L_k
      t = 1.0 / L;

      // (13) gamma_(k+1) = ProjectionOperator(y_k - t_k * g)
      cusp::blas::axpby(y,g,gammaNew,1.0,-t);
      project2<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);

      // Update the components of the while condition
      performSchurComplementProduct(gammaNew); //cusp::multiply(system->N,gammaNew,gammaTmp); //
      obj1 = 0.5 * cusp::blas::dot(gammaNew,gammaTmp) + cusp::blas::dot(gammaNew,system->r);
      performSchurComplementProduct(y); //cusp::multiply(system->N,y,gammaTmp); //
      obj2 = 0.5 * cusp::blas::dot(y,gammaTmp) + cusp::blas::dot(y,system->r);
      cusp::blas::axpby(gammaNew,y,gammaTmp,1.0,-1.0);
      obj2 += cusp::blas::dot(g,gammaTmp) + 0.5 * L * pow(cusp::blas::nrm2(gammaTmp),2.0);

      // (14) endwhile
    }

    // (15) theta_(k+1) = (-theta_k^2 + theta_k * sqrt(theta_k^2 + 4)) / 2
    thetaNew = (-pow(theta, 2.0) + theta * sqrt(pow(theta, 2.0) + 4.0)) / 2.0;

    // (16) Beta_(k+1) = theta_k * (1 - theta_k) / (theta_k^2 + theta_(k+1))
    Beta = theta * (1.0 - theta) / (pow(theta, 2.0) + thetaNew);

    // (17) y_(k+1) = gamma_(k+1) + Beta_(k+1) * (gamma_(k+1) - gamma_k)
    cusp::blas::axpby(gammaNew,system->gamma,yNew,(1.0+Beta),-Beta);

    // (18) r = r(gamma_(k+1))
    double res = getResidual(gammaNew);

    // (19) if r < epsilon_min
    if (res < residual) {
      // (20) r_min = r
      residual = res;

      // (21) gamma_hat = gamma_(k+1)
      cusp::blas::copy(gammaNew,gammaHat);

      // (22) endif
    }

    cusp::print(gammaNew);
    updateActiveSet<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d),CASTD1(system->friction_d),CASTI1(activeSetNormalNew_d),CASTI1(activeSetTangentNew_d),system->collisionDetector->numCollisions);
    sameActiveSetNormal = thrust::equal(activeSetNormalNew_d.begin(),activeSetNormalNew_d.end(), activeSetNormal_d.begin());
    sameActiveSetTangent = thrust::equal(activeSetTangentNew_d.begin(),activeSetTangentNew_d.end(), activeSetTangent_d.begin());
    activeSetNormal_d = activeSetNormalNew_d;
    activeSetTangent_d = activeSetTangentNew_d;

    // (23) if r < Tau
    if (residual < tolerance) {
      // (24) break
      break;

      // (25) endif
    }

    // (26) if g' * (gamma_(k+1) - gamma_k) > 0
    cusp::blas::axpby(gammaNew,system->gamma,gammaTmp,1.0,-1.0);
    if (cusp::blas::dot(g,gammaTmp) > 0) {
      // (27) y_(k+1) = gamma_(k+1)
      cusp::blas::copy(gammaNew,yNew);

      // (28) theta_(k+1) = 1
      thetaNew = 1.0;

      // (29) endif
    }

    // (30) L_k = 0.9 * L_k
    L = 0.9 * L;

    // (31) t_k = 1 / L_k
    t = 1.0 / L;

    // Update iterates
    theta = thetaNew;
    cusp::blas::copy(gammaNew,system->gamma);
    cusp::blas::copy(yNew,y);

    // (32) endfor
    cout << "  Iterations: " << k << " Residual: " << residual << " Same active set? " << (sameActiveSetNormal&&sameActiveSetTangent) << endl;
  }
  cout << "  Iterations: " << k << " Residual: " << residual << endl;
  cin.get();

  // (33) return Value at time step t_(l+1), gamma_(l+1) := gamma_hat
  iterations = k;
  cusp::blas::copy(gammaHat,system->gamma);

  // USE ACTIVE SET TO PERFORM EQP STAGE
  activeSet_h = activeSetNormal_d;
  for(int i=0;i<activeSetNormal_d.size();i++) {
    printf("activeSet_d[%d] = %u\n",i,activeSet_h[i]);
  }
  cin.get();

  activeSet_h = activeSetTangent_d;
  for(int i=0;i<activeSetTangent_d.size();i++) {
    printf("activeSet_d[%d] = %u\n",i,activeSet_h[i]);
  }
  cin.get();

  // Step 1: Identify the indices of the active constraints
  numActiveNormalConstraints = thrust::remove_copy(activeSetNormal_d.begin(),activeSetNormal_d.end(), activeSetNormalNew_d.begin(), 0)-activeSetNormalNew_d.begin();
  activeSetNormalNew_d.resize(numActiveNormalConstraints);
  numActiveTangentConstraints = thrust::remove_copy(activeSetTangent_d.begin(),activeSetTangent_d.end(), activeSetTangentNew_d.begin(), 0)-activeSetTangentNew_d.begin();
  activeSetTangentNew_d.resize(numActiveTangentConstraints);

  activeSet_h = activeSetNormalNew_d;
  for(int i=0;i<numActiveNormalConstraints;i++) {
    printf("activeSet_d[%d] = %u\n",i,activeSet_h[i]);
  }
  cin.get();

  activeSet_h = activeSetTangentNew_d;
  for(int i=0;i<numActiveTangentConstraints;i++) {
    printf("activeSet_d[%d] = %u\n",i,activeSet_h[i]);
  }
  cin.get();

  for(int k=0; k<maxIterations; k++) {
    // Step 2: Build the constraint gradient
    initializeActiveConstraintGradient();
    cusp::print(grad_f);
    cin.get();

    cusp::print(grad_f_T);
    cin.get();

    // Step 3: Build the residual vector
    //system->buildSchurMatrix(); // THIS ONLY NEEDS TO BE DONE ONCE
    //cusp::print(system->N);
    lambda_d.resize(numActiveNormalConstraints+numActiveTangentConstraints);
    thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
    lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());
    updateResidualVector();

    cusp::print(res);
    cin.get();

    // Step 4: Build the Schur complement product matrix
    //buildSchurMatrix(); // TODO: Build the full matrix, needed if I want to use preconditioning...

    // Step 5: Solve the A*delta = res
    delta_d.resize(3*system->collisionDetector->numCollisions+numActiveTangentConstraints+numActiveNormalConstraints);
    thrust::device_ptr<double> wrapped_device_delta(CASTD1(delta_d));
    delta = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
    delta_gamma = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + system->gamma_d.size());
    delta_lambda = DeviceValueArrayView(wrapped_device_delta + system->gamma_d.size(), wrapped_device_delta + delta_d.size());

    m_spmv = new MySpmv(system->mass, system->D, system->DT, grad_f, grad_f_T, system->tmp, gammaTmp, delta);
    mySolver = new SpikeSolver(partitions, solverOptions);
    mySolver->setup(system->mass); //TODO: Use preconditioning here! Need to build full matrix...

    cusp::blas::fill(delta, 0);
    bool success = mySolver->solve(*m_spmv, res, delta);
    spike::Stats stats = mySolver->getStats();

    cusp::print(delta);
    cin.get();

    // Step 6: Update the gamma and lambda vectors with delta
    cusp::blas::axpy(delta_gamma, system->gamma, -1.0);
    cusp::blas::axpy(delta_lambda, lambda, -1.0);

    cusp::print(system->gamma);
    cin.get();

    // Step 7: Calculate infinity norm of the correction and check for convergence
    double delta_nrm = cusp::blas::nrmmax(delta);
    printf("delta_nrm: %f\n",delta_nrm);
    if (delta_nrm <= tolerance) break;
  }
  // END EQP STAGE

  return 0;
}
