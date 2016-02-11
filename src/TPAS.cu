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
  residual = 10e30;
  totalKrylovIterations = 0;

  tolerance = 1e-4;
  maxIterations = 1000;
  iterations = 0;

  tol_p = 1e-2; // TODO: What is best here?
  epsilon = 1e-12;

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
  x_d = system->a_h;
  xNew_d = system->a_h;
  x0_d = system->a_h;
  d_d = system->a_h;
  r_d = system->a_h;
  xTmp_d = system->a_h;
  xTmp2_d = system->a_h;

  breakpoints_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_x(CASTD1(x_d));
  thrust::device_ptr<double> wrapped_device_xNew(CASTD1(xNew_d));
  thrust::device_ptr<double> wrapped_device_x0(CASTD1(x0_d));
  thrust::device_ptr<double> wrapped_device_d(CASTD1(d_d));
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  thrust::device_ptr<double> wrapped_device_xTmp(CASTD1(xTmp_d));
  thrust::device_ptr<double> wrapped_device_xTmp2(CASTD1(xTmp2_d));
  x = DeviceValueArrayView(wrapped_device_x, wrapped_device_x + x_d.size());
  xNew = DeviceValueArrayView(wrapped_device_xNew, wrapped_device_xNew + xNew_d.size());
  x0 = DeviceValueArrayView(wrapped_device_x0, wrapped_device_x0 + x0_d.size());
  d = DeviceValueArrayView(wrapped_device_d, wrapped_device_d + d_d.size());
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  xTmp = DeviceValueArrayView(wrapped_device_xTmp, wrapped_device_xTmp + xTmp_d.size());
  xTmp2 = DeviceValueArrayView(wrapped_device_xTmp2, wrapped_device_xTmp2 + xTmp2_d.size());

  /*
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
*/
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

/*
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

  // Update residual vector associated with gammas
  performSchurComplementProduct(gammaNew); //cusp::multiply(system->N,system->gamma,gammaTmp);
  cusp::multiply(grad_f_T,lambda,res_gamma);
  cusp::blas::axpbypcz(gammaTmp, system->r, res_gamma, res_gamma, 1.0, 1.0, 1.0);

  // Update residual vector associated with lambdas
  updateConstraintVectorTangent<<<BLOCKS(numActiveTangentConstraints),THREADS>>>(CASTD1(gammaNew_d), CASTD1(system->friction_d), CASTD1(res_d), CASTI1(activeSetTangentNew_d), numActiveTangentConstraints, system->collisionDetector->numCollisions);
  updateConstraintVectorNormal<<<BLOCKS(numActiveNormalConstraints),THREADS>>>(CASTD1(gammaNew_d), CASTD1(res_d), CASTI1(activeSetNormalNew_d), numActiveTangentConstraints, numActiveNormalConstraints, system->collisionDetector->numCollisions);

  return 0;
}

int TPAS::initializeActiveConstraintGradient() {
  constructActiveConstraintGradientTangent<<<BLOCKS(numActiveTangentConstraints),THREADS>>>(CASTI1(grad_fI_d), CASTI1(grad_fJ_d), CASTD1(grad_f_d), CASTD1(gammaNew_d), CASTD1(system->friction_d), CASTI1(activeSetTangentNew_d), numActiveTangentConstraints);
  constructActiveConstraintGradientNormal<<<BLOCKS(numActiveNormalConstraints),THREADS>>>(CASTI1(grad_fI_d), CASTI1(grad_fJ_d), CASTD1(grad_f_d), CASTI1(activeSetNormalNew_d), numActiveTangentConstraints, numActiveNormalConstraints);

  initializeConstraintGradientTranspose();

  return 0;
}

int TPAS::initializeConstraintGradientTranspose() {
  constructActiveConstraintGradientTransposeTangent<<<BLOCKS(numActiveTangentConstraints),THREADS>>>(CASTI1(grad_fI_T_d), CASTI1(grad_fJ_T_d), CASTD1(grad_f_T_d), CASTD1(gammaNew_d), CASTD1(system->friction_d), CASTI1(activeSetTangentNew_d), numActiveTangentConstraints);
  constructActiveConstraintGradientTransposeNormal<<<BLOCKS(numActiveNormalConstraints),THREADS>>>(CASTI1(grad_fI_T_d), CASTI1(grad_fJ_T_d), CASTD1(grad_f_T_d), CASTI1(activeSetNormalNew_d), numActiveTangentConstraints, numActiveNormalConstraints);

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

__global__ void project2(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 1.0;
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

int TPAS::performEQPStage(int currentIterate) {
  // USE ACTIVE SET TO PERFORM EQP STAGE

  // Step 1: Identify the indices of the active constraints (this copies the non-zero elements from the old to the new)
  numActiveNormalConstraints = thrust::remove_copy(activeSetNormal_d.begin(),activeSetNormal_d.end(), activeSetNormalNew_d.begin(), 0)-activeSetNormalNew_d.begin();
  activeSetNormalNew_d.resize(numActiveNormalConstraints);
  numActiveTangentConstraints = thrust::remove_copy(activeSetTangent_d.begin(),activeSetTangent_d.end(), activeSetTangentNew_d.begin(), 0)-activeSetTangentNew_d.begin();
  activeSetTangentNew_d.resize(numActiveTangentConstraints);

  // Step 2: Update size of delta vector and set up submatrix views
  delta_d.resize(3*system->collisionDetector->numCollisions+numActiveTangentConstraints+numActiveNormalConstraints);
  thrust::device_ptr<double> wrapped_device_delta(CASTD1(delta_d));
  delta = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
  delta_gamma = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + system->gamma_d.size());
  delta_lambda = DeviceValueArrayView(wrapped_device_delta + system->gamma_d.size(), wrapped_device_delta + delta_d.size());

  // Step 3: Update the size of the lambda vector
  lambda_d.resize(numActiveNormalConstraints+numActiveTangentConstraints);
  thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
  lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());

  // Step 4: Initialize size of the grad_f matrix
  grad_fI_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_fJ_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_f_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);

  // create constraint gradient using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(grad_fI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + grad_fI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(grad_fJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + grad_fJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(grad_f_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + grad_f_d.size());

  grad_f = DeviceView(numActiveTangentConstraints+numActiveNormalConstraints, 3*system->collisionDetector->numCollisions, grad_f_d.size(), row_indices, column_indices, values);
  // end create constraint gradient

  // Step 5: Initialize the size of the grad_f_T matrix
  grad_fI_T_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_fJ_T_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);
  grad_f_T_d.resize(3*numActiveTangentConstraints+numActiveNormalConstraints);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_IT(CASTI1(grad_fI_T_d));
  DeviceIndexArrayView row_indicesT = DeviceIndexArrayView(wrapped_device_IT, wrapped_device_IT + grad_fI_T_d.size());

  thrust::device_ptr<int> wrapped_device_JT(CASTI1(grad_fJ_T_d));
  DeviceIndexArrayView column_indicesT = DeviceIndexArrayView(wrapped_device_JT, wrapped_device_JT + grad_fJ_T_d.size());

  thrust::device_ptr<double> wrapped_device_VT(CASTD1(grad_f_T_d));
  DeviceValueArrayView valuesT = DeviceValueArrayView(wrapped_device_VT, wrapped_device_VT + grad_f_T_d.size());

  grad_f_T = DeviceView(3*system->collisionDetector->numCollisions, numActiveTangentConstraints+numActiveNormalConstraints, grad_f_T_d.size(), row_indicesT, column_indicesT, valuesT);
  // end create constraint gradient transpose

  // Step 6: Initialize the size of the residual matrix
  res_d.resize(3*system->collisionDetector->numCollisions+numActiveTangentConstraints+numActiveNormalConstraints);
  thrust::device_ptr<double> wrapped_device_res(CASTD1(res_d));
  res = DeviceValueArrayView(wrapped_device_res, wrapped_device_res + res_d.size());
  res_gamma = DeviceValueArrayView(wrapped_device_res, wrapped_device_res + system->gamma_d.size());
  res_lambda = DeviceValueArrayView(wrapped_device_res + system->gamma_d.size(), wrapped_device_res + res_d.size());

  bool sameActiveSetNormal = false;
  bool sameActiveSetTangent = false;

  int h;
  for(h=currentIterate+1; h<maxIterations; h++) {
    // Step 7: Build the constraint gradient
    initializeActiveConstraintGradient();

    // Step 8: Build the residual vector
    updateResidualVector();

    // Step 9: Build the Schur complement product matrix
    //buildSchurMatrix(); // TODO: Build the full matrix, needed if I want to use preconditioning...

    // Step 10: Solve the A*delta = res
    cusp::blas::fill(delta, 0);
    bool success = mySolver->solve(*m_spmv, res, delta);
    spike::Stats stats = mySolver->getStats();

    // Step 11: Update the gamma and lambda vectors with delta
    cusp::blas::axpy(delta_gamma, gammaNew, -1.0);
    cusp::blas::axpy(delta_lambda, lambda, -1.0);

    double res = getResidual(gammaNew); // TODO: Get rid of this, for debugging only
    printf("   Iteration %d, Residual (EQP): %f\n",h,res);

//    // Step 12: Check for incorrect active set TODO: How do I do this?
//    updateActiveSet<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d),CASTD1(system->friction_d),CASTI1(activeSetNormalEQP_d),CASTI1(activeSetTangentEQP_d),system->collisionDetector->numCollisions);
//    sameActiveSetNormal = thrust::equal(activeSetNormalEQP_d.begin(),activeSetNormalEQP_d.end(), activeSetNormal_d.begin());
//    sameActiveSetTangent = thrust::equal(activeSetTangentEQP_d.begin(),activeSetTangentEQP_d.end(), activeSetTangent_d.begin());
//    if(!(sameActiveSetNormal&&sameActiveSetTangent)) {
//      tol_p = 0.5*tol_p;
//      break;
//    }

    // Step 13: Calculate infinity norm of the correction and check for convergence
    double delta_nrm = cusp::blas::nrmmax(delta);
    if (delta_nrm <= tolerance) break;
  }

  // Step 14: Ensure that a better gamma was found
  double res = getResidual(gammaNew);
  if (res < residual) {
    // r_min = r
    residual = res;

    // gamma_hat = gamma_(k+1)
    cusp::blas::copy(gammaNew,gammaHat);
  }
  else {
    // gamma_(k+1) = gamma_hat
    cusp::blas::copy(gammaHat,gammaNew);
  }

  return h;
  // END EQP STAGE
}
*/

__global__ void getFeasible_TPAS(double* src, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double xn = src[3*index];
  double xt1 = src[3*index+1];
  double xt2 = src[3*index+2];

  xn = xn-sqrt(pow(xt1,2.0)+pow(xt2,2.0));
  if(xn!=xn) xn = 0.0;
  dst[3*index] = -fmin(0.0,xn);
  dst[3*index+1] = -10e30;
  dst[3*index+2] = -10e30;
}

__global__ void constructScalingMatrices(int* invTxI, int* invTxJ, double* invTx, int* TyI, int* TyJ, double* Ty, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index];

  invTxI[3*index] = 3*index;
  invTxJ[3*index] = 3*index;
  invTx[3*index] = 1.0/mu;

  invTxI[3*index+1] = 3*index+1;
  invTxJ[3*index+1] = 3*index+1;
  invTx[3*index+1] = 1.0;

  invTxI[3*index+2] = 3*index+2;
  invTxJ[3*index+2] = 3*index+2;
  invTx[3*index+2] = 1.0;

  TyI[3*index] = 3*index;
  TyJ[3*index] = 3*index;
  Ty[3*index] = 1.0;

  TyI[3*index+1] = 3*index+1;
  TyJ[3*index+1] = 3*index+1;
  Ty[3*index+1] = mu;

  TyI[3*index+2] = 3*index+2;
  TyJ[3*index+2] = 3*index+2;
  Ty[3*index+2] = mu;
}

int TPAS::initializeScalingMatrices() {
  constructScalingMatrices<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(invTxI_d), CASTI1(invTxJ_d), CASTD1(invTx_d), CASTI1(TyI_d), CASTI1(TyJ_d), CASTD1(Ty_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);

  {
    // create invTx using cusp library
    thrust::device_ptr<int> wrapped_device_I(CASTI1(invTxI_d));
    DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + invTxI_d.size());

    thrust::device_ptr<int> wrapped_device_J(CASTI1(invTxJ_d));
    DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + invTxJ_d.size());

    thrust::device_ptr<double> wrapped_device_V(CASTD1(invTx_d));
    DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + invTx_d.size());

    invTx = DeviceView(3*system->collisionDetector->numCollisions, 3*system->collisionDetector->numCollisions, invTx_d.size(), row_indices, column_indices, values);
    // end create invTx
  }

  {
    // create Ty using cusp library
    thrust::device_ptr<int> wrapped_device_I(CASTI1(TyI_d));
    DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + TyI_d.size());

    thrust::device_ptr<int> wrapped_device_J(CASTI1(TyJ_d));
    DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + TyJ_d.size());

    thrust::device_ptr<double> wrapped_device_V(CASTD1(Ty_d));
    DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Ty_d.size());

    Ty = DeviceView(3*system->collisionDetector->numCollisions, 3*system->collisionDetector->numCollisions, Ty_d.size(), row_indices, column_indices, values);
    // end create Ty
  }

  return 0;
}

__global__ void initializeImpulseVector_TPAS(double* src, double* dst, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  dst[3*index] = friction[index] * src[3*index];
}

__global__ void project_TPAS(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double3 gamma = make_double3(src[3*index],src[3*index+1],src[3*index+2]);
  double gamma_n = gamma.x;
  double gamma_t = sqrt(pow(gamma.y,2.0)+pow(gamma.z,2.0));

  if(gamma_t < gamma_n) {
    // Don't touch gamma!
  }
  else if((gamma_t < -gamma_n) || (abs(gamma_n) < 10e-15)) {
    gamma = make_double3(0,0,0);
  }
  else {
    double gamma_n_proj = (gamma_t + gamma_n)/(2.0);
    double gamma_t_proj = gamma_n_proj;
    double tproj_div_t = gamma_t_proj/gamma_t;
    double gamma_u_proj = tproj_div_t * gamma.y;
    double gamma_v_proj = tproj_div_t * gamma.z;
    gamma = make_double3(gamma_n_proj, gamma_u_proj, gamma_v_proj);
  }

  src[3*index  ] = gamma.x;
  src[3*index+1] = gamma.y;
  src[3*index+2] = gamma.z;
}

int TPAS::performSchurComplementProduct(DeviceValueArrayView src) {
  // NOTE: system->gamma is destroyed! We restore it at the end of the solver
  cusp::multiply(invTx,src,xTmp);
  cusp::multiply(system->DT,xTmp,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,system->gamma);
  cusp::multiply(Ty,system->gamma,xTmp);

  return 0;
}

__global__ void processBreakpoints(double* x, double* d, double* breakpoints, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double3 xi = make_double3(x[3*index],x[3*index+1],x[3*index+2]);
  double3 di = make_double3(d[3*index],d[3*index+1],d[3*index+2]);
  double xt = sqrt(pow(xi.y,2.0)+pow(xi.z,2.0));
  double dt = sqrt(pow(di.y,2.0)+pow(di.z,2.0));

  double a = pow(dt,2.0)-pow(di.x,2.0);
  double b = 2.0*(xi.y*di.y+xi.z*di.z-xi.x*di.x);
  double c = pow(xt,2.0)-pow(xi.x,2.0);
  double discriminant = pow(b,2.0)-4.0*a*c;

  if(discriminant < 0) {
    breakpoints[2*index] = 1.0;
    breakpoints[2*index+1] = 1.0;
  } else if (discriminant == 0) {
    double tmp = -b/(2.0*a);
    if(tmp<0) tmp = 1.0;
    breakpoints[2*index] = tmp;
    breakpoints[2*index+1] = 1.0;
  } else {
    double tmp = (-b+sqrt(pow(b,2.0)-4*a*c))/(2.0*a);
    if(tmp<0) tmp = 1.0;
    breakpoints[2*index] = tmp;
    tmp = (-b-sqrt(pow(b,2.0)-4*a*c))/(2.0*a);
    if(tmp<0) tmp = 1.0;
    breakpoints[2*index+1] = tmp;
  }
}

int TPAS::evaluateBreakpoints() {
  processBreakpoints<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(d_d), CASTD1(breakpoints_d), system->collisionDetector->numCollisions);
  thrust::sort(breakpoints_d.begin(),breakpoints_d.end());
  breakpoints_h = breakpoints_d;
  breakpoints_h.push_back(1.0);

  return 0;
}

__global__ void getWorkingSet(double* x, int* W0, int* WP, double epsilon, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double3 xi = make_double3(x[3*index],x[3*index+1],x[3*index+2]);
  double xt = sqrt(pow(xi.y,2.0)+pow(xi.z,2.0));

  if( xt >= xi.x - epsilon && xi.x - epsilon > 0 ) {
    WP[index] = index;
  } else {
    WP[index] = 0;
  }

  if( xt <= xi.x && xi.x <= epsilon ) {
    W0[index] = index;
  } else {
    W0[index] = 0;
  }
}

int TPAS::updateWorkingSet() {
  W0_d.resize(system->collisionDetector->numCollisions);
  WP_d.resize(system->collisionDetector->numCollisions);

  getWorkingSet<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTI1(W0_d), CASTI1(WP_d), epsilon, system->collisionDetector->numCollisions);

  // Identify the indices of the active constraints (this copies the non-zero elements from the old to the new)
  int numActiveNormalConstraints = thrust::remove_copy(W0_d.begin(),W0_d.end(), W0_d.begin(), 0)-W0_d.begin();
  W0_d.resize(numActiveNormalConstraints);
  int numActiveTangentConstraints = thrust::remove_copy(WP_d.begin(),WP_d.end(), WP_d.begin(), 0)-WP_d.begin();
  WP_d.resize(numActiveTangentConstraints);

  return 0;
}

__global__ void cancelOutWorkingSet(double* src, int* W0, uint numWorkingConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numWorkingConstraints);

  int i = W0[index];
  //src[3*i  ] = 0;
  src[3*i+1] = 0;
  src[3*i+2] = 0;
}

double TPAS::getDirectionalDerivative() {
  return 0;
}

double TPAS::backtrackLinesearch(double alpha0) {
  double c = 0.5;
  double alpha = fmin(alpha0,1.0);

  cusp::blas::axpby(x,d,xTmp2,1.0,alpha);
  project_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(xTmp2_d), system->collisionDetector->numCollisions);

  performSchurComplementProduct(xTmp2); // xTmp = N*xTmp2
  double obj1 = 0.5 * cusp::blas::dot(xTmp2,xTmp) + cusp::blas::dot(xTmp2,r);
  performSchurComplementProduct(x); // xTmp = N*x
  double obj2 = 0.5 * cusp::blas::dot(x,xTmp) + cusp::blas::dot(x,r);
  cusp::blas::axpby(x,xTmp2,xTmp,1.0,-1.0);
  obj2 -= 0.5*pow(cusp::blas::nrm2(xTmp),2.0)/alpha;
  while(obj1>obj2) {
    alpha = c*alpha;

    cusp::blas::axpby(x,d,xTmp2,1.0,alpha);
    project_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(xTmp2_d), system->collisionDetector->numCollisions);

    performSchurComplementProduct(xTmp2); // xTmp = N*xTmp2
    obj1 = 0.5 * cusp::blas::dot(xTmp2,xTmp) + cusp::blas::dot(xTmp2,r);
    performSchurComplementProduct(x); // xTmp = N*x
    obj2 = 0.5 * cusp::blas::dot(x,xTmp) + cusp::blas::dot(x,r);
    cusp::blas::axpby(x,xTmp2,xTmp,1.0,-1.0);
    obj2 -= 0.5*pow(cusp::blas::nrm2(xTmp),2.0)/alpha;
  }

  return alpha;
}

int TPAS::PG() {
  evaluateBreakpoints();
  cusp::blas::copy(x,x0);
  double tau = breakpoints_h[0];

  cusp::blas::axpby(x0,d,x,1.0,tau);
  project_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), system->collisionDetector->numCollisions);

  performSchurComplementProduct(x); // xTmp = N*x
  double obj = 0.5 * cusp::blas::dot(x,xTmp) + cusp::blas::dot(x,r);

  for(int j=0; j<breakpoints_h.size()-1; j++) {
    double tauNew = breakpoints_h[j+1];
    cusp::blas::axpby(x0,d,xNew,1.0,tauNew);
    project_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(xNew_d), system->collisionDetector->numCollisions);

    performSchurComplementProduct(xNew); // xTmp = N*xNew
    double objNew = 0.5 * cusp::blas::dot(xNew,xTmp) + cusp::blas::dot(xNew,r);

    updateWorkingSet();

    cancelOutWorkingSet<<<BLOCKS(W0_d.size()),THREADS>>>(CASTD1(d_d), CASTI1(W0_d), W0_d.size());

    if(getDirectionalDerivative()>0) {
      break;
    } else if (objNew > obj) {
      double t = 1.0;
      if(!WP_d.size()) {
        performSchurComplementProduct(x); // xTmp = N*x
        cusp::blas::axpby(xTmp,r,xTmp,1.0,1.0);
        t = -cusp::blas::dot(xTmp,d);
        performSchurComplementProduct(d); // xTmp = N*d
        t = t/cusp::blas::dot(xTmp,d);
      } else {
        t = backtrackLinesearch(tauNew-tau);
      }

      cusp::blas::axpby(x,d,x,1.0,t);
      project_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), system->collisionDetector->numCollisions);
      break;
    }

    tau = tauNew;
    obj = objNew;
    cusp::blas::copy(xNew,x);
  }

  return 0;
}

int TPAS::solve() {
  // vectors
  system->gamma_d.resize(3*system->collisionDetector->numCollisions);
  x_d.resize(3*system->collisionDetector->numCollisions);
  xNew_d.resize(3*system->collisionDetector->numCollisions);
  x0_d.resize(3*system->collisionDetector->numCollisions);
  d_d.resize(3*system->collisionDetector->numCollisions);
  r_d.resize(3*system->collisionDetector->numCollisions);
  xTmp_d.resize(3*system->collisionDetector->numCollisions);
  xTmp2_d.resize(3*system->collisionDetector->numCollisions);

  // matrices: invTx, Ty
  invTxI_d.resize(3*system->collisionDetector->numCollisions);
  invTxJ_d.resize(3*system->collisionDetector->numCollisions);
  invTx_d.resize(3*system->collisionDetector->numCollisions);

  TyI_d.resize(3*system->collisionDetector->numCollisions);
  TyJ_d.resize(3*system->collisionDetector->numCollisions);
  Ty_d.resize(3*system->collisionDetector->numCollisions);

  breakpoints_d.resize(2*system->collisionDetector->numCollisions);

  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_x(CASTD1(x_d));
  thrust::device_ptr<double> wrapped_device_xNew(CASTD1(xNew_d));
  thrust::device_ptr<double> wrapped_device_x0(CASTD1(x0_d));
  thrust::device_ptr<double> wrapped_device_d(CASTD1(d_d));
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  thrust::device_ptr<double> wrapped_device_xTmp(CASTD1(xTmp_d));
  thrust::device_ptr<double> wrapped_device_xTmp2(CASTD1(xTmp2_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  x = DeviceValueArrayView(wrapped_device_x, wrapped_device_x + x_d.size());
  xNew = DeviceValueArrayView(wrapped_device_x, wrapped_device_x + x_d.size());
  x0 = DeviceValueArrayView(wrapped_device_x0, wrapped_device_x0 + x0_d.size());
  d = DeviceValueArrayView(wrapped_device_d, wrapped_device_d + d_d.size());
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  xTmp = DeviceValueArrayView(wrapped_device_xTmp, wrapped_device_xTmp + xTmp_d.size());
  xTmp2 = DeviceValueArrayView(wrapped_device_xTmp2, wrapped_device_xTmp2 + xTmp2_d.size());

  // initialize matrices and vectors
  initializeScalingMatrices();
  cusp::multiply(Ty,system->r,r); // get a scaled version of r
  initializeImpulseVector_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), CASTD1(x_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);

  int k;
  for (k=0; k < maxIterations; k++) {
    performSchurComplementProduct(x); // xTmp = N*x
    cusp::blas::axpby(xTmp,r,d,-1.0,-1.0);
    PG();

    // check feasible and optimal
    getFeasible_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(xTmp_d), system->collisionDetector->numCollisions);
    double feasibleX = Thrust_Max(xTmp_d);

    performSchurComplementProduct(x); // xTmp = N*x
    cusp::blas::axpby(xTmp,r,xTmp,1.0,1.0);
    double optim = cusp::blas::dot(x,xTmp);

    getFeasible_TPAS<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(xTmp_d), CASTD1(xTmp_d), system->collisionDetector->numCollisions);
    double feasibleY = Thrust_Max(xTmp_d);

    residual = fmax(feasibleX,feasibleY);
    residual = fmax(residual,optim);
    if (residual < tolerance) break;
  }
  cout << "  Iterations: " << k << " Residual: " << residual << endl;
  iterations = k;

  /*
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
  activeSetNormalEQP_d.resize(system->collisionDetector->numCollisions);
  activeSetTangentEQP_d.resize(system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...

  thrust::device_ptr<double> wrapped_device_gammaHat(CASTD1(gammaHat_d));
  thrust::device_ptr<double> wrapped_device_gammaNew(CASTD1(gammaNew_d));
  thrust::device_ptr<double> wrapped_device_g(CASTD1(g_d));
  thrust::device_ptr<double> wrapped_device_y(CASTD1(y_d));
  thrust::device_ptr<double> wrapped_device_yNew(CASTD1(yNew_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));

  gammaHat = DeviceValueArrayView(wrapped_device_gammaHat, wrapped_device_gammaHat + gammaHat_d.size());
  gammaNew = DeviceValueArrayView(wrapped_device_gammaNew, wrapped_device_gammaNew + gammaNew_d.size());
  g = DeviceValueArrayView(wrapped_device_g, wrapped_device_g + g_d.size());
  y = DeviceValueArrayView(wrapped_device_y, wrapped_device_y + y_d.size());
  yNew = DeviceValueArrayView(wrapped_device_yNew, wrapped_device_yNew + yNew_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());

  // Set up the linear solver TODO: could move this to the outside of everything
  m_spmv = new MySpmv(system->mass, system->D, system->DT, grad_f, grad_f_T, system->tmp, gammaTmp, delta);
  mySolver = new SpikeSolver(partitions, solverOptions);
  mySolver->setup(system->mass); //TODO: Use preconditioning here! Need to build full matrix...

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
  residual = 10e30;
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
    printf("   Iteration %d, Residual (PG): %f\n",k,residual);

    updateActiveSet<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d),CASTD1(system->friction_d),CASTI1(activeSetNormalNew_d),CASTI1(activeSetTangentNew_d),system->collisionDetector->numCollisions);
    sameActiveSetNormal = thrust::equal(activeSetNormalNew_d.begin(),activeSetNormalNew_d.end(), activeSetNormal_d.begin());
    sameActiveSetTangent = thrust::equal(activeSetTangentNew_d.begin(),activeSetTangentNew_d.end(), activeSetTangent_d.begin());
    activeSetNormal_d = activeSetNormalNew_d;
    activeSetTangent_d = activeSetTangentNew_d;
    printf("   Same active set? %d\n",sameActiveSetNormal && sameActiveSetTangent);

    if(residual < tol_p && sameActiveSetNormal && sameActiveSetTangent) {
      k = performEQPStage(k);

      // the new active set vector is destroyed in the EQP stage, need to rebuild it
      activeSetNormalNew_d = activeSetNormal_d;
      activeSetTangentNew_d = activeSetTangent_d;
    }

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
    //cout << "  Iterations: " << k << " Residual: " << residual << " Same active set? " << (sameActiveSetNormal&&sameActiveSetTangent) << endl;
  }
  //cout << "  Iterations: " << k << " Residual: " << residual << endl;
  //cin.get();

  // (33) return Value at time step t_(l+1), gamma_(l+1) := gamma_hat
  iterations = k;
  cusp::blas::copy(gammaHat,system->gamma);
*/

  cusp::multiply(invTx,x,system->gamma);
  return 0;
}
