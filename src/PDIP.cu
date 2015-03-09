#include <algorithm>
#include <vector>
#include "include.cuh"
#include "PDIP.cuh"

PDIP::PDIP(System* sys)
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

int PDIP::setup()
{
  f_d = system->a_h;
  lambda_d = system->a_h;
  lambdaTmp_d = system->a_h;
  ones_d = system->a_h;
  r_t_d = system->a_h;
  delta_d = system->a_h;
  gammaTmp_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
  thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
  thrust::device_ptr<double> wrapped_device_lambdaTmp(CASTD1(lambdaTmp_d));
  thrust::device_ptr<double> wrapped_device_ones(CASTD1(ones_d));
  thrust::device_ptr<double> wrapped_device_r_t(CASTD1(r_t_d));
  thrust::device_ptr<double> wrapped_device_delta(CASTD1(delta_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));

  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());
  lambdaTmp = DeviceValueArrayView(wrapped_device_lambdaTmp, wrapped_device_lambdaTmp + lambdaTmp_d.size());
  ones = DeviceValueArrayView(wrapped_device_ones, wrapped_device_ones + ones_d.size());
  r_d = DeviceValueArrayView(wrapped_device_r_t, wrapped_device_r_t + r_t_d.size());
  r_g = DeviceValueArrayView(wrapped_device_r_t, wrapped_device_r_t + r_t_d.size());
  r_t = DeviceValueArrayView(wrapped_device_r_t, wrapped_device_r_t + r_t_d.size());
  delta_gamma = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
  delta_lambda = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
  delta = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());

  return 0;
}

void PDIP::setSolverType(int solverType)
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

void PDIP::setPrecondType(int useSpike)
{
  solverOptions.precondType = useSpike ? spike::Spike : spike::None;
}

void PDIP::printSolverParams()
{
  //  printf("Step size: %e\n", h);
  //  printf("Newton tolerance: %e\n", tol);
  printf("Krylov relTol: %e  abdTol: %e\n", solverOptions.relTol, solverOptions.absTol);
  printf("Max. Krylov iterations: %d\n", solverOptions.maxNumIterations);
  printf("----------------------------\n");
}

__global__ void initializeImpulseVector(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index  ] = 1.0;
  src[3*index+1] = 0.0;
  src[3*index+2] = 0.0;
}

__global__ void updateConstraintVector(double* src, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  dst[index] = 0.5 * (pow(src[3*index+1], 2.0) + pow(src[3*index+2], 2.0) - pow(mu, 2.0) * pow(src[3*index], 2.0));
  dst[index + numCollisions] = -src[3*index];
}

__global__ void initializeLambda(double* src, double* dst, uint numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  dst[index] = -1.0/src[index];
}

__global__ void getSupremum(double* lambdaTmp, double* lambda, double* delta, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, 2*numCollisions);

  double dLambda = delta[3*numCollisions+index];
  double tmp = -lambda[index]/dLambda;
  if(dLambda > 0) tmp = 1.0;

  lambdaTmp[index] = tmp;
}

__global__ void constructConstraintGradient(int* grad_fI, int* grad_fJ, double* grad_f, double* gamma, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  grad_fI[3*index  ] = index;
  grad_fI[3*index+1] = index;
  grad_fI[3*index+2] = index;
  grad_fI[3*numCollisions+index] = index+numCollisions;

  grad_fJ[3*index  ] = 3*index;
  grad_fJ[3*index+1] = 3*index+1;
  grad_fJ[3*index+2] = 3*index+2;
  grad_fJ[3*numCollisions+index] = 3*index;

  grad_f[3*index  ] = -pow(mu,2.0)*gamma[3*index];
  grad_f[3*index+1] = gamma[3*index+1];
  grad_f[3*index+2] = gamma[3*index+2];
  grad_f[3*numCollisions+index] = -1.0;
}

__global__ void constructConstraintGradientTranspose(int* grad_fI, int* grad_fJ, double* grad_f, double* gamma, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  grad_fI[4*index  ] = 3*index;
  grad_fI[4*index+1] = 3*index;
  grad_fI[4*index+2] = 3*index+1;
  grad_fI[4*index+3] = 3*index+2;

  grad_fJ[4*index  ] = index;
  grad_fJ[4*index+1] = index+numCollisions;
  grad_fJ[4*index+2] = index;
  grad_fJ[4*index+3] = index;

  grad_f[4*index  ] = -pow(mu,2.0)*gamma[3*index];
  grad_f[4*index+1] = -1.0;
  grad_f[4*index+2] = gamma[3*index+1];
  grad_f[4*index+3] = gamma[3*index+2];
}

int PDIP::initializeConstraintGradient() {
  constructConstraintGradient<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(grad_fI_d), CASTI1(grad_fJ_d), CASTD1(grad_f_d), CASTD1(system->gamma_d), system->collisionDetector->numCollisions);

  // create constraint gradient using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(grad_fI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + grad_fI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(grad_fJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + grad_fJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(grad_f_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + grad_f_d.size());

  grad_f = DeviceView(2*system->collisionDetector->numCollisions, 3*system->collisionDetector->numCollisions, grad_f_d.size(), row_indices, column_indices, values);
  grad_f_global = DeviceView(2*system->collisionDetector->numCollisions, 5*system->collisionDetector->numCollisions, grad_f_d.size(), row_indices, column_indices, values);
  // end create constraint gradient

  initializeConstraintGradientTranspose();

  return 0;
}

int PDIP::initializeConstraintGradientTranspose() {
  constructConstraintGradientTranspose<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(grad_fI_T_d), CASTI1(grad_fJ_T_d), CASTD1(grad_f_T_d), CASTD1(system->gamma_d), system->collisionDetector->numCollisions);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(grad_fI_T_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + grad_fI_T_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(grad_fJ_T_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + grad_fJ_T_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(grad_f_T_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + grad_f_T_d.size());

  grad_f_T = DeviceView(3*system->collisionDetector->numCollisions, 2*system->collisionDetector->numCollisions, grad_f_T_d.size(), row_indices, column_indices, values);
  // end create constraint gradient transpose

  return 0;
}

__global__ void constructB(int* BI, int* BJ, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  BI[4*index  ] = 3*index;
  BI[4*index+1] = 3*index;
  BI[4*index+2] = 3*index+1;
  BI[4*index+3] = 3*index+2;

  BJ[4*index  ] = 3*numCollisions+index;
  BJ[4*index+1] = 3*numCollisions+index+numCollisions;
  BJ[4*index+2] = 3*numCollisions+index;
  BJ[4*index+3] = 3*numCollisions+index;
}

int PDIP::initializeB() {
  constructB<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(BI_d), CASTI1(BJ_d), system->collisionDetector->numCollisions);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(BI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + BI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(BJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + BJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(grad_f_T_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + grad_f_T_d.size());

  B = DeviceView(5*system->collisionDetector->numCollisions, 5*system->collisionDetector->numCollisions, grad_f_T_d.size(), row_indices, column_indices, values);
  // end create constraint gradient transpose

  return 0;
}

__global__ void constructM_hat(int* i_indices, int* j_indices, double* values, double* lambda, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  i_indices[3*index  ] = 3*index;
  i_indices[3*index+1] = 3*index+1;
  i_indices[3*index+2] = 3*index+2;

  j_indices[3*index  ] = 3*index;
  j_indices[3*index+1] = 3*index+1;
  j_indices[3*index+2] = 3*index+2;

  double l = lambda[index];
  values[3*index  ] = -pow(mu,2.0)*l;
  values[3*index+1] = l;
  values[3*index+2] = l;
}

int PDIP::initializeM_hat() {
  constructM_hat<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(MhatI_d), CASTI1(MhatJ_d), CASTD1(Mhat_d), CASTD1(lambda_d), system->collisionDetector->numCollisions);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(MhatI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + MhatI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(MhatJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + MhatJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(Mhat_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Mhat_d.size());

  M_hat = DeviceView(5*system->collisionDetector->numCollisions, 5*system->collisionDetector->numCollisions, Mhat_d.size(), row_indices, column_indices, values);
  // end create constraint gradient transpose

  return 0;
}

__global__ void constructDinv(int* i_indices, int* j_indices, double* values, double* f, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, 2*numCollisions);

  i_indices[index] = 3*numCollisions+index;
  j_indices[index] = 3*numCollisions+index;

  values[index] = -f[index];
}

__global__ void updateDinv(double* Dinv, double* f, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, 2*numCollisions);

  Dinv[index] = -f[index];
}

int PDIP::initializeDinv() {
  constructDinv<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTI1(DinvI_d), CASTI1(DinvJ_d), CASTD1(Dinv_d), CASTD1(f_d), system->collisionDetector->numCollisions);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(DinvI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + DinvI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(DinvJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + DinvJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(Dinv_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Dinv_d.size());

  Dinv = DeviceView(5*system->collisionDetector->numCollisions, 5*system->collisionDetector->numCollisions, Dinv_d.size(), row_indices, column_indices, values);
  // end create constraint gradient transpose

  return 0;
}

__global__ void constructDiagLambda(int* i_indices, int* j_indices, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, 2*numCollisions);

  i_indices[index] = 3*numCollisions+index;
  j_indices[index] = index;
}

int PDIP::initializeDiagLambda() {
  constructDiagLambda<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTI1(lambdaI_d), CASTI1(lambdaJ_d), system->collisionDetector->numCollisions);

  // create constraint gradient transpose using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(lambdaI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + lambdaI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(lambdaJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + lambdaJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(lambda_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Dinv_d.size());

  diagLambda = DeviceView(5*system->collisionDetector->numCollisions, 2*system->collisionDetector->numCollisions, lambda_d.size(), row_indices, column_indices, values);
  // end create constraint gradient transpose

  return 0;
}

__global__ void updateConstraintGradient(double* grad_f, double* grad_f_T, double* gamma, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  grad_f[3*index  ] = -pow(mu,2.0)*gamma[3*index];
  grad_f[3*index+1] = gamma[3*index+1];
  grad_f[3*index+2] = gamma[3*index+2];
  grad_f[3*numCollisions+index] = -1.0;

  grad_f_T[4*index  ] = -pow(mu,2.0)*gamma[3*index];
  grad_f_T[4*index+1] = -1.0;
  grad_f_T[4*index+2] = gamma[3*index+1];
  grad_f_T[4*index+3] = gamma[3*index+2];
}

int PDIP::performSchurComplementProduct(DeviceValueArrayView src) {
  cusp::multiply(system->DT,src,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,gammaTmp);

  return 0;
}

int PDIP::updateNewtonStepVector(DeviceValueArrayView gamma, DeviceValueArrayView lambda, DeviceValueArrayView f, double t) {
  performSchurComplementProduct(gamma); // gammaTmp = N*gamma
  //cusp::multiply(system->N,gamma,rhs);

  cusp::multiply(grad_f_T,lambda,r_d);
  cusp::blas::axpbypcz(gammaTmp, system->r, r_d, r_d, 1.0, 1.0, 1.0);
  cusp::blas::xmy(lambda,f,r_g);
  cusp::blas::axpby(ones,r_g,r_g,-1.0/t,-1.0);

  return 0;
}

__global__ void updateM_hat(double* M_hat, double* lambda, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  double l = lambda[index];
  M_hat[3*index  ] = -pow(mu,2.0)*l;
  M_hat[3*index+1] = l;
  M_hat[3*index+2] = l;
}

int PDIP::buildSchurComplementMatrix() {
  // create contact jacobian using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(system->DI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + system->DI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(system->DJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + system->DJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(system->D_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + system->D_d.size());

  D = DeviceView(5*system->collisionDetector->numCollisions, 3*system->bodies.size(), system->D_d.size(), row_indices, column_indices, values);
  // end create contact jacobian

  // create contact jacobian using cusp library
  thrust::device_ptr<int> wrapped_device_I2(CASTI1(system->DTI_d));
  DeviceIndexArrayView row_indices2 = DeviceIndexArrayView(wrapped_device_I2, wrapped_device_I2 + system->DI_d.size());

  thrust::device_ptr<int> wrapped_device_J2(CASTI1(system->DTJ_d));
  DeviceIndexArrayView column_indices2 = DeviceIndexArrayView(wrapped_device_J2, wrapped_device_J2 + system->DJ_d.size());

  thrust::device_ptr<double> wrapped_device_V2(CASTD1(system->DT_d));
  DeviceValueArrayView values2 = DeviceValueArrayView(wrapped_device_V2, wrapped_device_V2 + system->D_d.size());

  DT = DeviceView(3*system->bodies.size(), 5*system->collisionDetector->numCollisions, system->DT_d.size(), row_indices2, column_indices2, values2);
  // end create contact jacobian

  DT.sort_by_row(); // TODO: Do I need this?

  // build N
  cusp::multiply(system->mass,DT,MinvDT);
  cusp::multiply(D,MinvDT,N);

  return 0;
}

int PDIP::buildAMatrix() {
  cusp::add(N,M_hat,A);
  cusp::add(A,B,A);
  cusp::add(A,Dinv,A);
  cusp::multiply(diagLambda,grad_f_global,C);
  cusp::subtract(A,C,A);

  return 0;
}

int PDIP::solve() {
  solverOptions.relTol = std::min(0.01 * tolerance, 1e-6);
  solverOptions.absTol = 1e-10;

  // Initialize scalars
  double eta_hat = 0.0;
  double t = 0.0;
  double s = 1.0;
  double s_max = 1.0;
  double norm_rt = 0.0;
  double residual = 10e30;


  system->gamma_d.resize(3*system->collisionDetector->numCollisions);
  gammaTmp_d.resize(3*system->collisionDetector->numCollisions);
  f_d.resize(2*system->collisionDetector->numCollisions);
  lambda_d.resize(2*system->collisionDetector->numCollisions);
  lambdaTmp_d.resize(2*system->collisionDetector->numCollisions);
  ones_d.resize(2*system->collisionDetector->numCollisions);
  r_t_d.resize(5*system->collisionDetector->numCollisions);
  delta_d.resize(5*system->collisionDetector->numCollisions);

  grad_fI_d.resize(4*system->collisionDetector->numCollisions);
  grad_fJ_d.resize(4*system->collisionDetector->numCollisions);
  grad_f_d.resize(4*system->collisionDetector->numCollisions);

  grad_fI_T_d.resize(4*system->collisionDetector->numCollisions);
  grad_fJ_T_d.resize(4*system->collisionDetector->numCollisions);
  grad_f_T_d.resize(4*system->collisionDetector->numCollisions);

  MhatI_d.resize(3*system->collisionDetector->numCollisions);
  MhatJ_d.resize(3*system->collisionDetector->numCollisions);
  Mhat_d.resize(3*system->collisionDetector->numCollisions);

  lambdaI_d.resize(2*system->collisionDetector->numCollisions);
  lambdaJ_d.resize(2*system->collisionDetector->numCollisions);

  DinvI_d.resize(2*system->collisionDetector->numCollisions);
  DinvJ_d.resize(2*system->collisionDetector->numCollisions);
  Dinv_d.resize(2*system->collisionDetector->numCollisions);

  BI_d.resize(4*system->collisionDetector->numCollisions);
  BJ_d.resize(4*system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
  thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
  thrust::device_ptr<double> wrapped_device_lambdaTmp(CASTD1(lambdaTmp_d));
  thrust::device_ptr<double> wrapped_device_ones(CASTD1(ones_d));
  thrust::device_ptr<double> wrapped_device_r_t(CASTD1(r_t_d));
  thrust::device_ptr<double> wrapped_device_delta(CASTD1(delta_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());
  lambdaTmp = DeviceValueArrayView(wrapped_device_lambdaTmp, wrapped_device_lambdaTmp + lambdaTmp_d.size());
  ones = DeviceValueArrayView(wrapped_device_ones, wrapped_device_ones + ones_d.size());
  r_d = DeviceValueArrayView(wrapped_device_r_t, wrapped_device_r_t + 3*system->collisionDetector->numCollisions);
  r_g = DeviceValueArrayView(wrapped_device_r_t + 3*system->collisionDetector->numCollisions, wrapped_device_r_t + r_t_d.size());
  r_t = DeviceValueArrayView(wrapped_device_r_t, wrapped_device_r_t + r_t_d.size());
  delta_gamma = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + 3*system->collisionDetector->numCollisions);
  delta_lambda = DeviceValueArrayView(wrapped_device_delta + 3*system->collisionDetector->numCollisions, wrapped_device_delta + delta_d.size());
  delta = DeviceValueArrayView(wrapped_device_delta, wrapped_device_delta + delta_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());

  // build the Schur Complement Matrix (padded with zeros)
  buildSchurComplementMatrix();

  // Provide an initial guess for gamma
  initializeImpulseVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), system->collisionDetector->numCollisions);

  // Initialize the constraint gradient and constraint gradient transpose
  initializeConstraintGradient();
  initializeM_hat();
  initializeDinv();
  initializeDiagLambda();
  initializeB();

  // (1) f = f(gamma_0)
  updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), CASTD1(f_d), system->collisionDetector->numCollisions);

  // (2) lambda_0 = -1/f
  initializeLambda<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTD1(f_d), CASTD1(lambda_d), 2*system->collisionDetector->numCollisions);
  cusp::blas::fill(ones,1.0);

  // (3) for k := 0 to N_max
  int k;
  for (k=0; k < maxIterations; k++) {
    // (4) f = f(gamma_k)
    updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), CASTD1(f_d), system->collisionDetector->numCollisions);

    // (5) eta_hat = -f^T * lambda_k
    eta_hat = -cusp::blas::dot(f,lambda);

    // (6) t = mu*m/eta_hat
    t = mu_pdip * f.size() / eta_hat;

    // (7) A = A(gamma_k, lambda_k, f)
    updateDinv<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTD1(Dinv_d), CASTD1(f_d), system->collisionDetector->numCollisions);
    updateM_hat<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(Mhat_d), CASTD1(lambda_d), system->collisionDetector->numCollisions);

    // (8) r_t = r_t(gamma_k, lambda_k, t)
    updateConstraintGradient<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(grad_f_d), CASTD1(grad_f_T_d), CASTD1(system->gamma_d), system->collisionDetector->numCollisions);
    updateNewtonStepVector(system->gamma, lambda, f, t);

    // (9) Solve the linear system A * y = -r_t
    buildAMatrix();

    delete mySolver;
    mySolver = new SpikeSolver(partitions, solverOptions);
    mySolver->setup(A);

    cusp::blas::fill(delta,0.0);
    bool success = mySolver->solve(A, r_t, delta);
    cusp::blas::scal(delta,-1.0);
    spike::Stats stats = mySolver->getStats();

    // (10) s_max = sup{s in [0,1]|lambda+s*delta_lambda>=0} = min{1,min{-lambda_i/delta_lambda_i|delta_lambda_i < 0 }}
    getSupremum<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTD1(lambdaTmp_d), CASTD1(lambda_d), CASTD1(delta_d), system->collisionDetector->numCollisions);
    s_max = Thrust_Min(lambdaTmp_d);
    s_max = fmin(1.0,s_max);

    // (11) s = 0.99 * s_max
    s = 0.99 * s_max;

    // (12) while max(f(gamma_k + s * delta_gamma) > 0)
    cusp::blas::axpby(system->gamma,delta_gamma,gammaTmp,1.0,s);
    updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(lambdaTmp_d), system->collisionDetector->numCollisions);
    while(Thrust_Max(lambdaTmp_d) > 0) {
      // (13) s = beta * s
      s = beta * s;

      cusp::blas::axpby(system->gamma,delta_gamma,gammaTmp,1.0,s);
      updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(lambdaTmp_d), system->collisionDetector->numCollisions);

      // (14) endwhile
    }

    // (15) while norm(r_t(gamma_k + s * delta_gamma, lambda_k + s * delta_lambda),2) > (1-alpha*s)*norm(r_t,2)
    norm_rt = cusp::blas::nrm2(r_t);
    cusp::blas::axpby(lambda,delta_lambda,lambdaTmp,1.0,s);
    updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(f_d), system->collisionDetector->numCollisions);
    updateConstraintGradient<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(grad_f_d), CASTD1(grad_f_T_d), CASTD1(gammaTmp_d), system->collisionDetector->numCollisions);
    updateNewtonStepVector(gammaTmp, lambdaTmp, f, t);
    while (cusp::blas::nrm2(r_t) > (1.0 - alpha * s) * norm_rt) {

      // (16) s = beta * s
      s = beta * s;

      cusp::blas::axpby(system->gamma,delta_gamma,gammaTmp,1.0,s);
      cusp::blas::axpby(lambda,delta_lambda,lambdaTmp,1.0,s);
      updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(f_d), system->collisionDetector->numCollisions);
      updateConstraintGradient<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(grad_f_d), CASTD1(grad_f_T_d), CASTD1(gammaTmp_d), system->collisionDetector->numCollisions);
      updateNewtonStepVector(gammaTmp, lambdaTmp, f, t);

      // (17) endwhile
    }

    // (18) gamma_(k+1) = gamma_k + s * delta_gamma
    cusp::blas::axpy(delta_gamma,system->gamma,s);

    // (19) lambda_(k+1) = lamda_k + s * delta_lambda
    cusp::blas::axpy(delta_lambda,lambda,s);

    // (20) r = r(gamma_(k+1))
    //residual = cusp::blas::nrm2(r_g);///system->collisionDetector->numCollisions;
    //residual = cusp::blas::nrm2(r_g);
    //cusp::multiply(system->N,system->gamma,gammaTmp);
    performSchurComplementProduct(system->gamma);
    cusp::blas::axpy(system->r,gammaTmp,1.0);
    residual = cusp::blas::nrmmax(r_g)/fmax(1.0,cusp::blas::nrmmax(gammaTmp));
    // (21) if r < tau
    if (residual < tolerance) {
      // (22) break
      break;

      // (23) endif
    }

    // (24) endfor
    cout << "  Iterations: " << k << " Residual: " << residual << " Krylov: " << stats.numIterations << endl;
  }

  // (25) return Value at time step t_(l+1), gamma_(l+1) := gamma_(k+1)
  iterations = k;
  cout << "  Iterations: " << k << " Residual: " << residual << endl;

  return 0;
}
