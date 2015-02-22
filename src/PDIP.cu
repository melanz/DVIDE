#include <algorithm>
#include <vector>
#include "include.cuh"
#include "PDIP.cuh"

PDIP::PDIP(System* sys)
{
  system = sys;
}

int PDIP::setup()
{
  f_d = system->a_h;
  lambda_d = system->a_h;
  lambdaTmp_d = system->a_h;
  ones_d = system->a_h;
  r_d_d = system->a_h;
  r_g_d = system->a_h;
  delta_gamma_d = system->a_h;
  delta_lambda_d = system->a_h;
  gammaTmp_d = system->a_h;
  Dinv_d = system->a_h;
  Mhat_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
  thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
  thrust::device_ptr<double> wrapped_device_lambdaTmp(CASTD1(lambdaTmp_d));
  thrust::device_ptr<double> wrapped_device_ones(CASTD1(ones_d));
  thrust::device_ptr<double> wrapped_device_r_d(CASTD1(r_d_d));
  thrust::device_ptr<double> wrapped_device_r_g(CASTD1(r_g_d));
  thrust::device_ptr<double> wrapped_device_delta_gamma(CASTD1(delta_gamma_d));
  thrust::device_ptr<double> wrapped_device_delta_lambda(CASTD1(delta_lambda_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_Dinv(CASTD1(Dinv_d));
  thrust::device_ptr<double> wrapped_device_Mhat(CASTD1(Mhat_d));

  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());
  lambdaTmp = DeviceValueArrayView(wrapped_device_lambdaTmp, wrapped_device_lambdaTmp + lambdaTmp_d.size());
  ones = DeviceValueArrayView(wrapped_device_ones, wrapped_device_ones + ones_d.size());
  r_d = DeviceValueArrayView(wrapped_device_r_d, wrapped_device_r_d + r_d_d.size());
  r_g = DeviceValueArrayView(wrapped_device_r_g, wrapped_device_r_g + r_g_d.size());
  delta_gamma = DeviceValueArrayView(wrapped_device_delta_gamma, wrapped_device_delta_gamma + delta_gamma_d.size());
  delta_lambda = DeviceValueArrayView(wrapped_device_delta_lambda, wrapped_device_delta_lambda + delta_lambda_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  Dinv = DeviceValueArrayView(wrapped_device_Dinv, wrapped_device_Dinv + Dinv_d.size());
  M_hat = DeviceValueArrayView(wrapped_device_Mhat, wrapped_device_Mhat + Mhat_d.size());

  return 0;
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

  dst[index] = 0.5 * (pow(src[index+1], 2.0) + pow(src[index+2], 2.0) - pow(mu, 2.0) * pow(src[index], 2.0));
  dst[index + numCollisions] = -src[index];
}

__global__ void initializeLambda(double* src, double* dst, uint numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  dst[index] = -1.0/src[index];
}

__global__ void getSupremum(double* lambdaTmp, double* lambda, double* delta_lambda, uint numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  double dLambda = delta_lambda[index];
  double tmp = -lambda[index]/dLambda;
  if(dLambda >= 0) tmp = 1.0;

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
  cusp::multiply(grad_f_T,lambda,r_d);
  cusp::blas::axpbypcz(gammaTmp, system->r, r_d, r_d, 1.0, 1.0, 1.0);
  cusp::blas::xmy(lambda,f,r_g);
  cusp::blas::axpby(ones,r_g,r_g,-1.0/t,-1.0);

  return 0;
}

__global__ void updateM_hat(double* M_hat, double* lambda, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0.1; // TODO: Put this in material library!

  double l = lambda[3*index];
  M_hat[3*index  ] = -pow(mu,2.0)*l;
  M_hat[3*index+1] = l;
  M_hat[3*index+2] = l;
}

int PDIP::solve() {
  int maxIterations = 1000;
  double tolerance = 1e-4;

  // Initialize scalars
  double mu_pdip = 10.0;
  double alpha = 0.01; // should be [0.01, 0.1]
  double beta = 0.8; // should be [0.3, 0.8]
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
  ones.resize(2*system->collisionDetector->numCollisions);
  r_d_d.resize(3*system->collisionDetector->numCollisions);
  r_g_d.resize(2*system->collisionDetector->numCollisions);
  delta_gamma_d.resize(3*system->collisionDetector->numCollisions);
  delta_lambda_d.resize(3*system->collisionDetector->numCollisions);
  Dinv_d.resize(2*system->collisionDetector->numCollisions);
  Mhat_d.resize(3*system->collisionDetector->numCollisions);

  grad_fI_d.resize(4*system->collisionDetector->numCollisions);
  grad_fJ_d.resize(4*system->collisionDetector->numCollisions);
  grad_f_d.resize(4*system->collisionDetector->numCollisions);

  grad_fI_T_d.resize(4*system->collisionDetector->numCollisions);
  grad_fJ_T_d.resize(4*system->collisionDetector->numCollisions);
  grad_f_T_d.resize(4*system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
  thrust::device_ptr<double> wrapped_device_lambda(CASTD1(lambda_d));
  thrust::device_ptr<double> wrapped_device_lambdaTmp(CASTD1(lambdaTmp_d));
  thrust::device_ptr<double> wrapped_device_ones(CASTD1(ones_d));
  thrust::device_ptr<double> wrapped_device_r_d(CASTD1(r_d_d));
  thrust::device_ptr<double> wrapped_device_r_g(CASTD1(r_g_d));
  thrust::device_ptr<double> wrapped_device_delta_gamma(CASTD1(delta_gamma_d));
  thrust::device_ptr<double> wrapped_device_delta_lambda(CASTD1(delta_lambda_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_Dinv(CASTD1(Dinv_d));
  thrust::device_ptr<double> wrapped_device_Mhat(CASTD1(Mhat_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  lambda = DeviceValueArrayView(wrapped_device_lambda, wrapped_device_lambda + lambda_d.size());
  lambdaTmp = DeviceValueArrayView(wrapped_device_lambdaTmp, wrapped_device_lambdaTmp + lambdaTmp_d.size());
  ones = DeviceValueArrayView(wrapped_device_ones, wrapped_device_ones + ones_d.size());
  r_d = DeviceValueArrayView(wrapped_device_r_d, wrapped_device_r_d + r_d_d.size());
  r_g = DeviceValueArrayView(wrapped_device_r_g, wrapped_device_r_g + r_g_d.size());
  delta_gamma = DeviceValueArrayView(wrapped_device_delta_gamma, wrapped_device_delta_gamma + delta_gamma_d.size());
  delta_lambda = DeviceValueArrayView(wrapped_device_delta_lambda, wrapped_device_delta_lambda + delta_lambda_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  Dinv = DeviceValueArrayView(wrapped_device_Dinv, wrapped_device_Dinv + Dinv_d.size());
  M_hat = DeviceValueArrayView(wrapped_device_Mhat, wrapped_device_Mhat + Mhat_d.size());

  // Provide an initial guess for gamma
  initializeImpulseVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), system->collisionDetector->numCollisions);

  // Initialize the constraint gradient and constraint gradient transpose
  initializeConstraintGradient();

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
    initializeLambda<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTD1(f_d), CASTD1(Dinv_d), 2*system->collisionDetector->numCollisions);
    updateM_hat<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(Mhat_d), CASTD1(lambda_d), system->collisionDetector->numCollisions);

    // (8) r_t = r_t(gamma_k, lambda_k, t)
    updateConstraintGradient<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(grad_f_d), CASTD1(grad_f_T_d), CASTD1(system->gamma_d), system->collisionDetector->numCollisions);
    updateNewtonStepVector(system->gamma, lambda, f, t);

    // (9) Solve the linear system A * y = -r_t TODO

    // (10) s_max = sup{s in [0,1]|lambda+s*delta_lambda>=0} = min{1,min{-lambda_i/delta_lambda_i|delta_lambda_i < 0 }}
    getSupremum<<<BLOCKS(2*system->collisionDetector->numCollisions),THREADS>>>(CASTD1(lambdaTmp_d), CASTD1(lambda_d), CASTD1(delta_lambda_d), 2*system->collisionDetector->numCollisions);
    s_max = Thrust_Min(lambdaTmp_d);
    s_max = fmin(1.0,s_max);

    // (11) s = 0.99 * s_max
    s = 0.99 * s_max;

    // (12) while max(f(gamma_k + s * delta_gamma) > 0)
    cusp::blas::axpby(system->gamma,delta_gamma,gammaTmp,1.0,s);
    updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(lambdaTmp_d), system->collisionDetector->numCollisions);
    while(cusp::blas::nrmmax(lambdaTmp) > 0) {
      // (13) s = beta * s
      s = beta * s;

      cusp::blas::axpby(system->gamma,delta_gamma,gammaTmp,1.0,s);
      updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(lambdaTmp_d), system->collisionDetector->numCollisions);

      // (14) endwhile
    }

    // (15) while norm(r_t(gamma_k + s * delta_gamma, lambda_k + s * delta_lambda),2) > (1-alpha*s)*norm(r_t,2)
    norm_rt = sqrt(cusp::blas::dot(r_d,r_d) + cusp::blas::dot(r_g,r_g));
    cusp::blas::axpby(lambda,delta_lambda,lambdaTmp,1.0,s);
    updateConstraintVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(f_d), system->collisionDetector->numCollisions);
    updateConstraintGradient<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(grad_f_d), CASTD1(grad_f_T_d), CASTD1(gammaTmp_d), system->collisionDetector->numCollisions);
    updateNewtonStepVector(gammaTmp, lambdaTmp, f, t);
    while (sqrt(cusp::blas::dot(r_d,r_d) + cusp::blas::dot(r_g,r_g)) > (1.0 - alpha * s) * norm_rt) {
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
    residual = cusp::blas::nrm2(r_g);

    // (21) if r < tau

    if (residual < tolerance) {
      // (22) break
      break;

      // (23) endif
    }

    // (24) endfor
  }

  // (25) return Value at time step t_(l+1), gamma_(l+1) := gamma_(k+1)
  cout << "  Iterations: " << k << " Residual: " << residual << endl;

  return 0;
}
