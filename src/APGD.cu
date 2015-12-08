#include <algorithm>
#include <vector>
#include "include.cuh"
#include "APGD.cuh"

APGD::APGD(System* sys)
{
  system = sys;

  tolerance = 1e-4;
  maxIterations = 100000;
  iterations = 0;
}

int APGD::setup()
{
  gammaHat_d = system->a_h;
  gammaNew_d = system->a_h;
  g_d = system->a_h;
  y_d = system->a_h;
  yNew_d = system->a_h;
  gammaTmp_d = system->a_h;

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

  return 0;
}

__global__ void project(double* src, double* friction, uint offsetBilateralConstraints, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing
  src = &src[3*index+offsetBilateralConstraints];
  double3 gamma = make_double3(src[0],src[1],src[2]);
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

  src[0] = gamma.x;
  src[1] = gamma.y;
  src[2] = gamma.z;
}


int APGD::performSchurComplementProduct(DeviceValueArrayView src) {
  cusp::multiply(system->DT,src,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,gammaTmp);

  return 0;
}


double APGD::getResidual(DeviceValueArrayView src) {
  double gdiff = 1.0 / pow(system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size(),2.0);
  performSchurComplementProduct(src); //cusp::multiply(system->N,src,gammaTmp); //
  cusp::blas::axpy(system->r,gammaTmp,1.0);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0,-gdiff);
  if(system->collisionDetector->numCollisions) project<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), system->collisionDetector->numCollisions);
  cusp::blas::axpby(src,gammaTmp,gammaTmp,1.0/gdiff,-1.0/gdiff);

  return cusp::blas::nrmmax(gammaTmp);
}

__global__ void updateAntiRelaxationVector(double* s, double* friction, double* antiRelaxation, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double s_v = s[3*index+1];
  double s_w = s[3*index+2];
  double mu = friction[index]; // TODO: Keep an eye on friction indexing

  antiRelaxation[3*index] = sqrt(pow(s_v,2.0)+pow(s_w,2.0))*mu;
  antiRelaxation[3*index+1] = 0;
  antiRelaxation[3*index+2] = 0;
}

__global__ void initializeImpulseVector_APGD(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index  ] = 1.0;
  src[3*index+1] = 0.0;
  src[3*index+2] = 0.0;
}

__global__ void getResidual_APGD(double* src, double* gamma, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index] = src[3*index]*gamma[3*index]+src[3*index+1]*gamma[3*index+1]+src[3*index+2]*gamma[3*index+2];
  src[3*index+1] = 0;
  src[3*index+2] = 0;
}

__global__ void getFeasibleX_APGD(double* src, double* dst, double* friction, uint numBilaterals, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing

  double xn = src[3*index+numBilaterals];
  double xt1 = src[3*index+1+numBilaterals];
  double xt2 = src[3*index+2+numBilaterals];

  xn = mu*xn-sqrt(pow(xt1,2.0)+pow(xt2,2.0));
  if(xn!=xn) xn = 0.0;
  dst[3*index+numBilaterals] = -fmin(0.0,xn);
  dst[3*index+1+numBilaterals] = -10e30;
  dst[3*index+2+numBilaterals] = -10e30;
}

__global__ void getFeasibleY_APGD(double* src, double* dst, double* friction, uint numBilaterals, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index]; // TODO: Keep an eye on friction indexing

  double xn = src[3*index+numBilaterals];
  double xt1 = src[3*index+1+numBilaterals];
  double xt2 = src[3*index+2+numBilaterals];

  xn = (1.0/mu)*xn-sqrt(pow(xt1,2.0)+pow(xt2,2.0));
  if(xn!=xn) xn = 0.0;
  dst[3*index+numBilaterals] = -fmin(0.0,xn);
  dst[3*index+1+numBilaterals] = -10e30;
  dst[3*index+2+numBilaterals] = -10e30;
}

int APGD::solve() {

  system->gamma_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  gammaHat_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  gammaNew_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  g_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  y_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  yNew_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  gammaTmp_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());
  antiRelaxation_d.resize(3*system->collisionDetector->numCollisions+system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size());

  // TODO: There's got to be a better way to do this...
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_gammaHat(CASTD1(gammaHat_d));
  thrust::device_ptr<double> wrapped_device_gammaNew(CASTD1(gammaNew_d));
  thrust::device_ptr<double> wrapped_device_g(CASTD1(g_d));
  thrust::device_ptr<double> wrapped_device_y(CASTD1(y_d));
  thrust::device_ptr<double> wrapped_device_yNew(CASTD1(yNew_d));
  thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));
  thrust::device_ptr<double> wrapped_device_antiRelaxation(CASTD1(antiRelaxation_d));
  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  gammaHat = DeviceValueArrayView(wrapped_device_gammaHat, wrapped_device_gammaHat + gammaHat_d.size());
  gammaNew = DeviceValueArrayView(wrapped_device_gammaNew, wrapped_device_gammaNew + gammaNew_d.size());
  g = DeviceValueArrayView(wrapped_device_g, wrapped_device_g + g_d.size());
  y = DeviceValueArrayView(wrapped_device_y, wrapped_device_y + y_d.size());
  yNew = DeviceValueArrayView(wrapped_device_yNew, wrapped_device_yNew + yNew_d.size());
  gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());
  antiRelaxation = DeviceValueArrayView(wrapped_device_antiRelaxation, wrapped_device_antiRelaxation + antiRelaxation_d.size());

  // (1) gamma_0 = zeros(nc,1)
  cusp::blas::fill(antiRelaxation,0.0);
  //cusp::blas::fill(system->gamma,0);

  // Provide an initial guess for gamma
  //initializeImpulseVector_APGD<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(system->gamma_d), system->collisionDetector->numCollisions);

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
    if(system->collisionDetector->numCollisions) project<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d), CASTD1(system->friction_d), system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), system->collisionDetector->numCollisions);

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
      if(system->collisionDetector->numCollisions) project<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d), CASTD1(system->friction_d), system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), system->collisionDetector->numCollisions);

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
//    getFeasibleX_APGD<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaNew_d), CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), system->collisionDetector->numCollisions);
//    thrust::fill(gammaTmp_d.begin(), gammaTmp_d.begin() + system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), -10e30);
//    double feasibleX = Thrust_Max(gammaTmp_d);
//
//    performSchurComplementProduct(gammaNew);
//    cusp::blas::axpy(system->r,gammaTmp,1.0);
//    double res3 = abs(cusp::blas::dot(gammaNew,gammaTmp));
//
//    getFeasibleY_APGD<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp_d), CASTD1(gammaTmp_d), CASTD1(system->friction_d), system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), system->collisionDetector->numCollisions);
//    thrust::fill(gammaTmp_d.begin(), gammaTmp_d.begin() + system->constraintsBilateralDOF_d.size()+3*system->constraintsSpherical_ShellNodeToBody2D_d.size(), -10e30);
//    double feasibleY = Thrust_Max(gammaTmp_d);
//
//    double res = fmax(feasibleX,feasibleY);
//    res = fmax(res,res3);

    // (19) if r < epsilon_min
    if (res < residual) {
      // (20) r_min = r
      residual = res;

      // (21) gamma_hat = gamma_(k+1)
      cusp::blas::copy(gammaNew,gammaHat);

      // (22) endif
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

//    // Apply anti-relaxation
//    cusp::blas::axpy(antiRelaxation,system->r,-1.0);
//    cusp::multiply(system->DT,system->gamma,system->f_contact);
//    cusp::blas::axpby(system->k,system->f_contact,system->tmp,1.0,1.0);
//    cusp::multiply(system->mass,system->tmp,system->v);
//    cusp::multiply(system->D,system->v,gammaTmp);
//    updateAntiRelaxationVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(gammaTmp), CASTD1(system->friction_d), CASTD1(antiRelaxation), system->collisionDetector->numCollisions);
//    cusp::blas::axpy(antiRelaxation,system->r,1.0);
//    // End apply anti-relaxation

    // (32) endfor
    //cout << "  Iterations: " << k << " Residual: " << residual << endl;
  }
  cout << "  Iterations: " << k << " Residual: " << residual << endl;

  // (33) return Value at time step t_(l+1), gamma_(l+1) := gamma_hat
  iterations = k;
  cusp::blas::copy(gammaHat,system->gamma);

  return 0;
}
