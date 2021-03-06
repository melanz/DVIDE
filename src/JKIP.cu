#include <algorithm>
#include <vector>
#include "include.cuh"
#include "JKIP.cuh"

JKIP::JKIP(System* sys)
{
  system = sys;

  careful = false;
  tolerance = 1e-4;
  maxIterations = 1000;
  iterations = 0;
  totalKrylovIterations = 0;

  // spike stuff
  partitions = 1;
  solverOptions.safeFactorization = true;
  solverOptions.trackReordering = true;
  solverOptions.maxNumIterations = 5000;
  solverOptions.precondType = spike::None;
  preconditionerUpdateModulus = -1; // the preconditioner updates every ___ time steps
  preconditionerMaxKrylovIterations = -1; // the preconditioner updates if Krylov iterations are greater than ____ iterations
  mySolver = new SpikeSolver(partitions, solverOptions);
  m_spmv = new MySpmvJKIP(system->mass, system->D, system->DT, Pw, Pinv, Ty, invTx, system->tmp, system->f_contact, tmp);
  //m_spmv = new MySpmv(grad_f, grad_f_T, system->D, system->DT, system->mass, lambda, lambdaTmp, Dinv, M_hat, gammaTmp, system->f_contact, system->tmp);
  stepKrylovIterations = 0;
  precUpdated = 0;
  // end spike stuff
}

int JKIP::setup()
{
  // vectors:  x, y, dx, dy, d, b
  x_d = system->a_h;
  y_d = system->a_h;
  dx_d = system->a_h;
  dy_d = system->a_h;
  d_d = system->a_h;
  b_d = system->a_h;
  r_d = system->a_h;
  tmp_d = system->a_h;

  thrust::device_ptr<double> wrapped_device_x(CASTD1(x_d));
  thrust::device_ptr<double> wrapped_device_y(CASTD1(y_d));
  thrust::device_ptr<double> wrapped_device_dx(CASTD1(dx_d));
  thrust::device_ptr<double> wrapped_device_dy(CASTD1(dy_d));
  thrust::device_ptr<double> wrapped_device_d(CASTD1(d_d));
  thrust::device_ptr<double> wrapped_device_b(CASTD1(b_d));
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  thrust::device_ptr<double> wrapped_device_tmp(CASTD1(tmp_d));

  x = DeviceValueArrayView(wrapped_device_x, wrapped_device_x + x_d.size());
  y = DeviceValueArrayView(wrapped_device_y, wrapped_device_y + y_d.size());
  dx = DeviceValueArrayView(wrapped_device_dx, wrapped_device_dx + dx_d.size());
  dy = DeviceValueArrayView(wrapped_device_dy, wrapped_device_dy + dy_d.size());
  d = DeviceValueArrayView(wrapped_device_d, wrapped_device_d + d_d.size());
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  tmp = DeviceValueArrayView(wrapped_device_tmp, wrapped_device_tmp + tmp_d.size());

  return 0;
}

void JKIP::setSolverType(int solverType)
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

void JKIP::setPrecondType(int useSpike)
{
  solverOptions.precondType = useSpike ? spike::Spike : spike::None;
}

void JKIP::printSolverParams()
{
  //  printf("Step size: %e\n", h);
  //  printf("Newton tolerance: %e\n", tol);
  printf("Krylov relTol: %e  abdTol: %e\n", solverOptions.relTol, solverOptions.absTol);
  printf("Max. Krylov iterations: %d\n", solverOptions.maxNumIterations);
  printf("----------------------------\n");
}

__global__ void initialize_d(double* x, double* y, double* d, double s, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double xn = x[3*index];
  double xt1 = x[3*index+1];
  double xt2 = x[3*index+2];

  double y0 = y[3*index];
  double y1 = y[3*index+1];
  double y2 = y[3*index+2];

  double yBar0 = s/(xn-(pow(xt1,2.0)+pow(xt2,2.0))/xn);
  double yBar1 = -(yBar0/xn)*xt1;
  double yBar2 = -(yBar0/xn)*xt2;

  d[3*index] = (yBar0-y0)/s;
  d[3*index+1] = (yBar1-y1)/s;
  d[3*index+2] = (yBar2-y2)/s;
}

__global__ void getInverse(double* src, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double xn = src[3*index];
  double xt1 = src[3*index+1];
  double xt2 = src[3*index+2];

  double d = 0.5*(pow(xn,2.0)-(pow(xt1,2.0)+pow(xt2,2.0)));

  dst[3*index] = xn/d;
  dst[3*index+1] = -xt1/d;
  dst[3*index+2] = -xt2/d;
}

__global__ void getFeasible(double* src, double* dst, uint numCollisions) {
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

__global__ void getResidual_JKIP(double* src, double* gamma, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  src[3*index] = src[3*index]*gamma[3*index]+src[3*index+1]*gamma[3*index+1]+src[3*index+2]*gamma[3*index+2];
  src[3*index+1] = 0;
  src[3*index+2] = 0;
}

__global__ void getStepLength(double* x, double* dx, double* y, double* dy, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double tx;
  double ty;
  double aux;

  double xn = x[3*index];
  double xt1 = x[3*index+1];
  double xt2 = x[3*index+2];
  double detx = 0.5*(pow(xn,2.0)-(pow(xt1,2.0)+pow(xt2,2.0)));

  double dxn = dx[3*index];
  double dxt1 = dx[3*index+1];
  double dxt2 = dx[3*index+2];
  double detdx = 0.5*(pow(dxn,2.0)-(pow(dxt1,2.0)+pow(dxt2,2.0)));

  if(detdx>0 && dxn>0) {
    tx = 1.0;
  } else {
    tx = 0.5*(dxn*xn-dxt1*xt1-dxt2*xt2);
    aux = pow(tx,2.0)-detdx*detx;
    if(aux>0.0) {
      tx = 0.99*detx/(sqrt(aux)-tx);
    } else {
      tx = -0.99*detx/tx;
    }
  }

  xn = y[3*index];
  xt1 = y[3*index+1];
  xt2 = y[3*index+2];
  detx = 0.5*(pow(xn,2.0)-(pow(xt1,2.0)+pow(xt2,2.0)));

  dxn = dy[3*index];
  dxt1 = dy[3*index+1];
  dxt2 = dy[3*index+2];
  detdx = 0.5*(pow(dxn,2.0)-(pow(dxt1,2.0)+pow(dxt2,2.0)));

  if(detdx>0 && dxn>0) {
    ty = 1.0;
  } else {
    ty = 0.5*(dxn*xn-dxt1*xt1-dxt2*xt2);
    aux = pow(ty,2.0)-detdx*detx;
    if(aux>0.0) {
      ty = 0.99*detx/(sqrt(aux)-ty);
    } else {
      ty = -0.99*detx/ty;
    }
  }

  dst[3*index] = tx;
  dst[3*index+1] = ty;
  dst[3*index+2] = 1.0;
}

__global__ void getDeterminant(double* src, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double xn = src[3*index];
  double xt1 = src[3*index+1];
  double xt2 = src[3*index+2];

  double d = 0.5*(pow(xn,2.0)-(pow(xt1,2.0)+pow(xt2,2.0)));

  dst[3*index] = log(2.0*d);
  dst[3*index+1] = 0.0;
  dst[3*index+2] = 0.0;
}

__global__ void constructPw(int* PwI, int* PwJ, double* Pw, double* x, double* y, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double x0 = x[3*index];
  double x1 = x[3*index+1];
  double x2 = x[3*index+2];

  double y0 = y[3*index];
  double y1 = y[3*index+1];
  double y2 = y[3*index+2];

  double sqrtDetx = sqrt(0.5*(pow(x0,2.0) - (pow(x1,2.0)+pow(x2,2.0))));
  double sqrtDety = sqrt(0.5*(pow(y0,2.0) - (pow(y1,2.0)+pow(y2,2.0))));
  if(sqrtDetx!=sqrtDetx) sqrtDetx = 0.0;
  if(sqrtDety!=sqrtDety) sqrtDety = 0.0;

  PwI[9*index] = 3*index;
  PwJ[9*index] = 3*index;
  Pw[9*index] = pow(y0 + (sqrtDety*x0)/sqrtDetx,2.0)/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety) - sqrtDety/sqrtDetx;

  PwI[9*index+1] = 3*index;
  PwJ[9*index+1] = 3*index+1;
  Pw[9*index+1] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y1 - (sqrtDety*x1)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);

  PwI[9*index+2] = 3*index;
  PwJ[9*index+2] = 3*index+2;
  Pw[9*index+2] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);

  PwI[9*index+3] = 3*index+1;
  PwJ[9*index+3] = 3*index;
  Pw[9*index+3] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y1 - (sqrtDety*x1)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);

  PwI[9*index+4] = 3*index+1;
  PwJ[9*index+4] = 3*index+1;
  Pw[9*index+4] = pow(y1 - (sqrtDety*x1)/sqrtDetx,2.0)/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety) + sqrtDety/sqrtDetx;

  PwI[9*index+5] = 3*index+1;
  PwJ[9*index+5] = 3*index+2;
  Pw[9*index+5] = ((y1 - (sqrtDety*x1)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);

  PwI[9*index+6] = 3*index+2;
  PwJ[9*index+6] = 3*index;
  Pw[9*index+6] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);

  PwI[9*index+7] = 3*index+2;
  PwJ[9*index+7] = 3*index+1;
  Pw[9*index+7] = ((y1 - (sqrtDety*x1)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);

  PwI[9*index+8] = 3*index+2;
  PwJ[9*index+8] = 3*index+2;
  Pw[9*index+8] = pow(y2 - (sqrtDety*x2)/sqrtDetx,2.0)/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety) + sqrtDety/sqrtDetx;
}

__global__ void updatePw(double* Pw, double* x, double* y, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double x0 = x[3*index];
  double x1 = x[3*index+1];
  double x2 = x[3*index+2];

  double y0 = y[3*index];
  double y1 = y[3*index+1];
  double y2 = y[3*index+2];

  double sqrtDetx = sqrt(abs(0.5*(pow(x0,2.0) - (pow(x1,2.0)+pow(x2,2.0)))));
  double sqrtDety = sqrt(abs(0.5*(pow(y0,2.0) - (pow(y1,2.0)+pow(y2,2.0)))));
  //if(sqrtDetx!=sqrtDetx) sqrtDetx = 1e-4;//0.0;
  //if(sqrtDety!=sqrtDety) sqrtDety = 0.0;

  Pw[9*index] =   pow(y0 + (sqrtDety*x0)/sqrtDetx,2.0)/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety) - sqrtDety/sqrtDetx;
  Pw[9*index+1] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y1 - (sqrtDety*x1)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);
  Pw[9*index+2] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);
  Pw[9*index+3] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y1 - (sqrtDety*x1)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);
  Pw[9*index+4] = pow(y1 - (sqrtDety*x1)/sqrtDetx,2.0)/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety) + sqrtDety/sqrtDetx;
  Pw[9*index+5] = ((y1 - (sqrtDety*x1)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);
  Pw[9*index+6] = ((y0 + (sqrtDety*x0)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);
  Pw[9*index+7] = ((y1 - (sqrtDety*x1)/sqrtDetx)*(y2 - (sqrtDety*x2)/sqrtDetx))/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety);
  Pw[9*index+8] = pow(y2 - (sqrtDety*x2)/sqrtDetx,2.0)/(x0*y0 + x1*y1 + x2*y2 + 2*sqrtDetx*sqrtDety) + sqrtDety/sqrtDetx;
}

int JKIP::initializePw() {
  constructPw<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(PwI_d), CASTI1(PwJ_d), CASTD1(Pw_d), CASTD1(x_d), CASTD1(y_d), system->collisionDetector->numCollisions);

  // create Pw using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(PwI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + PwI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(PwJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + PwJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(Pw_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Pw_d.size());

  Pw = DeviceView(3*system->collisionDetector->numCollisions, 3*system->collisionDetector->numCollisions, Pw_d.size(), row_indices, column_indices, values);
  // end create Pw

  return 0;
}

__global__ void constructPinv(int* PinvI, int* PinvJ, double* Pinv, double* x, double* y, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  PinvI[9*index] = 3*index;
  PinvJ[9*index] = 3*index;
  Pinv[9*index] = 0;

  PinvI[9*index+1] = 3*index;
  PinvJ[9*index+1] = 3*index+1;
  Pinv[9*index+1] = 0;

  PinvI[9*index+2] = 3*index;
  PinvJ[9*index+2] = 3*index+2;
  Pinv[9*index+2] = 0;

  PinvI[9*index+3] = 3*index+1;
  PinvJ[9*index+3] = 3*index;
  Pinv[9*index+3] = 0;

  PinvI[9*index+4] = 3*index+1;
  PinvJ[9*index+4] = 3*index+1;
  Pinv[9*index+4] = 0;

  PinvI[9*index+5] = 3*index+1;
  PinvJ[9*index+5] = 3*index+2;
  Pinv[9*index+5] = 0;

  PinvI[9*index+6] = 3*index+2;
  PinvJ[9*index+6] = 3*index;
  Pinv[9*index+6] = 0;

  PinvI[9*index+7] = 3*index+2;
  PinvJ[9*index+7] = 3*index+1;
  Pinv[9*index+7] = 0;

  PinvI[9*index+8] = 3*index+2;
  PinvJ[9*index+8] = 3*index+2;
  Pinv[9*index+8] = 0;
}

int JKIP::initializePinv() {
  constructPinv<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(PinvI_d), CASTI1(PinvJ_d), CASTD1(Pinv_d), CASTD1(x_d), CASTD1(y_d), system->collisionDetector->numCollisions);
  Pinv_h = Pinv_d; // save on host for reset at each iteration TODO: reset on the GPU (might be faster to just recompute)

  // create Pinv using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(PinvI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + PinvI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(PinvJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + PinvJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(Pinv_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + Pinv_d.size());

  Pinv = DeviceView(3*system->collisionDetector->numCollisions, 3*system->collisionDetector->numCollisions, Pinv_d.size(), row_indices, column_indices, values);
  // end create Pinv

  return 0;
}

__global__ void updatePinv(double* Pinv, double* Pw, uint* bodyIdentifiersA, uint* bodyIdentifiersB, double* D, double* mass, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  int indexA = bodyIdentifiersA[index];
  int indexB = bodyIdentifiersB[index];

  double3 D_n = make_double3(D[18*index+0], D[18*index+1], D[18*index+2]);
  double3 D_u = make_double3(D[18*index+6], D[18*index+7], D[18*index+8]);
  double3 D_v = make_double3(D[18*index+12],D[18*index+13],D[18*index+14]);

  double Minv = mass[3*indexA]+mass[3*indexB];

  double P0 = Minv*D_n.x*D_n.x + Minv*D_n.y*D_n.y + Minv*D_n.z*D_n.z + Pw[9*index];
  double P1 = Pw[9*index+1] + D_n.x*D_u.x*Minv + D_n.y*D_u.y*Minv + D_n.z*D_u.z*Minv;
  double P2 = Pw[9*index+2] + D_n.x*D_v.x*Minv + D_n.y*D_v.y*Minv + D_n.z*D_v.z*Minv;
  double P3 = Pw[9*index+3] + D_n.x*D_u.x*Minv + D_n.y*D_u.y*Minv + D_n.z*D_u.z*Minv;
  double P4 = Minv*D_u.x*D_u.x + Minv*D_u.y*D_u.y + Minv*D_u.z*D_u.z + Pw[9*index+4];
  double P5 = Pw[9*index+5] + D_u.x*D_v.x*Minv + D_u.y*D_v.y*Minv + D_u.z*D_v.z*Minv;
  double P6 = Pw[9*index+6] + D_n.x*D_v.x*Minv + D_n.y*D_v.y*Minv + D_n.z*D_v.z*Minv;
  double P7 = Pw[9*index+7] + D_u.x*D_v.x*Minv + D_u.y*D_v.y*Minv + D_u.z*D_v.z*Minv;
  double P8 = Minv*D_v.x*D_v.x + Minv*D_v.y*D_v.y + Minv*D_v.z*D_v.z + Pw[9*index+8];

  double detP = (P0*P4*P8 - P0*P5*P7 - P1*P3*P8 + P1*P5*P6 + P2*P3*P7 - P2*P4*P6);

  Pinv[9*index] =   (P4*P8 - P5*P7)/detP;
  Pinv[9*index+1] = -(P1*P8 - P2*P7)/detP;
  Pinv[9*index+2] = (P1*P5 - P2*P4)/detP;
  Pinv[9*index+3] = -(P3*P8 - P5*P6)/detP;
  Pinv[9*index+4] = (P0*P8 - P2*P6)/detP;
  Pinv[9*index+5] = -(P0*P5 - P2*P3)/detP;
  Pinv[9*index+6] = (P3*P7 - P4*P6)/detP;
  Pinv[9*index+7] = -(P0*P7 - P1*P6)/detP;
  Pinv[9*index+8] = (P0*P4 - P1*P3)/detP;
}

__global__ void initializeImpulseVector(double* src, double* friction, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = friction[index];

  src[3*index  ] = mu;
  src[3*index+1] = 0.0;
  src[3*index+2] = 0.0;
}

__global__ void constructT(int* invTxI, int* invTxJ, double* invTx, int* TyI, int* TyJ, double* Ty, double* friction, uint numCollisions) {
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

int JKIP::initializeT() {
  constructT<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTI1(invTxI_d), CASTI1(invTxJ_d), CASTD1(invTx_d), CASTI1(TyI_d), CASTI1(TyJ_d), CASTD1(Ty_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);

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

double JKIP::updateAlpha(double s) {
  double alphaCen = 0.1;
  double alpha1 = 10;
  double betaCen;
  double beta1;
  double betaBD;
  double fcentering;
  double barrier;
  double beta;

  if(careful) {
    betaCen=0.1;
    beta1=0.5;
    betaBD=1.0;
  } else {
    betaCen=1e-30;
    beta1=0.1;
    betaBD=0.5;
  }

  // choose centering force
  double n = system->collisionDetector->numCollisions;
  double dotprod = cusp::blas::dot(x,y)+s;
  if(s > 0) {
    fcentering=2.0*(n+1.0)*log(dotprod/(n+1.0));
    barrier = log(pow(s,2.0));
  } else {
    fcentering=2.0*(n)*log(dotprod/n);
    barrier = 0.0;
  }

  getDeterminant<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(tmp_d), system->collisionDetector->numCollisions);
  barrier+= Thrust_Total(tmp_d);
  getDeterminant<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(y_d), CASTD1(tmp_d), system->collisionDetector->numCollisions);
  barrier+= Thrust_Total(tmp_d);

  fcentering = fcentering - barrier;

  if(fcentering != fcentering) fcentering = 0.0;

  if(fcentering < alphaCen) {
    beta = betaCen;
  } else if(fcentering <= alpha1) {
    beta = beta1;
  } else {
    beta = betaBD;
  }

  return 0.5*beta*dotprod/(n+1.0);
}

int JKIP::performSchurComplementProduct(DeviceValueArrayView src, DeviceValueArrayView tmp2) {
  cusp::multiply(invTx,src,tmp);
  cusp::multiply(system->DT,tmp,system->f_contact);
  cusp::multiply(system->mass,system->f_contact,system->tmp);
  cusp::multiply(system->D,system->tmp,tmp2);
  cusp::multiply(Ty,tmp2,tmp);

  return 0;
}

int JKIP::buildSchurMatrix() {
  // build N
  cusp::multiply(system->mass,system->DT,system->MinvDT);
  cusp::multiply(system->D,system->MinvDT,system->N);
  cusp::multiply(system->N,invTx,system->MinvDT);
  cusp::multiply(Ty,system->MinvDT,system->N);

  return 0;
}

int JKIP::solve() {
  solverOptions.relTol = std::min(0.01 * tolerance, 1e-6);
  solverOptions.absTol = 1e-10;

  // Initialize scalars
  double residual = 10e30;

  // vectors:  x, y, dx, dy, d, b
  system->gamma_d.resize(3*system->collisionDetector->numCollisions);
  x_d.resize(3*system->collisionDetector->numCollisions);
  y_d.resize(3*system->collisionDetector->numCollisions);
  dx_d.resize(3*system->collisionDetector->numCollisions);
  dy_d.resize(3*system->collisionDetector->numCollisions);
  d_d.resize(3*system->collisionDetector->numCollisions);
  b_d.resize(3*system->collisionDetector->numCollisions);
  r_d.resize(3*system->collisionDetector->numCollisions);
  tmp_d.resize(3*system->collisionDetector->numCollisions);

  // matrices: invTx, Ty, Pw, R(?)
  invTxI_d.resize(3*system->collisionDetector->numCollisions);
  invTxJ_d.resize(3*system->collisionDetector->numCollisions);
  invTx_d.resize(3*system->collisionDetector->numCollisions);

  TyI_d.resize(3*system->collisionDetector->numCollisions);
  TyJ_d.resize(3*system->collisionDetector->numCollisions);
  Ty_d.resize(3*system->collisionDetector->numCollisions);

  PwI_d.resize(9*system->collisionDetector->numCollisions);
  PwJ_d.resize(9*system->collisionDetector->numCollisions);
  Pw_d.resize(9*system->collisionDetector->numCollisions);

  PinvI_d.resize(9*system->collisionDetector->numCollisions);
  PinvJ_d.resize(9*system->collisionDetector->numCollisions);
  Pinv_d.resize(9*system->collisionDetector->numCollisions);

  // TODO: There's got to be a better way to do this...
  // vectors:  x, y, dx, dy, d, b
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(system->gamma_d));
  thrust::device_ptr<double> wrapped_device_x(CASTD1(x_d));
  thrust::device_ptr<double> wrapped_device_y(CASTD1(y_d));
  thrust::device_ptr<double> wrapped_device_dx(CASTD1(dx_d));
  thrust::device_ptr<double> wrapped_device_dy(CASTD1(dy_d));
  thrust::device_ptr<double> wrapped_device_d(CASTD1(d_d));
  thrust::device_ptr<double> wrapped_device_b(CASTD1(b_d));
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  thrust::device_ptr<double> wrapped_device_tmp(CASTD1(tmp_d));

  system->gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + system->gamma_d.size());
  x = DeviceValueArrayView(wrapped_device_x, wrapped_device_x + x_d.size());
  y = DeviceValueArrayView(wrapped_device_y, wrapped_device_y + y_d.size());
  dx = DeviceValueArrayView(wrapped_device_dx, wrapped_device_dx + dx_d.size());
  dy = DeviceValueArrayView(wrapped_device_dy, wrapped_device_dy + dy_d.size());
  d = DeviceValueArrayView(wrapped_device_d, wrapped_device_d + d_d.size());
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  tmp = DeviceValueArrayView(wrapped_device_tmp, wrapped_device_tmp + tmp_d.size());

  // initialize matrices and vectors
  initializeT();

  if(verbose) {
    cusp::print(invTx);
    cusp::print(Ty);

    cin.get();
  }

  initializePw();
  initializePinv();
  //buildSchurMatrix();

  cusp::multiply(Ty,system->r,r);
  initializeImpulseVector<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(system->friction_d), system->collisionDetector->numCollisions);
  performSchurComplementProduct(x,y); //NOTE: y is destroyed here
  cusp::blas::axpby(tmp,r,y,1.0,1.0);

  // determine initial alpha
  double alpha = cusp::blas::dot(x,y);
  alpha = 0.5*abs(alpha)/((double) system->collisionDetector->numCollisions);

  if(verbose) {
    cusp::print(system->DT);
    cusp::print(system->mass);
    cusp::print(system->D);
    cusp::print(r);
    cusp::print(x);
    cusp::print(y);

    cout << "alpha: " << alpha << endl;
  }

  // determine d vector
  double s = 2*alpha;
  initialize_d<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(y_d), CASTD1(d_d), s, system->collisionDetector->numCollisions);
  cusp::blas::axpy(d,y,s);

  bool feasible = false;

  alpha = updateAlpha(s);

  if(verbose) {
    cusp::print(x);
    cusp::print(y);
    cusp::print(d);
    cout << "alpha: " << alpha << endl;
  }

  double ds = 0;
  int k;
  totalKrylovIterations = 0;
  for (k=0; k < maxIterations; k++) {
    updatePw<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(Pw_d), CASTD1(x_d), CASTD1(y_d), system->collisionDetector->numCollisions);
    updatePinv<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(Pinv_d), CASTD1(Pw_d), CASTU1(system->collisionDetector->collisionIdentifierA_d), CASTU1(system->collisionDetector->collisionIdentifierB_d), CASTD1(system->D_d), CASTD1(system->mass_d), system->collisionDetector->numCollisions);

    if(verbose) {
      cusp::print(Pw);
      cusp::print(Pinv);
      cin.get();
    }

    if(feasible) {
      ds = 0.0;
    } else {
      ds = 2.0*alpha-s;
    }
    if(verbose) {
      cout << "ds: " << ds << endl;
      cin.get();
    }

    getInverse<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(b_d), system->collisionDetector->numCollisions);
    cusp::blas::axpbypcz(b,y,d,dx,alpha,-1.0,-ds); //b is in dx (dx is temporary here)
    cusp::multiply(Pinv,dx,b); // multiply by preconditioner!

    if(verbose) {
      cusp::print(b);
      cin.get();

      system->buildSchurMatrix(); //TODO: remove this!
      cusp::print(system->N);
      cin.get();
    }

    // solve system
    //delete mySolver;
    //m_spmv = new MySpmvJKIP(system->mass, system->D, system->DT, Pw, Pinv, Ty, invTx, system->tmp, system->f_contact, tmp);
    //mySolver = new SpikeSolver(partitions, solverOptions);
    mySolver->setup(Pw); // This should do nothing (minor setup, isn't building any preconditioner)

    cusp::blas::fill(dx, 0.0);
    bool success = mySolver->solve(*m_spmv, b, dx);
    spike::Stats stats = mySolver->getStats();
    if(verbose) {
      cusp::print(dx);
      cin.get();
    }

    performSchurComplementProduct(dx,dy); //NOTE: dy is destroyed here
    cusp::blas::axpby(tmp,d,dy,1.0,ds);
    if(verbose) {
      cusp::print(dy);
      cin.get();
    }

    getStepLength<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(dx_d), CASTD1(y_d), CASTD1(dy_d), CASTD1(tmp_d), system->collisionDetector->numCollisions);
    double theta = fmin(Thrust_Min(tmp_d),1.0);

    if(verbose) {
      cusp::print(x);
      cusp::print(dx);
      cusp::print(tmp);
      std::cout << "theta: " << theta << std::endl;
      cin.get();
    }

    cusp::blas::axpy(dx,x,theta);
    cusp::blas::axpy(dy,y,theta);
    s+=theta*ds;

    // check feasible and optimal
    getFeasible<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(x_d), CASTD1(tmp_d), system->collisionDetector->numCollisions);
    double feasibleX = Thrust_Max(tmp_d);
    cusp::blas::axpby(y,d,tmp,1.0,-s);

    //double optim = abs(cusp::blas::dot(x,tmp))/((double) system->collisionDetector->numCollisions);
    getResidual_JKIP<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(tmp_d), CASTD1(x_d), system->collisionDetector->numCollisions);
    double optim = cusp::blas::nrmmax(tmp);

    cusp::blas::axpby(y,d,tmp,1.0,-s);
    getFeasible<<<BLOCKS(system->collisionDetector->numCollisions),THREADS>>>(CASTD1(tmp_d), CASTD1(tmp_d), system->collisionDetector->numCollisions);
    double feasibleY = Thrust_Max(tmp_d);
    if(feasible==false && feasibleY == 0) {
      cusp::blas::axpy(d,y,-s);
      s = 0.0;
      feasible = true;
    }

    residual = fmax(feasibleX,feasibleY);
    residual = fmax(residual,optim);
    if (residual < tolerance) break;
    if(verbose) cusp::print(x);

    alpha = updateAlpha(s);
    totalKrylovIterations += stats.numIterations;

    if(verbose) {
      cout << "  Iterations: " << k << " Residual: " << residual << " Total Krylov iters: " << totalKrylovIterations << endl;
      cin.get();
    }
  }
  cusp::multiply(invTx,x,system->gamma);

  iterations = k;
  cout << "  Iterations: " << k << " Residual: " << residual << " Total Krylov iters: " << totalKrylovIterations << endl;
  return 0;
}
