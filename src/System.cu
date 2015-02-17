#include <algorithm>
#include <vector>
#include "include.cuh"
#include "System.cuh"

System::System()
{
  gravity = make_double3(0,-9.81,0);
  tol = 1e-8;
  h = 1e-3;
  timeIndex = 0;
  time = 0;

	// spike stuff
	partitions = 1;
	solverOptions.safeFactorization = true;
	solverOptions.trackReordering = true;
	solverOptions.maxNumIterations = 5000;
	preconditionerUpdateModulus = -1; // the preconditioner updates every ___ time steps
	preconditionerMaxKrylovIterations = -1; // the preconditioner updates if Krylov iterations are greater than ____ iterations
	//mySolver = new SpikeSolver(partitions, solverOptions);
	//m_spmv = new MySpmv(mass);
  stepKrylovIterations = 0;
  precUpdated = 0;
	// end spike stuff

  collisionDetector = new CollisionDetector(this);
}

void System::setSolverType(int solverType)
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
	}
}

void System::setPrecondType(int useSpike)
{
	solverOptions.precondType = useSpike ? spike::Spike : spike::None;
}

void System::setTimeStep(double step_size, double precision)
{
	h = step_size;

	// Set tolerance for Newton iteration based on the precision in positions
	// and integration step-size.
	double safety = 1;////0.5;
	tol = safety * precision / (h * h);

	// Set the tolerances for Krylov
	solverOptions.relTol = std::min(0.01 * tol, 1e-6);
	solverOptions.absTol = 1e-10;
}

void System::printSolverParams()
{
	printf("Step size: %e\n", h);
	printf("Newton tolerance: %e\n", tol);
	printf("Krylov relTol: %e  abdTol: %e\n", solverOptions.relTol, solverOptions.absTol);
	printf("Max. Krylov iterations: %d\n", solverOptions.maxNumIterations);
	printf("----------------------------\n");
}

int System::add(Body* body) {
  // TODO: make this function general for any Body
	//add the element
  body->setIndex(p_h.size()); // Indicates the Body's location in the position array
  indices_h.push_back(p_h.size()); // Push Body's location to global library
  body->setIdentifier(bodies.size()); // Indicates the number that the Body was added
	bodies.push_back(body);

	// update p
	p_h.push_back(body->pos.x);
	p_h.push_back(body->pos.y);
	p_h.push_back(body->pos.z);

  // update v
  v_h.push_back(body->vel.x);
  v_h.push_back(body->vel.y);
  v_h.push_back(body->vel.z);

  // update a
  a_h.push_back(body->acc.x);
  a_h.push_back(body->acc.y);
  a_h.push_back(body->acc.z);

	// update external force vector (gravity)
  if(body->isFixed()) {
    f_h.push_back(0);
    f_h.push_back(0);
    f_h.push_back(0);
  }
  else {
    f_h.push_back(body->mass * this->gravity.x);
    f_h.push_back(body->mass * this->gravity.y);
    f_h.push_back(body->mass * this->gravity.z);
  }

  f_contact_h.push_back(0);
  f_contact_h.push_back(0);
  f_contact_h.push_back(0);

  tmp_h.push_back(0);
  tmp_h.push_back(0);
  tmp_h.push_back(0);

  r_h.push_back(0);
  r_h.push_back(0);
  r_h.push_back(0);

  k_h.push_back(0);
  k_h.push_back(0);
  k_h.push_back(0);

  for(int i=0; i<3; i++) {
    gamma_h.push_back(0);
    gammaHat_h.push_back(0);
    gammaNew_h.push_back(0);
    g_h.push_back(0);
    y_h.push_back(0);
    yNew_h.push_back(0);
    gammaTmp_h.push_back(0);
  }

	// update the mass matrix
	for (int i = 0; i < body->numDOF; i++) {
      massI_h.push_back(i + body->numDOF * (bodies.size() - 1));
      massJ_h.push_back(i + body->numDOF * (bodies.size() - 1));
      if(body->isFixed()) {
        mass_h.push_back(0);
      }
      else {
        mass_h.push_back(1.0/body->mass);
      }
	}

	contactGeometry_h.push_back(body->contactGeometry);

	return bodies.size();
}

int System::initializeDevice() {
  indices_d = indices_h;
	p_d = p_h;
	v_d = v_h;
	a_d = a_h;
	f_d = f_h;
	f_contact_d = f_contact_h;
	tmp_d = tmp_h;
	r_d = r_h;
	k_d = k_h;
  gamma_d = gamma_h;
  gammaHat_d = gammaHat_h;
  gammaNew_d = gammaNew_h;
  g_d = g_h;
  y_d = y_h;
  yNew_d = yNew_h;
  gammaTmp_d = gammaTmp_h;

	massI_d = massI_h;
	massJ_d = massJ_h;
	mass_d = mass_h;

	contactGeometry_d = contactGeometry_h;
	fixedBodies_d = fixedBodies_h;

	thrust::device_ptr<double> wrapped_device_p(CASTD1(p_d));
	thrust::device_ptr<double> wrapped_device_v(CASTD1(v_d));
	thrust::device_ptr<double> wrapped_device_a(CASTD1(a_d));
	thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
	thrust::device_ptr<double> wrapped_device_f_contact(CASTD1(f_contact_d));
	thrust::device_ptr<double> wrapped_device_tmp(CASTD1(tmp_d));
	thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
	thrust::device_ptr<double> wrapped_device_k(CASTD1(k_d));
	thrust::device_ptr<double> wrapped_device_gamma(CASTD1(gamma_d));
	thrust::device_ptr<double> wrapped_device_gammaHat(CASTD1(gammaHat_d));
	thrust::device_ptr<double> wrapped_device_gammaNew(CASTD1(gammaNew_d));
	thrust::device_ptr<double> wrapped_device_g(CASTD1(g_d));
	thrust::device_ptr<double> wrapped_device_y(CASTD1(y_d));
	thrust::device_ptr<double> wrapped_device_yNew(CASTD1(yNew_d));
	thrust::device_ptr<double> wrapped_device_gammaTmp(CASTD1(gammaTmp_d));

	p = DeviceValueArrayView(wrapped_device_p, wrapped_device_p + p_d.size());
	v = DeviceValueArrayView(wrapped_device_v, wrapped_device_v + v_d.size());
	a = DeviceValueArrayView(wrapped_device_a, wrapped_device_a + a_d.size());
	f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
	f_contact = DeviceValueArrayView(wrapped_device_f_contact, wrapped_device_f_contact + f_contact_d.size());
	tmp = DeviceValueArrayView(wrapped_device_tmp, wrapped_device_tmp + tmp_d.size());
	r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
	k = DeviceValueArrayView(wrapped_device_k, wrapped_device_k + k_d.size());
	gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + gamma_d.size());
	gammaHat = DeviceValueArrayView(wrapped_device_gammaHat, wrapped_device_gammaHat + gammaHat_d.size());
	gammaNew = DeviceValueArrayView(wrapped_device_gammaNew, wrapped_device_gammaNew + gammaNew_d.size());
	g = DeviceValueArrayView(wrapped_device_g, wrapped_device_g + g_d.size());
	y = DeviceValueArrayView(wrapped_device_y, wrapped_device_y + y_d.size());
	yNew = DeviceValueArrayView(wrapped_device_yNew, wrapped_device_yNew + yNew_d.size());
	gammaTmp = DeviceValueArrayView(wrapped_device_gammaTmp, wrapped_device_gammaTmp + gammaTmp_d.size());

	// create mass matrix using cusp library (shouldn't change)
	thrust::device_ptr<int> wrapped_device_I(CASTI1(massI_d));
	DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + massI_d.size());

	thrust::device_ptr<int> wrapped_device_J(CASTI1(massJ_d));
	DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + massJ_d.size());

	thrust::device_ptr<double> wrapped_device_V(CASTD1(mass_d));
	DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + mass_d.size());

	mass = DeviceView(a_d.size(), a_d.size(), mass_d.size(), row_indices, column_indices, values);
	// end create mass matrix

	return 0;
}

int System::initializeSystem() {

  // update the contact geometry and fixed bodies
  for(int i=0; i<bodies.size(); i++) {
    contactGeometry_h[i] = bodies[i]->contactGeometry;
    if(bodies[i]->isFixed()) fixedBodies_h.push_back(i);
  }

	initializeDevice();

	// create and setup the Spike::GPU solver
	//m_spmv = new MySpmv(mass);
	//mySolver = new SpikeSolver(partitions, solverOptions);
	//mySolver->setup(mass);

	//bool success = mySolver->solve(*m_spmv, f, a);

	return 0;
}

int System::DoTimeStep() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// Perform collision detection
	collisionDetector->generateAxisAlignedBoundingBoxes();
	collisionDetector->detectPossibleCollisions_spatialSubdivision();
  collisionDetector->detectCollisions();

  buildAppliedImpulseVector();
  if(collisionDetector->numCollisions) {
    // Set up the QOCC
    buildContactJacobian();
    buildRightHandSideVector();

    // Solve the QOCC
    solve_APGD();

    // Perform time integration (contacts)
    cusp::multiply(DT,gamma,v);
    cusp::blas::axpby(k,v,tmp,1.0,1.0);
    cusp::multiply(mass,tmp,v);
  }
  else {
    // Perform time integration (no contacts)
    cusp::multiply(mass,k,v);
  }

  cusp::blas::axpy(v, p, h);

  time += h;
  timeIndex++;
  p_h = p_d;

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	printf("Time: %f (Exec. Time: %f), Collisions: %d (%d possible)\n",time,elapsedTime,collisionDetector->numCollisions, (int)collisionDetector->numPossibleCollisions);

	return 0;
}

int System::applyContactForces_CPU() {
  Thrust_Fill(f_contact_h,0);

  for(int i=0; i<collisionDetector->normalsAndPenetrations_h.size(); i++) {
    uint bodyA = collisionDetector->bodyIdentifierA_h[i];
    uint bodyB = collisionDetector->bodyIdentifierB_h[i];
    double4 nAndP = collisionDetector->normalsAndPenetrations_h[i];
    double3 normal = make_double3(nAndP.x,nAndP.y,nAndP.z);
    double penetration = nAndP.w;

    double sigmaA = (1.0-0.25)/2.0e7;
    double sigmaB = sigmaA;
    double rA = 0.4;
    double rB = 0.4;
    double3 contactForce = 4.0/(3.0*(sigmaA+sigmaB))*sqrt(rA*rB/(rA+rB))*pow(penetration,1.5)*normal;

    // Add damping
    v_h = v_d;
    double3 v = make_double3(v_h[indices_h[bodyB]]-v_h[indices_h[bodyA]],v_h[indices_h[bodyB]+1]-v_h[indices_h[bodyA]+1],v_h[indices_h[bodyB]+2]-v_h[indices_h[bodyA]+2]);
    double b = 250;
    double3 damping;
    damping.x = b * normal.x * normal.x * v.x + b * normal.x * normal.y * v.y + b * normal.x * normal.z * v.z;
    damping.y = b * normal.x * normal.y * v.x + b * normal.y * normal.y * v.y + b * normal.y * normal.z * v.z;
    damping.z = b * normal.x * normal.z * v.x + b * normal.y * normal.z * v.y + b * normal.z * normal.z * v.z;
    if(penetration>=0) contactForce -= damping;

    f_contact_h[indices_h[bodyA]]   -= contactForce.x;
    f_contact_h[indices_h[bodyA]+1] -= contactForce.y;
    f_contact_h[indices_h[bodyA]+2] -= contactForce.z;

    f_contact_h[indices_h[bodyB]]   += contactForce.x;
    f_contact_h[indices_h[bodyB]+1] += contactForce.y;
    f_contact_h[indices_h[bodyB]+2] += contactForce.z;

  }
  f_contact_d = f_contact_h;

  return 0;
}

__global__ void constructContactJacobian(int* DI, int* DJ, double* D, double4* normalsAndPenetrations, uint* bodyIdentifierA, uint* bodyIdentifierB, int* indices, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double4 nAndP;
  double3 n, u, v;
  uint bodyA, bodyB;
  bodyA = bodyIdentifierA[index];
  bodyB = bodyIdentifierB[index];
  nAndP = normalsAndPenetrations[index];
  n = make_double3(nAndP.x,nAndP.y,nAndP.z);

  if(n.z != 0) {
    u = normalize(make_double3(1,0,-n.x/n.z));
  }
  else if(n.x != 0) {
    u = normalize(make_double3(-n.z/n.x,0,1));
  }
  else {
    u = normalize(make_double3(1,-n.x/n.y,0));
  }
  v = normalize(cross(n,u));

  // Add n, i indices
  DI[18*index+0 ] = 3*index+0;
  DI[18*index+1 ] = 3*index+0;
  DI[18*index+2 ] = 3*index+0;
  DI[18*index+3 ] = 3*index+0;
  DI[18*index+4 ] = 3*index+0;
  DI[18*index+5 ] = 3*index+0;

  // Add u, i indices
  DI[18*index+6 ] = 3*index+1;
  DI[18*index+7 ] = 3*index+1;
  DI[18*index+8 ] = 3*index+1;
  DI[18*index+9 ] = 3*index+1;
  DI[18*index+10] = 3*index+1;
  DI[18*index+11] = 3*index+1;

  // Add v, i indices
  DI[18*index+12] = 3*index+2;
  DI[18*index+13] = 3*index+2;
  DI[18*index+14] = 3*index+2;
  DI[18*index+15] = 3*index+2;
  DI[18*index+16] = 3*index+2;
  DI[18*index+17] = 3*index+2;

  // Add n, j indices
  DJ[18*index+0 ] = indices[bodyA]+0;
  DJ[18*index+1 ] = indices[bodyA]+1;
  DJ[18*index+2 ] = indices[bodyA]+2;
  DJ[18*index+3 ] = indices[bodyB]+0;
  DJ[18*index+4 ] = indices[bodyB]+1;
  DJ[18*index+5 ] = indices[bodyB]+2;

  // Add u, j indices
  DJ[18*index+6 ] = indices[bodyA]+0;
  DJ[18*index+7 ] = indices[bodyA]+1;
  DJ[18*index+8 ] = indices[bodyA]+2;
  DJ[18*index+9 ] = indices[bodyB]+0;
  DJ[18*index+10] = indices[bodyB]+1;
  DJ[18*index+11] = indices[bodyB]+2;

  // Add v, j indices
  DJ[18*index+12] = indices[bodyA]+0;
  DJ[18*index+13] = indices[bodyA]+1;
  DJ[18*index+14] = indices[bodyA]+2;
  DJ[18*index+15] = indices[bodyB]+0;
  DJ[18*index+16] = indices[bodyB]+1;
  DJ[18*index+17] = indices[bodyB]+2;

  // Add n, values
  D[18*index+0 ] = n.x;
  D[18*index+1 ] = n.y;
  D[18*index+2 ] = n.z;
  D[18*index+3 ] = -n.x;
  D[18*index+4 ] = -n.y;
  D[18*index+5 ] = -n.z;

  // Add u, values
  D[18*index+6 ] = u.x;
  D[18*index+7 ] = u.y;
  D[18*index+8 ] = u.z;
  D[18*index+9 ] = -u.x;
  D[18*index+10] = -u.y;
  D[18*index+11] = -u.z;

  // Add v, values
  D[18*index+12] = v.x;
  D[18*index+13] = v.y;
  D[18*index+14] = v.z;
  D[18*index+15] = -v.x;
  D[18*index+16] = -v.y;
  D[18*index+17] = -v.z;
}

int System::buildContactJacobian() {
  DI_d.resize(18*collisionDetector->numCollisions);
  DJ_d.resize(18*collisionDetector->numCollisions);
  D_d.resize(18*collisionDetector->numCollisions);
  constructContactJacobian<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), CASTD4(collisionDetector->normalsAndPenetrations_d), CASTU1(collisionDetector->bodyIdentifierA_d), CASTU1(collisionDetector->bodyIdentifierB_d), CASTI1(indices_d), collisionDetector->numCollisions);

  // create contact jacobian using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(DI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + DI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(DJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + DJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(D_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + D_d.size());

  D = DeviceView(3*collisionDetector->numCollisions, 3*bodies.size(), D_d.size(), row_indices, column_indices, values);
  // end create contact jacobian

  buildContactJacobianTranspose();

  return 0;
}

int System::buildContactJacobianTranspose() {
  DTI_d = DJ_d;
  DTJ_d = DI_d;
  DT_d = D_d;

  // create contact jacobian using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(DTI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + DI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(DTJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + DJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(DT_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + D_d.size());

  DT = DeviceView(3*bodies.size(), 3*collisionDetector->numCollisions, DT_d.size(), row_indices, column_indices, values);
  // end create contact jacobian

  DT.sort_by_row(); // TODO: Do I need this?

  return 0;
}

int System::performSchurComplementProduct(DeviceValueArrayView src) {
  cusp::multiply(DT,src,f_contact);
  cusp::multiply(mass,f_contact,tmp);
  cusp::multiply(D,tmp,gammaTmp);

  return 0;
}

int System::multiplyByMass_CPU(thrust::device_vector<double> src, thrust::device_vector<double> dst) {
  // TODO: perform in parallel
  thrust::host_vector<double> src_h = src;
  thrust::host_vector<double> dst_h = dst;

  for(int i=0;i<bodies.size();i++) {
    int index = indices_h[i];
    double mass = 0;
    if(!bodies[i]->fixed) mass = bodies[i]->mass;
    dst_h[index] = mass*src_h[index];
    dst_h[index+1] = bodies[i]->mass*src_h[index+1];
    dst_h[index+2] = bodies[i]->mass*src_h[index+2];
  }

  dst = dst_h;

  return 0;
}

__global__ void massMultiply(double* massInv, double* src, double* dst, uint numDOF) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numDOF);

  double mass = massInv[index];
  if(mass) mass = 1.0/mass;
  dst[index] = mass*dst[index];
}

int System::multiplyByMass(thrust::device_vector<double> src, thrust::device_vector<double> dst) {
  massMultiply<<<BLOCKS(v_d.size()),THREADS>>>(CASTD1(mass_d), CASTD1(src), CASTD1(dst), v_d.size());

  return 0;
}

int System::buildAppliedImpulseVector() {
  // build k
  multiplyByMass(v_d,k_d);
  cusp::blas::axpy(f,k,h);

  return 0;
}

__global__ void applyStabilization(double* r, double4* normalsAndPenetrations, double timeStep, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double penetration = normalsAndPenetrations[index].w;
  if(penetration>0) penetration = 0;

  r[3*index] += penetration/timeStep;
}

int System::buildRightHandSideVector() {
  // build r
  r_d.resize(3*collisionDetector->numCollisions);
  r.resize(3*collisionDetector->numCollisions);
  cusp::multiply(mass,k,tmp);
  cusp::multiply(D,tmp,r);

  applyStabilization<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(r_d), CASTD4(collisionDetector->normalsAndPenetrations_d), h, collisionDetector->numCollisions);

  return 0;
}

__global__ void performProjection(double* src, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double mu = 0;
  double3 gamma = make_double3(src[3*index],src[3*index+1],src[3*index+2]);
  double gamma_n = gamma.x;
  double gamma_t = sqrt(pow(gamma.y,2.0)+pow(gamma.z,2.0));

  if(gamma_n <= -mu*gamma_t) {
    gamma = make_double3(0,0,0);
  }
  else if(pow(gamma.y,2.0)+pow(gamma.z,2.0) > pow(mu*gamma_n,2.0)) {
    gamma = make_double3(
        (gamma_n+mu*gamma_t)/(mu*mu+1.0),
        gamma.y*mu*gamma_n/gamma_t,
        gamma.z*mu*gamma_n/gamma_t);
  }

  src[3*index] = gamma.x;
  src[3*index+1] = gamma.y;
  src[3*index+2] = gamma.z;
}

int System::project(thrust::device_vector<double> src) {
  performProjection<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(src), collisionDetector->numCollisions);

  return 0;
}

double System::getResidual(DeviceValueArrayView src) {
  double gdiff = 1.0 / pow(collisionDetector->numCollisions,2.0);
  performSchurComplementProduct(gamma);
  cusp::blas::axpy(r,gammaTmp,1.0);
  cusp::blas::axpby(gamma,gammaTmp,gammaTmp,1.0,-gdiff);
  project(gammaTmp_d);
  cusp::blas::axpby(gamma,gammaTmp,gammaTmp,1.0/gdiff,-1.0/gdiff);

  return cusp::blas::nrmmax(gammaTmp);
}

int System::solve_APGD() {
  int maxIterations = 1000;
  double tolerance = 1e-4;

  gamma_d.resize(3*collisionDetector->numCollisions);
  gammaHat_d.resize(3*collisionDetector->numCollisions);
  gammaNew_d.resize(3*collisionDetector->numCollisions);
  g_d.resize(3*collisionDetector->numCollisions);
  y_d.resize(3*collisionDetector->numCollisions);
  yNew_d.resize(3*collisionDetector->numCollisions);
  gammaTmp_d.resize(3*collisionDetector->numCollisions);

  gamma.resize(3*collisionDetector->numCollisions);
  gammaHat.resize(3*collisionDetector->numCollisions);
  gammaNew.resize(3*collisionDetector->numCollisions);
  g.resize(3*collisionDetector->numCollisions);
  y.resize(3*collisionDetector->numCollisions);
  yNew.resize(3*collisionDetector->numCollisions);
  gammaTmp.resize(3*collisionDetector->numCollisions);

  // (1) gamma_0 = zeros(nc,1)
  cusp::blas::fill(gamma,0);

  // (2) gamma_hat_0 = ones(nc,1)
  cusp::blas::fill(gammaHat,1.0);

  // (3) y_0 = gamma_0
  cusp::blas::copy(gamma,y);

  // (4) theta_0 = 1
  double theta = 1.0;
  double thetaNew = theta;
  double Beta = 0.0;
  double obj1 = 0.0;
  double obj2 = 0.0;
  double residual = 10e30;

  // (5) L_k = norm(N * (gamma_0 - gamma_hat_0)) / norm(gamma_0 - gamma_hat_0)
  cusp::blas::axpby(gamma,gammaHat,gammaTmp,1.0,-1.0);
  double L = cusp::blas::nrm2(gammaTmp);
  performSchurComplementProduct(gammaTmp);
  L = cusp::blas::nrm2(gammaTmp)/L;

  // (6) t_k = 1 / L_k
  double t = 1.0/L;

  // (7) for k := 0 to N_max
  int k;
  for (k=0; k < maxIterations; k++) {
    // (8) g = N * y_k - r
    performSchurComplementProduct(y);
    cusp::blas::copy(gammaTmp,g);
    cusp::blas::axpy(r,g,1.0);

    // (9) gamma_(k+1) = ProjectionOperator(y_k - t_k * g)
    cusp::blas::axpby(y,g,gammaNew,1.0,-t);
    project(gammaNew_d);

    // (10) while 0.5 * gamma_(k+1)' * N * gamma_(k+1) - gamma_(k+1)' * r >= 0.5 * y_k' * N * y_k - y_k' * r + g' * (gamma_(k+1) - y_k) + 0.5 * L_k * norm(gamma_(k+1) - y_k)^2
    performSchurComplementProduct(gammaNew);
    obj1 = 0.5 * cusp::blas::dot(gammaNew,gammaTmp) + cusp::blas::dot(gammaNew,r);
    performSchurComplementProduct(y);
    obj2 = 0.5 * cusp::blas::dot(y,gammaTmp) + cusp::blas::dot(y,r);
    cusp::blas::axpby(gammaNew,y,gammaTmp,1.0,-1.0);
    obj2 += cusp::blas::dot(g,gammaTmp) + 0.5 * L * pow(cusp::blas::nrm2(gammaTmp),2.0);

    while (obj1 >= obj2) {
      // (11) L_k = 2 * L_k
      L = 2.0 * L;

      // (12) t_k = 1 / L_k
      t = 1.0 / L;

      // (13) gamma_(k+1) = ProjectionOperator(y_k - t_k * g)
      cusp::blas::axpby(y,g,gammaNew,1.0,-t);
      project(gammaNew_d);

      // Update the components of the while condition
      performSchurComplementProduct(gammaNew);
      obj1 = 0.5 * cusp::blas::dot(gammaNew,gammaTmp) + cusp::blas::dot(gammaNew,r);
      performSchurComplementProduct(y);
      obj2 = 0.5 * cusp::blas::dot(y,gammaTmp) + cusp::blas::dot(y,r);
      cusp::blas::axpby(gammaNew,y,gammaTmp,1.0,-1.0);
      obj2 += cusp::blas::dot(g,gammaTmp) + 0.5 * L * pow(cusp::blas::nrm2(gammaTmp),2.0);

      // (14) endwhile
    }

    // (15) theta_(k+1) = (-theta_k^2 + theta_k * sqrt(theta_k^2 + 4)) / 2
    thetaNew = (-pow(theta, 2.0) + theta * sqrt(pow(theta, 2.0) + 4.0)) / 2.0;

    // (16) Beta_(k+1) = theta_k * (1 - theta_k) / (theta_k^2 + theta_(k+1))
    Beta = theta * (1.0 - theta) / (pow(theta, 2.0) + thetaNew);

    // (17) y_(k+1) = gamma_(k+1) + Beta_(k+1) * (gamma_(k+1) - gamma_k)
    cusp::blas::axpby(gammaNew,gamma,yNew,(1.0+Beta),-Beta);

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

    // (23) if r < Tau
    if (residual < tolerance) {
      // (24) break
      break;

      // (25) endif
    }

    // (26) if g' * (gamma_(k+1) - gamma_k) > 0
    cusp::blas::axpby(gammaNew,gamma,gammaTmp,1.0,-1.0);
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
    cusp::blas::copy(gammaNew,gamma);
    cusp::blas::copy(yNew,y);

    // (32) endfor
  }
  cout << "  Iterations: " << k << endl;

  // (33) return Value at time step t_(l+1), gamma_(l+1) := gamma_hat
  cusp::blas::copy(gammaHat,gamma);

  return 0;
}
