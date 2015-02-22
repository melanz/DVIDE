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

  collisionDetector = new CollisionDetector(this);
  solver = new PDIP(this);
}

void System::setTimeStep(double step_size, double precision)
{
	h = step_size;

//	// Set tolerance for Newton iteration based on the precision in positions
//	// and integration step-size.
//	double safety = 1;////0.5;
//	tol = safety * precision / (h * h);
//
//	// Set the tolerances for Krylov
//	solverOptions.relTol = std::min(0.01 * tol, 1e-6);
//	solverOptions.absTol = 1e-10;
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

//  for(int i=0; i<3; i++) {
//    gamma_h.push_back(0);
//    gammaHat_h.push_back(0);
//    gammaNew_h.push_back(0);
//    g_h.push_back(0);
//    y_h.push_back(0);
//    yNew_h.push_back(0);
//    gammaTmp_h.push_back(0);
//  }

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
  gamma_d = a_h;

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

	p = DeviceValueArrayView(wrapped_device_p, wrapped_device_p + p_d.size());
	v = DeviceValueArrayView(wrapped_device_v, wrapped_device_v + v_d.size());
	a = DeviceValueArrayView(wrapped_device_a, wrapped_device_a + a_d.size());
	f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
	f_contact = DeviceValueArrayView(wrapped_device_f_contact, wrapped_device_f_contact + f_contact_d.size());
	tmp = DeviceValueArrayView(wrapped_device_tmp, wrapped_device_tmp + tmp_d.size());
	r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
	k = DeviceValueArrayView(wrapped_device_k, wrapped_device_k + k_d.size());
	gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + gamma_d.size());

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
	solver->setup();

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
    solver->solve();

    // Perform time integration (contacts)
    cusp::multiply(DT,gamma,v);
    cusp::blas::axpby(k,v,tmp,1.0,1.0);
    cusp::multiply(mass,tmp,v);
  }
  else {
    // Perform time integration (no contacts)
    cusp::multiply(mass,k,v);
  }

  // Apply sinusoidal motion
  v_h = v_d;
  for(int i=0;i<5;i++) {
    v_h[3*i] = v_h[3*i]+4.0*sin(time*3.0);
    v_h[3*i+1] = 0;
    v_h[3*i+2] = 0;
  }
  v_d = v_h;

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

__global__ void multiplyByMass(double* massInv, double* src, double* dst, uint numDOF) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numDOF);

  double mass = massInv[index];
  if(mass) mass = 1.0/mass;
  dst[index] = mass*src[index];
}

int System::buildAppliedImpulseVector() {
  // build k
  multiplyByMass<<<BLOCKS(v_d.size()),THREADS>>>(CASTD1(mass_d), CASTD1(v_d), CASTD1(k_d), v_d.size());
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
  // TODO: There's got to be a better way to do this...
  //r.resize(3*collisionDetector->numCollisions);
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  cusp::multiply(mass,k,tmp);
  cusp::multiply(D,tmp,r);

  applyStabilization<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(r_d), CASTD4(collisionDetector->normalsAndPenetrations_d), h, collisionDetector->numCollisions);

  return 0;
}
