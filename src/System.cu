#include <algorithm>
#include <vector>
#include "include.cuh"
#include "System.cuh"
#include "Solver.cuh"
#include "APGD.cuh"
#include "PDIP.cuh"
#include "TPAS.cuh"
#include "JKIP.cuh"

System::System()
{
  gravity = make_double3(0,-9.81,0);
  tol = 1e-8;
  h = 1e-3;
  timeIndex = 0;
  time = 0;
  elapsedTime = 0;

  collisionDetector = new CollisionDetector(this);
  solver = new APGD(this);
}

System::System(int solverType)
{
  gravity = make_double3(0,-9.81,0);
  tol = 1e-8;
  h = 1e-3;
  timeIndex = 0;
  time = 0;
  elapsedTime = 0;

  collisionDetector = new CollisionDetector(this);

  switch(solverType) {
  case 1:
    solver = new APGD(this);
    break;
  case 2:
    solver = new PDIP(this);
    break;
  case 3:
    solver = new TPAS(this);
    break;
  case 4:
    solver = new JKIP(this);
    break;
  default:
    solver = new APGD(this);
  }
}

void System::setTimeStep(double step_size)
{
  h = step_size;
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

  fApplied_h.push_back(0);
  fApplied_h.push_back(0);
  fApplied_h.push_back(0);
  fApplied_d = fApplied_h;

  tmp_h.push_back(0);
  tmp_h.push_back(0);
  tmp_h.push_back(0);

  r_h.push_back(0);
  r_h.push_back(0);
  r_h.push_back(0);

  r_h.push_back(0);
  r_h.push_back(0);
  r_h.push_back(0);

  k_h.push_back(0);
  k_h.push_back(0);
  k_h.push_back(0);

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
  b_d = b_h;
  k_d = k_h;
  gamma_d = a_h;
  friction_d = a_h;
  fApplied_d = fApplied_h;

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
  thrust::device_ptr<double> wrapped_device_fApplied(CASTD1(fApplied_d));
  thrust::device_ptr<double> wrapped_device_tmp(CASTD1(tmp_d));
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  thrust::device_ptr<double> wrapped_device_b(CASTD1(b_d));
  thrust::device_ptr<double> wrapped_device_k(CASTD1(k_d));
  thrust::device_ptr<double> wrapped_device_gamma(CASTD1(gamma_d));

  p = DeviceValueArrayView(wrapped_device_p, wrapped_device_p + p_d.size());
  v = DeviceValueArrayView(wrapped_device_v, wrapped_device_v + v_d.size());
  a = DeviceValueArrayView(wrapped_device_a, wrapped_device_a + a_d.size());
  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  f_contact = DeviceValueArrayView(wrapped_device_f_contact, wrapped_device_f_contact + f_contact_d.size());
  fApplied = DeviceValueArrayView(wrapped_device_fApplied, wrapped_device_fApplied + fApplied_d.size());
  tmp = DeviceValueArrayView(wrapped_device_tmp, wrapped_device_tmp + tmp_d.size());
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
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
    buildSchurVector();

    // Solve the QOCC
    solver->solve();

    // Perform time integration (contacts)
    cusp::multiply(DT,gamma,f_contact);
    cusp::blas::axpby(k,f_contact,tmp,1.0,1.0);
    cusp::multiply(mass,tmp,v);
    cusp::blas::scal(f_contact,1.0/h);
  }
  else {
    // Perform time integration (no contacts)
    cusp::multiply(mass,k,v);

    cusp::blas::fill(f_contact,0.0);
  }

  if(time>1.5) {
    // Apply sinusoidal motion
    v_h = v_d;
    for(int i=0;i<1;i++) {
      v_h[3*i] = -0.20;//v_h[3*i]+4.0*sin((time-2)*3.0);
      v_h[3*i+1] = 0;
      v_h[3*i+2] = 0;
    }
    v_d = v_h;
  }

  cusp::blas::axpy(v, p, h);

  time += h;
  timeIndex++;
  p_h = p_d;

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float execTime;
  cudaEventElapsedTime(&execTime, start, stop);
  elapsedTime = execTime;

  printf("Time: %f (Exec. Time: %f), Collisions: %d (%d possible)\n",time,elapsedTime,collisionDetector->numCollisions, (int)collisionDetector->numPossibleCollisions);

  size_t avail;
  size_t total;
  cudaMemGetInfo( &avail, &total );
  size_t used = total - avail;
  totalGPUMemoryUsed = used/1000000.0;
  cout << "  Device memory used: " << totalGPUMemoryUsed << " MB (Avail: " << avail/1000000 << " MB)" << endl;

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

int System::applyForce(Body* body, double3 force) {
  int index = body->getIndex();
  cout << index << endl;

  fApplied_h = fApplied_d;
  fApplied_h[index]+=force.x;
  fApplied_h[index+1]+=force.y;
  fApplied_h[index+2]+=force.z;
  fApplied_d = fApplied_h;

  return 0;
}

int System::clearAppliedForces() {
  Thrust_Fill(fApplied_d,0.0);

  return 0;
}

__global__ void constructContactJacobian(int* DI, int* DJ, double* D, double* friction, double4* normalsAndPenetrations, uint* bodyIdentifierA, uint* bodyIdentifierB, int* indices, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  friction[index] = 0.25; // TODO: EDIT THIS TO BE MINIMUM OF FRICTION COEFFICIENTS

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
  friction_d.resize(collisionDetector->numCollisions);

  constructContactJacobian<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), CASTD1(friction_d), CASTD4(collisionDetector->normalsAndPenetrations_d), CASTU1(collisionDetector->bodyIdentifierA_d), CASTU1(collisionDetector->bodyIdentifierB_d), CASTI1(indices_d), collisionDetector->numCollisions);

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
  cusp::blas::axpbypcz(f,fApplied,k,k,h,h,1);

  return 0;
}

__global__ void buildStabilization(double* b, double4* normalsAndPenetrations, double timeStep, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double penetration = normalsAndPenetrations[index].w;
  if(penetration>0) penetration = 0; // TODO: is this correct?

  b[3*index] = penetration/timeStep;
  b[3*index+1] = 0;
  b[3*index+2] = 0;
}

int System::buildSchurVector() {
  // build r
  r_d.resize(3*collisionDetector->numCollisions);
  b_d.resize(3*collisionDetector->numCollisions);
  // TODO: There's got to be a better way to do this...
  //r.resize(3*collisionDetector->numCollisions);
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  thrust::device_ptr<double> wrapped_device_b(CASTD1(b_d));
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
  cusp::multiply(mass,k,tmp);
  cusp::multiply(D,tmp,r);

  buildStabilization<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(b_d), CASTD4(collisionDetector->normalsAndPenetrations_d), h, collisionDetector->numCollisions);
  cusp::blas::axpy(b,r,1.0);

  return 0;
}

int System::buildSchurMatrix() {
  // build N
  cusp::multiply(mass,DT,MinvDT);
  cusp::multiply(D,MinvDT,N);

  return 0;
}

__global__ void getNormalComponent(double* src, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  dst[index] = src[3*index];
}

__global__ void calculateConeViolation(double* gamma, double* friction, double* dst, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double gamma_t = sqrt(pow(gamma[3*index+1],2.0)+pow(gamma[3*index+2],2.0));
  double coneViolation = friction[index]*gamma[3*index] - gamma_t; // TODO: Keep the friction indexing in mind for bilaterals
  if(coneViolation>0) coneViolation = 0;
  dst[index] = coneViolation;
}

double4 System::getCCPViolation() {
  double4 violationCCP = make_double4(0,0,0,0);

  if(collisionDetector->numCollisions) {
    // Build normal impulse vector, gamma_n
    thrust::device_vector<double> gamma_n_d;
    gamma_n_d.resize(collisionDetector->numCollisions);
    thrust::device_ptr<double> wrapped_device_gamma_n(CASTD1(gamma_n_d));
    DeviceValueArrayView gamma_n = DeviceValueArrayView(wrapped_device_gamma_n, wrapped_device_gamma_n + gamma_n_d.size());
    getNormalComponent<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(gamma_d), CASTD1(gamma_n_d), collisionDetector->numCollisions);
    violationCCP.x = Thrust_Min(gamma_n_d);
    if(violationCCP.x > 0) violationCCP.x = 0;

    // Build normal velocity vector, v_n
    thrust::device_vector<double> tmp_gamma_d;
    tmp_gamma_d.resize(3*collisionDetector->numCollisions);
    thrust::device_ptr<double> wrapped_device_tmp_gamma(CASTD1(tmp_gamma_d));
    DeviceValueArrayView tmp_gamma = DeviceValueArrayView(wrapped_device_tmp_gamma, wrapped_device_tmp_gamma + tmp_gamma_d.size());

    thrust::device_vector<double> v_n_d;
    v_n_d.resize(collisionDetector->numCollisions);
    thrust::device_ptr<double> wrapped_device_v_n(CASTD1(v_n_d));
    DeviceValueArrayView v_n = DeviceValueArrayView(wrapped_device_v_n, wrapped_device_v_n + v_n_d.size());
    cusp::multiply(D,v,tmp_gamma);
    cusp::blas::axpy(b,tmp_gamma,1.0);
    getNormalComponent<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(tmp_gamma_d), CASTD1(v_n_d), collisionDetector->numCollisions);
    violationCCP.y = Thrust_Min(v_n_d);
    if(violationCCP.y > 0) violationCCP.y = 0;

    // Check complementarity condition
    violationCCP.z = cusp::blas::dot(gamma_n,v_n);

    // Check friction cone condition
    calculateConeViolation<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(gamma_d), CASTD1(friction_d), CASTD1(v_n_d), collisionDetector->numCollisions);
    violationCCP.w = cusp::blas::nrm2(v_n);
  }

  return violationCCP;
}

int System::exportSystem(string filename) {
  ofstream filestream;
  filestream.open(filename.c_str());

  p_h = p_d;
  v_h = v_d;
  filestream << "0, " << bodies.size() << ", 0, " << endl;
  for (int i = 0; i < bodies.size(); i++) {
    filestream
        << i << ", "
        << bodies[i]->isFixed() << ", "
        << p_h[3*i] << ", "
        << p_h[3*i+1] << ", "
        << p_h[3*i+2] << ", "
        << "1, "
        << "0, "
        << "0, "
        << "0, "
        << v_h[3*i] << ", "
        << v_h[3*i+1] << ", "
        << v_h[3*i+2] << ", ";

        if(contactGeometry_h[i].y == 0) {
          filestream
            << "0, "
            << contactGeometry_h[i].x << ", ";
        }
        else {
          filestream
            << "2, "
            << contactGeometry_h[i].x << ", "
            << contactGeometry_h[i].y << ", "
            << contactGeometry_h[i].z << ", ";
        }
        filestream
          << "\n";
  }
  filestream.close();

  return 0;
}

int System::importSystem(string filename) {
  double3 pos;
  double3 vel;
  double3 geometry = make_double3(0,0,0);
  int isFixed;
  string temp_data;
  int numBodies;
  double blah;
  int index;
  int shape;

  ifstream ifile(filename.c_str());
  getline(ifile,temp_data);
  for(int i=0; i<temp_data.size(); ++i){
    if(temp_data[i]==','){temp_data[i]=' ';}
  }
  stringstream ss1(temp_data);
  ss1>>blah>>numBodies>>blah;

  Body* bodyPtr;
  for(int i=0; i<numBodies; i++) {
    getline(ifile,temp_data);
    for(int i=0; i<temp_data.size(); ++i){
      if(temp_data[i]==','){temp_data[i]=' ';}
    }
    stringstream ss(temp_data);
    ss>>index>>isFixed>>pos.x>>pos.y>>pos.z>>blah>>blah>>blah>>blah>>vel.x>>vel.y>>vel.z>>shape;
    if(shape == 0) {
      ss>>geometry.x;
      geometry.y = 0;
      geometry.z = 0;
    } else {
      ss>>geometry.x>>geometry.y>>geometry.z;
    }

    bodyPtr = new Body(pos);
    bodyPtr->setBodyFixed(isFixed);
    bodyPtr->setGeometry(geometry);
    bodyPtr->setVelocity(vel);
    if(shape == 0) {
      bodyPtr->setMass(2600*4.0*3.14159*pow(geometry.x,3.0)/3.0);
    } else {
      bodyPtr->setMass(1.0);
    }
    add(bodyPtr);
    //cout << index << " " << isFixed << " " << pos.x << " " << pos.y << " " << pos.z << " " << "1 0 0 0 " << vel.x << " " << vel.y << " " << vel.z << " " << shape << " " << geometry.x << " " << geometry.y << " " << geometry.z << endl;
  }

  return 0;
}

int System::exportMatrices(string directory) {

  string filename = directory + "/D.mtx";
  cusp::io::write_matrix_market_file(D, filename);

  filename = directory + "/Minv.mtx";
  cusp::io::write_matrix_market_file(mass, filename);

  filename = directory + "/r.mtx";
  cusp::io::write_matrix_market_file(r, filename);

  filename = directory + "/b.mtx";
  cusp::io::write_matrix_market_file(b, filename);

  filename = directory + "/k.mtx";
  cusp::io::write_matrix_market_file(k, filename);

  return 0;
}
