#include <algorithm>
#include <vector>
#include "include.cuh"
#include "System.cuh"
#include "Solver.cuh"
#include "APGD.cuh"
#include "PDIP.cuh"
#include "TPAS.cuh"
#include "JKIP.cuh"
#include "PJKIP.cuh"
#include "PGJ.cuh"
#include "PGS.cuh"

System::System()
{
  gravity = make_double3(0,-9.81,0);
  tol = 1e-8;
  h = 1e-3;
  timeIndex = 0;
  time = 0;
  elapsedTime = 0;
  totalGPUMemoryUsed = 0;

  collisionDetector = new CollisionDetector(this);
  solver = new APGD(this);

  wt3.push_back(5.0 / 9.0);
  wt3.push_back(8.0 / 9.0);
  wt3.push_back(5.0 / 9.0);
  pt3.push_back(-sqrt(3.0 / 5.0));
  pt3.push_back(0.0);
  pt3.push_back(sqrt(3.0 / 5.0));

  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back(128. / 225.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
  pt5.push_back(-(sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(-(sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(0.);
  pt5.push_back((sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back((sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
}

System::System(int solverType)
{
  gravity = make_double3(0,-9.81,0);
  tol = 1e-8;
  h = 1e-3;
  timeIndex = 0;
  time = 0;
  elapsedTime = 0;
  totalGPUMemoryUsed = 0;

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
  case 5:
    solver = new PGJ(this);
    break;
  case 6:
    solver = new PGS(this);
    break;
  case 7:
    solver = new PJKIP(this);
    break;
  default:
    solver = new APGD(this);
  }

  wt3.push_back(5.0 / 9.0);
  wt3.push_back(8.0 / 9.0);
  wt3.push_back(5.0 / 9.0);
  pt3.push_back(-sqrt(3.0 / 5.0));
  pt3.push_back(0.0);
  pt3.push_back(sqrt(3.0 / 5.0));

  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back(128. / 225.);
  wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
  wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
  pt5.push_back(-(sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(-(sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back(0.);
  pt5.push_back((sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
  pt5.push_back((sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
}

void System::setTimeStep(double step_size)
{
  h = step_size;
}

int System::add(Body* body) {
  //add the body
  bodies.push_back(body);

  return bodies.size();
}

int System::add(Beam* beam) {
  //add the beam
  beam->sys = this;
  beams.push_back(beam);
  return beams.size();
}

int System::add(Plate* plate) {
  //add the plate
  plate->sys = this;
  plates.push_back(plate);
  return plates.size();
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
  fElastic_d = fElastic_h;

  massI_d = massI_h;
  massJ_d = massJ_h;
  mass_d = mass_h;

  contactGeometry_d = contactGeometry_h;
  collisionGeometry_d = collisionGeometry_h;
  collisionMap_d = collisionMap_h;
  materialsBeam_d = materialsBeam_h;
  materialsPlate_d = materialsPlate_h;
  fixedBodies_d = fixedBodies_h;
  constraintsBilateralDOF_d = constraintsBilateralDOF_h;

  strainDerivative_d = strainDerivative_h;
  strain_d = strain_h;
  Sx_d = Sx_h;
  Sxx_d = Sxx_h;

  thrust::device_ptr<double> wrapped_device_p(CASTD1(p_d));
  thrust::device_ptr<double> wrapped_device_v(CASTD1(v_d));
  thrust::device_ptr<double> wrapped_device_a(CASTD1(a_d));
  thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));
  thrust::device_ptr<double> wrapped_device_f_contact(CASTD1(f_contact_d));
  thrust::device_ptr<double> wrapped_device_fApplied(CASTD1(fApplied_d));
  thrust::device_ptr<double> wrapped_device_fElastic(CASTD1(fElastic_d));
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
  fElastic = DeviceValueArrayView(wrapped_device_fElastic, wrapped_device_fElastic + fElastic_d.size());
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

  for(int j=0; j<bodies.size(); j++) {
    Body* body = bodies[j];
    body->setIdentifier(j); // Indicates the number that the Body was added
    body->setIndex(p_h.size()); // Indicates the Body's location in the position array

    // Push Body's location to global library
    indices_h.push_back(p_h.size());

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

    fElastic_h.push_back(0);
    fElastic_h.push_back(0);
    fElastic_h.push_back(0);

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
    for(int i = 0; i < body->numDOF; i++) {
      massI_h.push_back(i + body->numDOF * j);
      massJ_h.push_back(i + body->numDOF * j);
      if(body->isFixed()) {
        mass_h.push_back(0);
      }
      else {
        mass_h.push_back(1.0/body->mass);
      }
    }

    contactGeometry_h.push_back(body->contactGeometry);
    collisionGeometry_h.push_back(body->contactGeometry);
    collisionMap_h.push_back(make_int4(body->getIdentifier(),0,0,body->getCollisionFamily()));

    if(body->isFixed()) fixedBodies_h.push_back(j);
  }

  for(int j=0; j<beams.size(); j++) {
    beams[j]->addBeam(j); //TODO: Make a function like this for body (makes code cleaner)
  }

  for(int j=0; j<plates.size(); j++) {
    plates[j]->addPlate(j); //TODO: Make a function like this for body (makes code cleaner)
  }

  initializeDevice();
  solver->setup();

  return 0;
}

int System::addBilateralConstraintDOF(int DOFA, int DOFB) {
  constraintsBilateralDOF_h.push_back(make_int2(DOFA,DOFB));
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
  if(collisionDetector->numCollisions||constraintsBilateralDOF_d.size()) {
    // Set up the QOCC
    buildContactJacobian();
    buildSchurVector();

    // Solve the QOCC
    solver->solve();

    // Perform time integration (contacts) TODO: Get rid of constraint forces in f_contact vector!
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
  cusp::blas::axpy(v, p, h);

  time += h;
  timeIndex++;
  //p_h = p_d;

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

int System::applyForce(Body* body, double3 force) {
  int index = body->getIndex();
  //cout << index << endl;

  fApplied_h[index]+=force.x;
  fApplied_h[index+1]+=force.y;
  fApplied_h[index+2]+=force.z;

  return 0;
}

int System::clearAppliedForces() {
  Thrust_Fill(fApplied_d,0.0);
  fApplied_h = fApplied_d;

  return 0;
}

__global__ void constructBilateralJacobian(int2* contraintBilateralDOF, int* DI, int* DJ, double* D, uint numConstraintsBilateral) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraintsBilateral);

  int2 bilateralDOFs = contraintBilateralDOF[index];

  DI[2*index] = index;
  DI[2*index+1] = index;

  DJ[2*index] = bilateralDOFs.x;
  DJ[2*index+1] = bilateralDOFs.y;

  D[2*index] = 1.0;
  D[2*index+1] = -1.0;
}

__global__ void constructContactJacobian(int* nonzerosPerContact_d, int4* collisionMap, double3* geometries, double3* collisionGeometry, int* DI, int* DJ, double* D, double* friction, double4* normalsAndPenetrations, uint* collisionIdentifierA, uint* collisionIdentifierB, int* indices, int numBodies, int numBeams, uint numConstraintsBilateral, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  friction[index] = 0.25; // TODO: EDIT THIS TO BE MINIMUM OF FRICTION COEFFICIENTS

  int offsetA = (!index) ? 0 : nonzerosPerContact_d[index - 1];
  offsetA+=2*numConstraintsBilateral; // add offset for bilateral constraints
  DI = &DI[offsetA];
  DJ = &DJ[offsetA];
  D = &D[offsetA];

  int bodyIdentifierA = collisionMap[collisionIdentifierA[index]].x;
  int bodyIdentifierB = collisionMap[collisionIdentifierB[index]].x;

  int endA = (bodyIdentifierA<numBodies) ? 3 : 12;
  int endB = (bodyIdentifierB<numBodies) ? 3 : 12;

  int indexA = indices[bodyIdentifierA];
  int indexB = indices[bodyIdentifierB];

  double xiA = static_cast<double>(collisionMap[collisionIdentifierA[index]].y)/(static_cast<double>(geometries[bodyIdentifierA].z-1));
  double lA = geometries[bodyIdentifierA].y;

  double xiB = static_cast<double>(collisionMap[collisionIdentifierB[index]].y)/(static_cast<double>(geometries[bodyIdentifierB].z-1));
  double lB = geometries[bodyIdentifierB].y;

  double4 nAndP;
  double3 n, u, v;
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
  int i;
  int end = endA;
  int j = 0;
  for(i=0;i<end;i++) {
    DI[i] = 3*index+0+numConstraintsBilateral;
    DJ[i] = indexA+j;
    j++;
  }
  end+=endB;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+0+numConstraintsBilateral;
    DJ[i] = indexB+j;
    j++;
  }

  // Add u, i indices
  end+=endA;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+1+numConstraintsBilateral;
    DJ[i] = indexA+j;
    j++;
  }
  end+=endB;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+1+numConstraintsBilateral;
    DJ[i] = indexB+j;
    j++;
  }

  // Add v, i indices
  end+=endA;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+2+numConstraintsBilateral;
    DJ[i] = indexA+j;
    j++;
  }
  end+=endB;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+2+numConstraintsBilateral;
    DJ[i] = indexB+j;
    j++;
  }

  // Add n, values
  int startIndex = 0;
  if(bodyIdentifierA<numBodies) {
    D[startIndex+0] = n.x;
    D[startIndex+1] = n.y;
    D[startIndex+2] = n.z;
    startIndex+=3;
  } else {
    D[startIndex+0 ] = n.x*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+1 ] = n.y*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+2 ] = n.z*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+3 ] = lA*n.x*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+4 ] = lA*n.y*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+5 ] = lA*n.z*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+6 ] = n.x*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+7 ] = n.y*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+8 ] = n.z*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+9 ] = -lA*n.x*(-xiA*xiA*xiA+xiA*xiA);
    D[startIndex+10] = -lA*n.y*(-xiA*xiA*xiA+xiA*xiA);
    D[startIndex+11] = -lA*n.z*(-xiA*xiA*xiA+xiA*xiA);
    startIndex+=12;
  }
  if(bodyIdentifierB<numBodies) {
    D[startIndex+0] = -n.x;
    D[startIndex+1] = -n.y;
    D[startIndex+2] = -n.z;
    startIndex+=3;
  } else {
    D[startIndex+0 ] = -n.x*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+1 ] = -n.y*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+2 ] = -n.z*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+3 ] = -lB*n.x*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+4 ] = -lB*n.y*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+5 ] = -lB*n.z*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+6 ] = -n.x*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+7 ] = -n.y*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+8 ] = -n.z*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+9 ] = lB*n.x*(-xiB*xiB*xiB+xiB*xiB);
    D[startIndex+10] = lB*n.y*(-xiB*xiB*xiB+xiB*xiB);
    D[startIndex+11] = lB*n.z*(-xiB*xiB*xiB+xiB*xiB);
    startIndex+=12;
  }

  // Add u, values
  if(bodyIdentifierA<numBodies) {
    D[startIndex+0] = u.x;
    D[startIndex+1] = u.y;
    D[startIndex+2] = u.z;
    startIndex+=3;
  } else {
    D[startIndex+0 ] = u.x*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+1 ] = u.y*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+2 ] = u.z*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+3 ] = lA*u.x*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+4 ] = lA*u.y*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+5 ] = lA*u.z*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+6 ] = u.x*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+7 ] = u.y*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+8 ] = u.z*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+9 ] = -lA*u.x*(-xiA*xiA*xiA+xiA*xiA);
    D[startIndex+10] = -lA*u.y*(-xiA*xiA*xiA+xiA*xiA);
    D[startIndex+11] = -lA*u.z*(-xiA*xiA*xiA+xiA*xiA);
    startIndex+=12;
  }
  if(bodyIdentifierB<numBodies) {
    D[startIndex+0] = -u.x;
    D[startIndex+1] = -u.y;
    D[startIndex+2] = -u.z;
    startIndex+=3;
  } else {
    D[startIndex+0 ] = -u.x*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+1 ] = -u.y*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+2 ] = -u.z*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+3 ] = -lB*u.x*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+4 ] = -lB*u.y*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+5 ] = -lB*u.z*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+6 ] = -u.x*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+7 ] = -u.y*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+8 ] = -u.z*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+9 ] = lB*u.x*(-xiB*xiB*xiB+xiB*xiB);
    D[startIndex+10] = lB*u.y*(-xiB*xiB*xiB+xiB*xiB);
    D[startIndex+11] = lB*u.z*(-xiB*xiB*xiB+xiB*xiB);
    startIndex+=12;
  }

  // Add v, values
  if(bodyIdentifierA<numBodies) {
    D[startIndex+0] = v.x;
    D[startIndex+1] = v.y;
    D[startIndex+2] = v.z;
    startIndex+=3;
  } else {
    D[startIndex+0 ] = v.x*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+1 ] = v.y*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+2 ] = v.z*(2.0*xiA*xiA*xiA-3.0*xiA*xiA+1.0);
    D[startIndex+3 ] = lA*v.x*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+4 ] = lA*v.y*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+5 ] = lA*v.z*(xiA*xiA*xiA-2.0*xiA*xiA+xiA);
    D[startIndex+6 ] = v.x*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+7 ] = v.y*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+8 ] = v.z*(-2.0*xiA*xiA*xiA+3.0*xiA*xiA);
    D[startIndex+9 ] = -lA*v.x*(-xiA*xiA*xiA+xiA*xiA);
    D[startIndex+10] = -lA*v.y*(-xiA*xiA*xiA+xiA*xiA);
    D[startIndex+11] = -lA*v.z*(-xiA*xiA*xiA+xiA*xiA);
    startIndex+=12;
  }
  if(bodyIdentifierB<numBodies) {
    D[startIndex+0] = -v.x;
    D[startIndex+1] = -v.y;
    D[startIndex+2] = -v.z;
    startIndex+=3;
  } else {
    D[startIndex+0 ] = -v.x*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+1 ] = -v.y*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+2 ] = -v.z*(2.0*xiB*xiB*xiB-3.0*xiB*xiB+1.0);
    D[startIndex+3 ] = -lB*v.x*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+4 ] = -lB*v.y*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+5 ] = -lB*v.z*(xiB*xiB*xiB-2.0*xiB*xiB+xiB);
    D[startIndex+6 ] = -v.x*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+7 ] = -v.y*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+8 ] = -v.z*(-2.0*xiB*xiB*xiB+3.0*xiB*xiB);
    D[startIndex+9 ] = lB*v.x*(-xiB*xiB*xiB+xiB*xiB);
    D[startIndex+10] = lB*v.y*(-xiB*xiB*xiB+xiB*xiB);
    D[startIndex+11] = lB*v.z*(-xiB*xiB*xiB+xiB*xiB);
    startIndex+=12;
  }
}

__global__ void updateNonzerosPerContact(int* nonzerosPerContact, int4* collisionMap, uint* collisionIdentifierA, uint* collisionIdentifierB, int numBodies, int numBeams, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  int numNonzeros = 0;
  int bodyIdentifierA = collisionMap[collisionIdentifierA[index]].x;
  int bodyIdentifierB = collisionMap[collisionIdentifierB[index]].x;

  if(bodyIdentifierA<numBodies) {
    numNonzeros+=9;
  }
  else if(bodyIdentifierA<(numBodies+numBeams)) {
    numNonzeros+=36;
  }
  else {
    numNonzeros+=108;
  }

  if(bodyIdentifierB<numBodies) {
    numNonzeros+=9;
  }
  else if(bodyIdentifierB<(numBodies+numBeams)) {
    numNonzeros+=36;
  }
  else {
    numNonzeros+=108;
  }

  nonzerosPerContact[index] = numNonzeros;
}

int System::buildContactJacobian() {
  // update nonzeros per contact
  int totalNonzeros = 0;
  nonzerosPerContact_d.resize(collisionDetector->numCollisions);
  if(collisionDetector->numCollisions) {
    updateNonzerosPerContact<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTI1(nonzerosPerContact_d), CASTI4(collisionMap_d), CASTU1(collisionDetector->collisionIdentifierA_d), CASTU1(collisionDetector->collisionIdentifierB_d), bodies.size(), beams.size(), collisionDetector->numCollisions);
    Thrust_Inclusive_Scan_Sum(nonzerosPerContact_d, totalNonzeros);
  }
  totalNonzeros+=2*constraintsBilateralDOF_d.size(); //Add in space for the bilateral entries

  DI_d.resize(totalNonzeros);
  DJ_d.resize(totalNonzeros);
  D_d.resize(totalNonzeros);
  friction_d.resize(collisionDetector->numCollisions);

  if(constraintsBilateralDOF_d.size()) constructBilateralJacobian<<<BLOCKS(constraintsBilateralDOF_d.size()),THREADS>>>(CASTI2(constraintsBilateralDOF_d), CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), constraintsBilateralDOF_d.size());
  if(collisionDetector->numCollisions) constructContactJacobian<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTI1(nonzerosPerContact_d), CASTI4(collisionMap_d), CASTD3(contactGeometry_d), CASTD3(collisionGeometry_d), CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), CASTD1(friction_d), CASTD4(collisionDetector->normalsAndPenetrations_d), CASTU1(collisionDetector->collisionIdentifierA_d), CASTU1(collisionDetector->collisionIdentifierB_d), CASTI1(indices_d), bodies.size(), beams.size(), constraintsBilateralDOF_d.size(), collisionDetector->numCollisions);

  // create contact jacobian using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(DI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + DI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(DJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + DJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(D_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + D_d.size());

  D = DeviceView(3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size(), 3*bodies.size()+12*beams.size(), D_d.size(), row_indices, column_indices, values);
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

  DT = DeviceView(3*bodies.size()+12*beams.size(), 3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size(), DT_d.size(), row_indices, column_indices, values);
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

__global__ void multiplyByBeamMass(double3* geometries, double3* materials, double* src, double* dst, uint numBodies, uint numBeams) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numBeams);

  double3 geometry = geometries[numBodies+index];
  double A = PI*geometry.x*geometry.x;
  double l = geometry.y;
  double rho = materials[index].x;

  uint offset = 3*numBodies+12*index;
  dst[offset+0 ] = (13.0*A*rho*src[0+offset])/35.0 + (9.0*A*rho*src[6+offset])/70.0 + (11.0*A*l*rho*src[3+offset])/210.0 - (13.0*A*l*rho*src[9 +offset])/420.0;
  dst[offset+1 ] = (13.0*A*rho*src[1+offset])/35.0 + (9.0*A*rho*src[7+offset])/70.0 + (11.0*A*l*rho*src[4+offset])/210.0 - (13.0*A*l*rho*src[10+offset])/420.0;
  dst[offset+2 ] = (13.0*A*rho*src[2+offset])/35.0 + (9.0*A*rho*src[8+offset])/70.0 + (11.0*A*l*rho*src[5+offset])/210.0 - (13.0*A*l*rho*src[11+offset])/420.0;
  dst[offset+3 ] = (A*l*l*rho*src[3+offset])/105.0 - (A*l*l*rho*src[9 +offset])/140.0 + (11.0*A*l*rho*src[0+offset])/210.0 + (13.0*A*l*rho*src[6+offset])/420.0;
  dst[offset+4 ] = (A*l*l*rho*src[4+offset])/105.0 - (A*l*l*rho*src[10+offset])/140.0 + (11.0*A*l*rho*src[1+offset])/210.0 + (13.0*A*l*rho*src[7+offset])/420.0;
  dst[offset+5 ] = (A*l*l*rho*src[5+offset])/105.0 - (A*l*l*rho*src[11+offset])/140.0 + (11.0*A*l*rho*src[2+offset])/210.0 + (13.0*A*l*rho*src[8+offset])/420.0;
  dst[offset+6 ] = (9.0*A*rho*src[0+offset])/70.0 + (13.0*A*rho*src[6+offset])/35.0 + (13.0*A*l*rho*src[3+offset])/420.0 - (11.0*A*l*rho*src[9 +offset])/210.0;
  dst[offset+7 ] = (9.0*A*rho*src[1+offset])/70.0 + (13.0*A*rho*src[7+offset])/35.0 + (13.0*A*l*rho*src[4+offset])/420.0 - (11.0*A*l*rho*src[10+offset])/210.0;
  dst[offset+8 ] = (9.0*A*rho*src[2+offset])/70.0 + (13.0*A*rho*src[8+offset])/35.0 + (13.0*A*l*rho*src[5+offset])/420.0 - (11.0*A*l*rho*src[11+offset])/210.0;
  dst[offset+9 ] = (A*l*l*rho*src[9 +offset])/105.0 - (A*l*l*rho*src[3+offset])/140.0 - (13.0*A*l*rho*src[0+offset])/420.0 - (11.0*A*l*rho*src[6+offset])/210.0;
  dst[offset+10] = (A*l*l*rho*src[10+offset])/105.0 - (A*l*l*rho*src[4+offset])/140.0 - (13.0*A*l*rho*src[1+offset])/420.0 - (11.0*A*l*rho*src[7+offset])/210.0;
  dst[offset+11] = (A*l*l*rho*src[11+offset])/105.0 - (A*l*l*rho*src[5+offset])/140.0 - (13.0*A*l*rho*src[2+offset])/420.0 - (11.0*A*l*rho*src[8+offset])/210.0;
}

__global__ void multiplyByPlateMass(double3* geometries, double4* materials, double* src, double* dst, uint numBodies, uint numBeams, uint numPlates) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numPlates);

  double3 geometry = geometries[numBodies+numBeams+index];
  double a = geometry.x;
  double b = geometry.y;
  double rho = materials[index].x;
  double th = materials[index].w;

  uint offset = 3*numBodies+12*numBeams+36*numPlates;
  dst[offset+0] = rho*th*src[0+offset]*1.370634920634921E-1+rho*th*src[9+offset]*4.865079365079365E-2+rho*th*src[18+offset]*1.563492063492063E-2+rho*th*src[27+offset]*4.865079365079365E-2+a*rho*th*src[3+offset]*1.829365079365079E-2-a*rho*th*src[12+offset]*1.087301587301587E-2-a*rho*th*src[21+offset]*4.603174603174603E-3+a*rho*th*src[30+offset]*7.896825396825397E-3+b*rho*th*src[6+offset]*1.829365079365079E-2+b*rho*th*src[15+offset]*7.896825396825397E-3-b*rho*th*src[24+offset]*4.603174603174603E-3-b*rho*th*src[33+offset]*1.087301587301587E-2;
  dst[offset+1] = rho*th*src[1+offset]*1.370634920634921E-1+rho*th*src[10+offset]*4.865079365079365E-2+rho*th*src[19+offset]*1.563492063492063E-2+rho*th*src[28+offset]*4.865079365079365E-2+a*rho*th*src[4+offset]*1.829365079365079E-2-a*rho*th*src[13+offset]*1.087301587301587E-2-a*rho*th*src[22+offset]*4.603174603174603E-3+a*rho*th*src[31+offset]*7.896825396825397E-3+b*rho*th*src[7+offset]*1.829365079365079E-2+b*rho*th*src[16+offset]*7.896825396825397E-3-b*rho*th*src[25+offset]*4.603174603174603E-3-b*rho*th*src[34+offset]*1.087301587301587E-2;
  dst[offset+2] = rho*th*src[2+offset]*1.370634920634921E-1+rho*th*src[11+offset]*4.865079365079365E-2+rho*th*src[20+offset]*1.563492063492063E-2+rho*th*src[29+offset]*4.865079365079365E-2+a*rho*th*src[5+offset]*1.829365079365079E-2-a*rho*th*src[14+offset]*1.087301587301587E-2-a*rho*th*src[23+offset]*4.603174603174603E-3+a*rho*th*src[32+offset]*7.896825396825397E-3+b*rho*th*src[8+offset]*1.829365079365079E-2+b*rho*th*src[17+offset]*7.896825396825397E-3-b*rho*th*src[26+offset]*4.603174603174603E-3-b*rho*th*src[35+offset]*1.087301587301587E-2;
  dst[offset+3] = a*rho*th*src[0+offset]*1.829365079365079E-2+a*rho*th*src[9+offset]*1.087301587301587E-2+a*rho*th*src[18+offset]*4.603174603174603E-3+a*rho*th*src[27+offset]*7.896825396825397E-3+(a*a)*rho*th*src[3+offset]*(1.0/3.15E2)-(a*a)*rho*th*src[12+offset]*(1.0/4.2E2)-(a*a)*rho*th*src[21+offset]*(1.0/8.4E2)+(a*a)*rho*th*src[30+offset]*(1.0/6.3E2)+a*b*rho*th*src[6+offset]*(1.0/4.0E2)+a*b*rho*th*src[15+offset]*(1.0/6.0E2)-a*b*rho*th*src[24+offset]*(1.0/9.0E2)-a*b*rho*th*src[33+offset]*(1.0/6.0E2);
  dst[offset+4] = a*rho*th*src[1+offset]*1.829365079365079E-2+a*rho*th*src[10+offset]*1.087301587301587E-2+a*rho*th*src[19+offset]*4.603174603174603E-3+a*rho*th*src[28+offset]*7.896825396825397E-3+(a*a)*rho*th*src[4+offset]*(1.0/3.15E2)-(a*a)*rho*th*src[13+offset]*(1.0/4.2E2)-(a*a)*rho*th*src[22+offset]*(1.0/8.4E2)+(a*a)*rho*th*src[31+offset]*(1.0/6.3E2)+a*b*rho*th*src[7+offset]*(1.0/4.0E2)+a*b*rho*th*src[16+offset]*(1.0/6.0E2)-a*b*rho*th*src[25+offset]*(1.0/9.0E2)-a*b*rho*th*src[34+offset]*(1.0/6.0E2);
  dst[offset+5] = a*rho*th*src[2+offset]*1.829365079365079E-2+a*rho*th*src[11+offset]*1.087301587301587E-2+a*rho*th*src[20+offset]*4.603174603174603E-3+a*rho*th*src[29+offset]*7.896825396825397E-3+(a*a)*rho*th*src[5+offset]*(1.0/3.15E2)-(a*a)*rho*th*src[14+offset]*(1.0/4.2E2)-(a*a)*rho*th*src[23+offset]*(1.0/8.4E2)+(a*a)*rho*th*src[32+offset]*(1.0/6.3E2)+a*b*rho*th*src[8+offset]*(1.0/4.0E2)+a*b*rho*th*src[17+offset]*(1.0/6.0E2)-a*b*rho*th*src[26+offset]*(1.0/9.0E2)-a*b*rho*th*src[35+offset]*(1.0/6.0E2);
  dst[offset+6] = b*rho*th*src[0+offset]*1.829365079365079E-2+b*rho*th*src[9+offset]*7.896825396825397E-3+b*rho*th*src[18+offset]*4.603174603174603E-3+b*rho*th*src[27+offset]*1.087301587301587E-2+(b*b)*rho*th*src[6+offset]*(1.0/3.15E2)+(b*b)*rho*th*src[15+offset]*(1.0/6.3E2)-(b*b)*rho*th*src[24+offset]*(1.0/8.4E2)-(b*b)*rho*th*src[33+offset]*(1.0/4.2E2)+a*b*rho*th*src[3+offset]*(1.0/4.0E2)-a*b*rho*th*src[12+offset]*(1.0/6.0E2)-a*b*rho*th*src[21+offset]*(1.0/9.0E2)+a*b*rho*th*src[30+offset]*(1.0/6.0E2);
  dst[offset+7] = b*rho*th*src[1+offset]*1.829365079365079E-2+b*rho*th*src[10+offset]*7.896825396825397E-3+b*rho*th*src[19+offset]*4.603174603174603E-3+b*rho*th*src[28+offset]*1.087301587301587E-2+(b*b)*rho*th*src[7+offset]*(1.0/3.15E2)+(b*b)*rho*th*src[16+offset]*(1.0/6.3E2)-(b*b)*rho*th*src[25+offset]*(1.0/8.4E2)-(b*b)*rho*th*src[34+offset]*(1.0/4.2E2)+a*b*rho*th*src[4+offset]*(1.0/4.0E2)-a*b*rho*th*src[13+offset]*(1.0/6.0E2)-a*b*rho*th*src[22+offset]*(1.0/9.0E2)+a*b*rho*th*src[31+offset]*(1.0/6.0E2);
  dst[offset+8] = b*rho*th*src[2+offset]*1.829365079365079E-2+b*rho*th*src[11+offset]*7.896825396825397E-3+b*rho*th*src[20+offset]*4.603174603174603E-3+b*rho*th*src[29+offset]*1.087301587301587E-2+(b*b)*rho*th*src[8+offset]*(1.0/3.15E2)+(b*b)*rho*th*src[17+offset]*(1.0/6.3E2)-(b*b)*rho*th*src[26+offset]*(1.0/8.4E2)-(b*b)*rho*th*src[35+offset]*(1.0/4.2E2)+a*b*rho*th*src[5+offset]*(1.0/4.0E2)-a*b*rho*th*src[14+offset]*(1.0/6.0E2)-a*b*rho*th*src[23+offset]*(1.0/9.0E2)+a*b*rho*th*src[32+offset]*(1.0/6.0E2);
  dst[offset+9] = rho*th*src[0+offset]*4.865079365079365E-2+rho*th*src[9+offset]*1.370634920634921E-1+rho*th*src[18+offset]*4.865079365079365E-2+rho*th*src[27+offset]*1.563492063492063E-2+a*rho*th*src[3+offset]*1.087301587301587E-2-a*rho*th*src[12+offset]*1.829365079365079E-2-a*rho*th*src[21+offset]*7.896825396825397E-3+a*rho*th*src[30+offset]*4.603174603174603E-3+b*rho*th*src[6+offset]*7.896825396825397E-3+b*rho*th*src[15+offset]*1.829365079365079E-2-b*rho*th*src[24+offset]*1.087301587301587E-2-b*rho*th*src[33+offset]*4.603174603174603E-3;
  dst[offset+10] = rho*th*src[1+offset]*4.865079365079365E-2+rho*th*src[10+offset]*1.370634920634921E-1+rho*th*src[19+offset]*4.865079365079365E-2+rho*th*src[28+offset]*1.563492063492063E-2+a*rho*th*src[4+offset]*1.087301587301587E-2-a*rho*th*src[13+offset]*1.829365079365079E-2-a*rho*th*src[22+offset]*7.896825396825397E-3+a*rho*th*src[31+offset]*4.603174603174603E-3+b*rho*th*src[7+offset]*7.896825396825397E-3+b*rho*th*src[16+offset]*1.829365079365079E-2-b*rho*th*src[25+offset]*1.087301587301587E-2-b*rho*th*src[34+offset]*4.603174603174603E-3;
  dst[offset+11] = rho*th*src[2+offset]*4.865079365079365E-2+rho*th*src[11+offset]*1.370634920634921E-1+rho*th*src[20+offset]*4.865079365079365E-2+rho*th*src[29+offset]*1.563492063492063E-2+a*rho*th*src[5+offset]*1.087301587301587E-2-a*rho*th*src[14+offset]*1.829365079365079E-2-a*rho*th*src[23+offset]*7.896825396825397E-3+a*rho*th*src[32+offset]*4.603174603174603E-3+b*rho*th*src[8+offset]*7.896825396825397E-3+b*rho*th*src[17+offset]*1.829365079365079E-2-b*rho*th*src[26+offset]*1.087301587301587E-2-b*rho*th*src[35+offset]*4.603174603174603E-3;
  dst[offset+12] = a*rho*th*src[0+offset]*(-1.087301587301587E-2)-a*rho*th*src[9+offset]*1.829365079365079E-2-a*rho*th*src[18+offset]*7.896825396825397E-3-a*rho*th*src[27+offset]*4.603174603174603E-3-(a*a)*rho*th*src[3+offset]*(1.0/4.2E2)+(a*a)*rho*th*src[12+offset]*(1.0/3.15E2)+(a*a)*rho*th*src[21+offset]*(1.0/6.3E2)-(a*a)*rho*th*src[30+offset]*(1.0/8.4E2)-a*b*rho*th*src[6+offset]*(1.0/6.0E2)-a*b*rho*th*src[15+offset]*(1.0/4.0E2)+a*b*rho*th*src[24+offset]*(1.0/6.0E2)+a*b*rho*th*src[33+offset]*(1.0/9.0E2);
  dst[offset+13] = a*rho*th*src[1+offset]*(-1.087301587301587E-2)-a*rho*th*src[10+offset]*1.829365079365079E-2-a*rho*th*src[19+offset]*7.896825396825397E-3-a*rho*th*src[28+offset]*4.603174603174603E-3-(a*a)*rho*th*src[4+offset]*(1.0/4.2E2)+(a*a)*rho*th*src[13+offset]*(1.0/3.15E2)+(a*a)*rho*th*src[22+offset]*(1.0/6.3E2)-(a*a)*rho*th*src[31+offset]*(1.0/8.4E2)-a*b*rho*th*src[7+offset]*(1.0/6.0E2)-a*b*rho*th*src[16+offset]*(1.0/4.0E2)+a*b*rho*th*src[25+offset]*(1.0/6.0E2)+a*b*rho*th*src[34+offset]*(1.0/9.0E2);
  dst[offset+14] = a*rho*th*src[2+offset]*(-1.087301587301587E-2)-a*rho*th*src[11+offset]*1.829365079365079E-2-a*rho*th*src[20+offset]*7.896825396825397E-3-a*rho*th*src[29+offset]*4.603174603174603E-3-(a*a)*rho*th*src[5+offset]*(1.0/4.2E2)+(a*a)*rho*th*src[14+offset]*(1.0/3.15E2)+(a*a)*rho*th*src[23+offset]*(1.0/6.3E2)-(a*a)*rho*th*src[32+offset]*(1.0/8.4E2)-a*b*rho*th*src[8+offset]*(1.0/6.0E2)-a*b*rho*th*src[17+offset]*(1.0/4.0E2)+a*b*rho*th*src[26+offset]*(1.0/6.0E2)+a*b*rho*th*src[35+offset]*(1.0/9.0E2);
  dst[offset+15] = b*rho*th*src[0+offset]*7.896825396825397E-3+b*rho*th*src[9+offset]*1.829365079365079E-2+b*rho*th*src[18+offset]*1.087301587301587E-2+b*rho*th*src[27+offset]*4.603174603174603E-3+(b*b)*rho*th*src[6+offset]*(1.0/6.3E2)+(b*b)*rho*th*src[15+offset]*(1.0/3.15E2)-(b*b)*rho*th*src[24+offset]*(1.0/4.2E2)-(b*b)*rho*th*src[33+offset]*(1.0/8.4E2)+a*b*rho*th*src[3+offset]*(1.0/6.0E2)-a*b*rho*th*src[12+offset]*(1.0/4.0E2)-a*b*rho*th*src[21+offset]*(1.0/6.0E2)+a*b*rho*th*src[30+offset]*(1.0/9.0E2);
  dst[offset+16] = b*rho*th*src[1+offset]*7.896825396825397E-3+b*rho*th*src[10+offset]*1.829365079365079E-2+b*rho*th*src[19+offset]*1.087301587301587E-2+b*rho*th*src[28+offset]*4.603174603174603E-3+(b*b)*rho*th*src[7+offset]*(1.0/6.3E2)+(b*b)*rho*th*src[16+offset]*(1.0/3.15E2)-(b*b)*rho*th*src[25+offset]*(1.0/4.2E2)-(b*b)*rho*th*src[34+offset]*(1.0/8.4E2)+a*b*rho*th*src[4+offset]*(1.0/6.0E2)-a*b*rho*th*src[13+offset]*(1.0/4.0E2)-a*b*rho*th*src[22+offset]*(1.0/6.0E2)+a*b*rho*th*src[31+offset]*(1.0/9.0E2);
  dst[offset+17] = b*rho*th*src[2+offset]*7.896825396825397E-3+b*rho*th*src[11+offset]*1.829365079365079E-2+b*rho*th*src[20+offset]*1.087301587301587E-2+b*rho*th*src[29+offset]*4.603174603174603E-3+(b*b)*rho*th*src[8+offset]*(1.0/6.3E2)+(b*b)*rho*th*src[17+offset]*(1.0/3.15E2)-(b*b)*rho*th*src[26+offset]*(1.0/4.2E2)-(b*b)*rho*th*src[35+offset]*(1.0/8.4E2)+a*b*rho*th*src[5+offset]*(1.0/6.0E2)-a*b*rho*th*src[14+offset]*(1.0/4.0E2)-a*b*rho*th*src[23+offset]*(1.0/6.0E2)+a*b*rho*th*src[32+offset]*(1.0/9.0E2);
  dst[offset+18] = rho*th*src[0+offset]*1.563492063492063E-2+rho*th*src[9+offset]*4.865079365079365E-2+rho*th*src[18+offset]*1.370634920634921E-1+rho*th*src[27+offset]*4.865079365079365E-2+a*rho*th*src[3+offset]*4.603174603174603E-3-a*rho*th*src[12+offset]*7.896825396825397E-3-a*rho*th*src[21+offset]*1.829365079365079E-2+a*rho*th*src[30+offset]*1.087301587301587E-2+b*rho*th*src[6+offset]*4.603174603174603E-3+b*rho*th*src[15+offset]*1.087301587301587E-2-b*rho*th*src[24+offset]*1.829365079365079E-2-b*rho*th*src[33+offset]*7.896825396825397E-3;
  dst[offset+19] = rho*th*src[1+offset]*1.563492063492063E-2+rho*th*src[10+offset]*4.865079365079365E-2+rho*th*src[19+offset]*1.370634920634921E-1+rho*th*src[28+offset]*4.865079365079365E-2+a*rho*th*src[4+offset]*4.603174603174603E-3-a*rho*th*src[13+offset]*7.896825396825397E-3-a*rho*th*src[22+offset]*1.829365079365079E-2+a*rho*th*src[31+offset]*1.087301587301587E-2+b*rho*th*src[7+offset]*4.603174603174603E-3+b*rho*th*src[16+offset]*1.087301587301587E-2-b*rho*th*src[25+offset]*1.829365079365079E-2-b*rho*th*src[34+offset]*7.896825396825397E-3;
  dst[offset+20] = rho*th*src[2+offset]*1.563492063492063E-2+rho*th*src[11+offset]*4.865079365079365E-2+rho*th*src[20+offset]*1.370634920634921E-1+rho*th*src[29+offset]*4.865079365079365E-2+a*rho*th*src[5+offset]*4.603174603174603E-3-a*rho*th*src[14+offset]*7.896825396825397E-3-a*rho*th*src[23+offset]*1.829365079365079E-2+a*rho*th*src[32+offset]*1.087301587301587E-2+b*rho*th*src[8+offset]*4.603174603174603E-3+b*rho*th*src[17+offset]*1.087301587301587E-2-b*rho*th*src[26+offset]*1.829365079365079E-2-b*rho*th*src[35+offset]*7.896825396825397E-3;
  dst[offset+21] = a*rho*th*src[0+offset]*(-4.603174603174603E-3)-a*rho*th*src[9+offset]*7.896825396825397E-3-a*rho*th*src[18+offset]*1.829365079365079E-2-a*rho*th*src[27+offset]*1.087301587301587E-2-(a*a)*rho*th*src[3+offset]*(1.0/8.4E2)+(a*a)*rho*th*src[12+offset]*(1.0/6.3E2)+(a*a)*rho*th*src[21+offset]*(1.0/3.15E2)-(a*a)*rho*th*src[30+offset]*(1.0/4.2E2)-a*b*rho*th*src[6+offset]*(1.0/9.0E2)-a*b*rho*th*src[15+offset]*(1.0/6.0E2)+a*b*rho*th*src[24+offset]*(1.0/4.0E2)+a*b*rho*th*src[33+offset]*(1.0/6.0E2);
  dst[offset+22] = a*rho*th*src[1+offset]*(-4.603174603174603E-3)-a*rho*th*src[10+offset]*7.896825396825397E-3-a*rho*th*src[19+offset]*1.829365079365079E-2-a*rho*th*src[28+offset]*1.087301587301587E-2-(a*a)*rho*th*src[4+offset]*(1.0/8.4E2)+(a*a)*rho*th*src[13+offset]*(1.0/6.3E2)+(a*a)*rho*th*src[22+offset]*(1.0/3.15E2)-(a*a)*rho*th*src[31+offset]*(1.0/4.2E2)-a*b*rho*th*src[7+offset]*(1.0/9.0E2)-a*b*rho*th*src[16+offset]*(1.0/6.0E2)+a*b*rho*th*src[25+offset]*(1.0/4.0E2)+a*b*rho*th*src[34+offset]*(1.0/6.0E2);
  dst[offset+23] = a*rho*th*src[2+offset]*(-4.603174603174603E-3)-a*rho*th*src[11+offset]*7.896825396825397E-3-a*rho*th*src[20+offset]*1.829365079365079E-2-a*rho*th*src[29+offset]*1.087301587301587E-2-(a*a)*rho*th*src[5+offset]*(1.0/8.4E2)+(a*a)*rho*th*src[14+offset]*(1.0/6.3E2)+(a*a)*rho*th*src[23+offset]*(1.0/3.15E2)-(a*a)*rho*th*src[32+offset]*(1.0/4.2E2)-a*b*rho*th*src[8+offset]*(1.0/9.0E2)-a*b*rho*th*src[17+offset]*(1.0/6.0E2)+a*b*rho*th*src[26+offset]*(1.0/4.0E2)+a*b*rho*th*src[35+offset]*(1.0/6.0E2);
  dst[offset+24] = b*rho*th*src[0+offset]*(-4.603174603174603E-3)-b*rho*th*src[9+offset]*1.087301587301587E-2-b*rho*th*src[18+offset]*1.829365079365079E-2-b*rho*th*src[27+offset]*7.896825396825397E-3-(b*b)*rho*th*src[6+offset]*(1.0/8.4E2)-(b*b)*rho*th*src[15+offset]*(1.0/4.2E2)+(b*b)*rho*th*src[24+offset]*(1.0/3.15E2)+(b*b)*rho*th*src[33+offset]*(1.0/6.3E2)-a*b*rho*th*src[3+offset]*(1.0/9.0E2)+a*b*rho*th*src[12+offset]*(1.0/6.0E2)+a*b*rho*th*src[21+offset]*(1.0/4.0E2)-a*b*rho*th*src[30+offset]*(1.0/6.0E2);
  dst[offset+25] = b*rho*th*src[1+offset]*(-4.603174603174603E-3)-b*rho*th*src[10+offset]*1.087301587301587E-2-b*rho*th*src[19+offset]*1.829365079365079E-2-b*rho*th*src[28+offset]*7.896825396825397E-3-(b*b)*rho*th*src[7+offset]*(1.0/8.4E2)-(b*b)*rho*th*src[16+offset]*(1.0/4.2E2)+(b*b)*rho*th*src[25+offset]*(1.0/3.15E2)+(b*b)*rho*th*src[34+offset]*(1.0/6.3E2)-a*b*rho*th*src[4+offset]*(1.0/9.0E2)+a*b*rho*th*src[13+offset]*(1.0/6.0E2)+a*b*rho*th*src[22+offset]*(1.0/4.0E2)-a*b*rho*th*src[31+offset]*(1.0/6.0E2);
  dst[offset+26] = b*rho*th*src[2+offset]*(-4.603174603174603E-3)-b*rho*th*src[11+offset]*1.087301587301587E-2-b*rho*th*src[20+offset]*1.829365079365079E-2-b*rho*th*src[29+offset]*7.896825396825397E-3-(b*b)*rho*th*src[8+offset]*(1.0/8.4E2)-(b*b)*rho*th*src[17+offset]*(1.0/4.2E2)+(b*b)*rho*th*src[26+offset]*(1.0/3.15E2)+(b*b)*rho*th*src[35+offset]*(1.0/6.3E2)-a*b*rho*th*src[5+offset]*(1.0/9.0E2)+a*b*rho*th*src[14+offset]*(1.0/6.0E2)+a*b*rho*th*src[23+offset]*(1.0/4.0E2)-a*b*rho*th*src[32+offset]*(1.0/6.0E2);
  dst[offset+27] = rho*th*src[0+offset]*4.865079365079365E-2+rho*th*src[9+offset]*1.563492063492063E-2+rho*th*src[18+offset]*4.865079365079365E-2+rho*th*src[27+offset]*1.370634920634921E-1+a*rho*th*src[3+offset]*7.896825396825397E-3-a*rho*th*src[12+offset]*4.603174603174603E-3-a*rho*th*src[21+offset]*1.087301587301587E-2+a*rho*th*src[30+offset]*1.829365079365079E-2+b*rho*th*src[6+offset]*1.087301587301587E-2+b*rho*th*src[15+offset]*4.603174603174603E-3-b*rho*th*src[24+offset]*7.896825396825397E-3-b*rho*th*src[33+offset]*1.829365079365079E-2;
  dst[offset+28] = rho*th*src[1+offset]*4.865079365079365E-2+rho*th*src[10+offset]*1.563492063492063E-2+rho*th*src[19+offset]*4.865079365079365E-2+rho*th*src[28+offset]*1.370634920634921E-1+a*rho*th*src[4+offset]*7.896825396825397E-3-a*rho*th*src[13+offset]*4.603174603174603E-3-a*rho*th*src[22+offset]*1.087301587301587E-2+a*rho*th*src[31+offset]*1.829365079365079E-2+b*rho*th*src[7+offset]*1.087301587301587E-2+b*rho*th*src[16+offset]*4.603174603174603E-3-b*rho*th*src[25+offset]*7.896825396825397E-3-b*rho*th*src[34+offset]*1.829365079365079E-2;
  dst[offset+29] = rho*th*src[2+offset]*4.865079365079365E-2+rho*th*src[11+offset]*1.563492063492063E-2+rho*th*src[20+offset]*4.865079365079365E-2+rho*th*src[29+offset]*1.370634920634921E-1+a*rho*th*src[5+offset]*7.896825396825397E-3-a*rho*th*src[14+offset]*4.603174603174603E-3-a*rho*th*src[23+offset]*1.087301587301587E-2+a*rho*th*src[32+offset]*1.829365079365079E-2+b*rho*th*src[8+offset]*1.087301587301587E-2+b*rho*th*src[17+offset]*4.603174603174603E-3-b*rho*th*src[26+offset]*7.896825396825397E-3-b*rho*th*src[35+offset]*1.829365079365079E-2;
  dst[offset+30] = a*rho*th*src[0+offset]*7.896825396825397E-3+a*rho*th*src[9+offset]*4.603174603174603E-3+a*rho*th*src[18+offset]*1.087301587301587E-2+a*rho*th*src[27+offset]*1.829365079365079E-2+(a*a)*rho*th*src[3+offset]*(1.0/6.3E2)-(a*a)*rho*th*src[12+offset]*(1.0/8.4E2)-(a*a)*rho*th*src[21+offset]*(1.0/4.2E2)+(a*a)*rho*th*src[30+offset]*(1.0/3.15E2)+a*b*rho*th*src[6+offset]*(1.0/6.0E2)+a*b*rho*th*src[15+offset]*(1.0/9.0E2)-a*b*rho*th*src[24+offset]*(1.0/6.0E2)-a*b*rho*th*src[33+offset]*(1.0/4.0E2);
  dst[offset+31] = a*rho*th*src[1+offset]*7.896825396825397E-3+a*rho*th*src[10+offset]*4.603174603174603E-3+a*rho*th*src[19+offset]*1.087301587301587E-2+a*rho*th*src[28+offset]*1.829365079365079E-2+(a*a)*rho*th*src[4+offset]*(1.0/6.3E2)-(a*a)*rho*th*src[13+offset]*(1.0/8.4E2)-(a*a)*rho*th*src[22+offset]*(1.0/4.2E2)+(a*a)*rho*th*src[31+offset]*(1.0/3.15E2)+a*b*rho*th*src[7+offset]*(1.0/6.0E2)+a*b*rho*th*src[16+offset]*(1.0/9.0E2)-a*b*rho*th*src[25+offset]*(1.0/6.0E2)-a*b*rho*th*src[34+offset]*(1.0/4.0E2);
  dst[offset+32] = a*rho*th*src[2+offset]*7.896825396825397E-3+a*rho*th*src[11+offset]*4.603174603174603E-3+a*rho*th*src[20+offset]*1.087301587301587E-2+a*rho*th*src[29+offset]*1.829365079365079E-2+(a*a)*rho*th*src[5+offset]*(1.0/6.3E2)-(a*a)*rho*th*src[14+offset]*(1.0/8.4E2)-(a*a)*rho*th*src[23+offset]*(1.0/4.2E2)+(a*a)*rho*th*src[32+offset]*(1.0/3.15E2)+a*b*rho*th*src[8+offset]*(1.0/6.0E2)+a*b*rho*th*src[17+offset]*(1.0/9.0E2)-a*b*rho*th*src[26+offset]*(1.0/6.0E2)-a*b*rho*th*src[35+offset]*(1.0/4.0E2);
  dst[offset+33] = b*rho*th*src[0+offset]*(-1.087301587301587E-2)-b*rho*th*src[9+offset]*4.603174603174603E-3-b*rho*th*src[18+offset]*7.896825396825397E-3-b*rho*th*src[27+offset]*1.829365079365079E-2-(b*b)*rho*th*src[6+offset]*(1.0/4.2E2)-(b*b)*rho*th*src[15+offset]*(1.0/8.4E2)+(b*b)*rho*th*src[24+offset]*(1.0/6.3E2)+(b*b)*rho*th*src[33+offset]*(1.0/3.15E2)-a*b*rho*th*src[3+offset]*(1.0/6.0E2)+a*b*rho*th*src[12+offset]*(1.0/9.0E2)+a*b*rho*th*src[21+offset]*(1.0/6.0E2)-a*b*rho*th*src[30+offset]*(1.0/4.0E2);
  dst[offset+34] = b*rho*th*src[1+offset]*(-1.087301587301587E-2)-b*rho*th*src[10+offset]*4.603174603174603E-3-b*rho*th*src[19+offset]*7.896825396825397E-3-b*rho*th*src[28+offset]*1.829365079365079E-2-(b*b)*rho*th*src[7+offset]*(1.0/4.2E2)-(b*b)*rho*th*src[16+offset]*(1.0/8.4E2)+(b*b)*rho*th*src[25+offset]*(1.0/6.3E2)+(b*b)*rho*th*src[34+offset]*(1.0/3.15E2)-a*b*rho*th*src[4+offset]*(1.0/6.0E2)+a*b*rho*th*src[13+offset]*(1.0/9.0E2)+a*b*rho*th*src[22+offset]*(1.0/6.0E2)-a*b*rho*th*src[31+offset]*(1.0/4.0E2);
  dst[offset+35] = b*rho*th*src[2+offset]*(-1.087301587301587E-2)-b*rho*th*src[11+offset]*4.603174603174603E-3-b*rho*th*src[20+offset]*7.896825396825397E-3-b*rho*th*src[29+offset]*1.829365079365079E-2-(b*b)*rho*th*src[8+offset]*(1.0/4.2E2)-(b*b)*rho*th*src[17+offset]*(1.0/8.4E2)+(b*b)*rho*th*src[26+offset]*(1.0/6.3E2)+(b*b)*rho*th*src[35+offset]*(1.0/3.15E2)-a*b*rho*th*src[5+offset]*(1.0/6.0E2)+a*b*rho*th*src[14+offset]*(1.0/9.0E2)+a*b*rho*th*src[23+offset]*(1.0/6.0E2)-a*b*rho*th*src[32+offset]*(1.0/4.0E2);
}

int System::buildAppliedImpulseVector() {
  // build k
  updateElasticForces();
  if(bodies.size()) multiplyByMass<<<BLOCKS(3*bodies.size()),THREADS>>>(CASTD1(mass_d), CASTD1(v_d), CASTD1(k_d), 3*bodies.size());
  if(beams.size()) multiplyByBeamMass<<<BLOCKS(beams.size()),THREADS>>>(CASTD3(contactGeometry_d), CASTD3(materialsBeam_d), CASTD1(v_d), CASTD1(k_d), bodies.size(), beams.size());
  if(plates.size()) multiplyByPlateMass<<<BLOCKS(plates.size()),THREADS>>>(CASTD3(contactGeometry_d), CASTD4(materialsPlate_d), CASTD1(v_d), CASTD1(k_d), bodies.size(), beams.size(), plates.size());
  //cusp::blas::axpy(fElastic,fApplied,-1.0); //TODO: Come up with a fix for applied forces
  cusp::blas::axpbypcz(f,fElastic,k,k,h,-h,1.0);

  return 0;
}

__global__ void buildStabilization(double* b, double4* normalsAndPenetrations, double timeStep, uint numBilateralConstraints, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double penetration = normalsAndPenetrations[index].w;

  b[3*index+numBilateralConstraints] = penetration/timeStep;
  b[3*index+1+numBilateralConstraints] = 0;
  b[3*index+2+numBilateralConstraints] = 0;
}

__global__ void buildStabilizationBilateral(double* b, int2* contraintBilateralDOF, double* p, double timeStep, uint numBilateralConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numBilateralConstraints);

  int2 constraintDOF = contraintBilateralDOF[index];
  double violation = p[constraintDOF.x]-p[constraintDOF.y];

  b[index] = violation/timeStep;
}

int System::buildSchurVector() {
  // build r
  r_d.resize(3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size());
  b_d.resize(3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size());
  // TODO: There's got to be a better way to do this...
  //r.resize(3*collisionDetector->numCollisions);
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  thrust::device_ptr<double> wrapped_device_b(CASTD1(b_d));
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
  cusp::multiply(mass,k,tmp);
  cusp::multiply(D,tmp,r);

  if(constraintsBilateralDOF_d.size()) buildStabilizationBilateral<<<BLOCKS(constraintsBilateralDOF_d.size()),THREADS>>>(CASTD1(b_d), CASTI2(constraintsBilateralDOF_d), CASTD1(p_d), h, constraintsBilateralDOF_d.size());
  if(collisionDetector->numCollisions) buildStabilization<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(b_d), CASTD4(collisionDetector->normalsAndPenetrations_d), h, constraintsBilateralDOF_d.size(), collisionDetector->numCollisions);
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
  filestream << "0, " << bodies.size() << ", " << beams.size() << ", " << endl;
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
  for (int i = 0; i < beams.size(); i++) {
    // TODO: Need to know collision family information, density, elastic modulus, number of contacts (especially important when importing)
    filestream
        << bodies.size()+i << ", "
        << contactGeometry_h[bodies.size()+i].x << ", "
        << contactGeometry_h[bodies.size()+i].y << ", "

        << p_h[3*bodies.size()+12*i] << ", "
        << p_h[3*bodies.size()+12*i+1] << ", "
        << p_h[3*bodies.size()+12*i+2] << ", "
        << p_h[3*bodies.size()+12*i+3] << ", "
        << p_h[3*bodies.size()+12*i+4] << ", "
        << p_h[3*bodies.size()+12*i+5] << ", "
        << p_h[3*bodies.size()+12*i+6] << ", "
        << p_h[3*bodies.size()+12*i+7] << ", "
        << p_h[3*bodies.size()+12*i+8] << ", "
        << p_h[3*bodies.size()+12*i+9] << ", "
        << p_h[3*bodies.size()+12*i+10] << ", "
        << p_h[3*bodies.size()+12*i+11] << ", "

        << v_h[3*bodies.size()+12*i] << ", "
        << v_h[3*bodies.size()+12*i+1] << ", "
        << v_h[3*bodies.size()+12*i+2] << ", "
        << v_h[3*bodies.size()+12*i+3] << ", "
        << v_h[3*bodies.size()+12*i+4] << ", "
        << v_h[3*bodies.size()+12*i+5] << ", "
        << v_h[3*bodies.size()+12*i+6] << ", "
        << v_h[3*bodies.size()+12*i+7] << ", "
        << v_h[3*bodies.size()+12*i+8] << ", "
        << v_h[3*bodies.size()+12*i+9] << ", "
        << v_h[3*bodies.size()+12*i+10] << ", "
        << v_h[3*bodies.size()+12*i+11] << ", ";

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
  int numBeams;
  double blah;
  int index;
  int shape;

  ifstream ifile(filename.c_str());
  getline(ifile,temp_data);
  for(int i=0; i<temp_data.size(); ++i){
    if(temp_data[i]==','){temp_data[i]=' ';}
  }
  stringstream ss1(temp_data);
  ss1>>blah>>numBodies>>numBeams;

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

  // TODO: IMPORT BEAMS

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
