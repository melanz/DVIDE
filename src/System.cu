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
  offsetConstraintsDOF = 0;
  objectiveCCP = 0;

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

  wt6.push_back(0.17132449);
  wt6.push_back(0.36076157);
  wt6.push_back(0.46791393);
  wt6.push_back(0.46791393);
  wt6.push_back(0.36076157);
  wt6.push_back(0.17132449);
  pt6.push_back(-0.93246951);
  pt6.push_back(-0.66120939);
  pt6.push_back(-0.23861918);
  pt6.push_back(0.23861918);
  pt6.push_back(0.66120939);
  pt6.push_back(0.93246951);
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
  offsetConstraintsDOF = 0;
  objectiveCCP = 0;

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

  wt6.push_back(0.17132449);
  wt6.push_back(0.36076157);
  wt6.push_back(0.46791393);
  wt6.push_back(0.46791393);
  wt6.push_back(0.36076157);
  wt6.push_back(0.17132449);
  pt6.push_back(-0.93246951);
  pt6.push_back(-0.66120939);
  pt6.push_back(-0.23861918);
  pt6.push_back(0.23861918);
  pt6.push_back(0.66120939);
  pt6.push_back(0.93246951);
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

int System::add(Body2D* body2D) {
  //add the plate
  body2D->sys = this;
  body2Ds.push_back(body2D);
  return body2Ds.size();
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
  materialsBody2D_d = materialsBody2D_h;
  fixedBodies_d = fixedBodies_h;

  strainDerivative_d = strainDerivative_h;
  strain_d = strain_h;
  strainEnergy_d = strainEnergy_h;
  strainPlate_d = strainPlate_h;
  strainEnergyPlate_d = strainEnergyPlate_h;
  strainDerivativePlate_d = strainDerivativePlate_h;
  curvatureDerivativePlate_d = curvatureDerivativePlate_h;
  Sx_d = Sx_h;
  Sxx_d = Sxx_h;
  Sy_d = Sy_h;
  Syy_d = Syy_h;
  strainPlate0_d = strainPlate0_h;
  curvaturePlate0_d = curvaturePlate0_h;
  strainBeam0_d = strainBeam0_h;
  curvatureBeam0_d = curvatureBeam0_h;

  // Shell Mesh Initialization
  fElasticShellMesh_d = fElasticShellMesh_h;
  strainShellMesh_d = strainShellMesh_h;
  strainEnergyShellMesh_d = strainEnergyShellMesh_h;
  strainDerivativeShellMesh_d = strainDerivativeShellMesh_h;
  curvatureDerivativeShellMesh_d = curvatureDerivativeShellMesh_h;
  Sx_shellMesh_d = Sx_shellMesh_h;
  Sxx_shellMesh_d = Sxx_shellMesh_h;
  Sy_shellMesh_d = Sy_shellMesh_h;
  Syy_shellMesh_d = Syy_shellMesh_h;
  strainShellMesh0_d = strainShellMesh0_h;
  curvatureShellMesh0_d = curvatureShellMesh0_h;
  // End Shell Mesh Initialization

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

  int offset_shellMesh = plates.size()*36+12*beams.size()+3*bodies.size()+3*body2Ds.size();
  p = DeviceValueArrayView(wrapped_device_p, wrapped_device_p + p_d.size());
  v = DeviceValueArrayView(wrapped_device_v, wrapped_device_v + v_d.size());
  v_shellMesh = DeviceValueArrayView(wrapped_device_v + offset_shellMesh, wrapped_device_v + v_d.size());
  a = DeviceValueArrayView(wrapped_device_a, wrapped_device_a + a_d.size());
  f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());
  f_contact = DeviceValueArrayView(wrapped_device_f_contact, wrapped_device_f_contact + f_contact_d.size());
  fApplied = DeviceValueArrayView(wrapped_device_fApplied, wrapped_device_fApplied + fApplied_d.size());
  fElastic = DeviceValueArrayView(wrapped_device_fElastic, wrapped_device_fElastic + fElastic_d.size());
  tmp = DeviceValueArrayView(wrapped_device_tmp, wrapped_device_tmp + tmp_d.size());
  tmp_shellMesh = DeviceValueArrayView(wrapped_device_tmp + offset_shellMesh, wrapped_device_tmp + tmp_d.size());
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
  k = DeviceValueArrayView(wrapped_device_k, wrapped_device_k + k_d.size());
  k_shellMesh = DeviceValueArrayView(wrapped_device_k + offset_shellMesh, wrapped_device_k + k_d.size());
  gamma = DeviceValueArrayView(wrapped_device_gamma, wrapped_device_gamma + gamma_d.size());

  // create mass matrix using cusp library (shouldn't change)
  thrust::device_ptr<int> wrapped_device_I(CASTI1(massI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + massI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(massJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + massJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(mass_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + mass_d.size());

  mass = DeviceView(a_d.size(), a_d.size(), mass_d.size(), row_indices, column_indices, values);
  mass.sort_by_row();
  // end create mass matrix

  // create shellMesh mass matrix using cusp
  thrust::device_ptr<int> wrapped_device_I_shell(CASTI1(massShellI_d));
  DeviceIndexArrayView row_indices_shell = DeviceIndexArrayView(wrapped_device_I_shell, wrapped_device_I_shell + massShellI_h.size());

  thrust::device_ptr<int> wrapped_device_J_shell(CASTI1(massShellJ_d));
  DeviceIndexArrayView column_indices_shell = DeviceIndexArrayView(wrapped_device_J_shell, wrapped_device_J_shell + massShellJ_h.size());

  thrust::device_ptr<double> wrapped_device_V_shell(CASTD1(massShell_d));
  DeviceValueArrayView values_shell = DeviceValueArrayView(wrapped_device_V_shell, wrapped_device_V_shell + massShell_d.size());

  mass_shellMesh = DeviceView(3*nodes_h.size(), 3*nodes_h.size(), massShell_d.size(), row_indices_shell, column_indices_shell, values_shell);
  mass_shellMesh.sort_by_row();
  // end create shellMesh mass matrix using cusp

  // calculate initialize strains and curvatures
  calculateInitialStrainAndCurvature();

  processConstraints();
  offsetBilaterals_d = offsetBilaterals_h;
  constraintsBilateralDOF_d = constraintsBilateralDOF_h;
  infoConstraintBilateralDOF_d = infoConstraintBilateralDOF_h;
  constraintsSpherical_ShellNodeToBody2D_d =constraintsSpherical_ShellNodeToBody2D_h;
  pSpherical_ShellNodeToBody2D_d = pSpherical_ShellNodeToBody2D_h;

  return 0;
}

int System::processConstraints() {
  // process the DOF bilaterals
  int offset = 0;
  for(int i=0;i<constraintsBilateralDOF_h.size();i++) {
    offsetBilaterals_h.push_back(offset);
    if(constraintsBilateralDOF_h[i].y<0) {
      offset+=1;
      infoConstraintBilateralDOF_h[i].z = p_h[constraintsBilateralDOF_h[i].x]; // need to know initial value
    } else {
      offset+=2;
    }
  }
  // end process the DOF bilaterals

  // process the ShellNodeToBody2D spherical constraints
  for(int i=0;i<constraintsSpherical_ShellNodeToBody2D_h.size();i++) {
    int indexA = constraintsSpherical_ShellNodeToBody2D_h[i].x;
    int nodeIndexA = constraintsSpherical_ShellNodeToBody2D_h[i].y;
    int indexB = constraintsSpherical_ShellNodeToBody2D_h[i].z;
    int offsetA;
    int offsetB;
    if(indexA==-1) {
      // shell mesh
      offsetA = 3*bodies.size()+12*beams.size()+36*plates.size()+3*body2Ds.size()+9*nodeIndexA;
      offsetB = 3*bodies.size()+12*beams.size()+36*plates.size()+3*indexB;
    } else {
      // plate
      offsetA = 3*bodies.size()+12*beams.size()+36*indexA+9*nodeIndexA;
      offsetB = 3*bodies.size()+12*beams.size()+36*plates.size()+3*indexB;
    }
    constraintsSpherical_ShellNodeToBody2D_h[i].x = offsetA; // NOTE: Reset value to offsets! Easier for later constraint processing
    constraintsSpherical_ShellNodeToBody2D_h[i].y = offsetB; // NOTE: Reset value to offsets! Easier for later constraint processing
    pSpherical_ShellNodeToBody2D_h.push_back(make_double3(p_h[offsetA]-p_h[offsetB],p_h[offsetA+1]-p_h[offsetB+1],p_h[offsetA+2]));
  }
  // end process the ShellNodeToBody2D spherical constraints

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
    plates[j]->addPlate(j);
  }

  for(int j=0; j<body2Ds.size(); j++) {
    body2Ds[j]->addBody2D(j);
  }

  // add shell mesh to system
  if(nodes_h.size()) {
    indices_h.push_back(p_h.size());

    // update p
    for(int i=0; i<nodes_h.size(); i++) {
      p_h.push_back(nodes_h[i].x);
      p_h.push_back(nodes_h[i].y);
      p_h.push_back(nodes_h[i].z);
    }

    // update fext
    for(int i=0; i<fextMesh_h.size(); i++) {
      f_h.push_back(fextMesh_h[i]);
    }

    // update zero vectors
    for(int i=0;i<3*nodes_h.size();i++) {
      v_h.push_back(0);
      a_h.push_back(0);
      f_contact_h.push_back(0);
      fApplied_h.push_back(0);
      fElastic_h.push_back(0);
      tmp_h.push_back(0);
      k_h.push_back(0);
      r_h.push_back(0);
    }

    // update the mass inverse
    int offset = plates.size()*36+12*beams.size()+3*bodies.size()+3*body2Ds.size();
    for(int i=0;i<invMassShellI_h.size();i++) {
      massI_h.push_back(invMassShellI_h[i]+offset);
      massJ_h.push_back(invMassShellJ_h[i]+offset);
      mass_h.push_back(invMassShell_h[i]);
    }

    for(int j=0;j<shellConnectivities_h.size();j++) {
      contactGeometry_h.push_back(make_double3(shellGeometries_h[j].x,shellGeometries_h[j].y,shellGeometries_h[j].w));

      for(int i=0;i<36;i++) {
        fElasticShellMesh_h.push_back(0);
        strainDerivativeShellMesh_h.push_back(make_double3(0,0,0));
        curvatureDerivativeShellMesh_h.push_back(make_double3(0,0,0));
      }

      strainEnergyShellMesh_h.push_back(0);
      strainShellMesh_h.push_back(make_double3(0,0,0));
      for(int i=0;i<wt6.size()*pt6.size();i++) strainShellMesh0_h.push_back(make_double3(0,0,0));
      for(int i=0;i<wt5.size()*pt5.size();i++) curvatureShellMesh0_h.push_back(make_double3(0,0,0));

      for(int i=0;i<12;i++) {
        Sx_shellMesh_h.push_back(0);
        Sxx_shellMesh_h.push_back(0);
        Sy_shellMesh_h.push_back(0);
        Syy_shellMesh_h.push_back(0);
      }
      for(int i=1;i<shellGeometries_h[j].w-1;i++) {
        for(int k=1;k<shellGeometries_h[j].w-1;k++) {
          collisionGeometry_h.push_back(make_double3(0.5*shellGeometries_h[j].z,0,0));
          collisionMap_h.push_back(make_int4(plates.size()+beams.size()+bodies.size()+body2Ds.size()+j,i,k,-2));
        }
      }
    }

  }

  initializeDevice();
  solver->setup();

  return 0;
}

int System::addBilateralConstraintDOF(int DOFA, int DOFB) {
  constraintsBilateralDOF_h.push_back(make_int2(DOFA,DOFB));
  infoConstraintBilateralDOF_h.push_back(make_double3(0,0,0));

  if(DOFB<0) {
    offsetConstraintsDOF=offsetConstraintsDOF+1;
  } else {
    offsetConstraintsDOF=offsetConstraintsDOF+2;
  }
  return 0;
}

int System::addBilateralConstraintDOF(int DOFA, int DOFB, double velocity, double startTime) {
  constraintsBilateralDOF_h.push_back(make_int2(DOFA,DOFB));
  infoConstraintBilateralDOF_h.push_back(make_double3(velocity,startTime,0));

  if(DOFB<0) {
    offsetConstraintsDOF=offsetConstraintsDOF+1;
  } else {
    offsetConstraintsDOF=offsetConstraintsDOF+2;
  }
  return 0;
}

int System::pinShellNodeToBody2D(int shellNodeIndex, int body2Dindex) {
  constraintsSpherical_ShellNodeToBody2D_h.push_back(make_int3(-1,shellNodeIndex,body2Dindex));
  return 0;
}

int System::pinPlateNodeToBody2D(int plateIndex, int plateNodeIndex, int body2Dindex) {
  constraintsSpherical_ShellNodeToBody2D_h.push_back(make_int3(plateIndex,plateNodeIndex,body2Dindex));
  return 0;
}

int System::DoTimeStep() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  objectiveCCP = 0;

  // Perform collision detection
  if(collisionGeometry_d.size()) {
    collisionDetector->generateAxisAlignedBoundingBoxes();
    collisionDetector->detectPossibleCollisions_spatialSubdivision();
    collisionDetector->detectCollisions();
  }

  buildAppliedImpulseVector();
  if(collisionDetector->numCollisions||constraintsBilateralDOF_d.size()||constraintsSpherical_ShellNodeToBody2D_d.size()) {

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

__global__ void constructBilateralJacobian(int2* constraintBilateralDOF, int* offsets, int* DI, int* DJ, double* D, uint numConstraintsBilateral) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraintsBilateral);

  int2 bilateralDOFs = constraintBilateralDOF[index];
  int offset = offsets[index];

  DI[offset] = index;
  DJ[offset] = bilateralDOFs.x;
  D[offset] = 1.0;

  if(bilateralDOFs.y>=0)
  {
    DI[offset+1] = index;
    DJ[offset+1] = bilateralDOFs.y;
    D[offset+1] = -1.0;
  }
}

__global__ void constructSpherical_ShellNodeToBody2DJacobian(int3* constraints, double3* pHats, double* p, int* DI, int* DJ, double* D, int numConstraintsDOF, int offsetConstraintsDOF, uint numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  int offset = offsetConstraintsDOF;
  int3 constraint = constraints[index];
  double3 pHat = pHats[index];

  int offsetS = constraint.x;
  int offsetB = constraint.y;

  DI[7*index+0+offset] = 3*index+numConstraintsDOF;
  DI[7*index+1+offset] = 3*index+numConstraintsDOF;
  DI[7*index+2+offset] = 3*index+numConstraintsDOF;
  DI[7*index+3+offset] = 3*index+1+numConstraintsDOF;
  DI[7*index+4+offset] = 3*index+1+numConstraintsDOF;
  DI[7*index+5+offset] = 3*index+1+numConstraintsDOF;
  DI[7*index+6+offset] = 3*index+2+numConstraintsDOF;

  DJ[7*index+0+offset] = offsetB;
  DJ[7*index+1+offset] = offsetB+2;
  DJ[7*index+2+offset] = offsetS;
  DJ[7*index+3+offset] = offsetB+1;
  DJ[7*index+4+offset] = offsetB+2;
  DJ[7*index+5+offset] = offsetS+1;
  DJ[7*index+6+offset] = offsetS+2;

  double phi = p[offsetB+2];
  D[7*index+0+offset] = 1.0;
  D[7*index+1+offset] = -pHat.x*sin(phi)-pHat.y*cos(phi);
  D[7*index+2+offset] = -1.0;
  D[7*index+3+offset] = 1.0;
  D[7*index+4+offset] = pHat.x*cos(phi)-pHat.y*sin(phi);
  D[7*index+5+offset] = -1.0;
  D[7*index+6+offset] = -1.0;
}

__global__ void constructContactJacobian(int* nonzerosPerContact_d, int4* collisionMap, int4* connectivities, double3* geometries, double3* collisionGeometry, int* DI, int* DJ, double* D, double* friction, double4* normalsAndPenetrations, uint* collisionIdentifierA, uint* collisionIdentifierB, int* indices, int numBodies, int numBeams, int numPlates, int numBodys2D, uint offsetConstraintsBilateral, uint numConstraintsBilateral, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  friction[index] = 0.25; // TODO: EDIT THIS TO BE MINIMUM OF FRICTION COEFFICIENTS
  bool shellMeshA = false;
  bool shellMeshB = false;
  int4 connectivityA;
  int4 connectivityB;
  int shellOffset = 3*numBodies+12*numBeams+36*numPlates+3*numBodys2D;

  int offsetA = (!index) ? 0 : nonzerosPerContact_d[index - 1];
  offsetA+=offsetConstraintsBilateral; // add offset for bilateral constraints
  DI = &DI[offsetA];
  DJ = &DJ[offsetA];
  D = &D[offsetA];

  int bodyIdentifierA = collisionMap[collisionIdentifierA[index]].x;
  int bodyIdentifierB = collisionMap[collisionIdentifierB[index]].x;

  int endA = 3;
  if(bodyIdentifierA<numBodies) {
    endA = 3; // body
  }
  else if(bodyIdentifierA<(numBodies+numBeams)) {
    endA = 12; // beam
  }
  else if(bodyIdentifierA<(numBodies+numBeams+numPlates)) {
    endA = 36; // plate
  }
  else if(bodyIdentifierA<(numBodies+numBeams+numPlates+numBodys2D)) {
    endA = 3; // body2D
  }
  else {
    endA = 36; // shellMesh
    shellMeshA = true;
    connectivityA = connectivities[bodyIdentifierA-(numBodies+numBeams+numPlates+numBodys2D)];
  }

  int endB = 3;
  if(bodyIdentifierB<numBodies) {
    endB = 3; // body
  }
  else if(bodyIdentifierB<(numBodies+numBeams)) {
    endB = 12; // beam
  }
  else if(bodyIdentifierB<(numBodies+numBeams+numPlates)) {
    endB = 36; // plate
  }
  else if(bodyIdentifierB<(numBodies+numBeams+numPlates+numBodys2D)) {
    endB = 3; // body2D
  }
  else {
    endB = 36; // shellMesh
    shellMeshB = true;
    connectivityB = connectivities[bodyIdentifierB-(numBodies+numBeams+numPlates+numBodys2D)];
  }

  int indexA = indices[bodyIdentifierA];
  int indexB = indices[bodyIdentifierB];

  double xiA = static_cast<double>(collisionMap[collisionIdentifierA[index]].y)/(static_cast<double>(geometries[bodyIdentifierA].z-1));
  double etaA = static_cast<double>(collisionMap[collisionIdentifierA[index]].z)/(static_cast<double>(geometries[bodyIdentifierA].z-1));
  double aA = geometries[bodyIdentifierA].x;
  double bA = geometries[bodyIdentifierA].y;
  double lA = bA;

  double xiB = static_cast<double>(collisionMap[collisionIdentifierB[index]].y)/(static_cast<double>(geometries[bodyIdentifierB].z-1));
  double etaB = static_cast<double>(collisionMap[collisionIdentifierB[index]].z)/(static_cast<double>(geometries[bodyIdentifierB].z-1));
  double aB = geometries[bodyIdentifierB].x;
  double bB = geometries[bodyIdentifierB].y;
  double lB = bB;

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
    if(shellMeshA) {
      if(j<9) {
        DJ[i] = shellOffset+9*connectivityA.x+j;
      }
      else if (j<18) {
        DJ[i] = shellOffset-9+9*connectivityA.y+j;
      }
      else if (j<27) {
        DJ[i] = shellOffset-18+9*connectivityA.z+j;
      }
      else {
        DJ[i] = shellOffset-27+9*connectivityA.w+j;
      }
    } else {
      DJ[i] = indexA+j;
    }
    j++;
  }
  end+=endB;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+0+numConstraintsBilateral;
    if(shellMeshB) {
      if(j<9) {
        DJ[i] = shellOffset+9*connectivityB.x+j;
      }
      else if (j<18) {
        DJ[i] = shellOffset-9+9*connectivityB.y+j;
      }
      else if (j<27) {
        DJ[i] = shellOffset-18+9*connectivityB.z+j;
      }
      else {
        DJ[i] = shellOffset-27+9*connectivityB.w+j;
      }
    } else {
      DJ[i] = indexB+j;
    }
    j++;
  }

  // Add u, i indices
  end+=endA;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+1+numConstraintsBilateral;
    if(shellMeshA) {
      if(j<9) {
        DJ[i] = shellOffset+9*connectivityA.x+j;
      }
      else if (j<18) {
        DJ[i] = shellOffset-9+9*connectivityA.y+j;
      }
      else if (j<27) {
        DJ[i] = shellOffset-18+9*connectivityA.z+j;
      }
      else {
        DJ[i] = shellOffset-27+9*connectivityA.w+j;
      }
    } else {
      DJ[i] = indexA+j;
    }
    j++;
  }
  end+=endB;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+1+numConstraintsBilateral;
    if(shellMeshB) {
      if(j<9) {
        DJ[i] = shellOffset+9*connectivityB.x+j;
      }
      else if (j<18) {
        DJ[i] = shellOffset-9+9*connectivityB.y+j;
      }
      else if (j<27) {
        DJ[i] = shellOffset-18+9*connectivityB.z+j;
      }
      else {
        DJ[i] = shellOffset-27+9*connectivityB.w+j;
      }
    } else {
      DJ[i] = indexB+j;
    }
    j++;
  }

  // Add v, i indices
  end+=endA;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+2+numConstraintsBilateral;
    if(shellMeshA) {
      if(j<9) {
        DJ[i] = shellOffset+9*connectivityA.x+j;
      }
      else if (j<18) {
        DJ[i] = shellOffset-9+9*connectivityA.y+j;
      }
      else if (j<27) {
        DJ[i] = shellOffset-18+9*connectivityA.z+j;
      }
      else {
        DJ[i] = shellOffset-27+9*connectivityA.w+j;
      }
    } else {
      DJ[i] = indexA+j;
    }
    j++;
  }
  end+=endB;
  j = 0;
  for(;i<end;i++) {
    DI[i] = 3*index+2+numConstraintsBilateral;
    if(shellMeshB) {
      if(j<9) {
        DJ[i] = shellOffset+9*connectivityB.x+j;
      }
      else if (j<18) {
        DJ[i] = shellOffset-9+9*connectivityB.y+j;
      }
      else if (j<27) {
        DJ[i] = shellOffset-18+9*connectivityB.z+j;
      }
      else {
        DJ[i] = shellOffset-27+9*connectivityB.w+j;
      }
    } else {
      DJ[i] = indexB+j;
    }
    j++;
  }

  // Add n, values
  int startIndex = 0;
  if(bodyIdentifierA<numBodies) {
    D[startIndex+0] = n.x;
    D[startIndex+1] = n.y;
    D[startIndex+2] = n.z;
    startIndex+=3;
  } else if (bodyIdentifierA<numBodies+numBeams) {
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
  } else {
    D[startIndex+0 ] = n.x*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+1 ] = n.y*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+2 ] = n.z*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+3 ] = -aA*n.x*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+4 ] = -aA*n.y*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+5 ] = -aA*n.z*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+6 ] = -bA*etaA*n.x*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+7 ] = -bA*etaA*n.y*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+8 ] = -bA*etaA*n.z*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+9 ] = -n.x*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+10] = -n.y*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+11] = -n.z*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+12] = -aA*n.x*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+13] = -aA*n.y*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+14] = -aA*n.z*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+15] = bA*etaA*n.x*xiA*pow(etaA-1.0,2.0);
    D[startIndex+16] = bA*etaA*n.y*xiA*pow(etaA-1.0,2.0);
    D[startIndex+17] = bA*etaA*n.z*xiA*pow(etaA-1.0,2.0);
    D[startIndex+18] = -etaA*n.x*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+19] = -etaA*n.y*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+20] = -etaA*n.z*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+21] = aA*etaA*n.x*(xiA*xiA)*(xiA-1.0);
    D[startIndex+22] = aA*etaA*n.y*(xiA*xiA)*(xiA-1.0);
    D[startIndex+23] = aA*etaA*n.z*(xiA*xiA)*(xiA-1.0);
    D[startIndex+24] = bA*(etaA*etaA)*n.x*xiA*(etaA-1.0);
    D[startIndex+25] = bA*(etaA*etaA)*n.y*xiA*(etaA-1.0);
    D[startIndex+26] = bA*(etaA*etaA)*n.z*xiA*(etaA-1.0);
    D[startIndex+27] = -etaA*n.x*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+28] = -etaA*n.y*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+29] = -etaA*n.z*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+30] = aA*etaA*n.x*xiA*pow(xiA-1.0,2.0);
    D[startIndex+31] = aA*etaA*n.y*xiA*pow(xiA-1.0,2.0);
    D[startIndex+32] = aA*etaA*n.z*xiA*pow(xiA-1.0,2.0);
    D[startIndex+33] = -bA*(etaA*etaA)*n.x*(etaA-1.0)*(xiA-1.0);
    D[startIndex+34] = -bA*(etaA*etaA)*n.y*(etaA-1.0)*(xiA-1.0);
    D[startIndex+35] = -bA*(etaA*etaA)*n.z*(etaA-1.0)*(xiA-1.0);
    startIndex+=36;
  }
  if(bodyIdentifierB<numBodies) {
    D[startIndex+0] = -n.x;
    D[startIndex+1] = -n.y;
    D[startIndex+2] = -n.z;
    startIndex+=3;
  } else if (bodyIdentifierB<numBodies+numBeams) {
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
  } else {
    D[startIndex+0 ] = -n.x*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+1 ] = -n.y*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+2 ] = -n.z*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+3 ] = aB*n.x*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+4 ] = aB*n.y*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+5 ] = aB*n.z*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+6 ] = bB*etaB*n.x*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+7 ] = bB*etaB*n.y*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+8 ] = bB*etaB*n.z*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+9 ] = n.x*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+10] = n.y*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+11] = n.z*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+12] = aB*n.x*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+13] = aB*n.y*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+14] = aB*n.z*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+15] = -bB*etaB*n.x*xiB*pow(etaB-1.0,2.0);
    D[startIndex+16] = -bB*etaB*n.y*xiB*pow(etaB-1.0,2.0);
    D[startIndex+17] = -bB*etaB*n.z*xiB*pow(etaB-1.0,2.0);
    D[startIndex+18] = etaB*n.x*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+19] = etaB*n.y*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+20] = etaB*n.z*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+21] = -aB*etaB*n.x*(xiB*xiB)*(xiB-1.0);
    D[startIndex+22] = -aB*etaB*n.y*(xiB*xiB)*(xiB-1.0);
    D[startIndex+23] = -aB*etaB*n.z*(xiB*xiB)*(xiB-1.0);
    D[startIndex+24] = -bB*(etaB*etaB)*n.x*xiB*(etaB-1.0);
    D[startIndex+25] = -bB*(etaB*etaB)*n.y*xiB*(etaB-1.0);
    D[startIndex+26] = -bB*(etaB*etaB)*n.z*xiB*(etaB-1.0);
    D[startIndex+27] = etaB*n.x*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+28] = etaB*n.y*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+29] = etaB*n.z*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+30] = -aB*etaB*n.x*xiB*pow(xiB-1.0,2.0);
    D[startIndex+31] = -aB*etaB*n.y*xiB*pow(xiB-1.0,2.0);
    D[startIndex+32] = -aB*etaB*n.z*xiB*pow(xiB-1.0,2.0);
    D[startIndex+33] = bB*(etaB*etaB)*n.x*(etaB-1.0)*(xiB-1.0);
    D[startIndex+34] = bB*(etaB*etaB)*n.y*(etaB-1.0)*(xiB-1.0);
    D[startIndex+35] = bB*(etaB*etaB)*n.z*(etaB-1.0)*(xiB-1.0);
    startIndex+=36;
  }

  // Add u, values
  if(bodyIdentifierA<numBodies) {
    D[startIndex+0] = u.x;
    D[startIndex+1] = u.y;
    D[startIndex+2] = u.z;
    startIndex+=3;
  } else if (bodyIdentifierA<numBodies+numBeams) {
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
  } else {
    D[startIndex+0 ] = u.x*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+1 ] = u.y*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+2 ] = u.z*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+3 ] = -aA*u.x*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+4 ] = -aA*u.y*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+5 ] = -aA*u.z*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+6 ] = -bA*etaA*u.x*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+7 ] = -bA*etaA*u.y*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+8 ] = -bA*etaA*u.z*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+9 ] = -u.x*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+10] = -u.y*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+11] = -u.z*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+12] = -aA*u.x*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+13] = -aA*u.y*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+14] = -aA*u.z*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+15] = bA*etaA*u.x*xiA*pow(etaA-1.0,2.0);
    D[startIndex+16] = bA*etaA*u.y*xiA*pow(etaA-1.0,2.0);
    D[startIndex+17] = bA*etaA*u.z*xiA*pow(etaA-1.0,2.0);
    D[startIndex+18] = -etaA*u.x*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+19] = -etaA*u.y*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+20] = -etaA*u.z*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+21] = aA*etaA*u.x*(xiA*xiA)*(xiA-1.0);
    D[startIndex+22] = aA*etaA*u.y*(xiA*xiA)*(xiA-1.0);
    D[startIndex+23] = aA*etaA*u.z*(xiA*xiA)*(xiA-1.0);
    D[startIndex+24] = bA*(etaA*etaA)*u.x*xiA*(etaA-1.0);
    D[startIndex+25] = bA*(etaA*etaA)*u.y*xiA*(etaA-1.0);
    D[startIndex+26] = bA*(etaA*etaA)*u.z*xiA*(etaA-1.0);
    D[startIndex+27] = -etaA*u.x*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+28] = -etaA*u.y*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+29] = -etaA*u.z*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+30] = aA*etaA*u.x*xiA*pow(xiA-1.0,2.0);
    D[startIndex+31] = aA*etaA*u.y*xiA*pow(xiA-1.0,2.0);
    D[startIndex+32] = aA*etaA*u.z*xiA*pow(xiA-1.0,2.0);
    D[startIndex+33] = -bA*(etaA*etaA)*u.x*(etaA-1.0)*(xiA-1.0);
    D[startIndex+34] = -bA*(etaA*etaA)*u.y*(etaA-1.0)*(xiA-1.0);
    D[startIndex+35] = -bA*(etaA*etaA)*u.z*(etaA-1.0)*(xiA-1.0);
    startIndex+=36;
  }
  if(bodyIdentifierB<numBodies) {
    D[startIndex+0] = -u.x;
    D[startIndex+1] = -u.y;
    D[startIndex+2] = -u.z;
    startIndex+=3;
  } else if (bodyIdentifierB<numBodies+numBeams) {
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
  } else {
    D[startIndex+0 ] = -u.x*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+1 ] = -u.y*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+2 ] = -u.z*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+3 ] = aB*u.x*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+4 ] = aB*u.y*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+5 ] = aB*u.z*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+6 ] = bB*etaB*u.x*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+7 ] = bB*etaB*u.y*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+8 ] = bB*etaB*u.z*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+9 ] = u.x*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+10] = u.y*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+11] = u.z*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+12] = aB*u.x*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+13] = aB*u.y*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+14] = aB*u.z*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+15] = -bB*etaB*u.x*xiB*pow(etaB-1.0,2.0);
    D[startIndex+16] = -bB*etaB*u.y*xiB*pow(etaB-1.0,2.0);
    D[startIndex+17] = -bB*etaB*u.z*xiB*pow(etaB-1.0,2.0);
    D[startIndex+18] = etaB*u.x*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+19] = etaB*u.y*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+20] = etaB*u.z*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+21] = -aB*etaB*u.x*(xiB*xiB)*(xiB-1.0);
    D[startIndex+22] = -aB*etaB*u.y*(xiB*xiB)*(xiB-1.0);
    D[startIndex+23] = -aB*etaB*u.z*(xiB*xiB)*(xiB-1.0);
    D[startIndex+24] = -bB*(etaB*etaB)*u.x*xiB*(etaB-1.0);
    D[startIndex+25] = -bB*(etaB*etaB)*u.y*xiB*(etaB-1.0);
    D[startIndex+26] = -bB*(etaB*etaB)*u.z*xiB*(etaB-1.0);
    D[startIndex+27] = etaB*u.x*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+28] = etaB*u.y*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+29] = etaB*u.z*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+30] = -aB*etaB*u.x*xiB*pow(xiB-1.0,2.0);
    D[startIndex+31] = -aB*etaB*u.y*xiB*pow(xiB-1.0,2.0);
    D[startIndex+32] = -aB*etaB*u.z*xiB*pow(xiB-1.0,2.0);
    D[startIndex+33] = bB*(etaB*etaB)*u.x*(etaB-1.0)*(xiB-1.0);
    D[startIndex+34] = bB*(etaB*etaB)*u.y*(etaB-1.0)*(xiB-1.0);
    D[startIndex+35] = bB*(etaB*etaB)*u.z*(etaB-1.0)*(xiB-1.0);
    startIndex+=36;
  }

  // Add v, values
  if(bodyIdentifierA<numBodies) {
    D[startIndex+0] = v.x;
    D[startIndex+1] = v.y;
    D[startIndex+2] = v.z;
    startIndex+=3;
  } else if (bodyIdentifierA<numBodies+numBeams) {
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
  } else {
    D[startIndex+0 ] = v.x*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+1 ] = v.y*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+2 ] = v.z*(etaA-1.0)*(xiA-1.0)*(etaA+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0+1.0);
    D[startIndex+3 ] = -aA*v.x*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+4 ] = -aA*v.y*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+5 ] = -aA*v.z*xiA*(etaA-1.0)*pow(xiA-1.0,2.0);
    D[startIndex+6 ] = -bA*etaA*v.x*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+7 ] = -bA*etaA*v.y*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+8 ] = -bA*etaA*v.z*pow(etaA-1.0,2.0)*(xiA-1.0);
    D[startIndex+9 ] = -v.x*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+10] = -v.y*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+11] = -v.z*xiA*(etaA-1.0)*(etaA+xiA*3.0-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+12] = -aA*v.x*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+13] = -aA*v.y*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+14] = -aA*v.z*(xiA*xiA)*(etaA-1.0)*(xiA-1.0);
    D[startIndex+15] = bA*etaA*v.x*xiA*pow(etaA-1.0,2.0);
    D[startIndex+16] = bA*etaA*v.y*xiA*pow(etaA-1.0,2.0);
    D[startIndex+17] = bA*etaA*v.z*xiA*pow(etaA-1.0,2.0);
    D[startIndex+18] = -etaA*v.x*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+19] = -etaA*v.y*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+20] = -etaA*v.z*xiA*(etaA*-3.0-xiA*3.0+(etaA*etaA)*2.0+(xiA*xiA)*2.0+1.0);
    D[startIndex+21] = aA*etaA*v.x*(xiA*xiA)*(xiA-1.0);
    D[startIndex+22] = aA*etaA*v.y*(xiA*xiA)*(xiA-1.0);
    D[startIndex+23] = aA*etaA*v.z*(xiA*xiA)*(xiA-1.0);
    D[startIndex+24] = bA*(etaA*etaA)*v.x*xiA*(etaA-1.0);
    D[startIndex+25] = bA*(etaA*etaA)*v.y*xiA*(etaA-1.0);
    D[startIndex+26] = bA*(etaA*etaA)*v.z*xiA*(etaA-1.0);
    D[startIndex+27] = -etaA*v.x*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+28] = -etaA*v.y*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+29] = -etaA*v.z*(xiA-1.0)*(etaA*3.0+xiA-(etaA*etaA)*2.0-(xiA*xiA)*2.0);
    D[startIndex+30] = aA*etaA*v.x*xiA*pow(xiA-1.0,2.0);
    D[startIndex+31] = aA*etaA*v.y*xiA*pow(xiA-1.0,2.0);
    D[startIndex+32] = aA*etaA*v.z*xiA*pow(xiA-1.0,2.0);
    D[startIndex+33] = -bA*(etaA*etaA)*v.x*(etaA-1.0)*(xiA-1.0);
    D[startIndex+34] = -bA*(etaA*etaA)*v.y*(etaA-1.0)*(xiA-1.0);
    D[startIndex+35] = -bA*(etaA*etaA)*v.z*(etaA-1.0)*(xiA-1.0);
    startIndex+=36;
  }
  if(bodyIdentifierB<numBodies) {
    D[startIndex+0] = -v.x;
    D[startIndex+1] = -v.y;
    D[startIndex+2] = -v.z;
    startIndex+=3;
  } else if (bodyIdentifierB<numBodies+numBeams) {
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
  } else {
    D[startIndex+0 ] = -v.x*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+1 ] = -v.y*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+2 ] = -v.z*(etaB-1.0)*(xiB-1.0)*(etaB+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0+1.0);
    D[startIndex+3 ] = aB*v.x*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+4 ] = aB*v.y*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+5 ] = aB*v.z*xiB*(etaB-1.0)*pow(xiB-1.0,2.0);
    D[startIndex+6 ] = bB*etaB*v.x*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+7 ] = bB*etaB*v.y*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+8 ] = bB*etaB*v.z*pow(etaB-1.0,2.0)*(xiB-1.0);
    D[startIndex+9 ] = v.x*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+10] = v.y*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+11] = v.z*xiB*(etaB-1.0)*(etaB+xiB*3.0-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+12] = aB*v.x*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+13] = aB*v.y*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+14] = aB*v.z*(xiB*xiB)*(etaB-1.0)*(xiB-1.0);
    D[startIndex+15] = -bB*etaB*v.x*xiB*pow(etaB-1.0,2.0);
    D[startIndex+16] = -bB*etaB*v.y*xiB*pow(etaB-1.0,2.0);
    D[startIndex+17] = -bB*etaB*v.z*xiB*pow(etaB-1.0,2.0);
    D[startIndex+18] = etaB*v.x*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+19] = etaB*v.y*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+20] = etaB*v.z*xiB*(etaB*-3.0-xiB*3.0+(etaB*etaB)*2.0+(xiB*xiB)*2.0+1.0);
    D[startIndex+21] = -aB*etaB*v.x*(xiB*xiB)*(xiB-1.0);
    D[startIndex+22] = -aB*etaB*v.y*(xiB*xiB)*(xiB-1.0);
    D[startIndex+23] = -aB*etaB*v.z*(xiB*xiB)*(xiB-1.0);
    D[startIndex+24] = -bB*(etaB*etaB)*v.x*xiB*(etaB-1.0);
    D[startIndex+25] = -bB*(etaB*etaB)*v.y*xiB*(etaB-1.0);
    D[startIndex+26] = -bB*(etaB*etaB)*v.z*xiB*(etaB-1.0);
    D[startIndex+27] = etaB*v.x*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+28] = etaB*v.y*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+29] = etaB*v.z*(xiB-1.0)*(etaB*3.0+xiB-(etaB*etaB)*2.0-(xiB*xiB)*2.0);
    D[startIndex+30] = -aB*etaB*v.x*xiB*pow(xiB-1.0,2.0);
    D[startIndex+31] = -aB*etaB*v.y*xiB*pow(xiB-1.0,2.0);
    D[startIndex+32] = -aB*etaB*v.z*xiB*pow(xiB-1.0,2.0);
    D[startIndex+33] = bB*(etaB*etaB)*v.x*(etaB-1.0)*(xiB-1.0);
    D[startIndex+34] = bB*(etaB*etaB)*v.y*(etaB-1.0)*(xiB-1.0);
    D[startIndex+35] = bB*(etaB*etaB)*v.z*(etaB-1.0)*(xiB-1.0);
    startIndex+=36;
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
  totalNonzeros+=offsetConstraintsDOF+7*constraintsSpherical_ShellNodeToBody2D_d.size(); //Add in space for the bilateralDOF entries

  DI_d.resize(totalNonzeros);
  DJ_d.resize(totalNonzeros);
  D_d.resize(totalNonzeros);
  friction_d.resize(collisionDetector->numCollisions);

  if(constraintsBilateralDOF_d.size()) constructBilateralJacobian<<<BLOCKS(constraintsBilateralDOF_d.size()),THREADS>>>(CASTI2(constraintsBilateralDOF_d), CASTI1(offsetBilaterals_d), CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), constraintsBilateralDOF_d.size());
  if(constraintsSpherical_ShellNodeToBody2D_d.size()) constructSpherical_ShellNodeToBody2DJacobian<<<BLOCKS(constraintsSpherical_ShellNodeToBody2D_d.size()),THREADS>>>(CASTI3(constraintsSpherical_ShellNodeToBody2D_d), CASTD3(pSpherical_ShellNodeToBody2D_d), CASTD1(p_d), CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), constraintsBilateralDOF_d.size(), offsetConstraintsDOF, constraintsSpherical_ShellNodeToBody2D_d.size());
  if(collisionDetector->numCollisions) constructContactJacobian<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTI1(nonzerosPerContact_d), CASTI4(collisionMap_d), CASTI4(shellConnectivities_d), CASTD3(contactGeometry_d), CASTD3(collisionGeometry_d), CASTI1(DI_d), CASTI1(DJ_d), CASTD1(D_d), CASTD1(friction_d), CASTD4(collisionDetector->normalsAndPenetrations_d), CASTU1(collisionDetector->collisionIdentifierA_d), CASTU1(collisionDetector->collisionIdentifierB_d), CASTI1(indices_d), bodies.size(), beams.size(), plates.size(), body2Ds.size(), offsetConstraintsDOF+7*constraintsSpherical_ShellNodeToBody2D_d.size(), constraintsBilateralDOF_d.size()+3*constraintsSpherical_ShellNodeToBody2D_d.size(), collisionDetector->numCollisions);

  // create contact jacobian using cusp library
  thrust::device_ptr<int> wrapped_device_I(CASTI1(DI_d));
  DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + DI_d.size());

  thrust::device_ptr<int> wrapped_device_J(CASTI1(DJ_d));
  DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + DJ_d.size());

  thrust::device_ptr<double> wrapped_device_V(CASTD1(D_d));
  DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + D_d.size());

  D = DeviceView(3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size()+3*constraintsSpherical_ShellNodeToBody2D_d.size(), 3*bodies.size()+12*beams.size()+36*plates.size()+3*body2Ds.size()+3*nodes_h.size(), D_d.size(), row_indices, column_indices, values);
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

  DT = DeviceView(3*bodies.size()+12*beams.size()+36*plates.size()+3*body2Ds.size()+3*nodes_h.size(), 3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size()+3*constraintsSpherical_ShellNodeToBody2D_d.size(), DT_d.size(), row_indices, column_indices, values);
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

  uint offset = 3*numBodies+12*numBeams+36*index;
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

__global__ void multiplyByBody2DMass(double2* materials, double* src, double* dst, int numBodies, int numBeams, int numPlates, int numBody2Ds) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numBody2Ds);

  double mass = materials[index].x;
  double inertia = materials[index].y;

  uint offset = 3*numBodies+12*numBeams+36*numPlates+3*index;
  dst[offset+0] = mass*src[offset+0];
  dst[offset+1] = mass*src[offset+1];
  dst[offset+2] = inertia*src[offset+2];
}

int System::buildAppliedImpulseVector() {
  // build k
  updateElasticForces();
  if(bodies.size()) multiplyByMass<<<BLOCKS(3*bodies.size()),THREADS>>>(CASTD1(mass_d), CASTD1(v_d), CASTD1(k_d), 3*bodies.size());
  if(beams.size()) multiplyByBeamMass<<<BLOCKS(beams.size()),THREADS>>>(CASTD3(contactGeometry_d), CASTD3(materialsBeam_d), CASTD1(v_d), CASTD1(k_d), bodies.size(), beams.size());
  if(plates.size()) multiplyByPlateMass<<<BLOCKS(plates.size()),THREADS>>>(CASTD3(contactGeometry_d), CASTD4(materialsPlate_d), CASTD1(v_d), CASTD1(k_d), bodies.size(), beams.size(), plates.size());
  if(body2Ds.size()) multiplyByBody2DMass<<<BLOCKS(body2Ds.size()),THREADS>>>(CASTD2(materialsBody2D_d), CASTD1(v_d), CASTD1(k_d), bodies.size(), beams.size(), plates.size(), body2Ds.size());
  if(nodes_h.size()) cusp::multiply(mass_shellMesh,v_shellMesh,k_shellMesh);
  //cusp::blas::axpy(fElastic,fApplied,-1.0); //TODO: Come up with a fix for applied forces
  cusp::blas::axpbypcz(f,fElastic,k,k,h,-h,1.0);

  return 0;
}

__global__ void buildStabilization(double* b, double4* normalsAndPenetrations, double timeStep, uint offsetBilateralConstraints, uint numCollisions) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numCollisions);

  double penetration = normalsAndPenetrations[index].w;

  b[3*index+offsetBilateralConstraints] = penetration/timeStep;
  b[3*index+1+offsetBilateralConstraints] = 0;
  b[3*index+2+offsetBilateralConstraints] = 0;
}

__global__ void buildStabilizationBilateral(double* b, double3* infoConstraintBilateralDOF, int2* constraintBilateralDOF, double* p, double timeStep, double time, uint numBilateralConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numBilateralConstraints);

  int2 constraintDOF = constraintBilateralDOF[index];
  double3 info = infoConstraintBilateralDOF[index];
  double tStart = info.y;
  double velocity = info.x;
  if(time<tStart) velocity = 0;
  double p0 = info.z;

  double violation = 0;
  if(constraintDOF.y<0) {
    violation = p[constraintDOF.x]-p0-velocity*(time-tStart);
  } else {
    violation = p[constraintDOF.x]-p[constraintDOF.y]-velocity*(time-tStart);
  }

  b[index] = violation/timeStep;
}

__global__ void buildStabilizationSpherical_ShellNodeToBody2D(double* b, int3* constraints, double3* pHats, double* p, double timeStep, uint numDOFConstraints, int numConstraints) {
  INIT_CHECK_THREAD_BOUNDED(INDEX1D, numConstraints);

  int offset = numDOFConstraints;
  int3 constraint = constraints[index];
  double3 pHat = pHats[index];
  int indexS = constraint.x;
  int indexB = constraint.y;

  double phi = p[indexB+2];
  b[3*index+offset+0] = (p[indexB]+pHat.x*cos(phi)-pHat.y*sin(phi)-p[indexS])/timeStep;
  b[3*index+offset+1] = (p[indexB+1]+pHat.x*sin(phi)+pHat.y*cos(phi)-p[indexS+1])/timeStep;
  b[3*index+offset+2] = (pHat.z-p[indexS+2])/timeStep;
}

int System::buildSchurVector() {
  // build r
  r_d.resize(3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size()+3*constraintsSpherical_ShellNodeToBody2D_d.size());
  b_d.resize(3*collisionDetector->numCollisions+constraintsBilateralDOF_d.size()+3*constraintsSpherical_ShellNodeToBody2D_d.size());
  // TODO: There's got to be a better way to do this...
  //r.resize(3*collisionDetector->numCollisions);
  thrust::device_ptr<double> wrapped_device_r(CASTD1(r_d));
  r = DeviceValueArrayView(wrapped_device_r, wrapped_device_r + r_d.size());
  thrust::device_ptr<double> wrapped_device_b(CASTD1(b_d));
  b = DeviceValueArrayView(wrapped_device_b, wrapped_device_b + b_d.size());
  cusp::multiply(mass,k,tmp);
  cusp::multiply(D,tmp,r);

  if(constraintsBilateralDOF_d.size()) buildStabilizationBilateral<<<BLOCKS(constraintsBilateralDOF_d.size()),THREADS>>>(CASTD1(b_d), CASTD3(infoConstraintBilateralDOF_d), CASTI2(constraintsBilateralDOF_d), CASTD1(p_d), h, time, constraintsBilateralDOF_d.size());
  if(constraintsSpherical_ShellNodeToBody2D_d.size()) buildStabilizationSpherical_ShellNodeToBody2D<<<BLOCKS(constraintsSpherical_ShellNodeToBody2D_d.size()),THREADS>>>(CASTD1(b_d), CASTI3(constraintsSpherical_ShellNodeToBody2D_d), CASTD3(pSpherical_ShellNodeToBody2D_d), CASTD1(p_d), h, constraintsBilateralDOF_d.size(), constraintsSpherical_ShellNodeToBody2D_d.size());
  if(collisionDetector->numCollisions) buildStabilization<<<BLOCKS(collisionDetector->numCollisions),THREADS>>>(CASTD1(b_d), CASTD4(collisionDetector->normalsAndPenetrations_d), h, constraintsBilateralDOF_d.size()+3*constraintsSpherical_ShellNodeToBody2D_d.size(), collisionDetector->numCollisions);
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

double System::getPotentialEnergy() {
  return -cusp::blas::dot(f,p);
}

double System::getKineticEnergy() {
  if(bodies.size()) multiplyByMass<<<BLOCKS(3*bodies.size()),THREADS>>>(CASTD1(mass_d), CASTD1(v_d), CASTD1(tmp_d), 3*bodies.size());
  if(beams.size()) multiplyByBeamMass<<<BLOCKS(beams.size()),THREADS>>>(CASTD3(contactGeometry_d), CASTD3(materialsBeam_d), CASTD1(v_d), CASTD1(tmp_d), bodies.size(), beams.size());
  if(plates.size()) multiplyByPlateMass<<<BLOCKS(plates.size()),THREADS>>>(CASTD3(contactGeometry_d), CASTD4(materialsPlate_d), CASTD1(v_d), CASTD1(tmp_d), bodies.size(), beams.size(), plates.size());
  if(body2Ds.size()) multiplyByBody2DMass<<<BLOCKS(body2Ds.size()),THREADS>>>(CASTD2(materialsBody2D_d), CASTD1(v_d), CASTD1(tmp_d), bodies.size(), beams.size(), plates.size(), body2Ds.size());
  if(nodes_h.size()) cusp::multiply(mass_shellMesh,v_shellMesh,tmp_shellMesh);

  return 0.5*cusp::blas::dot(v,tmp);
}

double System::getStrainEnergy() {
  double strainEnergy = 0;
  if(beams.size()) strainEnergy+=thrust::reduce(strainEnergy_d.begin(),strainEnergy_d.end());
  if(plates.size()) strainEnergy+=thrust::reduce(strainEnergyPlate_d.begin(),strainEnergyPlate_d.end());
  if(nodes_h.size()) strainEnergy+=thrust::reduce(strainEnergyShellMesh_d.begin(),strainEnergyShellMesh_d.end());

  return strainEnergy;
}

double System::getTotalEnergy() {
  return getPotentialEnergy()+getKineticEnergy()+getStrainEnergy();
}

int System::exportSystem(string filename) {
  ofstream filestream;
  filestream.open(filename.c_str());

  p_h = p_d;
  v_h = v_d;
  filestream << bodies.size() << ", " << beams.size() << ", " << plates.size()+shellConnectivities_h.size() << ", " << body2Ds.size() << ", " << endl;
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
  for (int i = 0; i < plates.size(); i++) {
    // TODO: Need to know collision family information, density, elastic modulus, number of contacts (especially important when importing)
    filestream
    << bodies.size()+beams.size()+i << ", "
    << contactGeometry_h[bodies.size()+beams.size()+i].x << ", "
    << contactGeometry_h[bodies.size()+beams.size()+i].y << ", "
    << plates[i]->getThickness() << ", ";

    for(int j=0;j<36;j++) {
      filestream << p_h[3*bodies.size()+12*beams.size()+36*i+j] << ", ";
    }

    for(int j=0;j<36;j++) {
      filestream << v_h[3*bodies.size()+12*beams.size()+36*i+j] << ", ";
    }

    filestream << "\n";
  }
  for (int i = 0; i < shellConnectivities_h.size(); i++) {
    filestream
    << bodies.size()+beams.size()+plates.size()+body2Ds.size()+i << ", "
    << shellGeometries_h[i].x << ", "
    << shellGeometries_h[i].y << ", "
    << shellGeometries_h[i].z << ", ";

    int offset = plates.size()*36+12*beams.size()+3*bodies.size()+3*body2Ds.size();
    double* p0 = &p_h[offset+9*shellConnectivities_h[i].x];
    double* p1 = &p_h[offset+9*shellConnectivities_h[i].y];
    double* p2 = &p_h[offset+9*shellConnectivities_h[i].z];
    double* p3 = &p_h[offset+9*shellConnectivities_h[i].w];

    double* v0 = &v_h[offset+9*shellConnectivities_h[i].x];
    double* v1 = &v_h[offset+9*shellConnectivities_h[i].y];
    double* v2 = &v_h[offset+9*shellConnectivities_h[i].z];
    double* v3 = &v_h[offset+9*shellConnectivities_h[i].w];

    for(int j=0;j<9;j++) filestream << p0[j] << ", ";
    for(int j=0;j<9;j++) filestream << p1[j] << ", ";
    for(int j=0;j<9;j++) filestream << p2[j] << ", ";
    for(int j=0;j<9;j++) filestream << p3[j] << ", ";

    for(int j=0;j<9;j++) filestream << v0[j] << ", ";
    for(int j=0;j<9;j++) filestream << v1[j] << ", ";
    for(int j=0;j<9;j++) filestream << v2[j] << ", ";
    for(int j=0;j<9;j++) filestream << v3[j] << ", ";

    filestream << "\n";
  }
  for (int i = 0; i < body2Ds.size(); i++) {
    // TODO: Need to know collision family information, density, elastic modulus, number of contacts (especially important when importing)
    filestream << bodies.size()+beams.size()+plates.size()+i << ", ";

    for(int j=0;j<3;j++) {
      filestream << p_h[3*bodies.size()+12*beams.size()+36*plates.size()+3*i+j] << ", ";
    }

    for(int j=0;j<3;j++) {
      filestream << v_h[3*bodies.size()+12*beams.size()+36*plates.size()+3*i+j] << ", ";
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

void System::importMesh(string filename, double stiffness, int numContactPointsPerElement) {
  string temp_data;
  int numShells;
  int numNodes;
  int numNonzeros_M;
  int numNonzeros_invM;
  double3 node;
  int4 connectivity;
  double4 material;
  double4 geometry;
  int map;
  double force;
  int iVal;
  int jVal;
  double val;

  ifstream ifile(filename.c_str());
  getline(ifile,temp_data);
  for(int i=0; i<temp_data.size(); ++i){
    if(temp_data[i]==','){temp_data[i]=' ';}
  }
  stringstream ss1(temp_data);
  ss1>>numNodes>>numShells>>numNonzeros_M>>numNonzeros_invM;

  // read nodes
  for(int i=0; i<3*numNodes; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>node.x>>node.y>>node.z;
    nodes_h.push_back(node);
  }

  // read shell connectivity
  for(int i=0; i<numShells; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>connectivity.x>>connectivity.y>>connectivity.z>>connectivity.w;
    shellConnectivities_h.push_back(connectivity);
  }
  shellConnectivities_d = shellConnectivities_h;

  // read shell materials
  for(int i=0; i<numShells; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>material.x>>material.y>>material.z>>material.w;
    material.y = stiffness;
    shellMaterials_h.push_back(material);
  }
  shellMaterials_d = shellMaterials_h;

  // read shell geometries
  for(int i=0; i<numShells; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>geometry.x>>geometry.y>>geometry.z>>geometry.w;
    geometry.w = numContactPointsPerElement;
    shellGeometries_h.push_back(geometry);
  }
  shellGeometries_d = shellGeometries_h;
  //cout << endl;

  // read shell map
  for(int i=0; i<numShells*36; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>map;
    shellMap_h.push_back(map);
  }
  shellMap_d = shellMap_h;
  shellMap0_d = shellMap_h;

  // read shell external force
  for(int i=0; i<numNodes*9; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>force;
    fextMesh_h.push_back(force);
  }

  // read shell mass matrix
  for(int i=0; i<numNonzeros_M; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>iVal>>jVal>>val;
    massShellI_h.push_back(iVal-1); // convert from 1-based indexing
    massShellJ_h.push_back(jVal-1); // convert from 1-based indexing
    massShell_h.push_back(val);
  }
  massShellI_d = massShellI_h;
  massShellJ_d = massShellJ_h;
  massShell_d = massShell_h;

  // read shell inverse mass matrix
  for(int i=0; i<numNonzeros_invM; i++) {
    getline(ifile,temp_data);
    for(int j=0; j<temp_data.size(); ++j){
      if(temp_data[j]==','){temp_data[j]=' ';}
    }
    stringstream ss(temp_data);
    ss>>iVal>>jVal>>val;
    invMassShellI_h.push_back(iVal-1); // convert from 1-based indexing
    invMassShellJ_h.push_back(jVal-1); // convert from 1-based indexing
    invMassShell_h.push_back(val);
  }
}

double3 System::transformNodalToCartesian_shellMesh(int shellIndex, double xi, double eta)
{
  double a = shellGeometries_h[shellIndex].x;
  double b = shellGeometries_h[shellIndex].y;

  int offset = plates.size()*36+12*beams.size()+3*bodies.size()+3*body2Ds.size();
  double* p0 = &p_h[offset+9*shellConnectivities_h[shellIndex].x];
  double* p1 = &p_h[offset+9*shellConnectivities_h[shellIndex].y];
  double* p2 = &p_h[offset+9*shellConnectivities_h[shellIndex].z];
  double* p3 = &p_h[offset+9*shellConnectivities_h[shellIndex].w];

  double3 pos;
  pos.x = -eta*p2[0]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p1[0]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p3[0]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p0[0]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p1[6]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p2[6]*xi*(eta-1.0)+a*eta*p2[3]*(xi*xi)*(xi-1.0)+a*eta*p3[3]*xi*pow(xi-1.0,2.0)-b*eta*p0[6]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p3[6]*(eta-1.0)*(xi-1.0)-a*p0[3]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p1[3]*(xi*xi)*(eta-1.0)*(xi-1.0);
  pos.y = -eta*p2[1]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p1[1]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p3[1]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p0[1]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p1[7]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p2[7]*xi*(eta-1.0)+a*eta*p2[4]*(xi*xi)*(xi-1.0)+a*eta*p3[4]*xi*pow(xi-1.0,2.0)-b*eta*p0[7]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p3[7]*(eta-1.0)*(xi-1.0)-a*p0[4]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p1[4]*(xi*xi)*(eta-1.0)*(xi-1.0);
  pos.z = -eta*p2[2]*xi*(eta*-3.0-xi*3.0+(eta*eta)*2.0+(xi*xi)*2.0+1.0)-p1[2]*xi*(eta-1.0)*(eta+xi*3.0-(eta*eta)*2.0-(xi*xi)*2.0)-eta*p3[2]*(xi-1.0)*(eta*3.0+xi-(eta*eta)*2.0-(xi*xi)*2.0)+p0[2]*(eta-1.0)*(xi-1.0)*(eta+xi-(eta*eta)*2.0-(xi*xi)*2.0+1.0)+b*eta*p1[8]*xi*pow(eta-1.0,2.0)+b*(eta*eta)*p2[8]*xi*(eta-1.0)+a*eta*p2[5]*(xi*xi)*(xi-1.0)+a*eta*p3[5]*xi*pow(xi-1.0,2.0)-b*eta*p0[8]*pow(eta-1.0,2.0)*(xi-1.0)-b*(eta*eta)*p3[8]*(eta-1.0)*(xi-1.0)-a*p0[5]*xi*(eta-1.0)*pow(xi-1.0,2.0)-a*p1[5]*(xi*xi)*(eta-1.0)*(xi-1.0);

  return pos;
}
