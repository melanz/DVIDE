#include "include.cuh"
#include <sys/stat.h>
#include <errno.h>
#include "System.cuh"
#include "Body.cuh"
#include "Beam.cuh"
#include "Plate.cuh"
#include "PDIP.cuh"
#include "TPAS.cuh"
#include "JKIP.cuh"

bool updateDraw = 1;
bool wireFrame = 1;
double desiredVelocity = 0.5;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;
std::string outDir = "../TEST_BIKEWHEEL/";
std::string povrayDir = outDir + "POVRAY/";
thrust::host_vector<double> p0_h;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(1,0,1),camreal3(0,0,0),camreal3(0,1,0),.01);

// OPENGL RENDERING CODE //
void changeSize(int w, int h) {
  if(h == 0) {h = 1;}
  float ratio = 1.0* w / h;
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glViewport(0, 0, w, h);
  gluPerspective(45,ratio,.1,1000);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  gluLookAt(0.0,0.0,0.0,    0.0,0.0,-7,   0.0f,1.0f,0.0f);
}

void initScene(){
  GLfloat light_position[] = { 1.0, 1.0, 1.0, 0.0 };
  glClearColor (1.0, 1.0, 1.0, 0.0);
  glShadeModel (GL_SMOOTH);
  glEnable(GL_COLOR_MATERIAL);
  glLightfv(GL_LIGHT0, GL_POSITION, light_position);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);
  glEnable (GL_POINT_SMOOTH);
  glEnable (GL_BLEND);
  glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glHint (GL_POINT_SMOOTH_HINT, GL_DONT_CARE);
}

void drawAll()
{
  if(updateDraw){
    sys->p_h = sys->p_d;
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glFrontFace(GL_CCW);
    glCullFace(GL_BACK);
    glEnable(GL_CULL_FACE);
    glDepthFunc(GL_LEQUAL);
    glClearDepth(1.0);

    glPointSize(2);
    glLoadIdentity();

    oglcamera.Update();

    for(int i=0;i<sys->bodies.size();i++)
    {
      if(wireFrame) {
        glPushMatrix();
        double3 position = sys->bodies[i]->getPosition();
        glTranslatef(sys->p_h[3*i],sys->p_h[3*i+1],sys->p_h[3*i+2]);
        double3 geometry = sys->bodies[i]->getGeometry();
        if(geometry.y) {
          glColor3f(0.0f,1.0f,0.0f);
          glScalef(2*geometry.x, 2*geometry.y, 2*geometry.z);
          glutWireCube(1.0);
        }
        else {
          glColor3f(0.0f,0.0f,1.0f);
          glutWireSphere(geometry.x,30,30);
        }
        glPopMatrix();
      }
      else {
        glPushMatrix();
        double3 position = sys->bodies[i]->getPosition();
        glTranslatef(sys->p_h[3*i],sys->p_h[3*i+1],sys->p_h[3*i+2]);
        double3 geometry = sys->bodies[i]->getGeometry();
        if(geometry.y) {
          glColor3f(0.0f,1.0f,0.0f);
          glScalef(2*geometry.x, 2*geometry.y, 2*geometry.z);
          glutSolidCube(1.0);
        }
        else {
          glColor3f(0.0f,0.0f,1.0f);
          glutSolidSphere(geometry.x,30,30);
        }
        glPopMatrix();
      }
    }

    for(int i=0;i<sys->beams.size();i++)
    {
      int xiDiv = sys->beams[i]->getGeometry().z;
      double xiInc = 1/(static_cast<double>(xiDiv-1));
      glColor3f(1.0f,0.0f,0.0f);
      for(int j=0;j<xiDiv;j++)
      {
        glPushMatrix();
        double3 position = sys->beams[i]->transformNodalToCartesian(xiInc*j);
        glTranslatef(position.x,position.y,position.z);
        glutSolidSphere(sys->beams[i]->getGeometry().x,10,10);
        glPopMatrix();
      }
    }

    for(int i=0;i<sys->plates.size();i++)
    {
      int xiDiv = sys->plates[i]->getGeometry().z;
      int etaDiv = sys->plates[i]->getGeometry().z;
      double xiInc = 1/(static_cast<double>(xiDiv-1));
      double etaInc = 1/(static_cast<double>(etaDiv-1));
      glColor3f(1.0f,0.0f,1.0f);
      for(int j=0;j<xiDiv;j++)
      {
        for(int k=0;k<etaDiv;k++) {
          glPushMatrix();
          double3 position = sys->plates[i]->transformNodalToCartesian(xiInc*j,etaInc*k);
          glTranslatef(position.x,position.y,position.z);
          glutSolidSphere(sys->plates[i]->getThickness(),10,10);
          glPopMatrix();
        }
      }
    }

    glutSwapBuffers();
  }
}

void renderSceneAll(){
  if(OGL){
    drawAll();
    p0_h = sys->p_d;
    sys->DoTimeStep();

    //    // TODO: This is a big no-no, need to enforce motion via constraints
    //    // Apply motion
    //    sys->v_h = sys->v_d;
    //    for(int i=0;i<1;i++) {
    //      sys->v_h[36*i] = 0;
    //      sys->v_h[36*i+1] = 0;
    //      sys->v_h[36*i+2] = 0;
    //    }
    //
    //    sys->p_d = p0_h;
    //    sys->v_d = sys->v_h;
    //    cusp::blas::axpy(sys->v, sys->p, sys->h);
    //    sys->p_h = sys->p_d;
    //    // End apply motion
  }
}

void CallBackKeyboardFunc(unsigned char key, int x, int y) {
  switch (key) {
  case 'w':
    oglcamera.Forward();
    break;

  case 's':
    oglcamera.Back();
    break;

  case 'd':
    oglcamera.Right();
    break;

  case 'a':
    oglcamera.Left();
    break;

  case 'q':
    oglcamera.Up();
    break;

  case 'e':
    oglcamera.Down();
    break;

  case 'i':
    if(wireFrame) {
      wireFrame = 0;
    }
    else {
      wireFrame = 1;
    }
  }
}

void CallBackMouseFunc(int button, int state, int x, int y) {
  oglcamera.SetPos(button, state, x, y);
}
void CallBackMotionFunc(int x, int y) {
  oglcamera.Move2D(x, y);
}
#endif
// END OPENGL RENDERING CODE //

double getRandomNumber(double min, double max)
{
  // x is in [0,1[
  double x = rand()/static_cast<double>(RAND_MAX);

  // [0,1[ * (max - min) + min is in [min,max[
  double that = min + ( x * (max - min) );

  return that;
}

int main(int argc, char** argv)
{
  // command line arguments
  // FlexibleNet <numPartitions> <numBeamsPerSide> <solverType> <usePreconditioning>
  // solverType: (0) BiCGStab, (1) BiCGStab1, (2) BiCGStab2, (3) MinRes, (4) CG, (5) CR

  double t_end = 10.0;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int precondType = 1;
  int numElementsPerSide = 4;
  int solverType = 4;
  int numPartitions = 1;
  double mu_pdip = 10.0;
  double alpha = 0.01; // should be [0.01, 0.1]
  double beta = 0.8; // should be [0.3, 0.8]
  int solverTypeQOCC = 1;
  int binsPerAxis = 20;
  double tolerance = 1.0;//e-2;
  double hh = 1e-4;

  if(argc > 1) {
    numElementsPerSide = atoi(argv[1]);
    solverTypeQOCC = atoi(argv[2]);
    tolerance = atof(argv[3]);
    hh = atof(argv[4]);
  }

#ifdef WITH_GLUT
  bool visualize = true;
#endif
  visualize = false;

  sys = new System(solverTypeQOCC);
  sys->setTimeStep(hh);
  sys->solver->tolerance = tolerance;

  // Create output directories
  std::stringstream outDirStream;
  outDirStream << "../TEST_BIKEWHEEL_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << "/";
  outDir = outDirStream.str();
  povrayDir = outDir + "POVRAY/";
  if(mkdir(outDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
  {
    if(errno != EEXIST)
    {
      printf("Error creating directory!n");
      exit(1);
    }
  }
  if(mkdir(povrayDir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1)
  {
    if(errno != EEXIST)
    {
      printf("Error creating directory!n");
      exit(1);
    }
  }

  sys->collisionDetector->setBinsPerAxis(make_uint3(binsPerAxis,binsPerAxis,binsPerAxis));
  if(solverTypeQOCC==2) {
    dynamic_cast<PDIP*>(sys->solver)->setPrecondType(precondType);
    dynamic_cast<PDIP*>(sys->solver)->setSolverType(solverType);
    dynamic_cast<PDIP*>(sys->solver)->setNumPartitions(numPartitions);
    dynamic_cast<PDIP*>(sys->solver)->alpha = alpha;
    dynamic_cast<PDIP*>(sys->solver)->beta = beta;
    dynamic_cast<PDIP*>(sys->solver)->mu_pdip = mu_pdip;
  }
  if(solverTypeQOCC==3) {
    dynamic_cast<TPAS*>(sys->solver)->setPrecondType(precondType);
    dynamic_cast<TPAS*>(sys->solver)->setSolverType(solverType);
    dynamic_cast<TPAS*>(sys->solver)->setNumPartitions(numPartitions);
  }
  if(solverTypeQOCC==4) {
    dynamic_cast<JKIP*>(sys->solver)->setPrecondType(precondType);
    dynamic_cast<JKIP*>(sys->solver)->setSolverType(solverType);
    dynamic_cast<JKIP*>(sys->solver)->setNumPartitions(numPartitions);
    dynamic_cast<JKIP*>(sys->solver)->careful = true;
  }

  //sys->solver->maxIterations = 40;
  //sys->gravity = make_double3(0,0,0);

  int numDiv = 7;
  double radianInc = 2.0*PI/((double) numDiv);
  double EM = 7.e7;
  double rho = 7810.0;
  double th = .01;
  double R = .2;
  double nu = .1;
  double fillet = .04;
  double beltWidth = .1;
  double B = .5*PI*beltWidth;//1.5*.5*PI*beltWidth;
  double L = 2.0*PI*(R+0.5*beltWidth)/((double) numDiv);//2*PI*(R+1.4*0.33*beltWidth)/((double) numDiv);
  int numContacts = 12;
  double depth = .2;
  double ditchLength = 2;
  double ditchWidth = 4*beltWidth;

  double r_rim = 0.011;
  double EM_rim = 2.e9;
  double rho_rim = 7200.0;
  double L_rim = 2.0*PI*R/((double) numDiv);//2*PI*(R+1.4*0.33*beltWidth)/((double) numDiv);

  // Create the nodes
  vector<double3> nodes;
  vector<double3> dxis;
  vector<double3> detas;
  for(int i=0;i<numDiv;i++)
  {
    nodes.push_back(make_double3(R*cos(radianInc*i),R*sin(radianInc*i),0));
    dxis.push_back(make_double3(-sin(radianInc*i),cos(radianInc*i),0));
    detas.push_back(make_double3(cos(radianInc*i),sin(radianInc*i),0));

    nodes.push_back(make_double3(R*cos(radianInc*i),R*sin(radianInc*i),beltWidth));
    dxis.push_back(make_double3(-sin(radianInc*i),cos(radianInc*i),0));
    detas.push_back(make_double3(-cos(radianInc*i),-sin(radianInc*i),0));
  }
  int i = 0;
  nodes.push_back(make_double3(R*cos(radianInc*i),R*sin(radianInc*i),0));
  dxis.push_back(make_double3(-sin(radianInc*i),cos(radianInc*i),0));
  detas.push_back(make_double3(cos(radianInc*i),sin(radianInc*i),0));

  nodes.push_back(make_double3(R*cos(radianInc*i),R*sin(radianInc*i),beltWidth));
  dxis.push_back(make_double3(-sin(radianInc*i),cos(radianInc*i),0));
  detas.push_back(make_double3(-cos(radianInc*i),-sin(radianInc*i),0));

  double3 center = make_double3(0,0,0);
  double3 center_back = make_double3(0,0,beltWidth);
  double3 detaStraight = make_double3(0,0,1);

  // Add tire elements
  Plate* plate;
  Beam* beam;
  for(int i=0;i<numDiv;i++)
  {
    plate = new Plate(L,B,nodes[2*i],dxis[2*i],detas[2*i],
        nodes[2*i+2],dxis[2*i+2],detas[2*i+2],
        nodes[2*i+3],dxis[2*i+3],detas[2*i+3],
        nodes[2*i+1],dxis[2*i+1],detas[2*i+1]);
    plate->setThickness(th);
    plate->setElasticModulus(EM);
    plate->setPoissonRatio(nu);
    plate->setDensity(rho);
    plate->setCollisionFamily(1);
    plate->setNumContactPoints(numContacts);
    sys->add(plate);

    beam = new Beam(center, detas[2*i],
        nodes[2*i],detas[2*i],R);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    sys->add(beam);

    beam = new Beam(nodes[2*i+1], detas[2*i+1],
        center_back,detas[2*i+1],R);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    sys->add(beam);

    beam = new Beam(nodes[2*i],dxis[2*i],
        nodes[2*i+2],dxis[2*i+2],L_rim);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    sys->add(beam);

    beam = new Beam(nodes[2*i+1],dxis[2*i+1],
        nodes[2*i+3],dxis[2*i+3],L_rim);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    sys->add(beam);

  }

  // Add hub
  Body* hub = new Body(center);
  //hub->setBodyFixed(true);
  hub->setGeometry(make_double3(2*r_rim,0,0));
  hub->setCollisionFamily(1);
  sys->add(hub);

  // Add hub
  Body* hub_back = new Body(center_back);
  //hub_back->setBodyFixed(true);
  hub_back->setGeometry(make_double3(2*r_rim,0,0));
  hub_back->setCollisionFamily(1);
  sys->add(hub_back);

  // Add ground
  Body* groundPtr = new Body(make_double3(0,-R-beltWidth-0.5*depth,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setCollisionFamily(2);
  groundPtr->setGeometry(make_double3(2*R,0.5*depth,0.5*ditchWidth));
  sys->add(groundPtr);

  // Add track
  Body* track = new Body(make_double3(0,-R-beltWidth-0.5*depth,beltWidth));
  track->setBodyFixed(true);
  track->setCollisionFamily(2);
  track->setGeometry(make_double3(th,0,0));
  sys->add(track);

  // Add ground
  Body* groundPtr2 = new Body(make_double3(4*R+ditchLength,-R-beltWidth-0.5*depth,0));
  groundPtr2->setBodyFixed(true);
  groundPtr2->setCollisionFamily(2);
  groundPtr2->setGeometry(make_double3(2*R,0.5*depth,0.5*ditchWidth));
  sys->add(groundPtr2);

  // Add ground
  Body* groundPtr3 = new Body(make_double3(2*R+0.5*ditchLength,-R-beltWidth-depth-th,0));
  groundPtr3->setBodyFixed(true);
  groundPtr3->setCollisionFamily(2);
  groundPtr3->setGeometry(make_double3(4*R+0.5*ditchLength,th,0.5*ditchWidth));
  sys->add(groundPtr3);

  // Add sides
  Body* right = new Body(make_double3(2*R+0.5*ditchLength,-R-beltWidth-0.5*depth,0.5*ditchWidth+th));
  right->setBodyFixed(true);
  right->setCollisionFamily(2);
  right->setGeometry(make_double3(4*R+0.5*ditchLength,depth+th,th));
  sys->add(right);

  // Add sides
  Body* left = new Body(make_double3(2*R+0.5*ditchLength,-R-beltWidth-0.5*depth,-0.5*ditchWidth-th));
  left->setBodyFixed(true);
  left->setCollisionFamily(2);
  left->setGeometry(make_double3(4*R+0.5*ditchLength,depth+th,th));
  sys->add(left);

  double rMin = 0.007;
  double rMax = 0.007;
  double density = 2600;
  double W = ditchWidth;
  double L_G = ditchLength;
  double H = 1.5*depth;
  double3 centerG = make_double3(2*R+0.5*ditchLength,-R-beltWidth-depth,0);
  Body* bodyPtr;
  double wiggle = 0.003;//0.003;//0.1;
  double numElementsPerSideX = L_G/(2.0*rMax+2.0*wiggle);
  double numElementsPerSideY = H/(2.0*rMax+2.0*wiggle);
  double numElementsPerSideZ = W/(2.0*rMax+2.0*wiggle);
  int numBodies = 0;
  // Add elements in x-direction
  for (int i = 0; i < (int) numElementsPerSideX; i++) {
    for (int j = 0; j < (int) numElementsPerSideY; j++) {
      for (int k = 0; k < (int) numElementsPerSideZ; k++) {

        double xWig = 0.8*getRandomNumber(-wiggle, wiggle);
        double yWig = 0.8*getRandomNumber(-wiggle, wiggle);
        double zWig = 0.8*getRandomNumber(-wiggle, wiggle);
        bodyPtr = new Body(centerG+make_double3((rMax+wiggle)*(2.0*((double)i)+1.0)-0.5*L_G+xWig,(rMax+wiggle)*(2.0*((double)j)+1.0)+yWig,(rMax+wiggle)*(2.0*((double)k)+1.0)-0.5*W+zWig));
        double rRand = getRandomNumber(rMin, rMax);
        bodyPtr->setMass(4.0*rRand*rRand*rRand*3.1415/3.0*density);
        bodyPtr->setGeometry(make_double3(rRand,0,0));
        //if(j==0)
        //bodyPtr->setBodyFixed(true);
        numBodies = sys->add(bodyPtr);

        if(numBodies%1000==0) printf("Bodies %d\n",numBodies);
      }
    }
  }

  // Add bilateral constraints
  for(int i=0;i<numDiv;i++)
  {
    int iNext = i+1;
    if(i==numDiv-1) iNext = 0;
    int numPlates = 1;
    int offsetA = 3*sys->bodies.size()+12*sys->beams.size()+36*(numPlates*i);
    int offsetB = 3*sys->bodies.size()+12*sys->beams.size()+36*(numPlates*iNext);
    int offsetFlatA = 3*sys->bodies.size()+12*sys->beams.size()+36*(numPlates*i+1);
    int offsetFlatB = 3*sys->bodies.size()+12*sys->beams.size()+36*(numPlates*iNext+1);

    int offsetBeamBackA = 3*sys->bodies.size()+12*(4*i);
    int offsetBeamFrontA = 3*sys->bodies.size()+12*(4*i+1);
    int offsetCurvBeamBackA = 3*sys->bodies.size()+12*(4*i+2);
    int offsetCurvBeamFrontA = 3*sys->bodies.size()+12*(4*i+3);

    int offsetBeamBackB = 3*sys->bodies.size()+12*(4*iNext);
    int offsetBeamFrontB = 3*sys->bodies.size()+12*(4*iNext+1);
    int offsetCurvBeamBackB = 3*sys->bodies.size()+12*(4*iNext+2);
    int offsetCurvBeamFrontB = 3*sys->bodies.size()+12*(4*iNext+3);

    // node 1 of plate A is fixed to node 0 of plate B
    sys->addBilateralConstraintDOF(offsetA+9*1, offsetB+9*0);
    sys->addBilateralConstraintDOF(offsetA+9*1+1, offsetB+9*0+1);
    sys->addBilateralConstraintDOF(offsetA+9*1+2, offsetB+9*0+2);
    sys->addBilateralConstraintDOF(offsetA+9*1+3, offsetB+9*0+3);
    sys->addBilateralConstraintDOF(offsetA+9*1+4, offsetB+9*0+4);
    sys->addBilateralConstraintDOF(offsetA+9*1+5, offsetB+9*0+5);
    sys->addBilateralConstraintDOF(offsetA+9*1+6, offsetB+9*0+6);
    sys->addBilateralConstraintDOF(offsetA+9*1+7, offsetB+9*0+7);
    sys->addBilateralConstraintDOF(offsetA+9*1+8, offsetB+9*0+8);

    // node 2 of plate A is fixed to node 3 of plate B
    sys->addBilateralConstraintDOF(offsetA+9*2, offsetB+9*3);
    sys->addBilateralConstraintDOF(offsetA+9*2+1, offsetB+9*3+1);
    sys->addBilateralConstraintDOF(offsetA+9*2+2, offsetB+9*3+2);
    sys->addBilateralConstraintDOF(offsetA+9*2+3, offsetB+9*3+3);
    sys->addBilateralConstraintDOF(offsetA+9*2+4, offsetB+9*3+4);
    sys->addBilateralConstraintDOF(offsetA+9*2+5, offsetB+9*3+5);
    sys->addBilateralConstraintDOF(offsetA+9*2+6, offsetB+9*3+6);
    sys->addBilateralConstraintDOF(offsetA+9*2+7, offsetB+9*3+7);
    sys->addBilateralConstraintDOF(offsetA+9*2+8, offsetB+9*3+8);
    //
    //    // node 1 of flat plate A is fixed to node 0 of flat plate B
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1, offsetFlatB+9*0);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+1, offsetFlatB+9*0+1);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+2, offsetFlatB+9*0+2);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+3, offsetFlatB+9*0+3);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+4, offsetFlatB+9*0+4);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+5, offsetFlatB+9*0+5);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+6, offsetFlatB+9*0+6);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+7, offsetFlatB+9*0+7);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*1+8, offsetFlatB+9*0+8);
    //
    //    // node 2 of flat plate A is fixed to node 3 of flat plate B
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2, offsetFlatB+9*3);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+1, offsetFlatB+9*3+1);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+2, offsetFlatB+9*3+2);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+3, offsetFlatB+9*3+3);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+4, offsetFlatB+9*3+4);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+5, offsetFlatB+9*3+5);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+6, offsetFlatB+9*3+6);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+7, offsetFlatB+9*3+7);
    //    sys->addBilateralConstraintDOF(offsetFlatA+9*2+8, offsetFlatB+9*3+8);

    // node 0 of curved plate A is fixed to node 0 of back curved plate A
    sys->addBilateralConstraintDOF(offsetA+9*0, offsetCurvBeamBackA+6*0);
    sys->addBilateralConstraintDOF(offsetA+9*0+1, offsetCurvBeamBackA+6*0+1);
    sys->addBilateralConstraintDOF(offsetA+9*0+2, offsetCurvBeamBackA+6*0+2);
    sys->addBilateralConstraintDOF(offsetA+9*0+3, offsetCurvBeamBackA+6*0+3);
    sys->addBilateralConstraintDOF(offsetA+9*0+4, offsetCurvBeamBackA+6*0+4);
    sys->addBilateralConstraintDOF(offsetA+9*0+5, offsetCurvBeamBackA+6*0+5);

    // node 3 of curved plate A is fixed to node 0 of front curved plate A
    sys->addBilateralConstraintDOF(offsetA+9*3, offsetCurvBeamFrontA+6*0);
    sys->addBilateralConstraintDOF(offsetA+9*3+1, offsetCurvBeamFrontA+6*0+1);
    sys->addBilateralConstraintDOF(offsetA+9*3+2, offsetCurvBeamFrontA+6*0+2);
    sys->addBilateralConstraintDOF(offsetA+9*3+3, offsetCurvBeamFrontA+6*0+3);
    sys->addBilateralConstraintDOF(offsetA+9*3+4, offsetCurvBeamFrontA+6*0+4);
    sys->addBilateralConstraintDOF(offsetA+9*3+5, offsetCurvBeamFrontA+6*0+5);

    // node 1 of back curved plate A is fixed to node 0 of back curved plate B
    sys->addBilateralConstraintDOF(offsetCurvBeamBackA+6*1, offsetCurvBeamBackB+6*0);
    sys->addBilateralConstraintDOF(offsetCurvBeamBackA+6*1+1, offsetCurvBeamBackB+6*0+1);
    sys->addBilateralConstraintDOF(offsetCurvBeamBackA+6*1+2, offsetCurvBeamBackB+6*0+2);
    sys->addBilateralConstraintDOF(offsetCurvBeamBackA+6*1+3, offsetCurvBeamBackB+6*0+3);
    sys->addBilateralConstraintDOF(offsetCurvBeamBackA+6*1+4, offsetCurvBeamBackB+6*0+4);
    sys->addBilateralConstraintDOF(offsetCurvBeamBackA+6*1+5, offsetCurvBeamBackB+6*0+5);

    // node 1 of front curved plate A is fixed to node 0 of front curved plate B
    sys->addBilateralConstraintDOF(offsetCurvBeamFrontA+6*1, offsetCurvBeamFrontB+6*0);
    sys->addBilateralConstraintDOF(offsetCurvBeamFrontA+6*1+1, offsetCurvBeamFrontB+6*0+1);
    sys->addBilateralConstraintDOF(offsetCurvBeamFrontA+6*1+2, offsetCurvBeamFrontB+6*0+2);
    sys->addBilateralConstraintDOF(offsetCurvBeamFrontA+6*1+3, offsetCurvBeamFrontB+6*0+3);
    sys->addBilateralConstraintDOF(offsetCurvBeamFrontA+6*1+4, offsetCurvBeamFrontB+6*0+4);
    sys->addBilateralConstraintDOF(offsetCurvBeamFrontA+6*1+5, offsetCurvBeamFrontB+6*0+5);

    // node 1 of back beam A is fixed to node 0 of curved back beam A
    sys->addBilateralConstraintDOF(offsetBeamBackA+6*1, offsetCurvBeamBackA+6*0);
    sys->addBilateralConstraintDOF(offsetBeamBackA+6*1+1, offsetCurvBeamBackA+6*0+1);
    sys->addBilateralConstraintDOF(offsetBeamBackA+6*1+2, offsetCurvBeamBackA+6*0+2);

    // node 0 of front beam A is fixed to node 0 of curved front beam A
    sys->addBilateralConstraintDOF(offsetBeamFrontA+6*0, offsetCurvBeamFrontA+6*0);
    sys->addBilateralConstraintDOF(offsetBeamFrontA+6*0+1, offsetCurvBeamFrontA+6*0+1);
    sys->addBilateralConstraintDOF(offsetBeamFrontA+6*0+2, offsetCurvBeamFrontA+6*0+2);

    // body 0 is fixed to node 0 of beam A back
    sys->addBilateralConstraintDOF(3*0, offsetBeamBackA+6*0);
    sys->addBilateralConstraintDOF(3*0+1, offsetBeamBackA+6*0+1);
    sys->addBilateralConstraintDOF(3*0+2, offsetBeamBackA+6*0+2);

    // body 1 is fixed to node 1 of beam A front
    sys->addBilateralConstraintDOF(3*1, offsetBeamFrontA+6*1);
    sys->addBilateralConstraintDOF(3*1+1, offsetBeamFrontA+6*1+1);
    sys->addBilateralConstraintDOF(3*1+2, offsetBeamFrontA+6*1+2);

    // node 0 of beam A is fixed to body 2 (z-direction)
    sys->addBilateralConstraintDOF(3*3+2, offsetBeamFrontA+6*0+2);

    // node 1 of beam A is fixed to body 2 (z-direction)
    sys->addBilateralConstraintDOF(3*3+2, offsetBeamFrontA+6*1+2);

    // node 0 of beam A is fixed to body 3 (z-direction)
    sys->addBilateralConstraintDOF(3*2+2, offsetBeamBackA+6*0+2);

    // node 1 of beam A is fixed to body 3 (z-direction)
    sys->addBilateralConstraintDOF(3*2+2, offsetBeamBackA+6*1+2);
  }

  sys->initializeSystem();
  printf("System initialized!\n");
  //sys->printSolverParams();

#ifdef WITH_GLUT
  if(visualize)
  {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(1024 ,512);
    glutCreateWindow("MAIN");
    glutDisplayFunc(renderSceneAll);
    glutIdleFunc(renderSceneAll);
    glutReshapeFunc(changeSize);
    glutIgnoreKeyRepeat(0);
    glutKeyboardFunc(CallBackKeyboardFunc);
    glutMouseFunc(CallBackMouseFunc);
    glutMotionFunc(CallBackMotionFunc);
    initScene();
    glutMainLoop();
  }
#endif

  // if you don't want to visualize, then output the data
  std::stringstream statsFileStream;
  statsFileStream << outDir << "statsBikeWheel_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << ".dat";
  ofstream statStream(statsFileStream.str().c_str());
  int fileIndex = 0;
  while(sys->time < t_end)
  {
    if(sys->timeIndex%200==0) {
      std::stringstream dataFileStream;
      dataFileStream << povrayDir << "data_" << fileIndex << ".dat";
      sys->exportSystem(dataFileStream.str());
      fileIndex++;
    }

    p0_h = sys->p_d;
    sys->DoTimeStep();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    double weight = 0;
    for(int i=0; i<1; i++) {
      weight += sys->f_contact_h[3*i+1];
    }
    cout << "  Weight: " << weight << endl;

    int numKrylovIter = 0;
    if(solverTypeQOCC==2) numKrylovIter = dynamic_cast<PDIP*>(sys->solver)->totalKrylovIterations;
    if(solverTypeQOCC==3) numKrylovIter = dynamic_cast<TPAS*>(sys->solver)->totalKrylovIterations;
    if(solverTypeQOCC==4) numKrylovIter = dynamic_cast<JKIP*>(sys->solver)->totalKrylovIterations;
    statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << weight << ", " << numKrylovIter << endl;

    // TODO: This is a big no-no, need to enforce motion via constraints
    // Apply motion
    sys->v_h = sys->v_d;
    if(sys->time>1.0) {
      for(int i=0;i<2;i++) {
        sys->v_h[3*i] = desiredVelocity;
        //sys->v_h[3*i+1] = 0;
        //sys->v_h[3*i+2] = 0;
      }
    }
    else {
      for(int i=0;i<1;i++) {
        //sys->v_h[3*i] = 0;
        //sys->v_h[3*i+1] = 0;
        //sys->v_h[3*i+2] = 0;
      }
    }

    sys->p_d = p0_h;
    sys->v_d = sys->v_h;
    cusp::blas::axpy(sys->v, sys->p, sys->h);
    sys->p_h = sys->p_d;
    // End apply motion
  }
  sys->exportMatrices(outDir.c_str());
  std::stringstream collisionFileStream;
  collisionFileStream << outDir << "collisionData.dat";
  sys->collisionDetector->exportSystem(collisionFileStream.str().c_str());

  return 0;
}
