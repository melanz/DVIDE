#include "include.cuh"
#include <sys/stat.h>
#include <errno.h>
#include "System.cuh"
#include "APGD.cuh"
#include "Body.cuh"
#include "Beam.cuh"
#include "Plate.cuh"

bool updateDraw = 1;
bool wireFrame = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;
std::string outDir = "../TEST_TWOPLATE/";
std::string povrayDir = outDir + "POVRAY/";

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
          glutSolidSphere(0.5*sys->plates[i]->getThickness(),10,10);
          glPopMatrix();
        }
      }
    }

    for(int i=0;i<sys->shellConnectivities_h.size();i++) {
      int xiDiv = sys->shellGeometries_h[i].w;
      int etaDiv = sys->shellGeometries_h[i].w;
      double xiInc = 1/(static_cast<double>(xiDiv-1));
      double etaInc = 1/(static_cast<double>(etaDiv-1));
      glColor3f(0.0f,1.0f,1.0f);
      for(int j=0;j<xiDiv;j++)
      {
        for(int k=0;k<etaDiv;k++) {
          glPushMatrix();
          double3 position = sys->transformNodalToCartesian_shellMesh(i,xiInc*j,etaInc*k);
          glTranslatef(position.x,position.y,position.z);
          glutSolidSphere(0.5*sys->shellGeometries_h[i].z,10,10);
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

    sys->DoTimeStep();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    double weight = 0;
    for(int i=0; i<1; i++) {
      weight += sys->f_contact_h[3*i+1];
    }
    cout << "  Weight: " << weight << endl;
    cout << "  Potential Energy: " << sys->getPotentialEnergy() << endl;
    cout << "  Kinetic Energy:   " << sys->getKineticEnergy() << endl;
    cout << "  Strain Energy:    " << sys->getStrainEnergy() << endl;
    cout << "  Total Energy:     " << sys->getTotalEnergy() << endl;
    cout << "  Objective CCP:    " << sys->objectiveCCP << endl;
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

  double t_end = 1.5;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int precondType = 1;
  int numElementsPerSide = 4;
  int solverType = 4;
  int solverTypeQOCC = 1;
  int binsPerAxis = 30;
  double tolerance = 1e-4;
  double hh = 1e-3;

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
  outDirStream << "../TEST_TWOPLATE_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << "/";
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
  if(solverTypeQOCC==1) {
    dynamic_cast<APGD*>(sys->solver)->setWarmStarting(true);
    dynamic_cast<APGD*>(sys->solver)->setAntiRelaxation(false);
  }

  //sys->solver->maxIterations = 40;
  //sys->gravity = make_double3(0,0,0);

//  double3 unitX = make_double3(1.0,0,0);
//  double3 unitZ = make_double3(0,0,1.0);
//  Plate* plate = new Plate(1.0,1.0,
//      make_double3(0,0,0), unitX, unitZ,
//      make_double3(1,0,0), unitX, unitZ,
//      make_double3(1,0,1), unitX, unitZ,
//      make_double3(0,0,1), unitX, unitZ);
//  plate->setThickness(0.02);
//  plate->setDensity(7200.0);
//  plate->setElasticModulus(2e7);
//  plate->setPoissonRatio(0.25);
//  plate->setCollisionFamily(1);
//  plate->setNumContactPoints(40);
//  sys->add(plate);

//  // Add ground
//  Body* groundPtr = new Body(make_double3(0,-0.3,0));
//  groundPtr->setBodyFixed(true);
//  groundPtr->setGeometry(make_double3(0.2,0,0));
//  sys->add(groundPtr);

  // Add plate element
  double EM = 2e6;
  double rho = 7200;
  double nu = 0.25;
  double length = 1.0;
  double width = 1.0;
  double thickness = 0.01;
  int numContactPoints = 30;

//  double3 unitX = make_double3(1.0,0,0);
//  double3 unitZ = make_double3(0,0,1.0);
//  double3 pos0 = make_double3(-0.5,0,-0.5);
//
//  Plate* plate;
//  length = length/((double) numElementsPerSide);
//  width = width/((double) numElementsPerSide);
//  for(int i=0; i<numElementsPerSide; i++) {
//    for(int j=0; j<numElementsPerSide; j++) {
//      double3 pos = pos0+i*length*unitX+j*width*unitZ;
//      plate = new Plate(length,width,pos,unitX,unitZ,
//                                pos+length*unitX,unitX,unitZ,
//                                pos+length*unitX+width*unitZ,unitX,unitZ,
//                                pos+width*unitZ,unitX,unitZ);
//      plate->setThickness(thickness);
//      plate->setCollisionFamily(-2);
//      plate->setNumContactPoints(numContactPoints);
//      plate->setElasticModulus(EM);
//      plate->setDensity(rho);
//      plate->setPoissonRatio(nu);
//      sys->add(plate);
//    }
//  }
//
//  // pin corners to the ground
//  sys->addBilateralConstraintDOF(0, -1);
//  sys->addBilateralConstraintDOF(1, -1);
//  sys->addBilateralConstraintDOF(2, -1);
//
//  sys->addBilateralConstraintDOF(0+3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size()+3*sys->body2Ds.size(), -1);
//  sys->addBilateralConstraintDOF(1+3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size()+3*sys->body2Ds.size(), -1);
//  sys->addBilateralConstraintDOF(2+3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size()+3*sys->body2Ds.size(), -1);
//
//  // Add j constraints
//  int offset = 3*sys->bodies.size();
//  for(int i=0; i<numElementsPerSide; i++) {
//    for(int j=0; j<numElementsPerSide-1; j++) {
//      int elementA = numElementsPerSide*i+j;
//      int elementB = elementA+1;
//
//      int nodeA = 3;
//      int nodeB = 0;
//      for(int k=0;k<9;k++) sys->addBilateralConstraintDOF(36*elementA+9*nodeA+k+offset, 36*elementB+9*nodeB+k+offset);
//
//      nodeA = 2;
//      nodeB = 1;
//      for(int k=0;k<9;k++) sys->addBilateralConstraintDOF(36*elementA+9*nodeA+k+offset, 36*elementB+9*nodeB+k+offset);
//    }
//  }
//
//  // Add i constraints
//  for(int i=0; i<numElementsPerSide-1; i++) {
//    for(int j=0; j<numElementsPerSide; j++) {
//      int elementA = numElementsPerSide*i+j;
//      int elementB = elementA+numElementsPerSide;
//
//      int nodeA = 1;
//      int nodeB = 0;
//      for(int k=0;k<9;k++) sys->addBilateralConstraintDOF(36*elementA+9*nodeA+k+offset, 36*elementB+9*nodeB+k+offset);
//
//      nodeA = 2;
//      nodeB = 3;
//      for(int k=0;k<9;k++) sys->addBilateralConstraintDOF(36*elementA+9*nodeA+k+offset, 36*elementB+9*nodeB+k+offset);
//    }
//  }

  std::stringstream inputFileStream;
  inputFileStream << "../twoPlate.dat";
  sys->importMesh(inputFileStream.str(),EM,numContactPoints);

  // pin corners to the ground
  sys->addBilateralConstraintDOF(0, -1);
  sys->addBilateralConstraintDOF(1, -1);
  sys->addBilateralConstraintDOF(2, -1);

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
  statsFileStream << outDir << "statsTwoPlate_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << ".dat";
  ofstream statStream(statsFileStream.str().c_str());
  int fileIndex = 0;
  while(sys->time < t_end)
  {
    if(sys->timeIndex%20==0) {
      std::stringstream dataFileStream;
      dataFileStream << povrayDir << "data_" << fileIndex << ".dat";
      sys->exportSystem(dataFileStream.str());
      fileIndex++;
    }

    sys->DoTimeStep();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    double weight = 0;
    for(int i=0; i<1; i++) {
      weight += sys->f_contact_h[3*i+1];
    }
    cout << "  Weight: " << weight << endl;

    sys->p_h = sys->p_d;
    statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << weight << ", " << sys->p_h[3] << ", " << sys->p_h[4] << ", " << sys->p_h[5] << endl;
  }
  sys->exportMatrices(outDir.c_str());
  std::stringstream collisionFileStream;
  collisionFileStream << outDir << "collisionData.dat";
  sys->collisionDetector->exportSystem(collisionFileStream.str().c_str());

  return 0;
}
