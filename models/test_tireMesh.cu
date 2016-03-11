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
std::string outDir = "../TEST_TIREMESH/";
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
          glutSolidSphere(sys->plates[i]->getThickness(),10,10);
          glPopMatrix();
        }
      }
    }

    for(int i=0;i<sys->body2Ds.size();i++)
    {
      double offset = 3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size();
      double3 pos = make_double3(sys->p_h[3*i+offset],sys->p_h[3*i+1+offset],sys->p_h[3*i+2+offset]);

      // Draw x-axis
      glLineWidth(2.5);
      glColor3f(1.0, 0.0, 0.0);
      glBegin(GL_LINES);
      glVertex3f(pos.x, pos.y, 0.0);
      glVertex3f(pos.x+cos(pos.z), pos.y+sin(pos.z), 0.0);
      glEnd();

      // Draw y-axis
      glLineWidth(2.5);
      glColor3f(0.0, 1.0, 0.0);
      glBegin(GL_LINES);
      glVertex3f(pos.x, pos.y, 0.0);
      glVertex3f(pos.x-sin(pos.z), pos.y+cos(pos.z), 0.0);
      glEnd();

      // Draw z-axis
      glLineWidth(2.5);
      glColor3f(0.0, 0.0, 1.0);
      glBegin(GL_LINES);
      glVertex3f(pos.x, pos.y, 0);
      glVertex3f(pos.x, pos.y, 1.0);
      glEnd();
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
    cout << "Weight: " << weight << ", Pos: (" << sys->p_h[3] << ", " << sys->p_h[4] << ", " << sys->p_h[5] << ")" << endl;

    cout << "  Weight:           " << weight << endl;
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

  double t_end = 20.0;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int precondType = 1;
  int numElementsPerSide = 3;
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
  //visualize = false;

  sys = new System(solverTypeQOCC);
  sys->setTimeStep(hh);
  sys->solver->tolerance = tolerance;

  // Create output directories
  std::stringstream outDirStream;
  outDirStream << "../TEST_TIREMESH_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << "/";
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

  //sys->solver->maxIterations = 200;
  sys->gravity = make_double3(0,0,0);

  // Add ground
  Body* groundPtr = new Body(make_double3(0,-0.1,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setGeometry(make_double3(100,0.3,100));
  //groundPtr->setCollisionFamily(-2);
  //sys->add(groundPtr);

  // Add hub
  Body2D* hub = new Body2D(make_double3(0,0.5,0),make_double3(0,0,0),1.0,1.0);
  hub->setMass(1);
  sys->add(hub);

  double R = 0.2;
  double beltWidth = 0.2;
  double slip = 0;
  double tStart = 2.0;
  double omega = 17.0*PI/180.0;
  double vel = (R+0.5*beltWidth)*omega*(1.0 - slip);
  double offsetHub = 3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size();
  sys->addBilateralConstraintDOF(offsetHub,-1, vel, tStart);
  sys->addBilateralConstraintDOF(offsetHub+1,-1);
  sys->addBilateralConstraintDOF(offsetHub+2,-1, -omega, tStart);

//  std::stringstream inputFileStream;
//  inputFileStream << "../shellMeshes/shellMesh" << numElementsPerSide << "x" << numElementsPerSide << ".txt";
//  sys->importMesh(inputFileStream.str(),2e6,6);
  sys->importMesh("../tireMesh_z10.txt",2e7,10);

  // Add bilateral constraints
  for(int i=0;i<2*numElementsPerSide;i++)
  {
    //pin tire nodes to hub
    sys->pinShellNodeToBody2D(i,0);
  }

////  sys->addBilateralConstraintDOF(3,-1);
////  sys->addBilateralConstraintDOF(4,-1);
////  sys->addBilateralConstraintDOF(5,-1);
////
//
//int node = numElementsPerSide;
////  sys->addBilateralConstraintDOF(3+9*node,-1);
////  sys->addBilateralConstraintDOF(4+9*node,-1);
////  sys->addBilateralConstraintDOF(5+9*node,-1);
////
//  node = (numElementsPerSide+1)*(numElementsPerSide+1)-1;
//  sys->addBilateralConstraintDOF(3+9*node,-1);
//  sys->addBilateralConstraintDOF(4+9*node,-1);
//  sys->addBilateralConstraintDOF(5+9*node,-1);
////
////  node = (numElementsPerSide+1)*(numElementsPerSide+1)-1-numEl;
////  sys->addBilateralConstraintDOF(3+9*node,-1);
////  sys->addBilateralConstraintDOF(4+9*node,-1);
////  sys->addBilateralConstraintDOF(5+9*node,-1);

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
  statsFileStream << outDir << "statsTireMesh_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << ".dat";
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
