#include "include.cuh"
#include <sys/stat.h>
#include <errno.h>
#include "System.cuh"
#include "Body.cuh"
#include "Beam.cuh"
#include "Plate.cuh"
#include "Body2D.cuh"
#include "APGD.cuh"
#include "PDIP.cuh"
#include "TPAS.cuh"
#include "JKIP.cuh"

bool updateDraw = 1;
bool wireFrame = 1;
double desiredVelocity = 0.5;
double tStartGlobal = 0;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;
std::string outDir = "../TEST_HUBMESHW/";
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
  gluLookAt(0.0,0.0,0.0,		0.0,0.0,-7,		0.0f,1.0f,0.0f);
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

double getRandomNumber(double min, double max)
{
  // x is in [0,1[
  double x = rand()/static_cast<double>(RAND_MAX);

  // [0,1[ * (max - min) + min is in [min,max[
  double that = min + ( x * (max - min) );

  return that;
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
	  //if(sys->timeIndex%20==0)
	    drawAll();
	  p0_h = sys->p_d;
    sys->DoTimeStep();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    sys->gamma_h = sys->gamma_d;
    sys->p_h = sys->p_d;
    double weight = sys->f_contact_h[3*0+1];
    double traction = sys->f_contact_h[3*0];
    double tractionLateral = sys->f_contact_h[3*0+2];
    double drawbar = sys->gamma_h[0];
    double torque = sys->gamma_h[1];
    int offsetHub = 3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size();
    double3 posHub = make_double3(sys->p_h[offsetHub+0],sys->p_h[offsetHub+1],sys->p_h[offsetHub+2]);
    cout << "  Weight: " << weight << " Traction: " << traction << " TractionLateral: " << tractionLateral << " Drawbar: " << drawbar << " Torque: " << torque << " Pos: (" << posHub.x << ", " << posHub.y << ", " << posHub.z << ")" << endl;


//    std::string loc = "..";
//    if(sys->collisionDetector->numCollisions) {//sys->solver->iterations>5000 || sys->time>0.074) {
//      sys->exportMatrices(loc);
//      cin.get();
//    }


    // TODO: This is a big no-no, need to enforce motion via constraints
    // Apply motion
    sys->v_h = sys->v_d;
    if(sys->time>tStartGlobal) {
      sys->v_h[0] = 0;
      sys->v_h[1] = 0;
      sys->v_h[2] = desiredVelocity;
    } else {
      sys->v_h[0] = 0;
      sys->v_h[1] = 0;
      sys->v_h[2] = 0;
    }
    sys->p_d = p0_h;
    sys->v_d = sys->v_h;
    cusp::blas::axpy(sys->v, sys->p, sys->h);
    sys->p_h = sys->p_d;
    // End apply motion

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

int main(int argc, char** argv)
{
  // command line arguments
  // FlexibleNet <numPartitions> <numBeamsPerSide> <solverType> <usePreconditioning>
  // solverType: (0) BiCGStab, (1) BiCGStab1, (2) BiCGStab2, (3) MinRes, (4) CG, (5) CR

  double t_end = 15.0;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int precondType = 1;
  int solverType = 4;
  int numPartitions = 1;
  double mu_pdip = 10.0;
  double alpha = 0.01; // should be [0.01, 0.1]
  double beta = 0.8; // should be [0.3, 0.8]
  int solverTypeQOCC = 1;
  int binsPerAxis = 10;
  double tolerance = 1e-4;
  double hh = 1e-4;
  int numDiv = 10;
  int numDivW = 1;
  double slip = 0.3;
  int numContacts = 12;
  double frictionCoefficient = 0.25;
  double slipAngle = 0;
  double load = 10; // N

  if(argc > 1) {
    numDiv = atoi(argv[1]);
    numDivW = atoi(argv[2]);
    load = atof(argv[3]);
    frictionCoefficient = atof(argv[4]);
    numContacts = atoi(argv[5]);
    tolerance = atof(argv[6]);
    hh = atof(argv[7]);
  }

#ifdef WITH_GLUT
  bool visualize = true;
#endif
  visualize = false;

  sys = new System(solverTypeQOCC);
  sys->setTimeStep(hh);
  sys->setFrictionCoefficient(frictionCoefficient);
  sys->solver->tolerance = tolerance;
  //sys->solver->maxIterations = 2000;

  // Create output directories
  std::stringstream outDirStream;
  outDirStream << "../TEST_HUBMESHW_n" << numDiv << "_nW" << numDivW << "_load" << load << "_mu" << frictionCoefficient << "_nC" << numContacts << "_h" << hh << "_tol" << tolerance << "/";
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

  sys->collisionDetector->setBinsPerAxis(make_uint3(binsPerAxis,10,binsPerAxis));
  if(solverTypeQOCC==1) {
    dynamic_cast<APGD*>(sys->solver)->setWarmStarting(true);
    dynamic_cast<APGD*>(sys->solver)->setAntiRelaxation(false);
  }
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

  //sys->solver->maxIterations = 200;
  //sys->gravity = make_double3(0,0,0);

  double radianInc = 2.0*PI/((double) numDiv);
  double EM = 2.e7;
  double rho = 7810.0;
  double th = .01;
  double R = .3;
  double R_o = 0.305;
  double nu = .1;
  double fillet = .04;
  double beltWidth = .2;
  double B = .5*PI*beltWidth;//1.5*.5*PI*beltWidth;
  double L = 2.0*PI*(R+0.5*beltWidth)/((double) numDiv);//2*PI*(R+1.4*0.33*beltWidth)/((double) numDiv);
  double depth = .2;
  double ditchLength = 2;
  double ditchWidth = 30.0*beltWidth;

  double3 center = make_double3(0,0,0);

  // Add hub
  Body2D* hub = new Body2D(center,make_double3(0,0,0),1.0,1.0);
  hub->setMass(1.0+load/9.81);
  sys->add(hub);

//  // Add ground
//  Body* groundPtr3 = new Body(make_double3(0,-1.2,0));
//  groundPtr3->setBodyFixed(true);
//  //groundPtr3->setCollisionFamily(2);
//  groundPtr3->setGeometry(make_double3(1,1,1));
//  sys->add(groundPtr3);

  // Add ground
  Body* groundPtr3 = new Body(make_double3(2*R+0.5*ditchLength,-R-3*th,0));
  //groundPtr3->setBodyFixed(true);
  //groundPtr3->setCollisionFamily(2);
  groundPtr3->setGeometry(make_double3(4*R+0.5*ditchLength,th,0.5*ditchWidth));
  sys->add(groundPtr3);

//  // Add ground
//  Body* groundPtr = new Body(make_double3(0,-R-beltWidth-0.5*depth,0));
//  groundPtr->setBodyFixed(true);
//  //groundPtr->setCollisionFamily(2);
//  groundPtr->setGeometry(make_double3(2*R,0.5*depth,0.5*ditchWidth));
//  sys->add(groundPtr);
//
//  // Add ground
//  Body* groundPtr2 = new Body(make_double3(4*R+ditchLength,-R-beltWidth-0.5*depth,0));
//  groundPtr2->setBodyFixed(true);
//  //groundPtr2->setCollisionFamily(2);
//  groundPtr2->setGeometry(make_double3(2*R,0.5*depth,0.5*ditchWidth));
//  sys->add(groundPtr2);
//
//  // Add ground
//  Body* groundPtr3 = new Body(make_double3(2*R+0.5*ditchLength,-R-beltWidth-depth-th,0));
//  groundPtr3->setBodyFixed(true);
//  //groundPtr3->setCollisionFamily(2);
//  groundPtr3->setGeometry(make_double3(4*R+0.5*ditchLength,th,0.5*ditchWidth));
//  sys->add(groundPtr3);
//
//  // Add sides
//  Body* right = new Body(make_double3(2*R+0.5*ditchLength,-R-beltWidth-0.5*depth,0.5*ditchWidth+th));
//  right->setBodyFixed(true);
//  //right->setCollisionFamily(2);
//  right->setGeometry(make_double3(4*R+0.5*ditchLength,depth+th,th));
//  sys->add(right);
//
//  // Add sides
//  Body* left = new Body(make_double3(2*R+0.5*ditchLength,-R-beltWidth-0.5*depth,-0.5*ditchWidth-th));
//  left->setBodyFixed(true);
//  //left->setCollisionFamily(2);
//  left->setGeometry(make_double3(4*R+0.5*ditchLength,depth+th,th));
//  sys->add(left);

//  Body* body = new Body(make_double3(2,0,0));
//  body->setGeometry(make_double3(R+0.5*beltWidth,0,0));
//  sys->add(body);
//
//  Body* body1 = new Body(make_double3(2,2.1*(R+0.5*beltWidth),0));
//  body1->setGeometry(make_double3(R+0.5*beltWidth,0,0));
//  sys->add(body1);
//
//  Body* body2 = new Body(make_double3(2,4.2*(R+0.5*beltWidth),0));
//  body2->setGeometry(make_double3(R+0.5*beltWidth,0,0));
//  sys->add(body2);
//
//  Body* body3 = new Body(make_double3(2,6.3*(R+0.5*beltWidth),0));
//  body3->setGeometry(make_double3(R+0.5*beltWidth,0,0));
//  sys->add(body3);
//
//  double rMin = 0.007;
//  double rMax = 0.007;
//  double density = 2600;
//  double W = ditchWidth;
//  double L_G = ditchLength;
//  double H = 3.0*depth;
//  double3 centerG = make_double3(2*R+0.5*ditchLength,-R-beltWidth-depth,0);
//  Body* bodyPtr;
//  double wiggle = 0.003;//0.1;
//  double numElementsPerSideX = L_G/(2.0*rMax+2.0*wiggle);
//  double numElementsPerSideY = H/(2.0*rMax+2.0*wiggle);
//  double numElementsPerSideZ = W/(2.0*rMax+2.0*wiggle);
//  int numBodies = 0;
//  // Add elements in x-direction
//  for (int i = 0; i < (int) numElementsPerSideX; i++) {
//    for (int j = 0; j < (int) numElementsPerSideY; j++) {
//      for (int k = 0; k < (int) numElementsPerSideZ; k++) {
//
//        double xWig = 0.8*getRandomNumber(-wiggle, wiggle);
//        double yWig = 0.8*getRandomNumber(-wiggle, wiggle);
//        double zWig = 0.8*getRandomNumber(-wiggle, wiggle);
//        bodyPtr = new Body(centerG+make_double3((rMax+wiggle)*(2.0*((double)i)+1.0)-0.5*L_G+xWig,(rMax+wiggle)*(2.0*((double)j)+1.0)+yWig,(rMax+wiggle)*(2.0*((double)k)+1.0)-0.5*W+zWig));
//        double rRand = getRandomNumber(rMin, rMax);
//        bodyPtr->setMass(4.0*rRand*rRand*rRand*PI/3.0*density);
//        bodyPtr->setGeometry(make_double3(rRand,0,0));
//        //if(j==0)
//        //bodyPtr->setBodyFixed(true);
//        numBodies = sys->add(bodyPtr);
//
//        if(numBodies%1000==0) printf("Bodies %d\n",numBodies);
//      }
//    }
//  }

  double tStart = 3.0;
  tStartGlobal = tStart;
  double omega = 17.0*PI/180.0;
  double vel = R_o*omega*(1.0 - slip);
  double velGround = tan(slipAngle)*vel;
  desiredVelocity = velGround;
  int offsetHub = 3*sys->bodies.size()+12*sys->beams.size()+36*sys->plates.size();
  sys->addBilateralConstraintDOF(offsetHub,-1, vel, tStart);
  //sys->addBilateralConstraintDOF(offsetHub+1,-1);
  sys->addBilateralConstraintDOF(offsetHub+2,-1, -omega, tStart);

  std::stringstream inputFileStream;
  inputFileStream << "../tireMeshes/tireMeshf_Ro0.3_Ri0.15_" << numDiv << "x" << numDivW << ".dat";
  sys->importMesh(inputFileStream.str(),EM,numContacts);

  // Add bilateral constraints
  for(int i=0;i<numDiv;i++)
  {
    //pin tire nodes to hub
    sys->pinShellNodeToBody2D((numDivW+1)*i,0);
    sys->pinShellNodeToBody2D((numDivW+1)*i+numDivW,0);
  }

//  // Add bilateral constraints
//  for(int i=0;i<2*numDiv;i++)
//  {
//    //pin tire nodes to hub
//    sys->pinShellNodeToBody2D(i,0);
//  }

  sys->initializeSystem();
  printf("System initialized!\n");
  //sys->printSolverParams();

#ifdef WITH_GLUT
  if(visualize)
  {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(0,0);
    glutInitWindowSize(1024	,512);
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
  statsFileStream << outDir << "statsHubMeshW_n" << numDiv << "_nW" << numDivW << "_load" << load << "_mu" << frictionCoefficient << "_nC" << numContacts << "_h" << hh << "_tol" << tolerance << ".dat";
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

    p0_h = sys->p_d;
    sys->DoTimeStep();
    //sys->exportMatrices(outDir.c_str());
    //cin.get();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    sys->gamma_h = sys->gamma_d;
    sys->p_h = sys->p_d;
    double weight = sys->f_contact_h[3*0+1];
    double traction = sys->f_contact_h[3*0];
    double tractionLateral = sys->f_contact_h[3*0+2];
    double drawbar = sys->gamma_h[0];
    double torque = sys->gamma_h[1];
    double3 posHub = make_double3(sys->p_h[offsetHub+0],sys->p_h[offsetHub+1],sys->p_h[offsetHub+2]);
    cout << "  Weight: " << weight << " Traction: " << traction << " TractionLateral: " << tractionLateral << " Drawbar: " << drawbar << " Torque: " << torque << " Pos: (" << posHub.x << ", " << posHub.y << ", " << posHub.z << ")" << endl;

    int numKrylovIter = 0;
    if(solverTypeQOCC==2) numKrylovIter = dynamic_cast<PDIP*>(sys->solver)->totalKrylovIterations;
    if(solverTypeQOCC==3) numKrylovIter = dynamic_cast<TPAS*>(sys->solver)->totalKrylovIterations;
    if(solverTypeQOCC==4) numKrylovIter = dynamic_cast<JKIP*>(sys->solver)->totalKrylovIterations;
    statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << weight << ", " << traction << ", " << tractionLateral << ", " << drawbar << ", " << torque << ", " << posHub.x << ", " << posHub.y << ", " << posHub.z << ", " << numKrylovIter << ", " << endl;

    // TODO: This is a big no-no, need to enforce motion via constraints
    // Apply motion
    sys->v_h = sys->v_d;
    if(sys->time>tStart) {
      sys->v_h[0] = 0;
      sys->v_h[1] = 0;
      sys->v_h[2] = velGround;
    } else {
      sys->v_h[0] = 0;
      sys->v_h[1] = 0;
      sys->v_h[2] = 0;
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

