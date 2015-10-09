#include "include.cuh"
#include "System.cuh"
#include "Body.cuh"
#include "PDIP.cuh"
#include "JKIP.cuh"

bool updateDraw = 1;
bool wireFrame = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;
double desiredVelocity = 0.166; // Needs to be global so that renderer can access it
thrust::host_vector<double> p0_h;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(0,0,-15),camreal3(0,0,0),camreal3(0,1,0),.01);

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

    glutSwapBuffers();
  }
}

void renderSceneAll(){
  if(OGL){
    //if(sys->timeIndex%10==0)
    drawAll();
    //char filename[100];
    //sprintf(filename, "../data/data_%03d.dat", sys->timeIndex);
    //sys->exportSystem(filename);
    p0_h = sys->p_d;
    sys->DoTimeStep();
//    if(sys->solver->iterations==1000) {
//      sys->exportMatrices("../data");
//      cin.get();
//    }

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    double shearForce = 0;
    for(int i=0; i<5; i++) {
      shearForce += sys->f_contact_h[3*i];
    }
    cout << "  Shear force: " << shearForce << endl;

    // TODO: This is a big no-no, need to enforce motion via constraints
    // Apply motion
    sys->v_h = sys->v_d;
    if(sys->time>1.5) {
      for(int i=0;i<5;i++) {
        sys->v_h[3*i] = desiredVelocity;
        sys->v_h[3*i+1] = 0;
        sys->v_h[3*i+2] = 0;
      }
    }
    // Constrain top body in x and z direction
    int i=5;
    sys->v_h[3*i] = 0;
    sys->v_h[3*i+2] = 0;
    sys->p_d = p0_h;
    sys->v_d = sys->v_h;
    cusp::blas::axpy(sys->v, sys->p, sys->h);
    sys->p_h = sys->p_d;
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
  int solverTypeQOCC = 1;
  double tolerance = 1e-4;
  double hh = 1e-4;

  double r = 0.8;
  double L = 6;
  double W = 6;
  double H = 3;
  double HTOP = 24;
  double TH = 3;
  double normalPressure = 168881; // 16,888.1 Pa // 44,127.0 Pa// 71,365.9 Pa
  double particleDensity = 2.6;
  double lengthToRun = 0.5;
  double gravity = 981;

#ifdef WITH_GLUT
  bool visualize = true;
#endif
  visualize = false;

  sys = new System(solverTypeQOCC);
  sys->setTimeStep(hh);
  sys->gravity = make_double3(0,-gravity,0);
  sys->collisionDetector->setBinsPerAxis(make_uint3(30,10,10));
  sys->solver->tolerance = tolerance;
  sys->solver->maxIterations = 1000;
  //sys->importSystem("../data_draft20K/data_129_overwrite.dat");

  // Bottom
  Body* groundPtr = new Body(make_double3(0,-0.5*TH,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setGeometry(make_double3(0.5*L+TH,0.5*TH,0.5*W+TH));
  sys->add(groundPtr);

  // Left Bottom
  Body* leftBottomPtr = new Body(make_double3(-0.5*L-0.5*TH,0.5*H,0));
  leftBottomPtr->setBodyFixed(true);
  leftBottomPtr->setGeometry(make_double3(0.5*TH,0.5*H,0.5*W+TH));
  sys->add(leftBottomPtr);

  // Right Bottom
  Body* rightBottomPtr = new Body(make_double3(0.5*L+0.5*TH,0.5*H,0));
  rightBottomPtr->setBodyFixed(true);
  rightBottomPtr->setGeometry(make_double3(0.5*TH,0.5*H,0.5*W+TH));
  sys->add(rightBottomPtr);

  // Back Bottom
  Body* backBottomPtr = new Body(make_double3(0,0.5*H,-0.5*W-0.5*TH));
  backBottomPtr->setBodyFixed(true);
  backBottomPtr->setGeometry(make_double3(0.5*L,0.5*H,0.5*TH));
  sys->add(backBottomPtr);

  // Front Bottom
  Body* frontBottomPtr = new Body(make_double3(0,0.5*H,0.5*W+0.5*TH));
  frontBottomPtr->setBodyFixed(true);
  frontBottomPtr->setGeometry(make_double3(0.5*L,0.5*H,0.5*TH));
  sys->add(frontBottomPtr);

  // Top
  Body* topPtr = new Body(make_double3(0,H+HTOP+0.5*TH,0));
  //topPtr->setBodyFixed(true);
  topPtr->setMass(normalPressure*L*W/gravity);
  topPtr->setGeometry(make_double3(0.5*L,0.5*TH,0.5*W));
  sys->add(topPtr);

  // Left Top
  Body* leftTopPtr = new Body(make_double3(-0.5*L-0.5*TH,0.5*HTOP+H,0));
  leftTopPtr->setBodyFixed(true);
  leftTopPtr->setGeometry(make_double3(0.5*TH,0.5*HTOP,0.5*W+TH));
  sys->add(leftTopPtr);

  // Right Top
  Body* rightTopPtr = new Body(make_double3(0.5*L+0.5*TH,0.5*HTOP+H,0));
  rightTopPtr->setBodyFixed(true);
  rightTopPtr->setGeometry(make_double3(0.5*TH,0.5*HTOP,0.5*W+TH));
  sys->add(rightTopPtr);

  // Back Top
  Body* backTopPtr = new Body(make_double3(0,0.5*HTOP+H,-0.5*W-0.5*TH));
  backTopPtr->setBodyFixed(true);
  backTopPtr->setGeometry(make_double3(0.5*L,0.5*HTOP,0.5*TH));
  sys->add(backTopPtr);

  // Front Top
  Body* frontTopPtr = new Body(make_double3(0,0.5*HTOP+H,0.5*W+0.5*TH));
  frontTopPtr->setBodyFixed(true);
  frontTopPtr->setGeometry(make_double3(0.5*L,0.5*HTOP,0.5*TH));
  sys->add(frontTopPtr);


  Body* bodyPtr;
  double wiggle = 0.4;
  int numElementsPerSideX = L/(2*r+2*wiggle);
  int numElementsPerSideY = (H+HTOP)/(2*r+2*wiggle);
  int numElementsPerSideZ = W/(2*r+2*wiggle);
  int numBodies = 0;
  // Add elements in x-direction
  for (int i = 0; i < numElementsPerSideX; i++) {
    for (int j = 0; j < numElementsPerSideY; j++) {
      for (int k = 0; k < numElementsPerSideZ; k++) {

        double xWig = getRandomNumber(-wiggle, wiggle);
        double yWig = getRandomNumber(-wiggle, wiggle); //0;//
        double zWig = getRandomNumber(-wiggle, wiggle);
        bodyPtr = new Body(make_double3(2*(r+wiggle)*i-0.5*L+(r+wiggle)+xWig,2*(r+wiggle)*j+(r+wiggle)+yWig,2*(r+wiggle)*k-0.5*W+(r+wiggle)+zWig));
        bodyPtr->setMass(4.0*r*r*r*3.1415/3.0*particleDensity);
        bodyPtr->setGeometry(make_double3(r,0,0));
        //if(j==0) bodyPtr->setBodyFixed(true);
        numBodies = sys->add(bodyPtr);

        if(numBodies%1000==0) printf("Bodies %d\n",numBodies);
      }
    }
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
  char filename[100];
  sprintf(filename, "../data/statsShear_tol%f_h%f_solver%d.dat", sys->solver->tolerance, hh, solverTypeQOCC);
  ofstream statStream(filename);

  int fileIndex = 0;
  while(sys->p_h[0] < lengthToRun)
  {
    if(sys->timeIndex%20==0) {
      char filename[100];
      sprintf(filename, "../data/data_%03d.dat", fileIndex);
      sys->exportSystem(filename);
      fileIndex++;
    }

    p0_h = sys->p_d;
    sys->DoTimeStep();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    double shearForce = 0;
    for(int i=0; i<5; i++) {
      shearForce += sys->f_contact_h[3*i];
    }
    cout << "  Shear force: " << shearForce << endl;

    // TODO: This is a big no-no, need to enforce motion via constraints
    // Apply motion
    sys->v_h = sys->v_d;
    if(sys->time>1.5) {
      for(int i=0;i<5;i++) {
        sys->v_h[3*i] = desiredVelocity;
        sys->v_h[3*i+1] = 0;
        sys->v_h[3*i+2] = 0;
      }
    }
    // Constrain top body in x and z direction
    int i=5;
    sys->v_h[3*i] = 0;
    sys->v_h[3*i+2] = 0;
    sys->p_d = p0_h;
    sys->v_d = sys->v_h;
    cusp::blas::axpy(sys->v, sys->p, sys->h);
    sys->p_h = sys->p_d;

    int numKrylovIter = 0;
    if(solverTypeQOCC==2) numKrylovIter = dynamic_cast<PDIP*>(sys->solver)->totalKrylovIterations;
    if(solverTypeQOCC==4) numKrylovIter = dynamic_cast<JKIP*>(sys->solver)->totalKrylovIterations;
    if(sys->timeIndex%10==0) statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << shearForce << ", " << sys->p_h[0] << ", " << numKrylovIter << ", " << endl;

//    if(sys->solver->iterations==1000) {
//      sys->exportSystem("../data/data_FAIL.dat");
//      sys->exportMatrices("../data");
//      cin.get();
//    }
  }

  return 0;
}
