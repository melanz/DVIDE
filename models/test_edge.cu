#include "include.cuh"
#include "System.cuh"
#include "Body.cuh"
#include "PDIP.cuh"

bool updateDraw = 1;
bool wireFrame = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(4,0.4,-14),camreal3(4,0.4,0),camreal3(0,1,0),.01);

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
		if(sys->timeIndex%10==0) drawAll();
		//char filename[100];
		//sprintf(filename, "../data/data_%03d.dat", sys->timeIndex);
		//sys->exportSystem(filename);
		sys->DoTimeStep();

    // Determine contact force on the container
    sys->f_contact_h = sys->f_contact_d;
    double weight = 0;
    for(int i=0; i<6; i++) {
      weight += sys->f_contact_h[3*i+1];
    }
    cout << "  Weight: " << weight << endl;
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

  double t_end = 3.0;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int precondType = 0;
  int numElementsPerSide = 4;
  int solverType = 3;
  int numPartitions = 1;
  double mu_pdip = 150.0;
  double alpha = 0.01; // should be [0.01, 0.1]
  double beta = 0.8; // should be [0.3, 0.8]
  int solverTypeQOCC = 1;
  int binsPerAxis = 10;

  if(argc > 1) {
    numPartitions = atoi(argv[1]);
    numElementsPerSide = atoi(argv[2]);
    solverType = atoi(argv[3]);
    precondType = atoi(argv[4]);
    mu_pdip = atof(argv[5]);
    alpha = atof(argv[6]);
    beta = atof(argv[7]);
    solverTypeQOCC = atoi(argv[8]);
    binsPerAxis = atoi(argv[9]);
  }

#ifdef WITH_GLUT
	bool visualize = true;
#endif
	//visualize = false;

	sys = new System(solverTypeQOCC);
  sys->setTimeStep(1e-4, 1e-10);

  int numElementsPerSideY = 10;
  sys->collisionDetector->setBinsPerAxis(make_uint3(binsPerAxis,numElementsPerSideY,binsPerAxis));
  if(solverTypeQOCC==2) {
    dynamic_cast<PDIP*>(sys->solver)->setPrecondType(precondType);
    dynamic_cast<PDIP*>(sys->solver)->setSolverType(solverType);
    dynamic_cast<PDIP*>(sys->solver)->setNumPartitions(numPartitions);
    dynamic_cast<PDIP*>(sys->solver)->alpha = alpha;
    dynamic_cast<PDIP*>(sys->solver)->beta = beta;
    dynamic_cast<PDIP*>(sys->solver)->mu_pdip = mu_pdip;
  }
  sys->solver->tolerance = 1e-4;
  //sys->solver->maxIterations = 10;

  double radius = 0.4;

  Body* ball1 = new Body(make_double3(5.2*radius,2.1*radius,5.2*radius));
  ball1->setGeometry(make_double3(radius,0,0));
  sys->add(ball1);

  Body* ball2 = new Body(make_double3(5.2*radius,2.1*radius,0));
  ball2->setGeometry(make_double3(radius,0,0));
  sys->add(ball2);

  Body* ball3 = new Body(make_double3(0,2.1*radius,0));
  ball3->setGeometry(make_double3(radius,0,0));
  sys->add(ball3);

  // Bottom
  Body* groundPtr = new Body(make_double3(0,0,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setGeometry(make_double3(5*radius,radius,5*radius));
  sys->add(groundPtr);

  Body* ball4 = new Body(make_double3(-5.2*radius,2.1*radius,-5.2*radius));
  ball4->setGeometry(make_double3(radius,0,0));
  sys->add(ball4);

  Body* ball5 = new Body(make_double3(-5.2*radius,2.1*radius,0));
  ball5->setGeometry(make_double3(radius,0,0));
  sys->add(ball5);

  Body* ball6 = new Body(make_double3(2.1*radius,2.1*radius,0));
  ball6->setGeometry(make_double3(radius,0,0));
  sys->add(ball6);



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
  char filename[100];
  sprintf(filename, "../data/stats_%d_%d_%d_%d_%.3f_%.3f_%.3f_%d_%d.dat",
      numPartitions,
      numElementsPerSide,
      solverType,
      precondType,
      mu_pdip,
      alpha,
      beta,
      solverTypeQOCC,
      binsPerAxis);
	ofstream statStream(filename);
	double maxVel = 0;
	while(sys->time < t_end)
	{
    char filename[100];
    sprintf(filename, "../data/data_%03d.dat", sys->timeIndex);
    sys->exportSystem(filename);

		sys->DoTimeStep();
		double maxVelNow = Thrust_Max(sys->v_d);
		if(maxVelNow>maxVel) maxVel = maxVelNow;

		// Determine contact force on the container
		sys->f_contact_h = sys->f_contact_d;
		double weight = 0;
		for(int i=0; i<6; i++) {
		  weight += sys->f_contact_h[3*i+1];
		}
		cout << "  Weight: " << weight << endl;

		int numKrylovIter = 0;
		if(solverTypeQOCC==2) numKrylovIter = dynamic_cast<PDIP*>(sys->solver)->totalKrylovIterations;
		statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << weight << ", " << numKrylovIter << ", " << endl;

	}
	cout << "Maximum Velocity = " << maxVel << endl;

	return 0;
}

