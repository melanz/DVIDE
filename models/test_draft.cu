#include "include.cuh"
#include "System.cuh"
#include "Body.cuh"
#include "PDIP.cuh"

bool updateDraw = 1;
bool wireFrame = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(4,5,-500),camreal3(4,5,0),camreal3(0,1,0),.01);

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
		//if(sys->timeIndex%10==0)
		drawAll();
		//char filename[100];
		//sprintf(filename, "../data/data_%03d.dat", sys->timeIndex);
		//sys->exportSystem(filename);
		sys->DoTimeStep();

//    // Determine contact force on the container
//    sys->f_contact_h = sys->f_contact_d;
//    double weight = 0;
//    for(int i=0; i<6; i++) {
//      weight += sys->f_contact_h[3*i+1];
//    }
//    cout << "  Weight: " << weight << endl;
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

  double t_end = 10;
  int    precUpdateInterval = -1;
  float  precMaxKrylov = -1;
  int precondType = 0;
  int numElementsPerSide = 4;
  int solverType = 1;
  int numPartitions = 1;
  double mu_pdip = 150.0;
  double alpha = 0.01; // should be [0.01, 0.1]
  double beta = 0.8; // should be [0.3, 0.8]
  int solverTypeQOCC = 1;
  int binsPerAxis = 10;


#ifdef WITH_GLUT
	bool visualize = true;
#endif
	visualize = false;

	double hh = 1e-3;
	sys = new System(solverTypeQOCC);
  sys->setTimeStep(hh);
  sys->gravity = make_double3(0,-981,0);

  sys->collisionDetector->setBinsPerAxis(make_uint3(30,10,10));
  sys->solver->tolerance = 5;
  //sys->solver->maxIterations = 10;

  double rMin = 0.8;
  double rMax = 1.6;
  double r = rMax;
  double L = 460;
  double W = 60;
  double H = 80;
  double bL = 1;
  double bH = 60;
  double bW = 20;
  double depth = 25;
  double th = 1;
  double density = 2.6;

  // Blade
  Body* bladePtr = new Body(make_double3(0.5*L+2*th,0.5*bH+depth,0));
  bladePtr->setBodyFixed(true);
  bladePtr->setGeometry(make_double3(0.5*bL,0.5*bH,0.5*bW));
  sys->add(bladePtr);

  // Bottom
  Body* groundPtr = new Body(make_double3(0,-th,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setGeometry(make_double3(0.5*L+th,th,0.5*W+th));
  sys->add(groundPtr);

  // Left
  Body* leftPtr = new Body(make_double3(-0.5*L-2*th,0.5*H+th,0));
  leftPtr->setBodyFixed(true);
  leftPtr->setGeometry(make_double3(th,0.5*H+th,0.5*W+th));
  sys->add(leftPtr);

  // Right
  Body* rightPtr = new Body(make_double3(0.5*L+2*th,0.5*H+th,0));
  rightPtr->setBodyFixed(true);
  rightPtr->setGeometry(make_double3(th,0.5*H+th,0.5*W+th));
  sys->add(rightPtr);

  // Back
  Body* backPtr = new Body(make_double3(0,0.5*H+th,-0.5*W-2*th));
  backPtr->setBodyFixed(true);
  backPtr->setGeometry(make_double3(0.5*L+th,0.5*H+th,th));
  sys->add(backPtr);

  // Front
  Body* frontPtr = new Body(make_double3(0,0.5*H+th,0.5*W+2*th));
  frontPtr->setBodyFixed(true);
  frontPtr->setGeometry(make_double3(0.5*L+th,0.5*H+th,th));
  sys->add(frontPtr);

  Body* bodyPtr;
  double wiggle = .3;//0.1;
  int numElementsPerSideX = (L+2*th)/(2*r+2*wiggle);
  int numElementsPerSideY = 2.5*H/(2*r+2*wiggle);
  int numElementsPerSideZ = (W+2*th)/(2*r+2*wiggle);
  int numBodies = 0;
  // Add elements in x-direction
  for (int i = 0; i < numElementsPerSideX; i++) {
    for (int j = 0; j < numElementsPerSideY; j++) {
      for (int k = 0; k < numElementsPerSideZ; k++) {

        double xWig = getRandomNumber(-wiggle, wiggle);
        double yWig = 0;//getRandomNumber(-.1, .1);
        double zWig = getRandomNumber(-wiggle, wiggle);
        bodyPtr = new Body(make_double3(2*(r+wiggle)*i-0.5*L-th+r+xWig,2*(r+wiggle)*j+r+yWig,2*(r+wiggle)*k-0.5*W-th+r+zWig));
        double rRand = getRandomNumber(rMin, rMax);
        bodyPtr->setMass(4.0*rRand*rRand*rRand*3.1415/3.0*density);
        bodyPtr->setGeometry(make_double3(rRand,0,0));
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
  sprintf(filename, "../data/stats_tol%f_h%f.dat",
      sys->solver->tolerance,
      hh);
	ofstream statStream(filename);
	int fileIndex = 0;
	while(sys->time < t_end)
	{
	  if(sys->timeIndex%20==0) {
      char filename[100];
      sprintf(filename, "../data/data_%03d.dat", fileIndex);
      sys->exportSystem(filename);
      fileIndex++;
	  }

		sys->DoTimeStep();

		// Determine contact force on the container
		sys->f_contact_h = sys->f_contact_d;
		double weight = 0;
		for(int i=0; i<1; i++) {
		  weight += sys->f_contact_h[3*i];
		}
		cout << "  Draft force: " << weight << endl;

		int numKrylovIter = 0;
		if(solverTypeQOCC==2) numKrylovIter = dynamic_cast<PDIP*>(sys->solver)->totalKrylovIterations;
		if(sys->timeIndex%10==0) statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << weight << ", " << numKrylovIter << ", " << endl;

	}

	return 0;
}

