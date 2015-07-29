#include "include.cuh"
#include "System.cuh"
#include "Body.cuh"
#include "TPAS.cuh"
#include "JKIP.cuh"

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
    for(int i=0; i<sys->f_contact_h.size(); i++) {
      cout << "f_contact_h[" << i << "] = " << sys->f_contact_h[i] << endl;
    }
    cin.get();

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
  // solverType: (0) BiCGStab, (1) BiCGStab1, (2) BiCGStab2, (3) MinRes, (4) CG, (5) CR

#ifdef WITH_GLUT
	bool visualize = true;
#endif
	//visualize = false;

	int solverTypeQOCC = 4;
	sys = new System(solverTypeQOCC);
  sys->setTimeStep(1e-2);
  sys->collisionDetector->setBinsPerAxis(make_uint3(10,10,10));
  sys->solver->tolerance = 1e-4;
  sys->solver->maxIterations = 100;
  int precondType = 1;
  int solverType = 2;
  int numPartitions = 1;
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

  // Bottom
  Body* groundPtr = new Body(make_double3(0,-0.5,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setGeometry(make_double3(4,0.5,1));
  sys->add(groundPtr);

  Body* ball1 = new Body(make_double3(3,1,0));
  ball1->setGeometry(make_double3(1,0,0));
  sys->add(ball1);
  sys->applyForce(ball1,make_double3(20,0,0));

  Body* ball2 = new Body(make_double3(0,1,0));
  ball2->setGeometry(make_double3(1,0,0));
  sys->add(ball2);
  sys->applyForce(ball2,make_double3(1,0,0));

  Body* ball3 = new Body(make_double3(-3,1,0));
  ball3->setGeometry(make_double3(1,0,0));
  sys->add(ball3);

  Body* ball4 = new Body(make_double3(0,-2,0));
  ball4->setGeometry(make_double3(1,0,0));
  sys->add(ball4);

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
	while(sys->time < 10)
	{

		sys->DoTimeStep();

    sys->f_contact_h = sys->f_contact_d;
    for(int i=0; i<sys->f_contact_h.size(); i++) {
      cout << "f_contact_h[" << i << "] = " << sys->f_contact_h[i] << endl;
    }
    cin.get();
	}

	return 0;
}

