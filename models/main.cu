#include "include.cuh"
#include "System.cuh"
#include "Element.cuh"
#include "Node.cuh"
#include "Particle.cuh"

bool updateDraw = 1;
bool showSphere = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System sys;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(-1,1,-1),camreal3(0,0,0),camreal3(0,1,0),.01);

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

		for(int i=0;i<sys.elements.size();i++)
		{
			int xiDiv = sys.numContactPoints;

			double xiInc = 1/(static_cast<double>(xiDiv-1));

			//if(showSphere)
			{
				glColor3f(0.0f,0.0f,1.0f);
				//for(int j=0;j<xiDiv;j++)
				{
					glPushMatrix();
					double3 position = sys.elements[i].getPosition();
					glTranslatef(sys.p_h[3*i],sys.p_h[3*i+1],sys.p_h[3*i+2]);
					glutSolidSphere(1,10,10);
					glPopMatrix();
				}
			}
//			else
//			{
//				int xiDiv = sys.numContactPoints;
//				double xiInc = 1/(static_cast<double>(xiDiv-1));
//				glLineWidth(sys.elements[i].getRadius()*500);
//				glColor3f(0.0f,1.0f,0.0f);
//				glBegin(GL_LINE_STRIP);
//				for(int j=0;j<sys.numContactPoints;j++)
//				{
//					float3 position = sys.getXYZPosition(i,xiInc*j);
//					glVertex3f(position.x,position.y,position.z);
//				}
//				glEnd();
//				glFlush();
//			}
		}

		glutSwapBuffers();
	}
}

void renderSceneAll(){
	if(OGL){
		//if(sys.timeIndex%10==0)
			drawAll();
		sys.DoTimeStep();
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
	// ImplicitBeamsGPU <numPartitions> <numBeamsPerSide> <solverType> <usePreconditioning> <elasticModulus> <dataFolder>
	// solverType: (0) BiCGStab, (1) BiCGStab1, (2) BiCGStab2, (3) MinRes

#ifdef WITH_GLUT
	bool visualize = true;
#endif

	sys.setTimeStep(1e-3, 1e-10);
	sys.setMaxNewtonIterations(20);
	sys.setMaxKrylovIterations(5000);
	sys.setNumPartitions((int)atoi(argv[1]));
	sys.numContactPoints = 30;

	double t_end = 5.0;
	int    precUpdateInterval = -1;
	float  precMaxKrylov = -1;
	int    outputInterval = 100;

	string data_folder;

	sys.fullJacobian = 1;
	double length = 1;
	double r = .02;
	double E = 2e11;
	double rho = 2200;
	double nu = .3;
	int numElementsPerSide = atoi(argv[2]);
	sys.setSolverType((int)atoi(argv[3]));
	sys.setPrecondType(atoi(argv[4]));
	if(atoi(argv[4])) {
	  sys.preconditionerUpdateModulus = precUpdateInterval;
	  sys.preconditionerMaxKrylovIterations = precMaxKrylov;
	}
	E = atof(argv[5]);
	data_folder = argv[6];

	Element element;
	int k = 0;
	// Add elements in x-direction
	for (int j = 0; j < numElementsPerSide+1; j++) {
	  for (int i = 0; i < numElementsPerSide; i++) {
	    element = Element(make_double3(i,0,j));
	    sys.add(&element);
	    k++;
	    if(k%100==0) printf("Elements %d\n",k);
	  }
	}

	printf("%d, %d, %d\n",sys.elements.size(),sys.constraints.size(),12*sys.elements.size()+sys.constraints.size());
	sys.initializeSystem();
	printf("System initialized!\n");
	sys.printSolverParams();
	
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

	stringstream ss_m;
	ss_m << data_folder << "/" << "timing_" << atoi(argv[1]) << "_" << atoi(argv[2]) << "_" << atoi(argv[3]) << "_" << atoi(argv[4]) << "_" << atof(argv[5]) << ".txt";
	string timing_file_name = ss_m.str();
	ofstream ofile(timing_file_name.c_str());
	
	// if you don't want to visualize, then output the data
	int fileIndex = 0;
	while(sys.time < t_end)
	{
		if(sys.getTimeIndex()%outputInterval==0)
		{
			stringstream ss;
			//cout << "Frame: " << fileIndex << endl;
			ss << data_folder << "/" << fileIndex << ".txt";
			sys.writeToFile(ss.str());
			fileIndex++;
		}
		sys.DoTimeStep();
		ofile << sys.time                 << ", "
		      << sys.stepTime             << ", "
		      << sys.stepNewtonIterations << ", "
		      << sys.stepKrylovIterations << ", "
		      << sys.precUpdated          << " ,     ";
		for (size_t i = 0; i < sys.stepNewtonIterations; ++i)
			ofile << sys.spikeSolveTime[i] << ", " << sys.spikeNumIter[i] << ",     ";
		ofile << endl;
	}
	printf("Total time to simulate: %f [s]\n",sys.timeToSimulate);
	ofile.close();

	return 0;
}

