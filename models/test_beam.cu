#include "include.cuh"
#include <sys/stat.h>
#include <errno.h>
#include "System.cuh"
#include "Body.cuh"
#include "Beam.cuh"
#include "APGD.cuh"
#include "PDIP.cuh"
#include "TPAS.cuh"
#include "JKIP.cuh"

bool updateDraw = 1;
bool wireFrame = 1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;
std::string outDir = "../TEST_BEAM/";
std::string povrayDir = outDir + "POVRAY/";
thrust::host_vector<double> p0_h;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(0,1,-3),camreal3(0,1,0),camreal3(0,1,0),.01);

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

		glutSwapBuffers();
	}
}

void renderSceneAll(){
	if(OGL){
		//if(sys->timeIndex%10==0)
		drawAll();
    //std::stringstream dataFileStream;
    //dataFileStream << povrayDir << "data_" << sys->timeIndex << ".dat";
    //sys->exportSystem(dataFileStream.str());
		sys->DoTimeStep();
//		cusp::print(sys->fElastic);
//		cin.get();

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

  double t_end = 8.0;
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
  double tolerance = 1e-4;
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
  outDirStream << "../TEST_BEAM_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << "/";
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
    dynamic_cast<APGD*>(sys->solver)->setAntiRelaxation(true);
    dynamic_cast<APGD*>(sys->solver)->setWarmStarting(true);
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
    dynamic_cast<TPAS*>(sys->solver)->alpha = alpha;
    dynamic_cast<TPAS*>(sys->solver)->beta = beta;
    dynamic_cast<TPAS*>(sys->solver)->mu_pdip = mu_pdip;
  }
  if(solverTypeQOCC==4) {
    dynamic_cast<JKIP*>(sys->solver)->setPrecondType(precondType);
    dynamic_cast<JKIP*>(sys->solver)->setSolverType(solverType);
    dynamic_cast<JKIP*>(sys->solver)->setNumPartitions(numPartitions);
    dynamic_cast<JKIP*>(sys->solver)->careful = true;
  }

  //sys->solver->maxIterations = 40;

  double radius = 0.4;

//  // Bottom
//  Body* groundPtr = new Body(make_double3(0,-radius+0.1,0));
//  groundPtr->setBodyFixed(true);
//  groundPtr->setGeometry(make_double3(1,radius,1));
//  sys->add(groundPtr);
//
//  // Pivot
//  Body* pivotPtr = new Body(make_double3(0,1,0));
//  pivotPtr->setBodyFixed(true);
//  pivotPtr->setGeometry(make_double3(0.025,0,0));
//  pivotPtr->setCollisionFamily(1);
//  sys->add(pivotPtr);
//
//  Beam* beamPtr;
//  double length = 2.0/static_cast<double>(numElementsPerSide);
//  for(int i=0; i<numElementsPerSide;i++) {
//    beamPtr = new Beam(make_double3(((double)i)*length,1,0),make_double3(((double)i+1.0)*length,1,0));
//    beamPtr->setRadius(0.02);
//    beamPtr->setCollisionFamily(1);
//    beamPtr->setElasticModulus(2.0e7);
//    sys->add(beamPtr);
//
//    if(i==0){
//      sys->addBilateralConstraintDOF(3,3*sys->bodies.size());
//      sys->addBilateralConstraintDOF(4,3*sys->bodies.size()+1);
//      sys->addBilateralConstraintDOF(5,3*sys->bodies.size()+2);
//    } else {
//      sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*(i-1)+6, 3*sys->bodies.size()+12*i);
//      sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*(i-1)+7, 3*sys->bodies.size()+12*i+1);
//      sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*(i-1)+8, 3*sys->bodies.size()+12*i+2);
//      sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*(i-1)+9, 3*sys->bodies.size()+12*i+3);
//      sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*(i-1)+10,3*sys->bodies.size()+12*i+4);
//      sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*(i-1)+11,3*sys->bodies.size()+12*i+5);
//    }
//  }

//  double rMin = 0.08;
//  double rMax = 0.08;
//  double density = 2600;
//  double L = 4.0;
//  double W = 2.0;
//  double H = 2.0;
//  double th = 0.1;
//
//  // Bottom
//  Body* groundPtr = new Body(make_double3(0,-th,0));
//  groundPtr->setBodyFixed(true);
//  groundPtr->setGeometry(make_double3(0.5*L+th,th,0.5*W+th));
//  sys->add(groundPtr);
//
//  // Left
//  Body* leftPtr = new Body(make_double3(-0.5*L-2*th,0.5*H+th,0));
//  leftPtr->setBodyFixed(true);
//  leftPtr->setGeometry(make_double3(th,0.5*H+th,0.5*W+th));
//  sys->add(leftPtr);
//
//  // Right
//  Body* rightPtr = new Body(make_double3(0.5*L+2*th,0.5*H+th,0));
//  rightPtr->setBodyFixed(true);
//  rightPtr->setGeometry(make_double3(th,0.5*H+th,0.5*W+th));
//  sys->add(rightPtr);
//
//  // Back
//  Body* backPtr = new Body(make_double3(0,0.5*H+th,-0.5*W-2*th));
//  backPtr->setBodyFixed(true);
//  backPtr->setGeometry(make_double3(0.5*L+th,0.5*H+th,th));
//  sys->add(backPtr);
//
//  // Front
//  Body* frontPtr = new Body(make_double3(0,0.5*H+th,0.5*W+2*th));
//  frontPtr->setBodyFixed(true);
//  frontPtr->setGeometry(make_double3(0.5*L+th,0.5*H+th,th));
//  sys->add(frontPtr);
//
//  Body* bodyPtr;
//  double wiggle = 0.02;//0.003;//0.1;
//  double numElementsPerSideX = L/(2.0*rMax+2.0*wiggle);
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
//        bodyPtr = new Body(make_double3((rMax+wiggle)*(2.0*((double)i)+1.0)-0.5*L+xWig,(rMax+wiggle)*(2.0*((double)j)+1.0)+yWig,(rMax+wiggle)*(2.0*((double)k)+1.0)-0.5*W+zWig));
//        double rRand = getRandomNumber(rMin, rMax);
//        bodyPtr->setMass(4.0*rRand*rRand*rRand*3.1415/3.0*density);
//        bodyPtr->setGeometry(make_double3(rRand,0,0));
//        //if(j==0) bodyPtr->setBodyFixed(true);
//        //numBodies = sys->add(bodyPtr);
//
//        if(numBodies%1000==0) printf("Bodies %d\n",numBodies);
//      }
//    }
//  }
//
//  double density_tire = 1100.0;
//  double radius_tire = 0.2;
//  double height_tire = 3.0;
//  double length = 2.0*PI*1.0*0.25;
//
//  double3 pos = make_double3(0,height_tire,0);
//  double3 node0 = make_double3(0,0,0)+pos;
//  double3 dnode0 = make_double3(1,0,0);
//  double3 node1 = make_double3(1,1,0)+pos;
//  double3 dnode1 = make_double3(0,1,0);
//  Beam* beam0 = new Beam(node0,dnode0,node1,dnode1,length);
//  beam0->setRadius(radius_tire);
//  beam0->setCollisionFamily(1);
//  beam0->setDensity(density_tire);
//  sys->add(beam0);
//
//  node0 = node1;
//  dnode0 = dnode1;
//  node1 = make_double3(0,2,0)+pos;
//  dnode1 = make_double3(-1,0,0);
//  Beam* beam1 = new Beam(node0,dnode0,node1,dnode1,length);
//  beam1->setCollisionFamily(1);
//  beam1->setRadius(radius_tire);
//  beam1->setDensity(density_tire);
//  sys->add(beam1);
//
//  node0 = node1;
//  dnode0 = dnode1;
//  node1 = make_double3(-1,1,0)+pos;
//  dnode1 = make_double3(0,-1,0);
//  Beam* beam2 = new Beam(node0,dnode0,node1,dnode1,length);
//  beam2->setCollisionFamily(1);
//  beam2->setRadius(radius_tire);
//  beam2->setDensity(density_tire);
//  sys->add(beam2);
//
//  node0 = node1;
//  dnode0 = dnode1;
//  node1 = make_double3(0,0,0)+pos;
//  dnode1 = make_double3(1,0,0);
//  Beam* beam3 = new Beam(node0,dnode0,node1,dnode1,length);
//  beam3->setCollisionFamily(1);
//  beam3->setRadius(radius_tire);
//  beam3->setDensity(density_tire);
//  sys->add(beam3);
//
//  for(int i=0;i<3;i++) {
//    sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*i+6, 3*sys->bodies.size()+12*(i+1));
//    sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*i+7, 3*sys->bodies.size()+12*(i+1)+1);
//    sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*i+8, 3*sys->bodies.size()+12*(i+1)+2);
//    sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*i+9, 3*sys->bodies.size()+12*(i+1)+3);
//    sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*i+10,3*sys->bodies.size()+12*(i+1)+4);
//    sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*i+11,3*sys->bodies.size()+12*(i+1)+5);
//  }
//  sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*3+6, 3*sys->bodies.size()+12*0);
//  sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*3+7, 3*sys->bodies.size()+12*0+1);
//  sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*3+8, 3*sys->bodies.size()+12*0+2);
//  sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*3+9, 3*sys->bodies.size()+12*0+3);
//  sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*3+10,3*sys->bodies.size()+12*0+4);
//  sys->addBilateralConstraintDOF(3*sys->bodies.size()+12*3+11,3*sys->bodies.size()+12*0+5);



  // Bottom
  Body* groundPtr = new Body(make_double3(0,-radius,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setGeometry(make_double3(0.5*numElementsPerSide,radius,0.5*numElementsPerSide));
  sys->add(groundPtr);

  // Left
  Body* leftPtr = new Body(make_double3(-0.5*numElementsPerSide-radius,0.5*numElementsPerSide+radius,0));
  leftPtr->setBodyFixed(true);
  leftPtr->setGeometry(make_double3(radius,0.5*numElementsPerSide+radius,0.5*numElementsPerSide));
  sys->add(leftPtr);

  // Right
  Body* rightPtr = new Body(make_double3(0.5*numElementsPerSide+radius,0.5*numElementsPerSide+radius,0));
  rightPtr->setBodyFixed(true);
  rightPtr->setGeometry(make_double3(radius,0.5*numElementsPerSide+radius,0.5*numElementsPerSide));
  sys->add(rightPtr);

  // Back
  Body* backPtr = new Body(make_double3(0,0.5*numElementsPerSide+radius,-0.5*numElementsPerSide-radius));
  backPtr->setBodyFixed(true);
  backPtr->setGeometry(make_double3(0.5*numElementsPerSide,0.5*numElementsPerSide+radius,radius));
  sys->add(backPtr);

  // Front
  Body* frontPtr = new Body(make_double3(0,0.5*numElementsPerSide+radius,0.5*numElementsPerSide+radius));
  frontPtr->setBodyFixed(true);
  frontPtr->setGeometry(make_double3(0.5*numElementsPerSide,0.5*numElementsPerSide+radius,radius));
  sys->add(frontPtr);

  Beam* beamPtr;
  Body* bodyPtr;
  int numBodies = 0;
  double wiggle = 0.1;
  // Add elements in x-direction
  for (int j = 0; j < 4*numElementsPerSide; j++) {
    for (int i = 0; i < numElementsPerSide; i++) {
      for (int k = 0; k < numElementsPerSide; k++) {
        double check = 0;//getRandomNumber(-1, 1);
        double xWig = getRandomNumber(-wiggle, wiggle);
        double yWig = 0;//getRandomNumber(-wiggle, wiggle);
        double zWig = getRandomNumber(-wiggle, wiggle);
        double length = 2*radius-2*wiggle;
        double3 center = make_double3(i-0.5*numElementsPerSide+radius+wiggle + xWig,j+wiggle+radius+yWig,k-0.5*numElementsPerSide+radius+wiggle+zWig);
        double3 dir = normalize(make_double3( getRandomNumber(-1, 1), getRandomNumber(-1, 1), getRandomNumber(-1, 1)));
        if(check<=0) {
          beamPtr = new Beam(center-0.5*length*dir,center+0.5*length*dir);
          beamPtr->setRadius(length/15);//0.1);
          beamPtr->setElasticModulus(2.0e5);
          beamPtr->setNumContactPoints(30);
          numBodies = sys->add(beamPtr);
        } else {
          bodyPtr = new Body(center);
          bodyPtr->setGeometry(make_double3(radius,0,0));
          bodyPtr->setMass(4.0/3.0*PI*radius*radius*radius*7200.0);
          numBodies = sys->add(bodyPtr);
        }

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
  std::stringstream statsFileStream;
  statsFileStream << outDir << "statsBeam_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << ".dat";
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
		for(int i=0; i<6; i++) {
		  weight += sys->f_contact_h[3*i+1];
		}
		cout << "  Weight: " << weight << endl;

		int numKrylovIter = 0;
		if(solverTypeQOCC==2) numKrylovIter = dynamic_cast<PDIP*>(sys->solver)->totalKrylovIterations;
		if(solverTypeQOCC==3) numKrylovIter = dynamic_cast<TPAS*>(sys->solver)->totalKrylovIterations;
		if(solverTypeQOCC==4) numKrylovIter = dynamic_cast<JKIP*>(sys->solver)->totalKrylovIterations;
		statStream << sys->time << ", " << sys->bodies.size() << ", " << sys->elapsedTime << ", " << sys->totalGPUMemoryUsed << ", " << sys->solver->iterations << ", " << sys->collisionDetector->numCollisions << ", " << weight << ", " << numKrylovIter << endl;

	}
	sys->exportMatrices(outDir.c_str());
  std::stringstream collisionFileStream;
  collisionFileStream << outDir << "collisionData.dat";
	sys->collisionDetector->exportSystem(collisionFileStream.str().c_str());

	return 0;
}

