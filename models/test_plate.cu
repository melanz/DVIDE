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
double desiredVelocity = 0.1;

// Create the system (placed outside of main so it is available to the OpenGL code)
System* sys;
std::string outDir = "../TEST_PLATE/";
std::string povrayDir = outDir + "POVRAY/";
thrust::host_vector<double> p0_h;

#ifdef WITH_GLUT
OpenGLCamera oglcamera(camreal3(0,0,5),camreal3(0,0,0),camreal3(0,1,0),.01);

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
  double tolerance = 1e-3;
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
  outDirStream << "../TEST_PLATE_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << "/";
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
  //sys->gravity = make_double3(0,0,0);

  int numDiv = 5;
  double radianInc = 2.0*PI/((double) numDiv);
  double EM = 2.e7;
  double rho = 7810.0;
  double th = .01;
  double R = .2;
  double nu = 0;
  double fillet = .04;
  double beltWidth = .1;
  double B = 1.5*.5*PI*beltWidth;
  double L = 2*PI*(R+1.4*0.33*beltWidth)/((double) numDiv);
  int numContacts = 12;

  double r_rim = 0.01;
  double EM_rim = 2.e9;
  double rho_rim = 7200.0;

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
  double3 center = make_double3(0,0,0);
  double3 center_back = make_double3(0,0,beltWidth);

  // Add tire elements
  Plate* plate;
  Beam* beam;
  for(int i=0;i<numDiv-1;i++)
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

    beam = new Beam(nodes[2*i+1],dxis[2*i+1],
        nodes[2*i+3],dxis[2*i+3],L);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    //sys->add(beam);

    beam = new Beam(nodes[2*i],dxis[2*i],
        nodes[2*i+2],dxis[2*i+2],L);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    //sys->add(beam);

    beam = new Beam(nodes[2*i+1],dxis[2*i+1],
        nodes[2*i+3],dxis[2*i+3],L);
    beam->setRadius(r_rim);
    beam->setElasticModulus(EM_rim);
    beam->setDensity(rho_rim);
    beam->setCollisionFamily(1);
    //sys->add(beam);
  }
  plate = new Plate(L,B,nodes[2*(numDiv-1)],dxis[2*(numDiv-1)],detas[2*(numDiv-1)],
      nodes[0],dxis[0],detas[0],
      nodes[1],dxis[1],detas[1],
      nodes[2*(numDiv-1)+1],dxis[2*(numDiv-1)+1],detas[2*(numDiv-1)+1]);
  plate->setThickness(th);
  plate->setElasticModulus(EM);
  plate->setPoissonRatio(nu);
  plate->setDensity(rho);
  plate->setCollisionFamily(1);
  plate->setNumContactPoints(numContacts);
  sys->add(plate);

  beam = new Beam(center, detas[2*(numDiv-1)],
      nodes[2*(numDiv-1)],detas[2*(numDiv-1)],R);
  beam->setRadius(r_rim);
  beam->setElasticModulus(EM_rim);
  beam->setDensity(rho_rim);
  beam->setCollisionFamily(1);
  sys->add(beam);

  beam = new Beam(nodes[2*(numDiv-1)+1], detas[2*(numDiv-1)+1],
      center_back,detas[2*(numDiv-1)+1],R);
  beam->setRadius(r_rim);
  beam->setElasticModulus(EM_rim);
  beam->setDensity(rho_rim);
  beam->setCollisionFamily(1);
  sys->add(beam);

  beam = new Beam(nodes[2*(numDiv-1)],dxis[2*(numDiv-1)],
      nodes[0],dxis[0],L);
  beam->setRadius(r_rim);
  beam->setElasticModulus(EM_rim);
  beam->setDensity(rho_rim);
  beam->setCollisionFamily(1);
  //sys->add(beam);

  beam = new Beam(nodes[2*(numDiv-1)+1],dxis[2*(numDiv-1)+1],
      nodes[1],dxis[1],L);
  beam->setRadius(r_rim);
  beam->setElasticModulus(EM_rim);
  beam->setDensity(rho_rim);
  beam->setCollisionFamily(1);
  //sys->add(beam);

  // Add hub
  Body* hub = new Body(center);
  //hub->setBodyFixed(true);
  hub->setGeometry(make_double3(r_rim,0,0));
  hub->setCollisionFamily(1);
  sys->add(hub);

  // Add hub
  Body* hub_back = new Body(center_back);
  //hub_back->setBodyFixed(true);
  hub_back->setGeometry(make_double3(r_rim,0,0));
  hub_back->setCollisionFamily(1);
  sys->add(hub_back);

  // Add ground
  Body* groundPtr = new Body(make_double3(0,-R-beltWidth-th,0));
  groundPtr->setBodyFixed(true);
  groundPtr->setCollisionFamily(2);
  groundPtr->setGeometry(make_double3(10,th,10));
  sys->add(groundPtr);

  // Add bump
  Body* bump = new Body(make_double3(2,-R-beltWidth-0.9,0));
  bump->setBodyFixed(true);
  bump->setCollisionFamily(2);
  bump->setGeometry(make_double3(1,1,10));
  sys->add(bump);


  // Add bilateral constraints
  for(int i=0;i<numDiv-1;i++)
  {
    int offsetA = 3*sys->bodies.size()+12*sys->beams.size()+36*i;
    int offsetB = 3*sys->bodies.size()+12*sys->beams.size()+36*(i+1);

    int offsetBeamA = 3*sys->bodies.size()+12*(2*i);
    int offsetBeamB = 3*sys->bodies.size()+12*(2*i+1);

    sys->addBilateralConstraintDOF(offsetA+9*1, offsetB+9*0);
    sys->addBilateralConstraintDOF(offsetA+9*1+1, offsetB+9*0+1);
    sys->addBilateralConstraintDOF(offsetA+9*1+2, offsetB+9*0+2);
    sys->addBilateralConstraintDOF(offsetA+9*1+3, offsetB+9*0+3);
    sys->addBilateralConstraintDOF(offsetA+9*1+4, offsetB+9*0+4);
    sys->addBilateralConstraintDOF(offsetA+9*1+5, offsetB+9*0+5);
    sys->addBilateralConstraintDOF(offsetA+9*1+6, offsetB+9*0+6);
    sys->addBilateralConstraintDOF(offsetA+9*1+7, offsetB+9*0+7);
    sys->addBilateralConstraintDOF(offsetA+9*1+8, offsetB+9*0+8);

    sys->addBilateralConstraintDOF(offsetA+9*2, offsetB+9*3);
    sys->addBilateralConstraintDOF(offsetA+9*2+1, offsetB+9*3+1);
    sys->addBilateralConstraintDOF(offsetA+9*2+2, offsetB+9*3+2);
    sys->addBilateralConstraintDOF(offsetA+9*2+3, offsetB+9*3+3);
    sys->addBilateralConstraintDOF(offsetA+9*2+4, offsetB+9*3+4);
    sys->addBilateralConstraintDOF(offsetA+9*2+5, offsetB+9*3+5);
    sys->addBilateralConstraintDOF(offsetA+9*2+6, offsetB+9*3+6);
    sys->addBilateralConstraintDOF(offsetA+9*2+7, offsetB+9*3+7);
    sys->addBilateralConstraintDOF(offsetA+9*2+8, offsetB+9*3+8);

    // node 0 of plate A is fixed to node 1 of beam A
    sys->addBilateralConstraintDOF(offsetA+9*0, offsetBeamA+6*1);
    sys->addBilateralConstraintDOF(offsetA+9*0+1, offsetBeamA+6*1+1);
    sys->addBilateralConstraintDOF(offsetA+9*0+2, offsetBeamA+6*1+2);

    // node 3 of plate A is fixed to node 0 of beam B
    sys->addBilateralConstraintDOF(offsetA+9*3, offsetBeamB+6*0);
    sys->addBilateralConstraintDOF(offsetA+9*3+1, offsetBeamB+6*0+1);
    sys->addBilateralConstraintDOF(offsetA+9*3+2, offsetBeamB+6*0+2);

    // body 0 is fixed to node 0 of beam A
    sys->addBilateralConstraintDOF(3*0, offsetBeamA+6*0);
    sys->addBilateralConstraintDOF(3*0+1, offsetBeamA+6*0+1);
    sys->addBilateralConstraintDOF(3*0+2, offsetBeamA+6*0+2);

    // body 1 is fixed to node 1 of beam B
    sys->addBilateralConstraintDOF(3*1, offsetBeamB+6*1);
    sys->addBilateralConstraintDOF(3*1+1, offsetBeamB+6*1+1);
    sys->addBilateralConstraintDOF(3*1+2, offsetBeamB+6*1+2);
  }
  int offsetA = 3*sys->bodies.size()+12*sys->beams.size()+36*(numDiv-1);
  int offsetB = 3*sys->bodies.size()+12*sys->beams.size()+36*0;

  int offsetBeamA = 3*sys->bodies.size()+12*(2*(numDiv-1));
  int offsetBeamB = 3*sys->bodies.size()+12*(2*(numDiv-1)+1);

  sys->addBilateralConstraintDOF(offsetA+9*1, offsetB+9*0);
  sys->addBilateralConstraintDOF(offsetA+9*1+1, offsetB+9*0+1);
  sys->addBilateralConstraintDOF(offsetA+9*1+2, offsetB+9*0+2);
  sys->addBilateralConstraintDOF(offsetA+9*1+3, offsetB+9*0+3);
  sys->addBilateralConstraintDOF(offsetA+9*1+4, offsetB+9*0+4);
  sys->addBilateralConstraintDOF(offsetA+9*1+5, offsetB+9*0+5);
  sys->addBilateralConstraintDOF(offsetA+9*1+6, offsetB+9*0+6);
  sys->addBilateralConstraintDOF(offsetA+9*1+7, offsetB+9*0+7);
  sys->addBilateralConstraintDOF(offsetA+9*1+8, offsetB+9*0+8);

  sys->addBilateralConstraintDOF(offsetA+9*2, offsetB+9*3);
  sys->addBilateralConstraintDOF(offsetA+9*2+1, offsetB+9*3+1);
  sys->addBilateralConstraintDOF(offsetA+9*2+2, offsetB+9*3+2);
  sys->addBilateralConstraintDOF(offsetA+9*2+3, offsetB+9*3+3);
  sys->addBilateralConstraintDOF(offsetA+9*2+4, offsetB+9*3+4);
  sys->addBilateralConstraintDOF(offsetA+9*2+5, offsetB+9*3+5);
  sys->addBilateralConstraintDOF(offsetA+9*2+6, offsetB+9*3+6);
  sys->addBilateralConstraintDOF(offsetA+9*2+7, offsetB+9*3+7);
  sys->addBilateralConstraintDOF(offsetA+9*2+8, offsetB+9*3+8);

  // node 0 of plate A is fixed to node 1 of beam A
  sys->addBilateralConstraintDOF(offsetA+9*0, offsetBeamA+6*1);
  sys->addBilateralConstraintDOF(offsetA+9*0+1, offsetBeamA+6*1+1);
  sys->addBilateralConstraintDOF(offsetA+9*0+2, offsetBeamA+6*1+2);

  // node 3 of plate A is fixed to node 0 of beam B
  sys->addBilateralConstraintDOF(offsetA+9*3, offsetBeamB+6*0);
  sys->addBilateralConstraintDOF(offsetA+9*3+1, offsetBeamB+6*0+1);
  sys->addBilateralConstraintDOF(offsetA+9*3+2, offsetBeamB+6*0+2);

  // body 0 is fixed to node 0 of beam A
  sys->addBilateralConstraintDOF(3*0, offsetBeamA+6*0);
  sys->addBilateralConstraintDOF(3*0+1, offsetBeamA+6*0+1);
  sys->addBilateralConstraintDOF(3*0+2, offsetBeamA+6*0+2);

  // body 1 is fixed to node 1 of beam B
  sys->addBilateralConstraintDOF(3*1, offsetBeamB+6*1);
  sys->addBilateralConstraintDOF(3*1+1, offsetBeamB+6*1+1);
  sys->addBilateralConstraintDOF(3*1+2, offsetBeamB+6*1+2);


//  Plate* plate = new Plate();
//  plate->setCollisionFamily(1);
//  sys->add(plate);
//
//  // Bottom
//  Body* groundPtr = new Body(make_double3(.5,-3,.5));
//  groundPtr->setBodyFixed(true);
//  //groundPtr->setGeometry(make_double3(.03,0,0));
//  groundPtr->setGeometry(make_double3(2,0,0));
//  //groundPtr->setCollisionFamily(1);
//  sys->add(groundPtr);
//
////  sys->addBilateralConstraintDOF(0, 3);
////  sys->addBilateralConstraintDOF(1, 4);
////  sys->addBilateralConstraintDOF(2, 5);

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
  statsFileStream << outDir << "statsPlate_n" << numElementsPerSide << "_h" << hh << "_tol" << tolerance << "_sol" << solverTypeQOCC << ".dat";
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
        sys->v_h[3*i+2] = 0;
      }
    }
    else {
      for(int i=0;i<1;i++) {
        sys->v_h[3*i] = 0;
        //sys->v_h[3*i+1] = 0;
        sys->v_h[3*i+2] = 0;
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

