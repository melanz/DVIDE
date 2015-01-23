#include <algorithm>
#include <vector>
#include "include.cuh"
#include "System.cuh"

#include <cusp/io/matrix_market.h>

System::System()
{
  gravity = make_double3(0,-9.81,0);

	// Set default solver parameters
	//setAlpha_HHT(-0.1);
	//setTimeStep(1e-3);
	maxNewtonIterations = 20;

	// spike stuff
	partitions = 1;
	solverOptions.safeFactorization = true;
	solverOptions.trackReordering = true;
	solverOptions.maxNumIterations = 5000;
	//mySpmv = new SpmvFunctor(lhs);
	// m_spmv = new MySpmv(lhs_mass, lhs_phiq, lhsVec);
	preconditionerUpdateModulus = -1; // the preconditioner updates every ___ time steps
	preconditionerMaxKrylovIterations = -1; // the preconditioner updates if Krylov iterations are greater than ____ iterations
	// end spike stuff

	this->timeIndex = 0;
	this->time = 0;
	timeToSimulate = 0;
	simTime = 0;
	fullJacobian = 1;

	wt3.push_back(5.0 / 9.0);
	wt3.push_back(8.0 / 9.0);
	wt3.push_back(5.0 / 9.0);

	pt3.push_back(-sqrt(3.0 / 5.0));
	pt3.push_back(0.0);
	pt3.push_back(sqrt(3.0 / 5.0));

	wt5.push_back((322. - 13. * sqrt(70.)) / 900.);
	wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
	wt5.push_back(128. / 225.);
	wt5.push_back((322. + 13. * sqrt(70.)) / 900.);
	wt5.push_back((322. - 13. * sqrt(70.)) / 900.);

	pt5.push_back(-(sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);
	pt5.push_back(-(sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
	pt5.push_back(0.);
	pt5.push_back((sqrt(5. - 2. * sqrt(10. / 7.))) / 3.);
	pt5.push_back((sqrt(5. + 2. * sqrt(10. / 7.))) / 3.);

	numCollisions = 0;
	numCollisionsSphere = 0;
	numContactPoints = 5;
	coefRestitution = .3;
	frictionCoef = .3;
	fileIndex = 0;

	// set up position files
	char filename1[100];
	char filename2[100];
	char filename3[100];
	sprintf(filename1, "position.dat");
	resultsFile1.open(filename1);
	sprintf(filename2, "energy.dat");
	resultsFile2.open(filename2);
	sprintf(filename3, "reactions.dat");
	resultsFile3.open(filename3);
}

void System::setSolverType(int solverType)
{
	switch(solverType) {
	case 0:
		solverOptions.solverType = spike::BiCGStab;
		break;
	case 1:
		solverOptions.solverType = spike::BiCGStab1;
		break;
	case 2:
		solverOptions.solverType = spike::BiCGStab2;
		break;
	case 3:
		solverOptions.solverType = spike::MINRES;
		break;
	}
}

void System::setPrecondType(int useSpike)
{
	solverOptions.precondType = useSpike ? spike::Spike : spike::None;
}

void System::setAlpha_HHT(double alpha) {
	// should be greater than -.3, usually set to -.1
	alphaHHT = alpha;
	betaHHT = (1 - alphaHHT) * (1 - alphaHHT) * .25;
	gammaHHT = 0.5 - alphaHHT;
}

void System::setTimeStep(double step_size,
                             double precision)
{
	h = step_size;

	// Set tolerance for Newton iteration based on the precision in positions
	// and integration step-size.
	double safety = 1;////0.5;
	tol = safety * precision / (h * h);

	// Set the tolerances for Krylov
	solverOptions.relTol = std::min(0.01 * tol, 1e-6);
	solverOptions.absTol = 1e-10;
}

void System::printSolverParams()
{
	printf("Step size: %e\n", h);
	printf("Newton tolerance: %e\n", tol);
	printf("Max. Newton iterations: %d\n", maxNewtonIterations);
	printf("Krylov relTol: %e  abdTol: %e\n", solverOptions.relTol, solverOptions.absTol);
	printf("Max. Krylov iterations: %d\n", solverOptions.maxNumIterations);
	printf("----------------------------\n");
}

int System::add(Element* element) {
	//add the element
	element->setIdentifier(elements.size());
	this->elements.push_back(*element);

	// update p
	p_h.push_back(element->pos.x);
	p_h.push_back(element->pos.y);
	p_h.push_back(element->pos.z);

  // update v
  v_h.push_back(element->vel.x);
  v_h.push_back(element->vel.y);
  v_h.push_back(element->vel.z);

  // update a
  a_h.push_back(element->acc.x);
  a_h.push_back(element->acc.y);
  a_h.push_back(element->acc.z);

	// update external force vector (gravity)
	f_h.push_back(element->mass * this->gravity.x);
	f_h.push_back(element->mass * this->gravity.y);
	f_h.push_back(element->mass * this->gravity.z);

	for (int i = 0; i < element->numDOF; i++) {
	  massI_h.push_back(i + element->numDOF * (elements.size() - 1));
		massJ_h.push_back(i + element->numDOF * (elements.size() - 1));
		mass_h.push_back(1.0/element->mass);
	}

	return elements.size();
}

int System::initializeDevice() {
	p_d = p_h;
	v_d = v_h;
	a_d = a_h;
	f_d = f_h;

	massI_d = massI_h;
	massJ_d = massJ_h;
	mass_d = mass_h;

	thrust::device_ptr<double> wrapped_device_p(CASTD1(p_d));
	thrust::device_ptr<double> wrapped_device_v(CASTD1(v_d));
	thrust::device_ptr<double> wrapped_device_a(CASTD1(a_d));
	thrust::device_ptr<double> wrapped_device_f(CASTD1(f_d));

	p = DeviceValueArrayView(wrapped_device_p, wrapped_device_p + p_d.size());
	v = DeviceValueArrayView(wrapped_device_v, wrapped_device_v + v_d.size());
	a = DeviceValueArrayView(wrapped_device_a, wrapped_device_a + a_d.size());
	f = DeviceValueArrayView(wrapped_device_f, wrapped_device_f + f_d.size());

	// create mass matrix using cusp library (shouldn't change)
	thrust::device_ptr<int> wrapped_device_I(CASTI1(massI_d));
	DeviceIndexArrayView row_indices = DeviceIndexArrayView(wrapped_device_I, wrapped_device_I + massI_d.size());

	thrust::device_ptr<int> wrapped_device_J(CASTI1(massJ_d));
	DeviceIndexArrayView column_indices = DeviceIndexArrayView(wrapped_device_J, wrapped_device_J + massJ_d.size());

	thrust::device_ptr<double> wrapped_device_V(CASTD1(mass_d));
	DeviceValueArrayView values = DeviceValueArrayView(wrapped_device_V, wrapped_device_V + mass_d.size());

	mass = DeviceView(anew_d.size(), anew_d.size(), mass_d.size(), row_indices, column_indices, values);
	// end create mass matrix

	return 0;
}

int System::initializeSystem() {

	initializeDevice();
//
//	// create and setup the Spike::GPU solver
//	m_spmv = new MySpmv(mass);
//	mySolver = new SpikeSolver(partitions, solverOptions);
//	mySolver->setup(mass);
//
//	bool success = mySolver->solve(*m_spmv, f, a);
//	spike::Stats stats = mySolver->getStats();
//
//	cout << endl
//	     << "Linear problem size:  " << eAll.size() << endl
//	     << "Number partitions:    " << stats.numPartitions << endl
//	     << "Bandwidth after MC64: " << stats.bandwidthMC64 << endl
//	     << "Bandwidth after RCM:  " << stats.bandwidthReorder << endl
//	     << "Bandwidth final:      " << stats.bandwidth << endl
//	     << "nuKf factor:          " << stats.nuKf << endl << endl;
//
//	// Vectors for Spike solver stats
//	spikeSolveTime.resize(maxNewtonIterations);
//	spikeNumIter.resize(maxNewtonIterations);

	return 0;
}

int System::DoTimeStep() {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	cusp::multiply(mass, f, a);
	cusp::blas::axpy(a, v, h);
	cusp::blas::axpy(v, p, h);

  time += h;
  timeIndex++;
  p_h = p_d;

  printf("Time: %f\n",time);

/*
	//System::updateParticleDynamics();
	stepKrylovIterations = 0;
	precUpdated = false;

//	// update q and q_dot for initial guess
//	cusp::blas::axpbypcz(p, v, a, pnew, 1, h, .5 * h * h);
//	cusp::blas::axpby(v, a, vnew, 1, h);
//
//	// Force a preconditioner update if needed
//	if ((preconditionerUpdateModulus > 0) && (timeIndex % preconditionerUpdateModulus == 0)) {
//		mySolver->update(lhs.values);
//		precUpdated = true;
//		printf("Preconditioner updated (step condition)!\n");
//	}
//
//	// Perform Newton iterations
//	int it;
//	for (it = 0; it < maxNewtonIterations; it++) {
//		System::updatePhi();
//		cusp::multiply(phiq, lambda, phiqlam);
//		System::resetLeftHandSideMatrix();
//		cusp::multiply(lhs_mass, anew, eTop); //cusp::multiply(mass,anew,eTop);
//		System::updateInternalForces();
//		cusp::blas::axpbypcz(eTop, fapp, fint, eTop, 1, -1, 1);
//		cusp::blas::axpby(eTop, fext, eTop, 1, -1);
//		cusp::blas::axpy(phiqlam, eTop, 1);
//		cusp::blas::copy(phi, eBottom);

		// SOLVE THE LINEAR SYSTEM USING SPIKE
		//cusp::blas::fill(delta, 0); // very important
		//stencil lhsStencil(anewAll.size(), lhs_mass, lhs_phiq, lhsVec);

		bool success = mySolver->solve(*m_spmv, f, a);
		spike::Stats stats = mySolver->getStats();

		if(!success) {
			printf("**********  DUMP DATA **************\n");

			char filename[100];
			
			sprintf(filename, "./data/lhs%d.mtx", timeIndex);
			cusp::io::write_matrix_market_file(lhs, filename);

			sprintf(filename, "./data/rhs%d.mtx", timeIndex);
			cusp::io::write_matrix_market_file(eAll, filename);

			sprintf(filename, "./data/stats%d.txt", timeIndex);
			ofstream file(filename);
			file << "Code: " << mySolver->getMonitorCode();
			file << "  " << mySolver->getMonitorMessage() << std::endl;
			file << "Number of iterations = " << stats.numIterations << std::endl;
			file << "RHS norm             = " << stats.rhsNorm << std::endl;
			file << "Residual norm        = " << stats.residualNorm << std::endl;
			file << "Rel. residual norm   = " << stats.relResidualNorm << std::endl;
			file.close();

			int code = mySolver->getMonitorCode();
			if (code == -1 || code == -2) {

				//// TODO:  clean this up...

				std::cout << "STOP" << std::endl;
				exit(0);
			}
		}

		spikeSolveTime[0] = stats.timeSolve;
		spikeNumIter[0] = stats.numIterations;
		stepKrylovIterations += stats.numIterations;
		// END SOLVE THE LINEAR SYSTEM

		// update anew
		//cusp::blas::axpy(delta, anewAll, -1);

		// update vnew
		//cusp::blas::axpbypcz(v, a, anew, vnew, 1, h * (1 - gammaHHT), h * gammaHHT);


		// update pnew
		//cusp::blas::axpbypcz(v, a, anew, pnew, h, h * h * .5 * (1 - 2 * betaHHT), h * h * .5 * 2 * betaHHT);


		// Calculate infinity norm of the correction and check for convergence
		//double delta_nrm = cusp::blas::nrmmax(delta);

		printf("         Krylov solver: %8.2f ms    %.2f iterations     ||delta||_inf = %e\n",
			stats.timeSolve, stats.numIterations, 0);

		//if (delta_nrm <= tol)
		//	break;
	//}

	// Number of Newton iterations and average number of Krylov iterations
	stepNewtonIterations = 1;
	float avgKrylov = stepKrylovIterations / stepNewtonIterations;

//	// If the average number of Krylov iterations per Newton iteration exceeds the specified limit,
//	// force a preconditioner update.
//	if ((preconditionerMaxKrylovIterations > 0) && (avgKrylov > preconditionerMaxKrylovIterations)) {
//		System::updateInternalForces();
//		mySolver->update(lhs.values);
//		precUpdated = true;
//		printf("Preconditioner updated! (krylov condition)\n");
//	}

//	cusp::copy(anew, a);
//	cusp::copy(vnew, v);
//	cusp::copy(pnew, p);
*/
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
/*
	stepTime = elapsedTime;
	timeToSimulate += elapsedTime / 1000.0;

	p_h = p_d;

	time += h;
	timeIndex++;

	printf("%f, Elapsed time = %8.2f ms, Newton = %2d, Ave. Krylov Per Newton = %.2f\n",
	       time, elapsedTime, stepNewtonIterations, avgKrylov);


*/
	return 0;
}

float3 System::getXYZPosition(int elementIndex, double xi) {
//	double a = elements[elementIndex].getLength_l();
//	double* p = CASTD1(p_h);
//	p = &p[12 * elementIndex];
	float3 pos;
//
//	pos.x = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[0]
//			+ a * (xi - 2 * xi * xi + pow(xi, 3)) * p[3]
//			+ (3 * xi * xi - 2 * pow(xi, 3)) * p[6]
//			+ a * (-xi * xi + pow(xi, 3)) * p[9];
//	pos.y = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[1]
//			+ a * (xi - 2 * xi * xi + pow(xi, 3)) * p[4]
//			+ (3 * xi * xi - 2 * pow(xi, 3)) * p[7]
//			+ a * (-xi * xi + pow(xi, 3)) * p[10];
//	pos.z = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[2]
//			+ a * (xi - 2 * xi * xi + pow(xi, 3)) * p[5]
//			+ (3 * xi * xi - 2 * pow(xi, 3)) * p[8]
//			+ a * (-xi * xi + pow(xi, 3)) * p[11];

	return pos;
}

float3 System::getXYZVelocity(int elementIndex, double xi) {
//	double a = elements[elementIndex].getLength_l();
//	double* p = CASTD1(v_h);
//	p = &p[12 * elementIndex];
	float3 pos;
//
//	pos.x = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[0]
//			+ a * (xi - 2 * xi * xi + pow(xi, 3)) * p[3]
//			+ (3 * xi * xi - 2 * pow(xi, 3)) * p[6]
//			+ a * (-xi * xi + pow(xi, 3)) * p[9];
//	pos.y = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[1]
//			+ a * (xi - 2 * xi * xi + pow(xi, 3)) * p[4]
//			+ (3 * xi * xi - 2 * pow(xi, 3)) * p[7]
//			+ a * (-xi * xi + pow(xi, 3)) * p[10];
//	pos.z = (1 - 3 * xi * xi + 2 * pow(xi, 3)) * p[2]
//			+ a * (xi - 2 * xi * xi + pow(xi, 3)) * p[5]
//			+ (3 * xi * xi - 2 * pow(xi, 3)) * p[8]
//			+ a * (-xi * xi + pow(xi, 3)) * p[11];

	return pos;
}

int System::saveLHS() {
	posFile.open("../lhs.dat");
	posFile << "symmetric" << endl;
	posFile << anew_h.size() << " " << anew_h.size() << " " << lhsI_h.size()
			<< endl;
	for (int i = 0; i < lhsI_h.size(); i++) {
		posFile << lhsI_h[i] << " " << lhsJ_h[i] << " " << lhs_h[i] << endl;
	}
	posFile.close();

	return 0;
}

int System::writeToFile(string fileName) {
//	posFile.open(fileName.c_str());
//	p_h = p_d;
//	double* posAll = CASTD1(p_h);
//	double* pos;
//	double l;
//	double r;
//	posFile << elements.size() << "," << endl;
//	for (int i = 0; i < elements.size(); i++) {
//		l = elements[i].getLength_l();
//		r = elements[i].getRadius();
//		pos = &posAll[12 * i];
//		posFile << r << "," << l;
//		for (int i = 0; i < 12; i++)
//			posFile << "," << pos[i];
//		posFile << "," << endl;
//	}
//	posFile.close();

	return 0;
}
