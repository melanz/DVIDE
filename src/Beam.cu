#include <algorithm>
#include <vector>
#include "include.cuh"
#include "Beam.cuh"
#include "System.cuh"

int Beam::addBeam(int j)
{
  setIdentifier(sys->bodies.size()+j); // Indicates the number that the Beam was added
  setIndex(sys->p_h.size()); // Indicates the Beam's location in the position array

  // Push Beam's location to global library
  sys->indices_h.push_back(sys->p_h.size());

  // update p
  sys->p_h.push_back(p_n0.x);
  sys->p_h.push_back(p_n0.y);
  sys->p_h.push_back(p_n0.z);
  sys->p_h.push_back(p_dn0.x);
  sys->p_h.push_back(p_dn0.y);
  sys->p_h.push_back(p_dn0.z);
  sys->p_h.push_back(p_n1.x);
  sys->p_h.push_back(p_n1.y);
  sys->p_h.push_back(p_n1.z);
  sys->p_h.push_back(p_dn1.x);
  sys->p_h.push_back(p_dn1.y);
  sys->p_h.push_back(p_dn1.z);

  // update v
  sys->v_h.push_back(v_n0.x);
  sys->v_h.push_back(v_n0.y);
  sys->v_h.push_back(v_n0.z);
  sys->v_h.push_back(v_dn0.x);
  sys->v_h.push_back(v_dn0.y);
  sys->v_h.push_back(v_dn0.z);
  sys->v_h.push_back(v_n1.x);
  sys->v_h.push_back(v_n1.y);
  sys->v_h.push_back(v_n1.z);
  sys->v_h.push_back(v_dn1.x);
  sys->v_h.push_back(v_dn1.y);
  sys->v_h.push_back(v_dn1.z);

  // update a
  sys->a_h.push_back(a_n0.x);
  sys->a_h.push_back(a_n0.y);
  sys->a_h.push_back(a_n0.z);
  sys->a_h.push_back(a_dn0.x);
  sys->a_h.push_back(a_dn0.y);
  sys->a_h.push_back(a_dn0.z);
  sys->a_h.push_back(a_n1.x);
  sys->a_h.push_back(a_n1.y);
  sys->a_h.push_back(a_n1.z);
  sys->a_h.push_back(a_dn1.x);
  sys->a_h.push_back(a_dn1.y);
  sys->a_h.push_back(a_dn1.z);

  // update external force vector (gravity)
  double rho = getDensity();
  double r = contactGeometry.x;
  double A = PI*r*r;
  double l = contactGeometry.y;
  sys->f_h.push_back(rho * A * l * sys->gravity.x / 0.2e1);
  sys->f_h.push_back(rho * A * l * sys->gravity.y / 0.2e1);
  sys->f_h.push_back(rho * A * l * sys->gravity.z / 0.2e1);
  sys->f_h.push_back(rho * A * l * l * sys->gravity.x / 0.12e2);
  sys->f_h.push_back(rho * A * l * l * sys->gravity.y / 0.12e2);
  sys->f_h.push_back(rho * A * l * l * sys->gravity.z / 0.12e2);
  sys->f_h.push_back(rho * A * l * sys->gravity.x / 0.2e1);
  sys->f_h.push_back(rho * A * l * sys->gravity.y / 0.2e1);
  sys->f_h.push_back(rho * A * l * sys->gravity.z / 0.2e1);
  sys->f_h.push_back(-rho * A * l * l * sys->gravity.x / 0.12e2);
  sys->f_h.push_back(-rho * A * l * l * sys->gravity.y / 0.12e2);
  sys->f_h.push_back(-rho * A * l * l * sys->gravity.z / 0.12e2);

  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);

  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);

  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);

  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);

  sys->r_h.push_back(0);
  sys->r_h.push_back(0);
  sys->r_h.push_back(0);
  sys->r_h.push_back(0);
  sys->r_h.push_back(0);
  sys->r_h.push_back(0);

  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);

  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);
  sys->strainDerivative_h.push_back(0);

  sys->strain_h.push_back(0);

  sys->Sx_h.push_back(0);
  sys->Sx_h.push_back(0);
  sys->Sx_h.push_back(0);
  sys->Sx_h.push_back(0);

  sys->Sxx_h.push_back(0);
  sys->Sxx_h.push_back(0);
  sys->Sxx_h.push_back(0);
  sys->Sxx_h.push_back(0);

  // update the sys->mass matrix
  int offset = j*12+3*sys->bodies.size();
  sys->massI_h.push_back(0+offset);
  sys->massJ_h.push_back(0+offset);
  sys->mass_h.push_back(16.0/(A*rho));
  sys->massI_h.push_back(0+offset);
  sys->massJ_h.push_back(3+offset);
  sys->mass_h.push_back(-120.0/(A*l*rho));
  sys->massI_h.push_back(0+offset);
  sys->massJ_h.push_back(6+offset);
  sys->mass_h.push_back(-4.0/(A*rho));
  sys->massI_h.push_back(0+offset);
  sys->massJ_h.push_back(9+offset);
  sys->mass_h.push_back(-60.0/(A*l*rho));
  sys->massI_h.push_back(1+offset);
  sys->massJ_h.push_back(1+offset);
  sys->mass_h.push_back(16.0/(A*rho));
  sys->massI_h.push_back(1+offset);
  sys->massJ_h.push_back(4+offset);
  sys->mass_h.push_back(-120.0/(A*l*rho));
  sys->massI_h.push_back(1+offset);
  sys->massJ_h.push_back(7+offset);
  sys->mass_h.push_back(-4.0/(A*rho));
  sys->massI_h.push_back(1+offset);
  sys->massJ_h.push_back(10+offset);
  sys->mass_h.push_back(-60.0/(A*l*rho));
  sys->massI_h.push_back(2+offset);
  sys->massJ_h.push_back(2+offset);
  sys->mass_h.push_back(16.0/(A*rho));
  sys->massI_h.push_back(2+offset);
  sys->massJ_h.push_back(5+offset);
  sys->mass_h.push_back(-120.0/(A*l*rho));
  sys->massI_h.push_back(2+offset);
  sys->massJ_h.push_back(8+offset);
  sys->mass_h.push_back(-4.0/(A*rho));
  sys->massI_h.push_back(2+offset);
  sys->massJ_h.push_back(11+offset);
  sys->mass_h.push_back(-60.0/(A*l*rho));
  sys->massI_h.push_back(3+offset);
  sys->massJ_h.push_back(0+offset);
  sys->mass_h.push_back(-120.0/(A*l*rho));
  sys->massI_h.push_back(3+offset);
  sys->massJ_h.push_back(3+offset);
  sys->mass_h.push_back(1200.0/(A*l*l*rho));
  sys->massI_h.push_back(3+offset);
  sys->massJ_h.push_back(6+offset);
  sys->mass_h.push_back(60.0/(A*l*rho));
  sys->massI_h.push_back(3+offset);
  sys->massJ_h.push_back(9+offset);
  sys->mass_h.push_back(840.0/(A*l*l*rho));
  sys->massI_h.push_back(4+offset);
  sys->massJ_h.push_back(1+offset);
  sys->mass_h.push_back(-120.0/(A*l*rho));
  sys->massI_h.push_back(4+offset);
  sys->massJ_h.push_back(4+offset);
  sys->mass_h.push_back(1200.0/(A*l*l*rho));
  sys->massI_h.push_back(4+offset);
  sys->massJ_h.push_back(7+offset);
  sys->mass_h.push_back(60.0/(A*l*rho));
  sys->massI_h.push_back(4+offset);
  sys->massJ_h.push_back(10+offset);
  sys->mass_h.push_back(840.0/(A*l*l*rho));
  sys->massI_h.push_back(5+offset);
  sys->massJ_h.push_back(2+offset);
  sys->mass_h.push_back(-120.0/(A*l*rho));
  sys->massI_h.push_back(5+offset);
  sys->massJ_h.push_back(5+offset);
  sys->mass_h.push_back(1200.0/(A*l*l*rho));
  sys->massI_h.push_back(5+offset);
  sys->massJ_h.push_back(8+offset);
  sys->mass_h.push_back(60.0/(A*l*rho));
  sys->massI_h.push_back(5+offset);
  sys->massJ_h.push_back(11+offset);
  sys->mass_h.push_back(840.0/(A*l*l*rho));
  sys->massI_h.push_back(6+offset);
  sys->massJ_h.push_back(0+offset);
  sys->mass_h.push_back(-4.0/(A*rho));
  sys->massI_h.push_back(6+offset);
  sys->massJ_h.push_back(3+offset);
  sys->mass_h.push_back(60.0/(A*l*rho));
  sys->massI_h.push_back(6+offset);
  sys->massJ_h.push_back(6+offset);
  sys->mass_h.push_back(16.0/(A*rho));
  sys->massI_h.push_back(6+offset);
  sys->massJ_h.push_back(9+offset);
  sys->mass_h.push_back(120.0/(A*l*rho));
  sys->massI_h.push_back(7+offset);
  sys->massJ_h.push_back(1+offset);
  sys->mass_h.push_back(-4.0/(A*rho));
  sys->massI_h.push_back(7+offset);
  sys->massJ_h.push_back(4+offset);
  sys->mass_h.push_back(60.0/(A*l*rho));
  sys->massI_h.push_back(7+offset);
  sys->massJ_h.push_back(7+offset);
  sys->mass_h.push_back(16.0/(A*rho));
  sys->massI_h.push_back(7+offset);
  sys->massJ_h.push_back(10+offset);
  sys->mass_h.push_back(120.0/(A*l*rho));
  sys->massI_h.push_back(8+offset);
  sys->massJ_h.push_back(2+offset);
  sys->mass_h.push_back(-4.0/(A*rho));
  sys->massI_h.push_back(8+offset);
  sys->massJ_h.push_back(5+offset);
  sys->mass_h.push_back(60.0/(A*l*rho));
  sys->massI_h.push_back(8+offset);
  sys->massJ_h.push_back(8+offset);
  sys->mass_h.push_back(16.0/(A*rho));
  sys->massI_h.push_back(8+offset);
  sys->massJ_h.push_back(11+offset);
  sys->mass_h.push_back(120.0/(A*l*rho));
  sys->massI_h.push_back(9+offset);
  sys->massJ_h.push_back(0+offset);
  sys->mass_h.push_back(-60.0/(A*l*rho));
  sys->massI_h.push_back(9+offset);
  sys->massJ_h.push_back(3+offset);
  sys->mass_h.push_back(840.0/(A*l*l*rho));
  sys->massI_h.push_back(9+offset);
  sys->massJ_h.push_back(6+offset);
  sys->mass_h.push_back(120.0/(A*l*rho));
  sys->massI_h.push_back(9+offset);
  sys->massJ_h.push_back(9+offset);
  sys->mass_h.push_back(1200.0/(A*l*l*rho));
  sys->massI_h.push_back(10+offset);
  sys->massJ_h.push_back(1+offset);
  sys->mass_h.push_back(-60.0/(A*l*rho));
  sys->massI_h.push_back(10+offset);
  sys->massJ_h.push_back(4+offset);
  sys->mass_h.push_back(840.0/(A*l*l*rho));
  sys->massI_h.push_back(10+offset);
  sys->massJ_h.push_back(7+offset);
  sys->mass_h.push_back(120.0/(A*l*rho));
  sys->massI_h.push_back(10+offset);
  sys->massJ_h.push_back(10+offset);
  sys->mass_h.push_back(1200.0/(A*l*l*rho));
  sys->massI_h.push_back(11+offset);
  sys->massJ_h.push_back(2+offset);
  sys->mass_h.push_back(-60.0/(A*l*rho));
  sys->massI_h.push_back(11+offset);
  sys->massJ_h.push_back(5+offset);
  sys->mass_h.push_back(840.0/(A*l*l*rho));
  sys->massI_h.push_back(11+offset);
  sys->massJ_h.push_back(8+offset);
  sys->mass_h.push_back(120.0/(A*l*rho));
  sys->massI_h.push_back(11+offset);
  sys->massJ_h.push_back(11+offset);
  sys->mass_h.push_back(1200.0/(A*l*l*rho));

  sys->contactGeometry_h.push_back(contactGeometry);
  sys->materialsBeam_h.push_back(make_double3(density,elasticModulus,0));

  for(int i=0;i<contactGeometry.z;i++) {
    sys->collisionGeometry_h.push_back(make_double3(contactGeometry.x,0,0));
    sys->collisionMap_h.push_back(make_int3(identifier,i,collisionFamily));
  }

  return 0;
}

double3 Beam::transformNodalToCartesian(double xi)
{
  double l = contactGeometry.y;
  double3 pos;
  pos.x = sys->p_h[index]*(2*xi*xi*xi - 3*xi*xi + 1) + sys->p_h[6+index]*(-2*xi*xi*xi + 3*xi*xi) + l*sys->p_h[3+index]*(xi*xi*xi - 2*xi*xi + xi) - l*sys->p_h[9+index]*(- xi*xi*xi + xi*xi);
  pos.y = sys->p_h[1+index]*(2*xi*xi*xi - 3*xi*xi + 1) + sys->p_h[7+index]*(-2*xi*xi*xi + 3*xi*xi) + l*sys->p_h[4+index]*(xi*xi*xi - 2*xi*xi + xi) - l*sys->p_h[10+index]*(- xi*xi*xi + xi*xi);
  pos.z = sys->p_h[2+index]*(2*xi*xi*xi - 3*xi*xi + 1) + sys->p_h[8+index]*(-2*xi*xi*xi + 3*xi*xi) + l*sys->p_h[5+index]*(xi*xi*xi - 2*xi*xi + xi) - l*sys->p_h[11+index]*(- xi*xi*xi + xi*xi);

  return pos;
}
