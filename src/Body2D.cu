#include <algorithm>
#include <vector>
#include "include.cuh"
#include "Body2D.cuh"
#include "System.cuh"

int Body2D::addBody2D(int j)
{
  setIdentifier(sys->bodies.size()+sys->beams.size()+sys->plates.size()+j); // Indicates the number that the Body2D was added
  setIndex(sys->p_h.size()); // Indicates the Beam's location in the position array

  // Push Beam's location to global library
  sys->indices_h.push_back(sys->p_h.size());

  // update p
  sys->p_h.push_back(p.x);
  sys->p_h.push_back(p.y);
  sys->p_h.push_back(p.z);

  // update v
  sys->v_h.push_back(v.x);
  sys->v_h.push_back(v.y);
  sys->v_h.push_back(v.z);

  // update a
  sys->a_h.push_back(a.x);
  sys->a_h.push_back(a.y);
  sys->a_h.push_back(a.z);

  // update external force vector (gravity)
  sys->f_h.push_back(getMass() * sys->gravity.x);
  sys->f_h.push_back(getMass() * sys->gravity.y);
  sys->f_h.push_back(0.0);

  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);
  sys->f_contact_h.push_back(0);

  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);
  sys->fApplied_h.push_back(0);

  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);
  sys->fElastic_h.push_back(0);

  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);
  sys->tmp_h.push_back(0);

  sys->r_h.push_back(0);
  sys->r_h.push_back(0);
  sys->r_h.push_back(0);

  sys->k_h.push_back(0);
  sys->k_h.push_back(0);
  sys->k_h.push_back(0);

  // update the sys->mass matrix
  int offset = j*3+36*sys->plates.size()+12*sys->beams.size()+3*sys->bodies.size();
  sys->massI_h.push_back(0+offset);
  sys->massJ_h.push_back(0+offset);
  sys->mass_h.push_back(1.0/getMass());
  sys->massI_h.push_back(1+offset);
  sys->massJ_h.push_back(1+offset);
  sys->mass_h.push_back(1.0/getMass());
  sys->massI_h.push_back(2+offset);
  sys->massJ_h.push_back(2+offset);
  sys->mass_h.push_back(1.0/getInertia());

  sys->materialsBody2D_h.push_back(make_double2(mass,inertia));

  sys->contactGeometry_h.push_back(make_double3(0,0,0));

  return 0;
}
