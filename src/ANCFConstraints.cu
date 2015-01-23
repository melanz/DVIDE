#include "include.cuh"
#include "System.cuh"

int System::addConstraint_AbsoluteX(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEX);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_AbsoluteY(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEY);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_AbsoluteZ(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEZ);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_AbsoluteDX1(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEDX1);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_AbsoluteDY1(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEDY1);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_AbsoluteDZ1(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEDZ1);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_RelativeX(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEX);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_RelativeY(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEY);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_RelativeZ(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEZ);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_RelativeDX1(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDX1);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_RelativeDY1(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDY1);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_RelativeDZ1(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDZ1);
	System::constraints.push_back(constraint);

	return 0;
}

int System::addConstraint_AbsoluteFixed(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEX);
	System::constraints.push_back(constraint);
	Constraint constraint2(nodeNum,CONSTRAINTABSOLUTEY);
	System::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum,CONSTRAINTABSOLUTEZ);
	System::constraints.push_back(constraint3);

	Constraint constraint4(nodeNum,CONSTRAINTABSOLUTEDX1);
	System::constraints.push_back(constraint4);
	Constraint constraint5(nodeNum,CONSTRAINTABSOLUTEDY1);
	System::constraints.push_back(constraint5);
	Constraint constraint6(nodeNum,CONSTRAINTABSOLUTEDZ1);
	System::constraints.push_back(constraint6);

	return 0;
}

int System::addConstraint_RelativeFixed(int nodeNum1,int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEX);
	System::constraints.push_back(constraint);
	Constraint constraint2(nodeNum1,nodeNum2,CONSTRAINTRELATIVEY);
	System::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum1,nodeNum2,CONSTRAINTRELATIVEZ);
	System::constraints.push_back(constraint3);

	Constraint constraint4(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDX1);
	System::constraints.push_back(constraint4);
	Constraint constraint5(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDY1);
	System::constraints.push_back(constraint5);
	Constraint constraint6(nodeNum1,nodeNum2,CONSTRAINTRELATIVEDZ1);
	System::constraints.push_back(constraint6);

	return 0;
}

int System::addConstraint_AbsoluteSpherical(int nodeNum)
{
	Constraint constraint(nodeNum,CONSTRAINTABSOLUTEX);
	System::constraints.push_back(constraint);
	Constraint constraint2(nodeNum,CONSTRAINTABSOLUTEY);
	System::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum,CONSTRAINTABSOLUTEZ);
	System::constraints.push_back(constraint3);

	return 0;
}

int System::addConstraint_RelativeSpherical(int nodeNum1, int nodeNum2)
{
	Constraint constraint(nodeNum1,nodeNum2,CONSTRAINTRELATIVEX);
	System::constraints.push_back(constraint);
	Constraint constraint2(nodeNum1,nodeNum2,CONSTRAINTRELATIVEY);
	System::constraints.push_back(constraint2);
	Constraint constraint3(nodeNum1,nodeNum2,CONSTRAINTRELATIVEZ);
	System::constraints.push_back(constraint3);

	return 0;
}

int System::addConstraint_AbsoluteX(Element& element, int node_local)
{
	addConstraint_AbsoluteX(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteY(Element& element, int node_local)
{
	addConstraint_AbsoluteY(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteZ(Element& element, int node_local)
{
	addConstraint_AbsoluteZ(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteDX1(Element& element, int node_local)
{
	addConstraint_AbsoluteDX1(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteDY1(Element& element, int node_local)
{
	addConstraint_AbsoluteDY1(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteDZ1(Element& element, int node_local)
{
	addConstraint_AbsoluteDZ1(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteFixed(Element& element, int node_local)
{
	addConstraint_AbsoluteFixed(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_AbsoluteSpherical(Element& element, int node_local)
{
	addConstraint_AbsoluteSpherical(element.getIndex()*2+node_local);
	return 0;
}

int System::addConstraint_RelativeX(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeX(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeY(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeY(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeZ(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeZ(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeDX1(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeDX1(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeDY1(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeDY1(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeDZ1(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeDZ1(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeFixed(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeFixed(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}

int System::addConstraint_RelativeSpherical(Element& element1, int node_local1, Element& element2, int node_local2)
{
	addConstraint_RelativeSpherical(element1.getIndex()*2+node_local1,element2.getIndex()*2+node_local2);
	return 0;
}
