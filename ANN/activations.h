#pragma once
class activations {
public:
	static double sigmoid(double);
	static double d_sigmoid(double);

	static double identity(double);
	static double d_identity(double);

	static double ReLU(double);
	static double d_ReLU(double);
};