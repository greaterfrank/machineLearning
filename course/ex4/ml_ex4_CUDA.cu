#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>
#include <map>
using namespace std;

typedef double FLOAT;

///  g++ ml_ex4.v2.cpp

struct Matrix {
	int h, w;
	FLOAT* p;
	string name;
	Matrix() {
		w = h = 0;
		p = 0;
		// printf(" .................. new    Matrix %p\n", this);
	}
	~Matrix()
	{
		// printf(" --------||||||||-- delete Matrix %p %d %d %p %s\n", this, h, w, p, name.c_str());
		delete p;
	}
	void init(int height, int width, const char* name_ = 0) {
		w = width;
		h = height;
		p = new FLOAT[w * h];
		memset(p, 0, w * h * sizeof(FLOAT));
		if (name_)
			name = name_;
		// printf(" ================== init   Matrix %p %d %d %p %s\n", this, h, w, p, name.c_str());
	}
	void initEye(int n)
	{
		init(n, n);
		for (int i = 0; i < n; i++)
			p[i * n + i] = 1;
	}
	void swap(Matrix& m) {
		if (h != m.h || w != m.w) {
			printf("swap failed: h != m.h || w != m.w \n");
			return;
		}
		FLOAT* tmp = p; p = m.p; m.p = tmp;
	}
	void move(Matrix& m) {
		if (p)
			delete p;
		h = m.h;
		w = m.w;
		p = m.p;
		name = m.name;
		m.h = m.w = 0;
		m.p = 0;
		m.name = "";
	}
	void print() {
		printf(" ===== %s ===== %p %p %d %d\n", name.c_str(), this, p, h, w);
		for (int i = 0; i < h; i++) {
			for (int j = 0; j < w; j++)
				printf(" %g", p[i * w + j]);
			printf("\n");
		}
		printf("\n");
	}
	void multipleTranspose(Matrix& m1, Matrix& m2) {
		if (m1.w != m2.w) {
			printf("multipleTranspose error: m1.w != m2.w\n");
			return;
		}
		if (!p)
			init(m1.h, m2.h);
		else if (h != m1.h) {
			printf("multipleTranspose error: h != m1.h\n");
			return;
		}
		else if (w != m2.h) {
			printf("multipleTranspose error: w != m2.h\n");
			return;
		}
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				p[i * w + j] = 0;
				for (int k = 0; k < m1.w; k++)
					p[i * w + j] += m1.p[i * m1.w + k] * m2.p[j * m2.w + k];
			}
	}
	void tranposeMultiple(Matrix& m1, Matrix& m2) {
		if (m1.h != m2.h) {
			printf("tranposeMultiple error: m1.h != m2.h\n");
			return;
		}
		if (!p)
			init(m1.w, m2.w);
		else if (h != m1.w) {
			printf("tranposeMultiple error: h != m1.w\n");
			return;
		}
		else if (w != m2.w) {
			printf("tranposeMultiple error: w != m2.w\n");
			return;
		}
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				p[i * w + j] = 0;
				for (int k = 0; k < m1.h; k++)
					p[i * w + j] += m1.p[k * m1.w + i] * m2.p[k * m2.w + j];
			}
	}
	FLOAT dotProduct(Matrix& m) {
		if (h * w != m.h * m.w) {
			printf("dotProduct error: h*w != m.h*m.w\n");
			return 0;
		}
		FLOAT prod = 0;
		for (int i = 0; i < h * w; i++)
			prod += p[i] * m.p[i];
		return prod;
	}
	void addition(Matrix& m1, FLOAT v1, Matrix& m2, FLOAT v2) {
		if (m1.h != m2.h) {
			printf("addition error: m1.h != m2.h\n");
			return;
		}
		if (m1.w != m2.w) {
			printf("addition error: m1.w != m2.w\n");
			return;
		}
		if (!p)
			init(m1.h, m1.w);
		else if (h != m1.h) {
			printf("addition error: h != m1.h\n");
			return;
		}
		else if (w != m1.w) {
			printf("addition error: w != m1.w\n");
			return;
		}
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] = m1.p[i * w + j] * v1 + m2.p[i * w + j] * v2;
	}
	void add(Matrix& m, FLOAT v) {
		if (h != m.h) {
			printf("add error: h != m.h\n");
			return;
		}
		if (w != m.w) {
			printf("add error: w != m.w\n");
			return;
		}
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] += m.p[i * w + j] * v;
	}
	void combine(FLOAT x, Matrix& m) {
		init(m.h, m.w + 1);
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] = ((j == 0) ? x : m.p[i * m.w + j - 1]);
	}
	void removeFirstColumn() {
		if (w < 2) {
			printf("removeFirstColumn error: w=%d is too small\n", w);
			return;
		}
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w - 1; j++)
				p[i * (w - 1) + j] = p[i * w + j + 1];
		w--;
	}
	void copy(Matrix& m) {
		if (p == 0)
			init(m.h, m.w);
		else if (m.h != h || m.w != w) {
			printf("Error in copy, (%d %d) != (%d %d)\n", h, w, m.h, m.w);
			return;
		}

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] = m.p[i * w + j];
	}
	void call(FLOAT(func(FLOAT))) {
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] = (func)(p[i * w + j]);
	}
	void call(FLOAT(func(FLOAT, FLOAT)), Matrix& m) {
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] = (func)(p[i * w + j], m.p[i * w + j]);
	}
	FLOAT sumSquare(bool ignoreFirstColumn = 0) {
		FLOAT sum = 0;
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				// printf(" sumSquare %d %d %d %f %f\n", i, j, i * w + j, p[i * w + j], sum);
				sum += ((ignoreFirstColumn && j == 0) ? 0 : (p[i * w + j] * p[i * w + j]));
			}
		return sum;
	}
	FLOAT sum(bool ignoreFirstColumn) {
		FLOAT sum = 0;
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				// printf(" sumSquare %d %d %d %f %f\n", i, j, i * w + j, p[i * w + j], sum);
				sum += ((ignoreFirstColumn && j == 0) ? 0 : (p[i * w + j]));
			}
		return sum;
	}
	void substraction(Matrix& m) {
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++)
				p[i * w + j] -= m.p[i * w + j];
	}
	void product(Matrix& m1, Matrix& m2) {
		if (m1.w != m2.h)
		{
			printf("Error in product m1.w != m2.h\n");
			return;
		}
		if (!p)
			init(m1.h, m2.w);
		else if (h != m1.h) {
			printf("Error in product h!= m1.h \n");
			return;
		}
		else if (w != m2.w) {
			printf("Error in product w!= m2.w \n");
			return;
		}
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				p[i * w + j] = 0;
				for (int k = 0; k < m1.w; k++)
					p[i * w + j] += m1.p[i * m1.w + k] * m2.p[k * m2.w + j];
			}
	}
	void multiple(Matrix& m) { // element multiple
		if (m.w != w)
		{
			printf("Error in multiple m.w != w\n");
			return;
		}
		if (m.h != h)
		{
			printf("Error in multiple m.h != h\n");
			return;
		}

		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				p[i * w + j] *= m.p[i * w + j];
			}
	}
	void multiple(FLOAT v) {
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				p[i * w + j] *= v;
			}
	}
	void setFirstColumnValue(FLOAT v) {
		for (int i = 0; i < h; i++)
			p[i * w + 0] = v;
	}
	FLOAT max() {
		FLOAT m = 0;
		for (int i = 0; i < h; i++)
			for (int j = 0; j < w; j++) {
				int k = i * w + j;
				if (k == 0)
					m = p[k];
				else if (m < p[k])
					m = p[k];
			}
		return m;
	}
};

FLOAT sigmoid(FLOAT z);
FLOAT sigmoidGradient(FLOAT z);
FLOAT costLogisticRegression(FLOAT Y, FLOAT h);

map<string, Matrix> mat;
int nitems = 0;
FLOAT realmin = 1.1755e-38f;

struct ModelSingleHiddenLayer {
	Matrix Theta1, Theta2, Theta1_grad, Theta2_grad, X, y;
	int input_layer_size, hidden_layer_size, num_labels;
	FLOAT lambda;
	void init(int input_layer_size, int hidden_layer_size, int num_labels, FLOAT lambda) {
		this->input_layer_size = input_layer_size;
		this->hidden_layer_size = hidden_layer_size;
		this->num_labels = num_labels;
		this->lambda = lambda;
	}
	FLOAT costFunction();
	void updateTheta();
	void predict(Matrix &data, Matrix &result);
};

bool read_file() {
	const char* filename = "ex4data1.txt";
	FILE* f = fopen(filename, "r");
	if (!f) { printf("Error in reading file %s\n", filename); return false; }
	char buffer[6000];
	int current_line = 0;
	int h = 0;
	int w = 0;

	Matrix* it = 0;
	while (fgets(buffer, 6000, f) != 0) {
		if (it != 0 && current_line > 0) {
			char* s = buffer, *s1=0;
			int i = 0;
			for (; i < w && s!=0; i++, s=s1)
				//sscanf(s, "%f", &it->p[(h - current_line) * w + i]); //  = strtof(s, &s1);
				it->p[(h - current_line) * w + i] = strtof(s, &s1);
				 
			//printf(" matrix %s, %d, %d\n", it->name.c_str(), current_line, i);
			current_line--;
			if (current_line == 0)
				it = 0;
			continue;
		}

		vector<char*> v;
		for (char* s = strtok(buffer, " "); s != 0; s = strtok(0, " "))
			v.push_back(s);
		if (v.size() >= 3) {
			if (!strcmp(v[0], "Item")) {
				if (!strcmp(v[1], "No") && !strcmp(v[2], "More"))
					return true; // file read completed
				if (!strcmp(v[1], "Matrix") && v.size() >= 5) {
					h = atoi(v[3]);
					w = atoi(v[4]);
					current_line = h;
					mat[v[2]] = Matrix();
					it = &mat[v[2]];
					it->init(h, w, v[2]);
					nitems++;
				}
			}
		}
	}
	fclose(f);
	return true;
}



FLOAT sigmoid(FLOAT z) {
	return 1.f / (1.f + (FLOAT)exp(-z));
}

FLOAT sigmoidGradient(FLOAT z) {
	return sigmoid(z) * (1 - sigmoid(z));
}

FLOAT costLogisticRegression(FLOAT Y, FLOAT h) {
	return (FLOAT)((-Y)*log(h) - (1.f - Y)*log(1.f - h));
}

FLOAT ModelSingleHiddenLayer::costFunction() {
	FLOAT J = 0;
	int m = X.h;

	// % recode y to Y
	Matrix I, Y;
	I.initEye(num_labels); // 10x10   // I = eye(num_labels);
	Y.init(m, num_labels); // 5000x10 // Y = zeros(m, num_labels);
	for (int i = 0; i < m; i++)
		for (int j = 0; j < num_labels; j++) // Y(i, :)= I(y(i), :);
		{
			FLOAT yy = y.p[i];
			int iy = int(y.p[i] - 1);
			Y.p[i * num_labels + j] = I.p[iy * num_labels + j];

		}

	// % feedforward
	Matrix a1, z2, z2sig, a2, z3, a3, h;
	a1.combine(1.f, X);               // 5000x401 // a1 = [ones(m, 1) X];
	z2.multipleTranspose(a1, Theta1); // 5000x25  // z2 = a1 * Theta1';
	z2sig.copy(z2);
	z2sig.call(sigmoid); 
	a2.combine(1.f, z2sig);           // 5000x26 // a2 = [ones(size(z2, 1), 1) sigmoid(z2)];
	z3.multipleTranspose(a2, Theta2); // 5000x10 // z3 = a2 * Theta2';
	a3.copy(z3);                      // 5000x10
	a3.call(sigmoid);                            // a3 = sigmoid(z3);
	h.copy(a3); // h = a3;            // 5000x10

	//% calculte penalty
	FLOAT p = Theta1.sumSquare(true) + Theta2.sumSquare(true); // p = sum(sum(Theta1(:, 2 : end). ^ 2, 2)) + sum(sum(Theta2(:, 2 : end). ^ 2, 2));


	// % calculate J // J = sum(sum((-Y).*log(h) - (1 - Y).*log(1 - h), 2)) / m + lambda * p / (2 * m);
	Matrix cLogR;
	cLogR.copy(Y);
	cLogR.call(costLogisticRegression, h);
	//Theta1.print();
	//Theta2.print();
	//h.name = "h";
	//h.print();
	//cLogR.name = "cLogR";
	//cLogR.print();
	FLOAT J0 = cLogR.sum(false);

	J = cLogR.sum(false) / m + lambda * p / (2 * m);

	// printf(" nnCost: p=%f J0 = %f J = %f\n", p, J0, J);

	// J should be 0.287629

	// % calculate sigmas
	Matrix sigma3, sigma2, z2sigGrad;
	sigma3.copy(a3);
	sigma3.substraction(Y);         // 5000x10 // sigma3 = a3. - Y;

	sigma2.product(sigma3, Theta2); // 5000x25 //sigma2 = (sigma3 * Theta2).*sigmoidGradient([ones(size(z2, 1), 1) z2]);
	z2sigGrad.combine(1, z2);
	z2sigGrad.call(sigmoidGradient);
	sigma2.multiple(z2sigGrad); // 5000x26
	sigma2.removeFirstColumn(); // 5000x25 //	sigma2 = sigma2(:, 2 : end);

	Matrix delta1, delta2;
	// % accumulate gradients
	delta1.tranposeMultiple(sigma2, a1); // 25x401 // delta_1 = (sigma2'*a1);
	delta2.tranposeMultiple(sigma3, a2); // 10x26  // delta_2 = (sigma3'*a2);
				
	Matrix p1, p2;
	//% calculate regularized gradient
	p1.copy(Theta1);
	p1.setFirstColumnValue(0);                                         // p1 = (lambda / m) * [zeros(size(Theta1, 1), 1) Theta1(:, 2 : end)];
	Theta1_grad.addition(delta1, 1.f / m, p1, lambda / m);  //25 x 401 // Theta1_grad = delta_1. / m + p1;

	p2.copy(Theta2);
	p2.setFirstColumnValue(0);                                         // p2 = (lambda / m) * [zeros(size(Theta2, 1), 1) Theta2(:, 2 : end)];
	Theta2_grad.addition(delta2, 1.f / m, p2, lambda / m);  //10 x 26  // Theta2_grad = delta_2. / m + p2;

	return J;

}
/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void debugInitializeWeights(Matrix& m, int fan_out, int fan_in) {
	m.init(fan_out, 1 + fan_in);
	for (int i = 0; i < m.h; i++) {
		for (int j = 0; j < m.w; j++) {
			m.p[i * m.w + j] = (FLOAT)sin((FLOAT)(j * m.h + i) + 1.f) / 10.f;
		}
	}
}

/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FLOAT computeNumericalGradient(ModelSingleHiddenLayer &model, Matrix& Theta1, Matrix& Theta2, Matrix &Theta1_grad, Matrix &Theta2_grad) {
	/*
	% COMPUTENUMERICALGRADIENT Computes the gradient using "finite differences"
	% andgives us a numerical estimate of the gradient.
	% numgrad = COMPUTENUMERICALGRADIENT(J, theta) computes the numerical
	% gradient of the function J around theta.Calling y = J(theta) should
	% return the function value at theta.

	% Notes: The following code implements numerical gradient checking, and
	% returns the numerical gradient.It sets numgrad(i) to(a numerical
		% approximation of) the partial derivative of J with respect to the
		% i - th input argument, evaluated at theta. (i.e., numgrad(i) should
			% be the(approximately) the partial derivative of J with respect
			% to theta(i).)
		%
	*/
	Matrix perturb1, perturb2;
	perturb1.init(Theta1.h, Theta1.w);
	perturb2.init(Theta2.h, Theta2.w);

	//numgrad = zeros(size(theta));
	//perturb = zeros(size(theta));
	FLOAT e = 1e-4f;
	int n = Theta1.h * Theta1.w + Theta2.h * Theta2.w;
	int n1 = Theta1.h * Theta1.w;

	model.Theta1_grad.init(Theta1.h, Theta1.w);
	model.Theta2_grad.init(Theta2.h, Theta2.w);

	FLOAT ft[][2] =
	{ { -0.0092782523, -0.0092782524}
	, { 0.0088991196, 0.0088991196 }
		, { -0.0083601076, -0.0083601076 }
		, { 0.0076281355, 0.0076281355 }
		, { -0.0067479837, -0.0067479837 }
		, { -0.0000030498, -0.0000030498 }
		, { 0.0000142869, 0.0000142869 }
		, { -0.0000259383, -0.0000259383 }
		, { 0.0000369883, 0.0000369883 }
		, { -0.0000468760, -0.0000468760 }
		, { -0.0001750601, -0.0001750601 }
		, { 0.0002331464, 0.0002331464 }
		, { -0.0002874687, -0.0002874687 }
		, { 0.0003353203, 0.0003353203 }
		, { -0.0003762156, -0.0003762156 }
		, { -0.0000962661, -0.0000962661 }
		, { 0.0001179827, 0.0001179827 }
		, { -0.0001371497, -0.0001371497 }
		, { 0.0001532471, 0.0001532471 }
		, { -0.0001665603, -0.0001665603 }
		, { 0.3145449700, 0.3145449701 }
		, { 0.1110565882, 0.1110565882 }
		, { 0.0974006970, 0.0974006970 }
		, { 0.1640908188, 0.1640908188 }
		, { 0.0575736493, 0.0575736493 }
		, { 0.0504575855, 0.0504575855 }
		, { 0.1645679323, 0.1645679323 }
		, { 0.0577867378, 0.0577867378 }
		, { 0.0507530173, 0.0507530173 }
		, { 0.1583393339, 0.1583393339 }
		, { 0.0559235296, 0.0559235296 }
		, { 0.0491620841, 0.0491620841 }
		, { 0.1511275275, 0.1511275275 }
		, { 0.0536967009, 0.0536967009 }
		, { 0.0471456249, 0.0471456249 }
		, { 0.1495683347, 0.1495683347 }
		, { 0.0531542052, 0.0531542052 }
		, { 0.0465597186, 0.0465597186 }
	};

	perturb1.name = "perturb1";
	perturb2.name = "perturb2";

	for (int p = 0; p < n; p++) {// for p = 1:numel(theta)
	// % Set perturbation vector
		int i = 0, j = 0, nn = 0;
		if (p < n1) {
			i = p % perturb1.h;
			j = p / perturb1.h;
			nn = i * perturb1.w + j;
			perturb1.p[nn] = e;
		}
		else {
			i = (p-n1) % perturb2.h;
			j = (p-n1) / perturb2.h;
			nn = i * perturb2.w + j;
			perturb2.p[nn] = e;       // perturb(p) = e;
		}

		model.Theta1.addition(Theta1, 1, perturb1, -1);
		model.Theta2.addition(Theta2, 1, perturb2, -1);

		FLOAT tt1 = model.Theta1.sum(0) + model.Theta2.sum(0);
		FLOAT loss1 = model.costFunction();

		model.Theta1.addition(Theta1, 1, perturb1, 1);
		model.Theta2.addition(Theta2, 1, perturb2, 1);
		FLOAT tt2 = model.Theta1.sum(0) + model.Theta2.sum(0);

		FLOAT loss2 = model.costFunction();
		printf(" %d === ttt %g %g loss1 = %f  loss2 = %f  %f ==================================================================\n", p, tt1, tt2, loss1, loss2, (loss2 - loss1) / (2 * e));

	// % Compute Numerical Gradient
		if (p < n1) {                 // numgrad(p) = (loss2 - loss1) / (2 * e);
			perturb1.p[nn] = 0; 
			Theta1_grad.p[nn] = (loss2 - loss1) / (2 * e); 
			printf(" in loop p=%d n1= %d t1 %d = %f %f %f %g\n", p, n1, p, Theta1_grad.p[nn], ft[p][1], ft[p][0], Theta1_grad.p[nn] - ft[p][1]);
		} else {                      // perturb(p) = 0;
			perturb2.p[nn] = 0; 
			Theta2_grad.p[nn] = (loss2 - loss1) / (2 * e);
			//printf(" in loop p=%d n1= %d t1 %d = %f %f %f %g\n", p, n1, p-n1, Theta2_grad.p[nn], ft[p][1], ft[p][0], Theta2_grad.p[nn] - ft[p][1]);
		}
	}

	return 0;
}


/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void checkNNGradients(FLOAT lambda = 0) {
	/*
	% CHECKNNGRADIENTS Creates a small neural network to check the
	% backpropagation gradients
	% CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
	% backpropagation gradients, it will output the analytical gradients
	% produced by your backprop codeand the numerical gradients(computed
		% using computeNumericalGradient).These two gradient computations should
		% result in very similar values.
		%
	*/

	ModelSingleHiddenLayer model;

	model.input_layer_size = 3;
	model.hidden_layer_size = 5;
	model.num_labels = 3;
	model.lambda = lambda;
	int m = 5;

	// % We generate some "random" test data
	debugInitializeWeights(model.Theta1, model.hidden_layer_size, model.input_layer_size);
	debugInitializeWeights(model.Theta2, model.num_labels, model.hidden_layer_size);
	Matrix t1, t2;
	t1.copy(model.Theta1);
	t2.copy(model.Theta2);

	// % Reusing debugInitializeWeights to generate X
	debugInitializeWeights(model.X, m, model.input_layer_size - 1);
	model.y.init(m, 1);
	for (int i = 0; i < m; i++)
		model.y.p[i] = (FLOAT)(1 + ((i + 1) % model.num_labels)); // y = 1 + mod(1:m, num_labels)';

	model.Theta1.name = "debug Theta1";
	model.Theta2.name = "debug Theta2";
	model.X.name = "debug X";
	model.y.name = "debug y";
	model.Theta1.print();
	model.Theta2.print();
	model.X.print();
	model.y.print();

	model.Theta1_grad.init(t1.h, t1.w);
	model.Theta2_grad.init(t2.h, t2.w);
	// FLOAT cost = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, Theta1_grad, Theta2_grad);
	FLOAT cost = model.costFunction();

	Matrix Theta1_grad, Theta2_grad, nTheta1_grad, nTheta2_grad;
	Theta1_grad.init(t1.h, t1.w);
	Theta2_grad.init(t2.h, t2.w);
	Theta1_grad.swap(model.Theta1_grad);
	Theta2_grad.swap(model.Theta2_grad);

	nTheta1_grad.init(t1.h, t1.w);
	nTheta2_grad.init(t2.h, t2.w);

	//computeNumericalGradient(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, nTheta1_grad, nTheta2_grad);
	computeNumericalGradient(model, t1, t2, nTheta1_grad, nTheta2_grad);

	// % Visually examine the two gradient computations.The two columns
	// % you get should be very similar.
	// disp ( [ numgrad grad ] );

	for (int i = 0; i < Theta1_grad.h * Theta1_grad.w; i++) {
		printf(" --- %g %g %g \n", Theta1_grad.p[i], nTheta1_grad.p[i], fabs(Theta1_grad.p[i] - nTheta1_grad.p[i]));
	}

	for (int i = 0; i < Theta2_grad.h * Theta2_grad.w; i++) {
		printf(" --- %g %g %g \n", Theta2_grad.p[i], nTheta2_grad.p[i], fabs(Theta2_grad.p[i] - nTheta2_grad.p[i]));
	}

	printf("The above two columns you get should be very similar.\n"
		   "(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n");

	// % Evaluate the norm of the difference between two solutions.
	//	 % If you have a correct implementation, and assuming you used EPSILON = 0.0001
	//	 % in computeNumericalGradient.m, then diff below should be less than 1e-9

	Matrix ng11, ng12, ng21, ng22;

	ng11.addition(Theta1_grad, 1, nTheta1_grad, -1);
	ng12.addition(Theta2_grad, 1, nTheta2_grad, -1);
	ng21.addition(Theta1_grad, 1, nTheta1_grad, 1);
	ng22.addition(Theta2_grad, 1, nTheta2_grad,  1);

	FLOAT m11 = ng11.max();
	FLOAT m12 = ng12.max();
	FLOAT m21 = ng21.max();
	FLOAT m22 = ng22.max();

	FLOAT m1 = (m11 > m12) ? m11 : m12;
	FLOAT m2 = (m21 > m22) ? m21 : m22;

	FLOAT diff = m1 / m2;

	printf("If your backpropagation implementation is correct, then \n"
		   "the relative difference will be small (less than 1e-9). \n"
		   "\nRelative Difference: %g\n", diff);
}



void randInitializeWeights(Matrix &m, int L_in, int L_out) {
	/*
	% RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
	% incoming connections and L_out outgoing connections
	% W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights
	% of a layer with L_in incoming connections and L_out outgoing
	% connections.
	%
	%Note that W should be set to a matrix of size(L_out, 1 + L_in) as
	% the first row of W handles the "bias" terms
	%

	%You need to return the following variables correctly
	*/
	m.init(L_out, 1 + L_in); // W = zeros(L_out, 1 + L_in);
	/*
	% ====================== YOUR CODE HERE ======================
	% Instructions: Initialize W randomly so that we break the symmetry while
	% training the neural network.
	%
	% Note : The first row of W corresponds to the parameters for the bias units
	%
	*/

	FLOAT epsilon_init = 0.12f;
	for (int i = 0; i < m.h * m.w; i++)
		m.p[i] = ((FLOAT)rand() / 32767.f) * 2.f * epsilon_init - epsilon_init; // W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
}
// % ======================================================================== =

/*
function[X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
% Minimize a continuous differentialble multivariate function.Starting point
% is given by "X" (D by 1), and the function named in the string "f", must
% return a function value and a vector of partial derivatives.The Polack -
% Ribiere flavour of conjugate gradients is used to compute search directions,
%and a line search using quadraticand cubic polynomial approximationsand the
% Wolfe - Powell stopping criteria is used together with the slope ratio method
% for guessing initial step sizes.Additionally a bunch of checks are made to
% make sure that exploration is taking placeand that extrapolation will not
% be unboundedly large.The "length" gives the length of the run : if it is
% positive, it gives the maximum number of line searches, if negative its
% absolute gives the maximum allowed number of function evaluations.You can
% (optionally)give "length" a second component, which will indicate the
% reduction in function value to be expected in the first line - search(defaults
	% to 1.0).The function returns when either its length is up, or if no further
	% progress can be made(ie, we are at a minimum, or so close that due to
		% numerical problems, we cannot get any closer).If the function terminates
	% within a few iterations, it could be an indication that the function value
	%and derivatives are not consistent(ie, there may be a bug in the
		% implementation of your "f" function).The function returns the found
	% solution "X", a vector of function values "fX" indicating the progress made
	% and"i" the number of iterations(line searches or function evaluations,
		% depending on the sign of "length") used.
	%
	%Usage: [X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
	%
	%See also : checkgrad
	%
	%Copyright(C) 2001 and 2002 by Carl Edward Rasmussen.Date 2002 - 02 - 13
	%
	%
	%(C)Copyright 1999, 2000 & 2001, Carl Edward Rasmussen
	%
	% Permission is granted for anyone to copy, use, or modify these
	% programs and accompanying documents for purposes of research or
	% education, provided this copyright notice is retained, and note is
	% made of any changes that have been made.
	%
	% These programsand documents are distributed without any warranty,
	% express or implied.As the programs were written for research
	% purposes only, they have not been tested to the degree that would be
	% advisable in any important application.All use of these programs is
	% entirely at the user's own risk.
	%
	%[ml - class] Changes Made :
% 1) Function nameand argument specifications
% 2) Output display
%

% Read options
if exist('options', 'var') && ~isempty(options) && isfield(options, 'MaxIter')
length = options.MaxIter;
else
length = 100;
end
*/

//function[X, fX, i] = fmincg(f, X, options, P1, P2, P3, P4, P5)
void fmincg(ModelSingleHiddenLayer &model, int maxIter) {

	FLOAT RHO = 0.01;// % a bunch of constants for line searches
	FLOAT SIG = 0.5;// % RHOand SIG are the constants in the Wolfe - Powell conditions
	FLOAT INT = 0.1;// % don't reevaluate within 0.1 of the limit of the current bracket
	FLOAT EXT = 3.0;// % extrapolate maximum 3 times the current bracket
	int MAX = 20;// % max 20 function evaluations per line search
	FLOAT RATIO = 100;// % maximum allowed slope ratio

	int length = maxIter;
	/*
	argstr = ['feval(f, X'];// % compose string used to call function
	for i = 1:(nargin - 3)
	argstr = [argstr, ',P', int2str(i)];
	end
	argstr = [argstr, ')'];

	if max(size(length)) == 2, red = length(2); length = length(1); else red = 1; end
	S = ['Iteration '];
	*/
	Matrix df11,df12,s1, s2, df01, df02, df21, df22;
	df11.init(model.Theta1.h, model.Theta1.w);
	df12.init(model.Theta2.h, model.Theta2.w);
	df21.init(model.Theta1.h, model.Theta1.w);
	df22.init(model.Theta2.h, model.Theta2.w);

	int i = 0;// % zero the run length counter
	int ls_failed = 0;// % no previous line search has failed
	//fX = [];  // cost for return
	//[f1 df1] = eval(argstr);// % get function valueand gradient

	FLOAT f2 = 0;
	FLOAT f0 = 0;
	FLOAT f1 = model.costFunction();
	df11.swap(model.Theta1_grad);
	df12.swap(model.Theta2_grad);

	i = i + (length < 0);// % count epochs ? !
	s1.copy(df11); ////s = -df1;// % search direction is steepest
	s2.copy(df12);
	s1.multiple(-1.f);
	s2.multiple(-1.f);
	FLOAT d1 = - s1.sumSquare(0) - s2.sumSquare(0);  // d1 = -s'*s;                                                 // % this is the slope
	FLOAT z1 = 1.f / (1.f - d1); // z1 = red / (1 - d1);// % initial step is red / (| s | +1)

	Matrix t01, t02;
	while( i < length) { // //while (i < abs(length)) { // % while not finished
		i = i + (length > 0);// % count iterations ? !

		t01.copy(model.Theta1); t02.copy(model.Theta2); f0 = f1; df01.copy(df11); df02.copy(df12); //X0 = X; f0 = f1; df0 = df1;// % make a copy of current values
		model.Theta1.add(s1, z1); model.Theta2.add(s2, z1);  // X = X + z1 * s;// % begin line search
		
		//[f2 df2] = eval(argstr);
		//f2 = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, df21, df22);
		f2 = model.costFunction();
		df21.swap(model.Theta1_grad);
		df22.swap(model.Theta2_grad);

		i = i + (length < 0);// % count epochs ? !
		FLOAT d2 = df21.dotProduct(s1) + df22.dotProduct(s2); //d2 = df2'*s;
		FLOAT f3 = f1; FLOAT d3 = d2; FLOAT z3 = -z1; // f3 = f1; d3 = d1; z3 = -z1;// % initialize point 3 equal to point 1
		FLOAT M = (length > 0) ? MAX : min(MAX, -length - i); // if length > 0, M = MAX; else M = min(MAX, -length - i); end
		FLOAT z2=1e31, A, B;
		int success = 0; FLOAT limit = -1.f;// % initialize quanteties
		while (1) {
			while (((f2 > f1 + z1 * RHO * d1) || (d2 > -SIG * d1)) && (M > 0)) {
				limit = z1;// % tighten the bracket
				if (f2 > f1) {
					z2 = z3 - (0.5 * d3 * z3 * z3) / (d3 * z3 + f2 - f3);// % quadratic fit
				}
				else {
					A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);// % cubic fit
					B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
					z2 = (sqrt(B * B - A * d2 * z3 * z3) - B) / A;// % numerical error possible - ok!
				} // end
				if(z2> 1e30) { // if (isnan(z2) || isinf(z2)) {
					z2 = z3 / 2;// % if we had a numerical problem then bisect
				} // end
				z2 = max(min(z2, INT * z3), (1 - INT) * z3);// % don't accept too close to limits
				z1 = z1 + z2;// % update the step
				model.Theta1.add(s1, z2); model.Theta2.add(s2, z2); // X = X + z2 * s;
				//[f2 df2] = eval(argstr);
				// f2 = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, df21, df22);
				f2 = model.costFunction();
				df21.swap(model.Theta1_grad);
				df22.swap(model.Theta2_grad);

				M = M - 1; i = i + (length < 0);// % count epochs ? !
				d2 = df21.dotProduct(s1) + df22.dotProduct(s2); //// d2 = df2'*s;
					z3 = z3 - z2;// % z3 is now relative to the location of z2
			} // end
			if (f2 > f1 + z1 * RHO * d1 || d2 > -SIG * d1) {
				break;// % this is a failure
			}
			else if (d2 > SIG * d1) {
				success = 1; break;// % success
			}
			else if (M == 0) {
				break;// % failure
			} //end

			A = 6 * (f2 - f3) / z3 + 3 * (d2 + d3);// % make cubic extrapolation
			B = 3 * (f3 - f2) - z3 * (d3 + 2 * d2);
			z2 = -d2 * z3 * z3 / (B + sqrt(B * B - A * d2 * z3 * z3));// % num.error possible - ok!
			if (z2 > 1e30) { // ~isreal(z2) || isnan(z2) || isinf(z2) || z2 < 0) { // % num prob or wrong sign ?
				if (limit < -0.5) { // % if we have no upper limit
					z2 = z1 * (EXT - 1);// % the extrapolate the maximum amount
				}
				else {
					z2 = (limit - z1) / 2;// % otherwise bisect
				} //end
			}
			else if ((limit > -0.5) && (z2 + z1 > limit)) { // % extraplation beyond max ?
				z2 = (limit - z1) / 2;// % bisect
			}
			else if ((limit < -0.5) && (z2 + z1 > z1 * EXT)) { // % extrapolation beyond limit
				z2 = z1 * (EXT - 1.0);// % set to extrapolation limit
			}
			else if (z2 < -z3 * INT) {
				z2 = -z3 * INT;
			}
			else if ((limit > -0.5) && (z2 < (limit - z1) * (1.0 - INT))) { // % too close to limit ?
				z2 = (limit - z1) * (1.0 - INT);
			} // end
			f3 = f2; d3 = d2; z3 = -z2;// % set point 3 equal to point 2
			z1 = z1 + z2; //X = X + z2 * s;// % update current estimates
			model.Theta1.add(s1, z2); model.Theta2.add(s2, z2);
			//[f2 df2] = eval(argstr);
			//f2 = nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, df21, df22);
			f2 = model.costFunction();
			df21.swap(model.Theta1_grad);
			df22.swap(model.Theta2_grad);

			M = M - 1; i = i + (length < 0);// % count epochs ? !
			d2 = df21.dotProduct(s1) + df22.dotProduct(s2); // d2 = df2'*s;
		} // end // % end of line search

		if (success) {// % if line search succeeded
			f1 = f2; // fX = [fX' f1]'; --- cost for return
			//printf("%s %4i | Cost: %4.6e\r", S, i, f1);
			printf("Epochs %4i | Cost: %4.6e\n", i, f1);
			// s = (df2'*df2-df1' * df2) / (df1'*df1)*s - df2;      // % Polack-Ribiere direction
			FLOAT factor = (df21.sumSquare() + df22.sumSquare()) - (df11.dotProduct(df21) + df12.dotProduct(df22)) / (df11.sumSquare() + df12.sumSquare());
			s1.multiple(1.f - factor); s1.substraction(df21);
			s2.multiple(1.f - factor); s2.substraction(df22);
			
			df11.swap(df21); df12.swap(df22); // tmp = df1; df1 = df2; df2 = tmp; // % swap derivatives
			
			d2 = df11.dotProduct(s1)  + df12.dotProduct(s2); // d2 = df1'*s;
			if (d2 > 0) { // % new slope must be negative
				s1.copy(df11); s2.copy(df12); // s = -df1;// % otherwise use steepest direction
				s1.multiple(-1.f); s2.multiple(-1.f);
				d2 = -s1.sumSquare() - s2.sumSquare(); // d2 = -s'*s;    
			} // end
			z1 = z1 * min(RATIO, d1 / (d2 - realmin));// % slope ratio but max RATIO
			d1 = d2;
			ls_failed = 0;// % this line search did not fail
		}
		else {
			model.Theta1.copy(t01); model.Theta2.copy(t02); f1 = f0; df11.copy(df01); df12.copy(df02); // X = X0; f1 = f0; df1 = df0; // % restore point from before failed line search
			if (ls_failed || i > length) {// % line search failed twice in a row
				break;// % or we ran out of time, so we give up
			} //end
			df11.swap(df21); df12.swap(df22); // tmp = df1; df1 = df2; df2 = tmp;// % swap derivatives
			s1.copy(df11); s2.copy(df12); // s = -df1;// % try steepest  
			s1.multiple(-1.f); s2.multiple(-1.f);
			d1 = -s1.sumSquare() - s2.sumSquare(); // d1 = -s'*s;
			z1 = 1 / (1 - d1);
			ls_failed = 1;// % this line search failed
		}
	} // end
	printf("\n");
}

// % ======================================================================== =

// function p = predict(Theta1, Theta2, X)
// % PREDICT Predict the label of an input given a trained neural network
// % p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
// % trained weights of a neural network(Theta1, Theta2)

void ModelSingleHiddenLayer::predict(Matrix &data, Matrix& p) {
	X.swap(data);

	// % Useful values
	int m = X.h; //  m = size(X, 1);
	int num_labels = Theta2.h; //  size(Theta2, 1);

	// % You need to return the following variables correctly
	// p = zeros(size(X, 1), 1);
	p.init(X.h, 1);

	Matrix h1, h11, h2, x;
	x.combine(1.f, X);
	h1.multipleTranspose(x, Theta1);  // h1 = sigmoid([ones(m, 1) X] * Theta1');
	h1.call(sigmoid);
	h11.combine(1.f, h1);
	h2.multipleTranspose(h11, Theta2); // h2 = sigmoid([ones(m, 1) h1] * Theta2');
	h2.call(sigmoid);
	
	//p.copy(h2);

	for (int i = 0; i < h2.h; i++) {
		int n = -1;
		FLOAT f = 0;
		for (int j = 0; j < h2.w; j++) {
			if (f < h2.p[i * h2.w + j]) {
				f = h2.p[i * h2.w + j];
				n = j;
			}
			p.p[i] = (FLOAT)(n+1);
		}
	}

	// % ======================================================================== =
} // end

/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// <summary>
/// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// </summary>
/// <returns></returns>
int main() {
	if(!read_file())
		return 1;

	printf(" number of items: %zu\n", mat.size());
	for (map<string, Matrix>::iterator it = mat.begin(); it != mat.end(); it++) {
		printf(" matrix: %s, %d %d\n", it->second.name.c_str(), it->second.h, it->second.w);
	}

	ModelSingleHiddenLayer model;

	model.Theta1.move(mat["Theta1"]);
	model.Theta2 = mat["Theta2"];
	model.X.move(mat["X"]);
	model.y.move(mat["y"]);

	Matrix Theta1_grad, Theta2_grad;
	Theta1_grad.init(model.Theta1.h, model.Theta1.w, "Theta1_grad");
	Theta2_grad.init(model.Theta2.h, model.Theta2.w, "Theta1_grad");


	//%% Setup the parameters you will use for this exercise
	model.input_layer_size = 400; //% 20x20 Input Images of Digits
	model.hidden_layer_size = 25; //% 25 hidden units
	model.num_labels = 10;		//% 10 labels, from 1 to 10
								//% (note that we have mapped "0" to label 10)

	printf("'\nFeedforward Using Neural Network ...\n");

/*
	%% ================ Part 3: Compute Cost(Feedforward) ================
		% To the neural network, you should first start by implementing the
		% feedforward part of the neural network that returns the cost only.You
		% should complete the code in nnCostFunction.m to return cost.After
		% implementing the feedforward to compute the cost, you can verify that
		% your implementation is correct by verifying that you get the same cost
		% as us for the fixed debugging parameters.
		%
		% We suggest implementing the feedforward cost* without* regularization
		% first so that it will be easier for you to debug.Later, in part 4, you
		% will get to implement the regularized cost.
		%
*/
	//% Weight regularization parameter(we set this to 0 here).
	
	model.lambda = 0.f;
	FLOAT J = model.costFunction(); //  nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, Theta1_grad, Theta2_grad);

	printf("Cost at parameters (loaded from ex4weights): %f "
		"\n(this value should be about 0.287629)\n", J);

	/*
	%% ============== = Part 4: Implement Regularization ============== =
		% Once your cost function implementation is correct, you should now
		% continue to implement the regularization with the cost.
		%
	*/
	printf("\nChecking Cost Function (w/ Regularization) ... \n");

	//	% Weight regularization parameter(we set this to 1 here).
	model.lambda = 1;

	//J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda);
	J = model.costFunction(); // nnCostFunction(Theta1, Theta2, input_layer_size, hidden_layer_size, num_labels, X, y, lambda, Theta1_grad, Theta2_grad);

	printf("Cost at parameters (loaded from ex4weights): %f "
		"\n(this value should be about 0.383770)\n", J);
/*	%% ================ Part 5: Sigmoid Gradient  ================
		% Before you start implementing the neural network, you will first
		% implement the gradient for the sigmoid function.You should complete the
		% code in the sigmoidGradient.m file.
		%
*/

	printf("\nEvaluating sigmoid gradient...\n");

	/*
	g = sigmoidGradient([1 - 0.5 0 0.5 1]);
	fprintf('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:\n  ');
	fprintf('%f ', g);
	fprintf('\n\n');
	*/
/*
	%% ================ Part 6: Initializing Pameters ================
		% In this part of the exercise, you will be starting to implment a two
		% layer neural network that classifies digits.You will start by
		% implementing a function to initialize the weights of the neural network
		% (randInitializeWeights.m)
*/
	printf("\nInitializing Neural Network Parameters ...\n");

	Matrix initial_Theta1, initial_Theta2;
	randInitializeWeights(initial_Theta1, model.input_layer_size, model.hidden_layer_size); // 25 x 401
	randInitializeWeights(initial_Theta2, model.hidden_layer_size, model.num_labels);       // 10 x 26

/*
	// % Unroll parameters
	//	initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];

	%% ============== = Part 7: Implement Backpropagation ============== =
		% Once your cost matches up with ours, you should proceed to implement the
		% backpropagation algorithm for the neural network.You should add to the
		% code you've written in nnCostFunction.m to return the partial
		% derivatives of the parameters.
		%
*/
	printf("\nChecking Backpropagation... \n");

	// % Check gradients by running checkNNGradients
	checkNNGradients();

/*

	%% ============== = Part 8: Implement Regularization ============== =
		% Once your backpropagation implementation is correct, you should now
		% continue to implement the regularization with the costand gradient.
		%

		fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

		% Check gradients by running checkNNGradients
*/
	FLOAT lambda = 3.f;
	checkNNGradients(lambda);

	// % Also output the costFunction debugging values
	lambda = 3;
	FLOAT debug_J = model.costFunction();

	printf("\n\nCost at (fixed) debugging parameters (w/ lambda = 10): %f "
		"\n(this value should be about 0.576051)\n\n", debug_J);

/*	%% ================== = Part 8: Training NN ================== =
		% You have now implemented all the code necessary to train a neural
		% network.To train your neural network, we will now use "fmincg", which
		% is a function which works similarly to "fminunc".Recall that these
		% advanced optimizers are able to train our cost functions efficiently as
		% long as we provide them with the gradient computations.
		% */
	printf("\nTraining Neural Network... \n");

		//% After you have completed the assignment, change the MaxIter to a larger
		//% value to see how more training helps.
		//options = optimset('MaxIter', 50);

	//% You should also try different values of lambda
	model.lambda = 1;

	/*% Create "short hand" for the cost function to be minimized
		costFunction = @(p)nnCostFunction(p, ...
			input_layer_size, ...
			hidden_layer_size, ...
			num_labels, X, y, lambda); 

	% Now, costFunction is a function that takes in only one argument(the
		% neural network parameters) 
		[nn_params, cost] = fmincg(costFunction, initial_nn_params, options); */

	int maxIter = 50;
	fmincg(model, maxIter);

	/* % Obtain Theta1and Theta2 back from nn_params
		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
			hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))) : end), ...
		num_labels, (hidden_layer_size + 1));

	fprintf('Program paused. Press enter to continue.\n');
	pause; */

/*
	%% ================ = Part 9: Visualize Weights ================ =
		% You can now "visualize" what the neural network is learning by
		% displaying the hidden units to see what features they are capturing in
		% the data.

		fprintf('\nVisualizing Neural Network... \n')

		displayData(Theta1(:, 2 : end));

	fprintf('\nProgram paused. Press enter to continue.\n');
	pause;

	%% ================ = Part 10: Implement Predict ================ =
		% After training the neural network, we would like to use it to predict
		% the labels.You will now implement the "predict" function to use the
		% neural network to predict the labels of the training set.This lets
		% you compute the training set accuracy.
*/
	Matrix pred;
	model.predict(model.X, pred);

	int correct = 0;
	for (int i = 0; i < pred.h; i++)
	{
		if (fabs(pred.p[i] - model.y.p[i]) < 0.01)
			correct++;

		/*
		printf("\n predict %3d %f",i , y.p[i]);
		for (int j = 0; j < pred.w; j+=10) {
			printf(" %f", pred.p[i * pred.w + j]);
		}
		*/
	}
	printf("\nTraining Set Accuracy: %d %d correct = %d %5.2f%% \n", pred.h, pred.w, correct, pred.h<1?0.f:(float)correct*100.f/(float)pred.h);

/*
	printf('Program paused. Press enter to continue.\n');
	pause;

	% To give you an idea of the network's output, you can also run
		% through the examples one at the a time to see what it is predicting.

		% Randomly permute examples
		rp = randperm(m);

	for i = 1:m
		% Display
		fprintf('\nDisplaying Example Image\n');
	displayData(X(rp(i), :));

	pred = predict(Theta1, Theta2, X(rp(i), :));
	fprintf('\nNeural Network Prediction: %d (digit %d)\n', pred, mod(pred, 10));

	% Pause
		fprintf('Program paused. Press enter to continue.\n');
	pause;
	end

*/


	return 0;
}