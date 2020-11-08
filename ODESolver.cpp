#include <vector>
#include <iostream>
#include <iomanip>

class ODESolver {
private:
	using Vector = std::vector<double>;
	Vector T;
	Vector X;
	Vector currX, nextX;
	Vector currFn, halfFn, nextFn;
	size_t nX = size_t();
	size_t nT = size_t();
	size_t currentStep = size_t();
	template <class Fn>
	void solveRK3(Fn&& f) {
		// Time-step handling
		double currT = T[currentStep];
		auto nextT = T[++currentStep];
		auto halfT = 0.5 * (currT + nextT);
		auto dT = nextT - currT;
		auto dT_2 = 0.5 * dT;
		auto dT_6 = dT / 6.0;
		// RK3 scheme
		for (auto i = size_t(); i < nX; ++i) {
			nextX[i] = currX[i] + dT_2 * currFn[i];
		}
		f(halfT, nextX, halfFn);
		for (auto i = size_t(); i < nX; ++i) {
			nextX[i] = currX[i] + dT * (2.0 * halfFn[i] - currFn[i]);
		}
		f(nextT, nextX, nextFn);
		for (auto i = size_t(); i < nX; ++i) {
			currX[i] = currX[i] + dT_6 * (currFn[i] + 4.0 * halfFn[i] + nextFn[i]);
		}
		// Update
		f(nextT, currX, currFn); 
		memcpy(&X[nX * currentStep], currX.data(), nX * sizeof(double));
	}
public:
	ODESolver() {};
	void setInitialCondition(Vector const& _X0) {
		nX = _X0.size();
		// Memory allocation for the solver.
		currX.resize(nX);
		nextX.resize(nX);
		currFn.resize(nX);
		halfFn.resize(nX);
		nextFn.resize(nX);
		// Storing the initial conditions of the ODE.
		currX = _X0;
	}
	void setTimeGrid(Vector const& _T) {
		T = _T;
		nT = T.size();
	}
	void setUniformTimeGrid(const double Tmin, const double Tmax, const size_t nSteps) {
		T.resize(nSteps + 1);
		auto dT = (Tmax - Tmin) / double(nSteps);
		for (auto i = size_t(); i < nSteps; ++i) {
			T[i] = Tmin + double(i) * dT;
		}
		T[0] = Tmin;
		T[nSteps] = Tmax;
		nT = nSteps + 1;
	}
	template <class Fn>
	void solveODE(Fn&& f) {
		// Memory allocation for the solver.
		X.resize(nX * nT);
		// Initial conditions of the ODE.
		currentStep = size_t();
		memcpy(X.data(), currX.data(), nX * sizeof(double));
		f(T[0], currX, currFn);
		// Loop on time steps.
		while (currentStep < nT - 1) {
			solveRK3(f);
		}
	}
	void printSolution() const {
		std::cout << std::setprecision(15);
		// Header
		std::cout << "T";
		for (auto i = size_t(); i < nX; ++i) {
			std::cout << ",X_" << i;
		}
		std::cout << std::endl;
		// Each line contains the solution at t: t, X(t)
		for (auto i = size_t(); i < nT; ++i) {
			std::cout << T[i];
			for (auto j = size_t(); j < nX; ++j) {
				std::cout << ',' << X[i * nX + j];
			}
			std::cout << std::endl;
		}
		(void)getchar();
	}
};

int main() {
	using Vector = std::vector<double>;
	for (int i = 0; i < 10; ++i) {
		Vector Initial_Condition(25000, 1.0);
		size_t nT = 10000;
		auto f = [&](double const T, Vector const& X, Vector & Y) {
			for (auto i = size_t(); i < X.size(); ++i)
				Y[i] = 5.0 * X[i];
		};
		ODESolver solver;
		solver.setInitialCondition(Initial_Condition);
		solver.setUniformTimeGrid(0.0, 1.0, nT);
		// Solves X'(t) = f(t, X(t))
		solver.solveODE(f);
		//solver.printSolution();
	}
	return 0;
}
