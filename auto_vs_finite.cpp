#include <adolc/adolc.h>
#include <iostream>
#include <iterator>
#include <random>
#include <cassert>

// TODO profile second evaluate of the jacobian.

/// This is a dense function R^n -> R^m. It is templatized so that we can
/// evaluate it on both double and adouble.
template <typename T>
void constraint_function_dense(int n, int m, const T* x, T* y) {
    for (int j = 0; j < m; ++j) y[j] = 0; 
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            y[j] += log(j + 2) * cos(x[i]) * x[(n - 1) - i];
        }
    }
}

/// This differentiates a function R^n -> R^m using ADOL-C.
void auto_jacobian(int n, int m, const double* px, double** J) {

    short int tag = 0;

    // Start recording.
    trace_on(tag);

    std::vector<adouble> x(n);
    std::vector<adouble> y(m);
    std::vector<double> py(m); // p for "passive variable;" ADOL-C terminology.

    // Indicate independent variables.
    for (int i = 0; i < n; ++i) x[i] <<= px[i];

    // Evaluate function.
    constraint_function_dense(n, m, x.data(), y.data());

    // Indicate dependent variables.
    for (int j = 0; j < m; ++j) y[j] >>= py[j];

    // Stop recording.
    trace_off();

    // Use the recorded tape to compute the jacobian.
    int success = jacobian(tag, m, n, px, J);
    assert(success == 3);
}

/// This differentiates a function R^n -> R^m using central finite differences.
void finite_jacobian(int n, int m, const double* x, double** J) {

    // Working memory for storing the perturbations.
    std::vector<double> f_perturb_right(m);
    std::vector<double> f_perturb_left(m);
    std::vector<double> x_perturb(x, x + n); 
    double h = 1e-9; // time step. sweet spot based on quick testing.
    // Loop through the independent variables.
    for (int j = 0; j < n; ++j) {
        x_perturb[j] += h;
        constraint_function_dense(n, m, x_perturb.data(), f_perturb_right.data());
        // 2 * h to cancel out the +h perturbation.
        x_perturb[j] -= 2 * h;
        constraint_function_dense(n, m, x_perturb.data(), f_perturb_left.data());
        // Restore this element to its original value, in preparation for the
        // next perturbation.
        x_perturb[j] = x[j];

        // Record the derivative for each row i for this column j.
        for (int i = 0; i < m; ++i) {
            // df/dx = [f(x + h) - f(x + h)] / (2h)
            J[i][j] = (f_perturb_right[i] - f_perturb_left[i]) / (2 * h);
        }
    }
}

std::ostream& operator<<(std::ostream& o, std::vector<double>& v) {
    for (const auto& e : v) { o << e << " "; }
    return o;
}

void print_and_clear(int n, int m, double* const * J) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) { 
            std::cout << J[i][j] << " ";
            // Clear for the next method.
            J[i][j] = 0;
        }
        std::cout << std::endl;
    }
}

template<typename Function>
double timeit(Function f) {
    using namespace std::chrono;
    auto start = steady_clock::now();
    // ------------------------------ //
    /* calling */   f();  /* function */
    // ------------------------------ //
    auto end = steady_clock::now();
    auto dur = duration_cast<milliseconds>(end - start).count();
    std::cout << "Duration: " << dur << " milliseconds" << std::endl;
    return dur;
}

int main() {
    // Benchmarking on a dense jacobian.
    int n = 1000; int m = 1000; // i.e., 100 nodes, 10 states

    // Generate a random point x.
    // For consistent results, use same seed each time.
    std::default_random_engine generator(0); 
    // Uniform distribution between 0.1 and 0.9.
    std::uniform_real_distribution<double> distribution(-0.5, 0.5);
    std::vector<double> x(n);
    std::generate(x.begin(), x.end(), [&]() { return distribution(generator); });

    // For fun, evaluate the constraint function at x.
    std::vector<double> y(m);
    constraint_function_dense(x.size(), y.size(), x.data(), y.data());
    // TODO std::cout << "f(x): " << y << std::endl;

    // TODO double** J = new double[x.size()][y.size()];
    double** J = new double*[y.size()];
    for (int i = 0; i < y.size(); ++i) J[i] = new double[x.size()];

    std::cout << "Computing with automatic differentiation... ";
    auto autodur = timeit([&]() {
            auto_jacobian(x.size(), y.size(), x.data(), J);
            });
    // std::cout << "J(x):" << std::endl;
    // print_and_clear(x.size(), y.size(), J);

    std::cout << "Computing with finite differences... ";
    auto finitedur = timeit([&]() {
            finite_jacobian(x.size(), y.size(), x.data(), J);
            });
    // std::cout << "J(x):" << std::endl;
    // print_and_clear(x.size(), y.size(), J);

    std::cout << "finite / auto: " << finitedur / autodur << std::endl;

    // TODO fix leak delete[][] J;
    return EXIT_SUCCESS;
}
