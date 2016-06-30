#include <adolc/adolc.h>
#include <adolc/adolc_sparse.h>
#include <iostream>
#include <iterator>
#include <random>
#include <cassert>
#include <algorithm>

// TODO profile repeated evaluated of the jacobian with autodiff, which might
// be even faster.

/// This is a dense function R^n -> R^m. It is templatized so that we can
/// evaluate it on both double and adouble.
template <typename T>
void constraint_function_dense(int n, int m, const T* x, T* y) {
    for (int j = 0; j < m; ++j) {
        y[j] = 0; // Clean up the given memory.
        for (int i = 0; i < n; ++i) {
            y[j] += log(j + 2) * cos(x[i]) * exp(-x[(n - 1) - i]);
        }
    }
}

/// This is a sparse tridiagonal function R^n -> R^m. It is templatized so that
/// we can evaluate it on both double and adouble.
template <typename T>
void constraint_function_tridiag(int n, int m, const T* x, T* y) {
    int dim = std::min(n, m);
    for (int j = 0; j < dim; ++j) {
        y[j] = 0; // Clean up the given memory.
        // For all but the first element, depende on the previous element.
        if (j > 0) y[j] += x[j - 1];/*cos(x[i + 1]);*/
        // The j-th dependent variable depends on the j-th independent var.
        y[j] += x[j]; // TODO
        // For all but the last element, depend on the next element.
        if (j < dim - 1) y[j] += x[j + 1]; /* TODO */
        /*if (j < dim) log(x[i]
        for (int i = j; i < n; ++i) {
            y[j] += log(j + 2) * cos(x[i]) * x[(n - 1) - i];
        }
        */
    }
}

/// This differentiates a function R^n -> R^m at px using ADOL-C.
void auto_jacobian(int n, int m, const double* px, double** J) {

    short int tag = 0;

    // Start recording information for computing derivatives.
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

/// Use ADOL-C's sparse jacobian driver to differentiate a function R^n -> R^m
/// at px.
void auto_sparse_jacobian(int n, int m, const double* px) {
    short int tag = 0;

    // Start recording information for computing derivatives.
    trace_on(tag);

    std::vector<adouble> x(n);
    std::vector<adouble> y(m);
    std::vector<double> py(m); // p for "passive variable;" ADOL-C terminology.

    // Indicate independent variables.
    for (int i = 0; i < n; ++i) x[i] <<= px[i];

    constraint_function_dense(n, m, x.data(), y.data());

    // Indicate dependent variables. Not actually interested in px.
    for (int j = 0; j < m; ++j) y[j] >>= py[j];

    // Stop recording.
    trace_off();
    // TODO myalloc2

    int repeatedCall = 0;
    int numNonZeros = -1;
    unsigned int* rowIndices = nullptr;
    unsigned int* colIndices = nullptr;
    double* J = nullptr;
    int options[4];
    options[0] = 0;          /* sparsity pattern by index domains (default) */ 
    options[1] = 0;          /*                         safe mode (default) */ 
    options[2] = 0;          /*              not required if options[0] = 0 */ 
    options[3] = 0;          /*                column compression (default) */ 

    int success = sparse_jac(tag, m, n, repeatedCall, px, &numNonZeros,
                             &rowIndices, &colIndices, &J, options);

    // TODO
    //printf("In sparse format:\n");
    //for (i=0;i<nnz;i++)
    //    printf("%2d %2d %10.6f\n\n",rind[i],cind[i],values[i]);

    //free(rind); rind=NULL;
    //free(cind); cind=NULL;
    //free(values); values=NULL;

}

void checkNumericallyEqual(int n, int m, double** J1, double** J2) {
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
            // Somewhat borrowed from Simbody.
            auto scale = std::max(std::abs(J1[i][j]), std::abs(J2[i][j]));
            if (std::abs(J1[i][j] - J2[i][j]) > scale * 1e-3)
                throw std::runtime_error("not numerically equal.");
        }
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

        // Perturb to the right.
        x_perturb[j] += h;
        constraint_function_dense(n, m, x_perturb.data(), f_perturb_right.data());
        
        // Perturb to the left. "2 * h" to cancel out the +h perturbation.
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
    for (const auto& e : v) o << e << " "; 
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
    int n = 20; int m = 10000; // i.e., 100 nodes, 10 states

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

    std::cout << "Computing with automatic differentiation... ";
    double** Jauto = myalloc(m, n);
    auto autodur = timeit([&]() {
            auto_jacobian(x.size(), y.size(), x.data(), Jauto);
            });
    // std::cout << "J(x):" << std::endl;
    // print_and_clear(x.size(), y.size(), Jauto);

    std::cout << "Computing with sparse automatic differentiation... ";
    // double** Jspauto = new double*[y.size()];
    // for (int i = 0; i < y.size(); ++i) Jspauto[i] = new double[x.size()];
    auto spautodur = timeit([&]() {
            auto_sparse_jacobian(x.size(), y.size(), x.data());
            });

    std::cout << "Computing with finite differences... ";
    double** Jfinite = myalloc(m, n);
    auto finitedur = timeit([&]() {
            finite_jacobian(x.size(), y.size(), x.data(), Jfinite);
            });
    // std::cout << "J(x):" << std::endl;
    // print_and_clear(x.size(), y.size(), Jfinite);

    std::cout << "finite / auto: " << finitedur / autodur << std::endl;

    checkNumericallyEqual(n, m, Jauto, Jfinite);

    myfree(Jauto);
    myfree(Jfinite);

    // TODO fix leak delete[][] J;
    return EXIT_SUCCESS;
}
