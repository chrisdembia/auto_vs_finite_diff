#include <adolc/adolc.h>
#include <iostream>
#include <iterator>

template <typename T>
void func(int n, int m, const T* x, T* y) {
    y[m-1] = 0;
    for (int i = 0; i < n; ++i) {
        y[m-1] += x[i] * x[i];
    }
}

void auto_diff(int n, int m, const double* px, double** J) {
    short int tag = 0;
    trace_on(tag);

    std::vector<adouble> x(n);
    std::vector<adouble> y(m);
    std::vector<double> py(m);

    for (int i = 0; i < n; ++i) x[i] <<= px[i];

    func(n, m, x.data(), y.data());

    for (int j = 0; j < m; ++j) y[j] >>= py[j];

    trace_off();

    int success = jacobian(tag, m, n, px, J);
}

void finite_diff(int n, int m, const double* x, double** J) {
    std::vector<double> f(m);
    func(n, m, x, f.data());
    std::vector<double> f_perturb(m);
    double h = 1e-9; // sweet spot!
    for (int j = 0; j < n; ++j) {
        // TODO use the memory once; just keep twiddling with the elements.
        std::vector<double> x_perturb(x, x + n);
        x_perturb[j] += h;
        func(n, m, x_perturb.data(), f_perturb.data());
        for (int i = 0; i < m; ++i) {
            J[i][j] = (f_perturb[i] - f[i]) / h;
        }
    }
}

std::ostream& operator<<(std::ostream& o, std::vector<double>& v) {
    for (const auto& e : v) { o << e; }
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

int main() {
    // TODO random.
    // TODO first do it on a dense matrix.
    std::vector<double> x {1, 2, 3, 4, 5};
    std::vector<double> y(1);
    func(x.size(), y.size(), x.data(), y.data());
    std::cout << "f(x): " << y << std::endl;

    // TODO  double** J = new double[x.size()][y.size()];
    double** J = new double*[y.size()];
    for (int i = 0; i < y.size(); ++i) J[i] = new double[x.size()];

    auto_diff(x.size(), y.size(), x.data(), J);
    print_and_clear(x.size(), y.size(), J);

    finite_diff(x.size(), y.size(), x.data(), J);
    print_and_clear(x.size(), y.size(), J);

    // TODO leak delete[][] J;
    return EXIT_SUCCESS;
}
