#include <iostream>
#include <string>
#include <chrono>
#include <pybind11/embed.h>

namespace py = pybind11;

PYBIND11_EMBEDDED_MODULE(sqrt_module_embedded, m) {
    m.def("sqrt", [](double x) {
        if (x < 0) throw std::domain_error("Cannot compute square root of negative number");
        return std::sqrt(x);
    });
}

int main() {
    double value = 16.0;

    py::scoped_interpreter guard{};

    // Using custom defined function from sqrt_module.py
    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, "/root/workspace/example");

    auto start_1 = std::chrono::high_resolution_clock::now();

    py::module_ sqrt_module = py::module_::import("sqrt_module");
    py::object result = sqrt_module.attr("sqrt")(value);

    double result_double = result.cast<double>();
    auto end_1 = std::chrono::high_resolution_clock::now();
    auto duration_1 = std::chrono::duration_cast<std::chrono::milliseconds>(end_1 - start_1);
    std::cout << "Square root of " << value << " is " << result_double << std::endl;
    std::cout << "Time taken: " << duration_1.count() << " milliseconds" << std::endl;


    // Using embedded python 
    auto start_2 = std::chrono::high_resolution_clock::now();
    py::object sqrt_py = py::module_::import("sqrt_module_embedded").attr("sqrt");
    double result_embedded = sqrt_py(value).cast<double>();
    auto end_2 = std::chrono::high_resolution_clock::now();
    auto duration_2 = std::chrono::duration_cast<std::chrono::milliseconds>(end_2 - start_2);
    std::cout << "Square root of " << value << " (embedded) is " << result_embedded << std::endl;
    std::cout << "Time taken: " << duration_2.count() << " milliseconds" << std::endl;
}
