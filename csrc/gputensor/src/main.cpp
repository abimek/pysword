#include <pybind11/pybind11.h>
#include "gputensor.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(gputensor, m) {
    m.doc() = R"pbdoc(add two numbers)pbdoc";

    m.def("add", &add, R"pbdoc(
        Add two numbers
        Some other explanation about the add function.
		)pbdoc");

    m.def("run", &run, R"pbdoc(
			run some cuda code and return 4
		)pbdoc");

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
