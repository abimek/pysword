#include <pybind11/attr.h>
#include <pybind11/buffer_info.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include "gputensor.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

PYBIND11_MODULE(gputensor, m) {
    py::class_<GPUTensor>(m, "GPUTensor", py::buffer_protocol())
        .def(py::init([](py::tuple tuple, py::buffer b){
            py::buffer_info info = b.request();
            if (info.ndim != 2) {
                throw std::runtime_error("Wrong Dimension");
            }
            int x = tuple[0].cast<int>();
            int y = tuple[1].cast<int>();
            Shape shape = Shape{x, y};
            GPUTensor g = GPUTensor(shape);
            g.populate_data(static_cast<float*>(info.ptr), x*y);
            return g;
        }))
        .def_buffer([](GPUTensor &g) -> py::buffer_info {
            Shape shape = g.get_shape();
            return py::buffer_info(
                g.data(),
                sizeof(float),
                py::format_descriptor<float>::format(),
                2,
                {shape.r, shape.c},
                {sizeof(float) * shape.c, sizeof(float)}
            );
        })
        .def_property_readonly("shape", [](GPUTensor& self) -> py::tuple {
            auto tuple = py::tuple(2);
            tuple[0] = self.get_shape().r;
            tuple[1] = self.get_shape().c;
            return tuple;
        })
        .def("sum", &GPUTensor::sum)
        .def("tranpose", &GPUTensor::tranpose)
        .def("__mul__", [](GPUTensor& self, float val){
            return self.mult(val);
        })
        .def("__mul__", [](GPUTensor& self, GPUTensor& other){
            return self.mult(other);
        })
        .def("__matmul__", [](GPUTensor& self, GPUTensor& other){
            return self.matmul(other);
        })
        .def("__rmul__", [](GPUTensor& self, float val){
            return self.mult(val);
        })
        .def("__add__", [](GPUTensor& self, GPUTensor& other){
            return self.add(other);
        });
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
