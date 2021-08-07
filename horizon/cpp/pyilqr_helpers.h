#ifndef PYILQR_HELPERS_H
#define PYILQR_HELPERS_H

#include "src/ilqr.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using namespace horizon;

casadi::Function to_cpp(py::object pyfn)
{
    // convert python's casadi.Function to cpp's casadi::Function
#if PYBIND11_VERSION_MINOR > 6
    auto cs = py::module_::import("casadi");
#else
    auto cs = py::module::import("casadi");
#endif
    auto Function = cs.attr("Function");
    auto serialize = Function.attr("serialize");
    auto fstr = serialize(pyfn).cast<std::string>();

    return casadi::Function::deserialize(fstr);
}

auto construct(py::object fdyn, int N)
{
    return std::make_unique<IterativeLQR>(to_cpp(fdyn), N);
}

auto set_inter_cost_wrapper(IterativeLQR& self, std::vector<py::object> flist)
{
    std::vector<casadi::Function> flist_cpp;
    for(auto pyfn : flist)
    {
        flist_cpp.push_back(to_cpp(pyfn));
    }

    self.setIntermediateCost(flist_cpp);
}

auto set_inter_cost_wrapper_single(IterativeLQR& self, int k, py::object f)
{
    self.setIntermediateCost(k, to_cpp(f));
}

auto set_final_cost_wrapper(IterativeLQR& self, py::object pyfn)
{
    self.setFinalCost(to_cpp(pyfn));
}

auto set_final_constraint_wrapper(IterativeLQR& self, py::object pyfn)
{
    self.setFinalConstraint(to_cpp(pyfn));
}

auto set_inter_constraint_wrapper(IterativeLQR& self, std::vector<py::object> flist)
{
    std::vector<casadi::Function> flist_cpp;
    for(auto pyfn : flist)
    {
        flist_cpp.push_back(to_cpp(pyfn));
    }

    self.setIntermediateConstraint(flist_cpp);
}

auto set_inter_constraint_wrapper_single(IterativeLQR& self, int k, py::object f)
{
    self.setIntermediateConstraint(k, to_cpp(f));
}

#endif // PYILQR_HELPERS_H
