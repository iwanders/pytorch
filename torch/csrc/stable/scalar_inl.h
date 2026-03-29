#pragma once


#include <torch/csrc/stable/c/shim.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/tensor_struct.h>

#include <torch/csrc/stable/stableivalue_conversions.h>



HIDDEN_NAMESPACE_BEGIN(torch, stable)

#define DEFINE_CONSTRUCTOR_BODY(cpp_type, _unused_scalartype) \
  inline Scalar::Scalar(cpp_type value): type_(c10::CppTypeToScalarType<cpp_type>::value), value_(torch::stable::detail::from(value)) {}

  AT_FORALL_SCALAR_TYPES(DEFINE_CONSTRUCTOR_BODY)



HIDDEN_NAMESPACE_END(torch, stable)
