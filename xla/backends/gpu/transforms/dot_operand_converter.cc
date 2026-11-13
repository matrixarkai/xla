/* Copyright 2020 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/transforms/dot_operand_converter.h"

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "tsl/platform/tensor_float_32_utils.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/rocm/rocm_compute_capability.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

bool IsIntegralExactInBf16(PrimitiveType type) {
  return primitive_util::IsIntegralType(type) &&
         primitive_util::CastPreservesValues(type, BF16);
}

bool HasDefaultPrecision(const HloInstruction* dot) {
  const PrecisionConfig& precision_config = dot->precision_config();
  return precision_config.algorithm() == PrecisionConfig::ALG_UNSET &&
         absl::c_all_of(precision_config.operand_precision(),
                        [](int p) { return p == PrecisionConfig::DEFAULT; });
}

// Returns the target element type to which the operands of `dot` should be
// converted. Every <=8-bit integer value (S1, U1, S2, U2, S4, U4, S8, U8) is
// exactly representable in BF16. When a default-precision dot on a BF16-capable
// GPU with TF32 enabled multiplies an <=8-bit integer operand by an F32 operand
// with an F32 or BF16 output, converting both operands to BF16 avoids falling
// back to a TF32 dot while preserving the exact integer values.
PrimitiveType DesiredOperandType(const HloInstruction* dot,
                                 const se::GpuComputeCapability& gpu_version) {
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  PrimitiveType lhs_type = lhs->shape().element_type();
  PrimitiveType rhs_type = rhs->shape().element_type();
  PrimitiveType out_type = dot->shape().element_type();
  if ((out_type == F32 || out_type == BF16) && HasDefaultPrecision(dot) &&
      tsl::tensor_float_32_execution_enabled() &&
      DotOperandConverter::SupportsBf16Dot(gpu_version) &&
      ((IsIntegralExactInBf16(lhs_type) && rhs_type == F32) ||
       (lhs_type == F32 && IsIntegralExactInBf16(rhs_type)))) {
    return BF16;
  }
  return ShapeUtil::HigherPrecisionElementType(lhs->shape(), rhs->shape());
}

// Returns true if every value produced by `instr` is exactly representable in
// BF16. Looks through data-movement ops to the producing upcast or
// reduce-precision.
//
// BF16 has 8 exponent bits (the same range as F32 and TF32) and 7 explicit
// mantissa bits (8 significand bits including the implicit leading one). Hence
// a value is BF16-exact iff it fits in that exponent range and its significand
// needs at most 8 bits. For example, every s8/u8 integer (|x| <= 255 < 2^8) and
// every bf16 value is BF16-exact, while f16 (10 mantissa bits) and s16 are not.
bool IsExactlyRepresentableInBf16(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kConvert:
      // Every instruction reached here has the dot operand's F32 element type
      // (the look-through cases below require an unchanged element type), and
      // F32 holds every BF16 value. So the result is BF16-exact iff the source
      // type casts to BF16 without loss (e.g. s8, u8, s4, bf16, f8e4m3fn).
      return primitive_util::CastPreservesValues(
          instr->operand(0)->shape().element_type(), BF16);
    case HloOpcode::kReducePrecision: {
      // reduce-precision rounds to a float format with `exponent_bits` exponent
      // bits and `mantissa_bits` explicit mantissa bits. Every value of that
      // format (including its denormals) is a BF16 value iff it has at most
      // ExponentWidth(BF16) = 8 exponent bits and at most
      // SignificandWidth(BF16) - 1 = 7 explicit mantissa bits; the `- 1` is
      // because SignificandWidth counts the implicit leading bit. NaN payloads
      // may not survive the demotion, but NaNs stay NaNs.
      const auto* reduce_precision = Cast<HloReducePrecisionInstruction>(instr);
      return reduce_precision->exponent_bits() <=
                 primitive_util::ExponentWidth(BF16) &&
             reduce_precision->mantissa_bits() <=
                 primitive_util::SignificandWidth(BF16) - 1;
    }
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kCopy:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
      // These ops only move values. A bitcast may also reinterpret the bits as
      // a different element type, which does not preserve values, so only look
      // through ops that keep the element type.
      if (instr->operand(0)->shape().element_type() !=
          instr->shape().element_type()) {
        return false;
      }
      return IsExactlyRepresentableInBf16(instr->operand(0));
    default:
      // Conservatively only trust the element type of any other producer.
      // Since the dot operand chain is F32, this is false in practice.
      return primitive_util::CastPreservesValues(instr->shape().element_type(),
                                                 BF16);
  }
}

// Returns true if the F32 x F32 `dot` can be demoted to an exact BF16 x BF16
// dot with the same output type. This holds when both operands only carry
// BF16-exact values (e.g. `convert<f32>(s8)` and `convert<f32>(bf16)`) and the
// dot already runs with TF32 inputs, i.e. it has no explicit algorithm, every
// operand precision is at most HIGH and TF32 execution is enabled (the same
// condition the cuBLAS, cuDNN and Triton emitters use to allow TF32).
//
// TF32 has 8 exponent bits and 10 explicit mantissa bits, a superset of BF16's
// 8 exponent and 7 mantissa bits, so the TF32 path already sees exactly the
// BF16-exact input values. The product of two values with 8-bit significands
// needs at most 16 significand bits, so it is exact in F32 (24 bits). BF16 and
// TF32 tensor-core MMAs both form these exact products and accumulate them in
// F32, so the demotion does not change the dot's numerics (beyond the
// accumulation order of whichever kernel config gets picked), while halving
// the operand footprint in registers and shared memory.
bool IsLosslessBf16Demotion(const HloInstruction* dot,
                            const se::GpuComputeCapability& gpu_version) {
  // Sparse and block-scaled dots carry extra (metadata or scale) operands; only
  // handle plain lhs x rhs dots.
  if (dot->operand_count() != 2) {
    return false;
  }
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  const PrimitiveType out_type = dot->shape().element_type();
  if (lhs->shape().element_type() != F32 ||
      rhs->shape().element_type() != F32 ||
      (out_type != F32 && out_type != BF16)) {
    return false;
  }
  const PrecisionConfig& precision_config = dot->precision_config();
  if (precision_config.algorithm() != PrecisionConfig::ALG_UNSET ||
      !absl::c_all_of(precision_config.operand_precision(),
                      [](int p) { return p <= PrecisionConfig::HIGH; }) ||
      !tsl::tensor_float_32_execution_enabled() ||
      !DotOperandConverter::SupportsBf16Dot(gpu_version)) {
    return false;
  }
  return IsExactlyRepresentableInBf16(lhs) && IsExactlyRepresentableInBf16(rhs);
}

// Returns a BF16 instruction with the same values as the BF16-exact F32
// `operand`, folding a direct upcast into a single convert from its source.
// Converting a BF16-exact value to BF16 is exact, since round-to-nearest-even
// leaves a value that is already representable unchanged.
HloInstruction* DemoteToBf16(HloInstruction* operand) {
  if (operand->opcode() == HloOpcode::kConvert &&
      primitive_util::CastPreservesValues(
          operand->operand(0)->shape().element_type(), BF16)) {
    operand = operand->mutable_operand(0);
    if (operand->shape().element_type() == BF16) {
      return operand;
    }
  }
  return operand->AddInstruction(HloInstruction::CreateConvert(
      ShapeUtil::ChangeElementType(operand->shape(), BF16), operand));
}

}  // namespace

bool DotOperandConverter::SupportsBf16Dot(
    const se::GpuComputeCapability& gpu_version) {
  if (const auto* cuda_cc = gpu_version.cuda_compute_capability()) {
    return cuda_cc->IsAtLeastAmpere();
  }
  if (const auto* rocm_cc = gpu_version.rocm_compute_capability()) {
    return rocm_cc->has_hipblaslt() && rocm_cc->gfx9_mi300_series();
  }
  return false;
}

bool DotOperandConverter::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (HloPredicateIsNotOp<HloOpcode::kDot>(instruction)) {
    return false;
  }
  HloInstruction* lhs = instruction->mutable_operand(0);
  HloInstruction* rhs = instruction->mutable_operand(1);

  PrimitiveType lhs_type = lhs->shape().element_type();
  PrimitiveType rhs_type = rhs->shape().element_type();

  if (lhs_type == rhs_type) {
    return IsLosslessBf16Demotion(instruction, gpu_version_);
  }

  // Exclude conversions between FP8 types.
  absl::flat_hash_set<PrimitiveType> non_converting = {F8E4M3FN, F8E5M2,
                                                       F8E4M3FNUZ, F8E5M2FNUZ};
  return !(non_converting.contains(lhs_type) &&
           non_converting.contains(rhs_type));
}

absl::StatusOr<HloInstruction*> DotOperandConverter::ExpandInstruction(
    HloInstruction* instruction) {
  if (IsLosslessBf16Demotion(instruction, gpu_version_)) {
    for (int i : {0, 1}) {
      ABSL_RETURN_IF_ERROR(instruction->ReplaceOperandWithDifferentShape(
          i, DemoteToBf16(instruction->mutable_operand(i))));
    }
    return nullptr;
  }
  PrimitiveType desired_type = DesiredOperandType(instruction, gpu_version_);
  for (int i : {0, 1}) {
    HloInstruction* operand = instruction->mutable_operand(i);
    if (operand->shape().element_type() == desired_type) {
      continue;
    }
    Shape new_shape =
        ShapeUtil::ChangeElementType(operand->shape(), desired_type);
    HloInstruction* convert_inst = instruction->AddInstruction(
        HloInstruction::CreateConvert(new_shape, operand));
    ABSL_RETURN_IF_ERROR(
        instruction->ReplaceOperandWithDifferentShape(i, convert_inst));
  }
  return nullptr;
}

}  // namespace xla::gpu
