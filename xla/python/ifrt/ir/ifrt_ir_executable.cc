/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/python/ifrt/ir/ifrt_ir_executable.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/WalkResult.h"
#include "tsl/profiler/lib/traceme.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array_spec.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/compilation/program_lowering.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ops.h"
#include "xla/python/ifrt/ir/program_interpreter.h"
#include "xla/python/ifrt/ir/utils.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/sharding.h"
#include "xla/python/ifrt/with_user_context.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

using ::tsl::profiler::TraceMe;

char IfrtIrExecutable::ID = 0;

IfrtIrExecutable::IfrtIrExecutable(
    std::string program_name,
    absl_nonnull std::shared_ptr<AtomExecutableMap>
        bound_atom_program_executables,
    absl_nonnull std::shared_ptr<CompiledAtomProgramMap> compiled_atom_programs,
    std::vector<ArraySpec> in_specs, std::vector<ArraySpec> out_specs,
    absl::Status layout_status, std::vector<int> donatable_input_indices,
    absl_nonnull std::shared_ptr<const IfrtIRProgram> program,
    DeviceListRef devices,
    absl_nonnull std::shared_ptr<IfrtIRCompileOptions> compile_options)
    : program_name_(std::move(program_name)),
      bound_atom_program_executables_(
          std::move(bound_atom_program_executables)),
      compiled_atom_programs_(std::move(compiled_atom_programs)),
      in_specs_(std::move(in_specs)),
      out_specs_(std::move(out_specs)),
      layout_status_(std::move(layout_status)),
      donatable_input_indices_(std::move(donatable_input_indices)),
      program_(std::move(program)),
      devices_(std::move(devices)),
      compile_options_(std::move(compile_options)) {
  CHECK_NE(bound_atom_program_executables_, nullptr);
  CHECK_NE(compiled_atom_programs_, nullptr);
  CHECK_NE(program_, nullptr);
  CHECK_NE(compile_options_, nullptr);
}

tsl::Future<ExecutableRef> IfrtIrExecutable::Create(
    absl_nonnull std::unique_ptr<IfrtIRProgram> ifrt_ir_program,
    absl_nonnull std::unique_ptr<IfrtIRCompileOptions> compile_options,
    Client* absl_nonnull client,
    absl_nonnull std::shared_ptr<AtomProgramCompiler> atom_program_compiler) {
  TraceMe traceme("IfrtIrExecutable::Create");

  ABSL_ASSIGN_OR_RETURN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(std::move(ifrt_ir_program), std::move(compile_options),
                         client, std::move(atom_program_compiler),
                         /*defer_atom_program_loading=*/true));
  CHECK_NE(state.compiled_atom_program_future_map, nullptr);

  std::vector<tsl::Future<>> ready_futures;
  ready_futures.reserve(state.atom_executable_future_map->size() +
                        state.compiled_atom_program_future_map->size());
  for (const auto& [key, executable_future] :
       tsl::KeySortedRange(*state.atom_executable_future_map)) {
    ready_futures.push_back(executable_future.GetReadyFuture());
  }
  for (const auto& [key, compiled_atom_future] :
       tsl::KeySortedRange(*state.compiled_atom_program_future_map)) {
    ready_futures.push_back(compiled_atom_future.GetReadyFuture());
  }
  // `client` outlives all compilation tasks per `xla::ifrt::Compiler`'s
  // lifetime contract (callers must await all returned compilation futures
  // before destroying the compiler and client).
  auto create_executable = [state = std::move(state),
                            client]() mutable -> absl::StatusOr<ExecutableRef> {
    auto bound_atom_map = std::make_shared<AtomExecutableMap>();
    for (const auto& [key, executable_future] :
         tsl::KeySortedRange(*state.atom_executable_future_map)) {
      CHECK(executable_future.IsReady());
      ABSL_ASSIGN_OR_RETURN(LoadedExecutableRef executable,
                            executable_future.Await());
      bound_atom_map->insert({key, std::move(executable)});
    }
    CHECK_NE(state.compiled_atom_program_future_map, nullptr);
    auto compiled_atom_map = std::make_shared<CompiledAtomProgramMap>();
    for (const auto& [key, compiled_atom_future] :
         tsl::KeySortedRange(*state.compiled_atom_program_future_map)) {
      CHECK(compiled_atom_future.IsReady());
      ABSL_ASSIGN_OR_RETURN(
          absl_nonnull std::shared_ptr<const CompiledAtomProgram> compiled_atom,
          compiled_atom_future.Await());
      if (compiled_atom == nullptr || compiled_atom->executable == nullptr ||
          compiled_atom->load_options == nullptr) {
        return absl::InternalError(absl::StrFormat(
            "AtomProgramCompiler returned null compiled atom program for `%s`",
            key));
      }
      if (bound_atom_map->contains(key)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Compiled atom program `%s` collides with a bound executable",
            key));
      }
      compiled_atom_map->insert({key, std::move(compiled_atom)});
    }

    mlir::SymbolTableCollection symbol_table;
    absl::Status call_validation_status = absl::OkStatus();
    GetMainFunction(state.ifrt_ir_program->mlir_module)
        .walk([&](CallLoadedExecutableOp call_op) -> mlir::WalkResult {
          LoadedExecutableOp loaded_exec_op = call_op.getCalleeOp(symbol_table);
          if (loaded_exec_op == nullptr) {
            call_validation_status = absl::InvalidArgumentError(
                absl::StrFormat("Could not find LoadedExecutableOp `%s`",
                                call_op.getCallee().getRootReference().str()));
            return mlir::WalkResult::interrupt();
          }
          absl::string_view atom_name = loaded_exec_op.getSymName();
          if (!bound_atom_map->contains(atom_name) &&
              !compiled_atom_map->contains(atom_name)) {
            call_validation_status = absl::InvalidArgumentError(absl::StrFormat(
                "Could not find bound or compiled executable for `%s`",
                atom_name));
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
    ABSL_RETURN_IF_ERROR(call_validation_status);

    absl::Status layout_status = PopulateIfrtIrProgramLayouts(
        state.ifrt_ir_program->mlir_module, client, *bound_atom_map,
        compiled_atom_map.get(), absl::MakeSpan(state.in_specs),
        absl::MakeSpan(state.out_specs), state.device_list);
    if (!layout_status.ok()) {
      for (ArraySpec& spec : state.in_specs) {
        spec.layout = nullptr;
      }
      for (ArraySpec& spec : state.out_specs) {
        spec.layout = nullptr;
      }
    }

    return ExecutableRef(new IfrtIrExecutable(
        std::move(state.program_name), std::move(bound_atom_map),
        std::move(compiled_atom_map), std::move(state.in_specs),
        std::move(state.out_specs), std::move(layout_status),
        std::move(state.donatable_input_indices),
        std::shared_ptr<const IfrtIRProgram>(std::move(state.ifrt_ir_program)),
        std::move(state.device_list), std::move(state.compile_options)));
  };
  return tsl::JoinFutures(ready_futures)
      .Map(GetIfrtIrCompilationExecutor(),
           WithCurrentUserContext(std::move(create_executable)));
}

absl::StatusOr<absl_nonnull std::shared_ptr<CompiledIfrtIrProgram>>
IfrtIrExecutable::CreateCompiledIfrtIrProgram(
    Client* absl_nonnull client,
    AtomExecutableMap newly_loaded_atom_executables) const {
  if (newly_loaded_atom_executables.size() != compiled_atom_programs_->size()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected %d loaded atom executables for program `%s`, got %d",
        compiled_atom_programs_->size(), program_name_,
        newly_loaded_atom_executables.size()));
  }

  auto atom_executable_map = std::make_shared<AtomExecutableMap>(
      bound_atom_program_executables_->begin(),
      bound_atom_program_executables_->end());
  for (auto& [name, loaded_executable] :
       tsl::KeySortedRange(newly_loaded_atom_executables)) {
    if (!compiled_atom_programs_->contains(name)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Unexpected atom executable `%s` for program `%s`",
                          name, program_name_));
    }
    if (loaded_executable == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Loaded atom executable `%s` for program `%s` must not be null", name,
          program_name_));
    }
    CHECK(atom_executable_map->insert({name, std::move(loaded_executable)})
              .second);
  }

  auto context = std::make_unique<mlir::MLIRContext>(
      mlir::MLIRContext::Threading::DISABLED);
  context->enableThreadSafeLocking();
  ABSL_ASSIGN_OR_RETURN(
      mlir::OwningOpRef<mlir::ModuleOp> cloned_module,
      CloneModuleIntoContext(program_->mlir_module, *context));
  auto cloned_program = std::make_unique<IfrtIRProgram>(
      std::move(context), std::move(cloned_module));

  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<ProgramInterpreter> interpreter,
                        ProgramInterpreter::Create(
                            client, program_name_, cloned_program->mlir_module,
                            atom_executable_map, devices_));
  ABSL_ASSIGN_OR_RETURN(ProgramInterpreter::ExecuteFn execute_fn,
                        interpreter->BuildExecuteFn());

  // Clone `compile_options_` so `IfrtIrLoadedExecutable` subclasses clearing
  // `loaded_exec_binding` on destruction do not mutate `compile_options_`
  // across multiple `Client::Load` calls on the same `IfrtIrExecutable`.
  auto cloned_compile_options =
      std::make_shared<IfrtIRCompileOptions>(*compile_options_);
  return std::make_shared<CompiledIfrtIrProgram>(CompiledIfrtIrProgram{
      /*program_name=*/program_name_,
      /*atom_program_executables=*/std::move(atom_executable_map),
      /*in_specs=*/in_specs_,
      /*out_specs=*/out_specs_,
      /*layout_status=*/layout_status_,
      /*donatable_input_indices=*/donatable_input_indices_,
      /*program=*/std::move(cloned_program),
      /*devices=*/devices_,
      /*compile_options=*/std::move(cloned_compile_options),
      /*execute_fn=*/std::move(execute_fn),
  });
}

absl::StatusOr<std::string> IfrtIrExecutable::Serialize() const {
  return absl::UnimplementedError(
      "IfrtIrExecutable::Serialize is not implemented");
}

absl::StatusOr<xla::CompiledMemoryStats>
IfrtIrExecutable::GetCompiledMemoryStats() const {
  return absl::UnimplementedError(
      "IfrtIrExecutable does not support GetCompiledMemoryStats()");
}

std::optional<std::vector<xla::OpSharding>>
IfrtIrExecutable::GetParameterShardings() const {
  std::vector<xla::OpSharding> parameter_shardings;
  parameter_shardings.reserve(in_specs_.size());
  for (size_t idx = 0; idx < in_specs_.size(); ++idx) {
    const ArraySpec& spec = in_specs_[idx];
    const auto* hlo_sharding =
        dyn_cast_or_null<HloSharding>(spec.sharding.get());
    if (hlo_sharding == nullptr) {
      LOG(ERROR) << "Failed to get sharding of parameter #" << idx
                 << ". Sharding is not an HloSharding.";
      return std::nullopt;
    }
    parameter_shardings.push_back(hlo_sharding->xla_hlo_sharding().ToProto());
  }
  return parameter_shardings;
}

std::optional<std::vector<xla::OpSharding>>
IfrtIrExecutable::GetOutputShardings() const {
  std::vector<xla::OpSharding> output_shardings;
  output_shardings.reserve(out_specs_.size());
  for (size_t idx = 0; idx < out_specs_.size(); ++idx) {
    const ArraySpec& spec = out_specs_[idx];
    const auto* hlo_sharding =
        dyn_cast_or_null<HloSharding>(spec.sharding.get());
    if (hlo_sharding == nullptr) {
      LOG(ERROR) << "Failed to get sharding of output #" << idx
                 << ". Sharding is not an HloSharding.";
      return std::nullopt;
    }
    output_shardings.push_back(hlo_sharding->xla_hlo_sharding().ToProto());
  }
  return output_shardings;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
IfrtIrExecutable::GetParameterLayouts() const {
  ABSL_RETURN_IF_ERROR(layout_status_);
  std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts;
  parameter_layouts.reserve(in_specs_.size());
  for (const ArraySpec& spec : in_specs_) {
    parameter_layouts.push_back(spec.layout);
  }
  return parameter_layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
IfrtIrExecutable::GetOutputLayouts() const {
  ABSL_RETURN_IF_ERROR(layout_status_);
  std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts;
  output_layouts.reserve(out_specs_.size());
  for (const ArraySpec& spec : out_specs_) {
    output_layouts.push_back(spec.layout);
  }
  return output_layouts;
}

absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>>
IfrtIrExecutable::GetHloModules() const {
  std::vector<std::shared_ptr<xla::HloModule>> all_modules;
  for (const auto& [unused_name, executable] :
       tsl::KeySortedRange(*bound_atom_program_executables_)) {
    ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<xla::HloModule>> modules,
                          executable->GetHloModules());
    all_modules.insert(all_modules.end(), modules.begin(), modules.end());
  }
  for (const auto& [unused_name, compiled_atom] :
       tsl::KeySortedRange(*compiled_atom_programs_)) {
    ABSL_ASSIGN_OR_RETURN(std::vector<std::shared_ptr<xla::HloModule>> modules,
                          compiled_atom->executable->GetHloModules());
    all_modules.insert(all_modules.end(), modules.begin(), modules.end());
  }
  return all_modules;
}

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
IfrtIrExecutable::GetOutputMemoryKinds() const {
  std::vector<absl::string_view> output_memory_kinds;
  output_memory_kinds.reserve(out_specs_.size());
  for (const ArraySpec& spec : out_specs_) {
    output_memory_kinds.push_back(spec.sharding->memory_kind().value());
  }
  return std::vector<std::vector<absl::string_view>>{
      std::move(output_memory_kinds)};
}

absl::StatusOr<AttributeMap> IfrtIrExecutable::GetCostAnalysis() const {
  return absl::UnimplementedError(
      "IfrtIrExecutable does not support GetCostAnalysis()");
}

}  // namespace ifrt
}  // namespace xla
