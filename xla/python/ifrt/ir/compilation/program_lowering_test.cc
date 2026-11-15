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

#include "xla/python/ifrt/ir/compilation/program_lowering.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/attribute_map.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/client.h"
#include "xla/python/ifrt/device.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/executable.h"
#include "xla/python/ifrt/hlo/hlo_program.h"
#include "xla/python/ifrt/host_callback.h"
#include "xla/python/ifrt/ir/compilation/atom_program_compiler.h"
#include "xla/python/ifrt/ir/compilation/passes.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/ir/import/passes.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/service/hlo_module_config.h"
#include "xla/tsl/concurrency/future.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/util/sorted_range.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;
using ::testing::SizeIs;

// Minimal unloaded `Executable` with configurable parameter and output layouts.
class FakeUnloadedExecutable final
    : public RTTIExtends<FakeUnloadedExecutable, Executable> {
 public:
  FakeUnloadedExecutable(
      std::string name,
      std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts,
      std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts,
      std::shared_ptr<xla::HloModule> hlo_module)
      : name_(std::move(name)),
        parameter_layouts_(std::move(parameter_layouts)),
        output_layouts_(std::move(output_layouts)),
        hlo_module_(std::move(hlo_module)) {}

  absl::string_view name() const override { return name_; }
  absl::StatusOr<std::optional<std::string>> Fingerprint() const override {
    return std::optional<std::string>("fp");
  }
  absl::StatusOr<std::string> Serialize() const override {
    return "serialized";
  }
  int num_devices() const override { return 2; }
  int64_t SizeOfGeneratedCodeInBytes() const override { return 32; }
  absl::StatusOr<xla::CompiledMemoryStats> GetCompiledMemoryStats()
      const override {
    if (hlo_module_ == nullptr) {
      return absl::UnimplementedError("Memory stats not supported");
    }
    xla::CompiledMemoryStats stats;
    stats.argument_size_in_bytes = 16;
    stats.output_size_in_bytes = 16;
    return stats;
  }
  std::optional<std::vector<xla::OpSharding>> GetParameterShardings()
      const override {
    return std::vector<xla::OpSharding>{xla::OpSharding()};
  }
  std::optional<std::vector<xla::OpSharding>> GetOutputShardings()
      const override {
    return std::vector<xla::OpSharding>{xla::OpSharding()};
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetParameterLayouts() const override {
    return parameter_layouts_;
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override {
    return output_layouts_;
  }
  absl::StatusOr<std::vector<std::shared_ptr<xla::HloModule>>> GetHloModules()
      const override {
    if (hlo_module_ == nullptr) {
      return absl::UnimplementedError("HloModules not supported");
    }
    return std::vector<std::shared_ptr<xla::HloModule>>{hlo_module_};
  }
  absl::StatusOr<std::vector<std::vector<absl::string_view>>>
  GetOutputMemoryKinds() const override {
    return std::vector<std::vector<absl::string_view>>{{"device"}};
  }
  absl::StatusOr<AttributeMap> GetCostAnalysis() const override {
    if (hlo_module_ == nullptr) {
      return absl::UnimplementedError("Cost analysis not supported");
    }
    AttributeMap::Map map;
    map.insert({"flops", AttributeMap::FloatValue(64.0f)});
    return AttributeMap(std::move(map));
  }

  static char ID;  // NOLINT

 private:
  std::string name_;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts_;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts_;
  std::shared_ptr<xla::HloModule> hlo_module_;
};

char FakeUnloadedExecutable::ID = 0;

// Counters tracking `AtomProgramCompiler` method invocations.
struct AtomCompilerCounters {
  std::atomic<int> compile_and_load_xla_calls{0};
  std::atomic<int> compile_xla_calls{0};
  std::atomic<int> compile_and_load_mpmd_reshard_calls{0};
  std::atomic<int> compile_mpmd_reshard_calls{0};
};

// `AtomProgramCompiler` fake supporting both compile-only and compile-and-load
// dispatches.
class FakeDeferredAtomProgramCompiler final : public AtomProgramCompiler {
 public:
  FakeDeferredAtomProgramCompiler(
      DeviceListRef devices,
      std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts,
      std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts,
      AtomCompilerCounters* absl_nonnull counters)
      : devices_(std::move(devices)),
        parameter_layouts_(std::move(parameter_layouts)),
        output_layouts_(std::move(output_layouts)),
        counters_(counters) {}

  tsl::Future<LoadedExecutableRef> CompileAndLoadXla(
      std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks)
      override {
    counters_->compile_and_load_xla_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
    auto executable = std::make_shared<NiceMock<MockLoadedExecutable>>();
    ON_CALL(*executable, GetParameterLayouts())
        .WillByDefault(Return(parameter_layouts_));
    ON_CALL(*executable, GetOutputLayouts())
        .WillByDefault(Return(output_layouts_));
    return LoadedExecutableRef(std::move(executable));
  }

  tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileXla(std::unique_ptr<HloProgram> computation,
             xla::CompileOptions options,
             std::vector<tsl::RCReference<LoadedHostCallback>>
                 loaded_host_callbacks) override {
    counters_->compile_xla_calls.fetch_add(1, std::memory_order_relaxed);
    std::string atom_name(computation->name());
    auto hlo_module =
        std::make_shared<xla::HloModule>(atom_name, xla::HloModuleConfig());
    ExecutableRef executable = std::make_shared<FakeUnloadedExecutable>(
        atom_name, parameter_layouts_, output_layouts_, std::move(hlo_module));
    auto load_options = std::make_unique<LoadOptions>(
        devices_, /*outputs_bundle_slice_sizes=*/std::nullopt);
    return std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
        /*executable=*/std::move(executable),
        /*load_options=*/std::move(load_options),
    });
  }

  tsl::Future<LoadedExecutableRef> CompileAndLoadMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) override {
    counters_->compile_and_load_mpmd_reshard_calls.fetch_add(
        1, std::memory_order_relaxed);
    return absl::UnimplementedError("Unexpected CompileAndLoadMpmdReshard");
  }

  tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileMpmdReshard(std::vector<DType> dtypes, std::vector<Shape> shapes,
                     std::vector<IfrtArrayType> in_array_types,
                     std::vector<IfrtArrayType> out_array_types) override {
    counters_->compile_mpmd_reshard_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
    ExecutableRef executable = std::make_shared<FakeUnloadedExecutable>(
        "mpmd_reshard", parameter_layouts_, output_layouts_,
        /*hlo_module=*/nullptr);
    auto load_options = std::make_unique<LoadOptions>(
        devices_, /*outputs_bundle_slice_sizes=*/std::nullopt);
    return std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
        /*executable=*/std::move(executable),
        /*load_options=*/std::move(load_options),
    });
  }

 private:
  DeviceListRef devices_;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> parameter_layouts_;
  std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts_;
  AtomCompilerCounters* absl_nonnull counters_;
};

class ProgramLoweringTest : public test_util::IfrtIrTestBase {
 protected:
  void SetUp() override {
    default_layout_ =
        std::make_shared<xla::PjRtLayout>(xla::LayoutUtil::MakeLayout({1, 0}));
    param_layout_ =
        std::make_shared<xla::PjRtLayout>(xla::LayoutUtil::MakeLayout({0, 1}));
    output_layout_ = std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeLayout({1, 0}, /*tiles=*/{xla::Tile({2, 2})}));
    ASSERT_NE(*default_layout_, *param_layout_);
    ASSERT_NE(*default_layout_, *output_layout_);
    ASSERT_NE(*param_layout_, *output_layout_);

    ON_CALL(device0_, Id()).WillByDefault(Return(DeviceId(10)));
    ON_CALL(device0_, PlatformName()).WillByDefault(Return("cpu"));
    ON_CALL(device1_, Id()).WillByDefault(Return(DeviceId(11)));
    ON_CALL(device1_, PlatformName()).WillByDefault(Return("cpu"));

    all_devices_ = {&device0_, &device1_};
    device_list_ = BasicDeviceList::Create(all_devices_);

    ON_CALL(client_, Attributes()).WillByDefault(ReturnRef(client_attributes_));
    ON_CALL(client_, LookupDevice(DeviceId(10)))
        .WillByDefault(Return(&device0_));
    ON_CALL(client_, LookupDevice(DeviceId(11)))
        .WillByDefault(Return(&device1_));
    ON_CALL(client_, MakeDeviceList(testing::_))
        .WillByDefault([](absl::Span<Device* const> devs) {
          return BasicDeviceList::Create(devs);
        });
    ON_CALL(client_, GetDefaultPjRtLayout(testing::_, testing::_, testing::_,
                                          testing::_))
        .WillByDefault(Return(default_layout_));
  }

  AttributeMap client_attributes_ = AttributeMap(AttributeMap::Map());
  NiceMock<MockClient> client_;
  NiceMock<MockDevice> device0_;
  NiceMock<MockDevice> device1_;
  std::vector<Device*> all_devices_;
  DeviceListRef device_list_;
  std::shared_ptr<const xla::PjRtLayout> default_layout_;
  std::shared_ptr<const xla::PjRtLayout> param_layout_;
  std::shared_ptr<const xla::PjRtLayout> output_layout_;
};

TEST_F(ProgramLoweringTest,
       DefersAtomProgramLoadingClonesContextDumpsDotAndPopulatesLayouts) {
  constexpr absl::string_view kSource = R"mlir(
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                      [0]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
module @deferred_lowering_module {
  func.func @main(%arg0: !array0 {ifrt.donated}) -> (!array1)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Reshard(%arg0) : (!array0) -> !array1
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0,1]
        : (!array1) -> !array1
    return %1 : !array1
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )mlir";

  AtomCompilerCounters counters;
  const std::string dot_dump_dir =
      absl::StrCat(testing::TempDir(), "/program_lowering_dot_dump");
  absl::StatusOr<LoweredIfrtIrProgramState> lowered_state;
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    auto compile_options = std::make_unique<IfrtIRCompileOptions>(
        std::vector<DeviceId>{DeviceId(10), DeviceId(11)});
    compile_options->dot_graph_dump_to = dot_dump_dir;
    lowered_state = LowerIfrtIrProgram(
        std::make_unique<IfrtIRProgram>(*module), std::move(compile_options),
        &client_,
        std::make_shared<FakeDeferredAtomProgramCompiler>(
            device_list_,
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{param_layout_},
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{output_layout_},
            &counters),
        /*defer_atom_program_loading=*/true);
  }
  ASSERT_OK(lowered_state);
  EXPECT_TRUE(lowered_state->ifrt_ir_program->OwnsMlirContext());
  EXPECT_EQ(counters.compile_xla_calls.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(counters.compile_mpmd_reshard_calls.load(std::memory_order_relaxed),
            1);
  EXPECT_EQ(counters.compile_and_load_xla_calls.load(std::memory_order_relaxed),
            0);
  EXPECT_EQ(counters.compile_and_load_mpmd_reshard_calls.load(
                std::memory_order_relaxed),
            0);
  EXPECT_THAT(lowered_state->donatable_input_indices, ElementsAre(0));
  ASSERT_NE(lowered_state->compiled_atom_program_future_map, nullptr);
  EXPECT_THAT(*lowered_state->compiled_atom_program_future_map, SizeIs(2));

  std::vector<std::string> dot_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      absl::StrCat(dot_dump_dir, "/ifrt_deferred_lowering_module_*.dot"),
      &dot_files));
  ASSERT_THAT(dot_files, SizeIs(1));
  std::string dot_contents;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), dot_files[0], &dot_contents));
  EXPECT_THAT(dot_contents, HasSubstr("add_one"));

  CompiledAtomProgramMap compiled_atoms;
  for (const auto& [name, future] :
       tsl::KeySortedRange(*lowered_state->compiled_atom_program_future_map)) {
    ASSERT_OK_AND_ASSIGN(
        absl_nonnull std::shared_ptr<const CompiledAtomProgram> compiled_atom,
        future.Await());
    compiled_atoms.insert({name, std::move(compiled_atom)});
  }
  AtomExecutableMap empty_bound_executables;
  ASSERT_OK(PopulateIfrtIrProgramLayouts(
      lowered_state->ifrt_ir_program->mlir_module, &client_,
      empty_bound_executables, &compiled_atoms,
      absl::MakeSpan(lowered_state->in_specs),
      absl::MakeSpan(lowered_state->out_specs), lowered_state->device_list));
  ASSERT_THAT(lowered_state->in_specs, SizeIs(1));
  EXPECT_EQ(*lowered_state->in_specs[0].layout, *param_layout_);
  ASSERT_THAT(lowered_state->out_specs, SizeIs(1));
  EXPECT_EQ(*lowered_state->out_specs[0].layout, *output_layout_);
}

TEST_F(ProgramLoweringTest,
       EagerLoweringDispatchesCompileAndLoadAndBorrowsExternalMlirContext) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @eager_lowering_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  ASSERT_OK_AND_ASSIGN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_,
          std::make_shared<FakeDeferredAtomProgramCompiler>(
              device_list_,
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                  param_layout_},
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                  output_layout_},
              &counters),
          /*defer_atom_program_loading=*/false));
  EXPECT_FALSE(state.ifrt_ir_program->OwnsMlirContext());
  EXPECT_EQ(state.compiled_atom_program_future_map, nullptr);
  EXPECT_EQ(counters.compile_and_load_xla_calls.load(std::memory_order_relaxed),
            1);
  EXPECT_EQ(counters.compile_xla_calls.load(std::memory_order_relaxed), 0);

  AtomExecutableMap loaded_atoms;
  for (const auto& [name, future] :
       tsl::KeySortedRange(*state.atom_executable_future_map)) {
    ASSERT_OK_AND_ASSIGN(LoadedExecutableRef executable, future.Await());
    loaded_atoms.insert({name, std::move(executable)});
  }
  ASSERT_OK(PopulateIfrtIrProgramLayouts(
      state.ifrt_ir_program->mlir_module, &client_, loaded_atoms,
      /*compiled_atom_programs=*/nullptr, absl::MakeSpan(state.in_specs),
      absl::MakeSpan(state.out_specs), state.device_list));
  ASSERT_THAT(state.in_specs, SizeIs(1));
  EXPECT_EQ(*state.in_specs[0].layout, *param_layout_);
  ASSERT_THAT(state.out_specs, SizeIs(1));
  EXPECT_EQ(*state.out_specs[0].layout, *output_layout_);
}

TEST_F(ProgramLoweringTest,
       OutlinedPipelineDumpsDotForBoundExecutablesWithoutAtomFutureMap) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @bound_dot_pipeline_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @bound_op(%arg0) : (!array) -> !array
    %1, %ctrl_1 = ifrt.Call @add_one(%0) after %ctrl_0 on devices [0,1]
        : (!array) -> !array
    return %1 : !array
  }

  ifrt.LoadedExecutable @bound_op on devices [0,1] : (!array) -> !array

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  auto bound_exec = std::make_shared<NiceMock<MockLoadedExecutable>>();
  ON_CALL(*bound_exec, name()).WillByDefault(Return("bound_op"));
  ON_CALL(*bound_exec, devices()).WillByDefault(Return(device_list_));
  const xla::OpSharding tile_2x1_sharding =
      xla::HloSharding::IotaTile({2, 1}).ToProto();
  ON_CALL(*bound_exec, GetParameterShardings())
      .WillByDefault(Return(std::vector<xla::OpSharding>{tile_2x1_sharding}));
  ON_CALL(*bound_exec, GetOutputShardings())
      .WillByDefault(Return(std::vector<xla::OpSharding>{tile_2x1_sharding}));
  xla::CompiledMemoryStats stats;
  stats.argument_size_in_bytes = 32;
  stats.output_size_in_bytes = 32;
  ON_CALL(*bound_exec, GetCompiledMemoryStats()).WillByDefault(Return(stats));
  AttributeMap::Map cost_map;
  cost_map.insert({"flops", AttributeMap::FloatValue(64.0f)});
  ON_CALL(*bound_exec, GetCostAnalysis())
      .WillByDefault(Return(AttributeMap(std::move(cost_map))));

  const std::string dot_dump_dir =
      absl::StrCat(testing::TempDir(), "/bound_dot_pipeline_dump");
  auto compile_options = std::make_shared<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)});
  compile_options->dot_graph_dump_to = dot_dump_dir;
  auto bound_map = std::make_shared<AtomExecutableMap>(
      AtomExecutableMap{{"bound_op", bound_exec}});
  auto compiled_future_map = std::make_shared<CompiledAtomProgramFutureMap>();
  AtomCompilerCounters counters;

  mlir::PassManager pm(module->getContext());
  createIfrtToOutlinedAtomProgramsPipeline(pm);
  createIfrtPopulateAtomProgramMetadataPipeline(pm);
  OutlinedAtomProgramsToCompiledPipelineOptions pipeline_options;
  pipeline_options.platform_names = {"cpu", "cpu"};
  ASSERT_OK(createOutlinedAtomProgramsToCompiledPipeline(
      pm,
      std::make_shared<FakeDeferredAtomProgramCompiler>(
          device_list_,
          std::vector<std::shared_ptr<const xla::PjRtLayout>>{param_layout_},
          std::vector<std::shared_ptr<const xla::PjRtLayout>>{output_layout_},
          &counters),
      pipeline_options, compile_options,
      /*atom_executable_future_map=*/nullptr, compiled_future_map, bound_map));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  std::vector<std::string> dot_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      absl::StrCat(dot_dump_dir, "/ifrt_bound_dot_pipeline_module_*.dot"),
      &dot_files));
  ASSERT_THAT(dot_files, SizeIs(1));
  std::string dot_contents;
  ASSERT_OK(
      tsl::ReadFileToString(tsl::Env::Default(), dot_files[0], &dot_contents));
  EXPECT_THAT(dot_contents, HasSubstr("bound_op"));
  EXPECT_THAT(dot_contents, HasSubstr("add_one"));
}

TEST_F(ProgramLoweringTest,
       PopulatesLayoutsAcrossCopyArraysWithCompiledAtomPrograms) {
  constexpr absl::string_view kSource = R"mlir(
!dev_array = !ifrt.array<tensor<2x2xi32>,
                         #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                         memory_kind = "device">
!host_array = !ifrt.array<tensor<2x2xi32>,
                          #ifrt.sharding_param<2x1 to [0] on 2>, [0,1],
                          memory_kind = "unpinned_host">
module @copy_arrays_layout_module {
  func.func @main(%arg0: !dev_array) -> (!dev_array, !host_array)
      attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!dev_array) -> !dev_array
    %dev_copy, %ctrl_1 = ifrt.CopyArrays(%0) : (!dev_array) -> !dev_array
    %host_copy, %ctrl_2 = ifrt.CopyArrays(%0) : (!dev_array) -> !host_array
    return %dev_copy, %host_copy : !dev_array, !host_array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    %0 = stablehlo.constant dense<1> : tensor<2x2xi32>
    %1 = stablehlo.add %arg0, %0 : tensor<2x2xi32>
    return %1 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  ASSERT_OK_AND_ASSIGN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_,
          std::make_shared<FakeDeferredAtomProgramCompiler>(
              device_list_,
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                  param_layout_},
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                  output_layout_},
              &counters),
          /*defer_atom_program_loading=*/true));
  CompiledAtomProgramMap compiled_atoms;
  for (const auto& [name, future] :
       tsl::KeySortedRange(*state.compiled_atom_program_future_map)) {
    ASSERT_OK_AND_ASSIGN(
        absl_nonnull std::shared_ptr<const CompiledAtomProgram> compiled_atom,
        future.Await());
    compiled_atoms.insert({name, std::move(compiled_atom)});
  }
  AtomExecutableMap empty_bound_executables;
  ASSERT_OK(PopulateIfrtIrProgramLayouts(
      state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
      &compiled_atoms, absl::MakeSpan(state.in_specs),
      absl::MakeSpan(state.out_specs), state.device_list));
  ASSERT_THAT(state.out_specs, SizeIs(2));
  EXPECT_EQ(*state.out_specs[0].layout, *output_layout_);
  EXPECT_EQ(*state.out_specs[1].layout, *default_layout_);
}

TEST_F(ProgramLoweringTest, RejectsNullLoadedExecBindingEntry) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @null_binding_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    return %arg0 : !array
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)},
      AtomExecutableMap{{"null_bound", nullptr}});
  EXPECT_THAT(
      LowerIfrtIrProgram(
          std::make_unique<IfrtIRProgram>(*module), std::move(compile_options),
          &client_,
          std::make_shared<FakeDeferredAtomProgramCompiler>(
              device_list_,
              /*parameter_layouts=*/
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{},
              /*output_layouts=*/
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{}, &counters),
          /*defer_atom_program_loading=*/true),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Bound executable `null_bound` must not be null")));
}

TEST_F(ProgramLoweringTest,
       RejectsOutOfBoundsOrNullAtomLayoutsInPopulateIfrtIrProgramLayouts) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @invalid_layouts_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @identity(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @identity(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  ASSERT_OK_AND_ASSIGN(
      LoweredIfrtIrProgramState state,
      LowerIfrtIrProgram(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_,
          std::make_shared<FakeDeferredAtomProgramCompiler>(
              device_list_,
              /*parameter_layouts=*/
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{},
              /*output_layouts=*/
              std::vector<std::shared_ptr<const xla::PjRtLayout>>{}, &counters),
          /*defer_atom_program_loading=*/true));
  ASSERT_NE(state.compiled_atom_program_future_map, nullptr);
  ASSERT_THAT(*state.compiled_atom_program_future_map, SizeIs(1));
  const std::string atom_name =
      state.compiled_atom_program_future_map->begin()->first;
  AtomExecutableMap empty_bound_executables;

  // Missing atom executable -> FailedPreconditionError.
  CompiledAtomProgramMap empty_compiled_map;
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &empty_compiled_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Could not find SPMD executable")));

  // Empty parameter layouts vector -> out-of-range error instead of OOB read.
  CompiledAtomProgramMap empty_param_layouts_map;
  empty_param_layouts_map.insert(
      {atom_name,
       std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
           /*executable=*/std::make_shared<FakeUnloadedExecutable>(
               atom_name, /*parameter_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{},
               /*output_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                   output_layout_},
               /*hlo_module=*/nullptr),
           /*load_options=*/std::make_unique<LoadOptions>(),
       })});
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &empty_param_layouts_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Parameter operand 0 out of range")));

  // Null parameter layout entry -> InternalError instead of null dereference.
  CompiledAtomProgramMap null_param_layout_map;
  null_param_layout_map.insert(
      {atom_name,
       std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
           /*executable=*/std::make_shared<FakeUnloadedExecutable>(
               atom_name,
               /*parameter_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{nullptr},
               /*output_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                   output_layout_},
               /*hlo_module=*/nullptr),
           /*load_options=*/std::make_unique<LoadOptions>(),
       })});
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &null_param_layout_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Parameter 0 layout")));

  // Empty output layouts vector -> out-of-range error instead of OOB read.
  CompiledAtomProgramMap empty_output_layouts_map;
  empty_output_layouts_map.insert(
      {atom_name,
       std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
           /*executable=*/std::make_shared<FakeUnloadedExecutable>(
               atom_name,
               /*parameter_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                   param_layout_},
               /*output_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{},
               /*hlo_module=*/nullptr),
           /*load_options=*/std::make_unique<LoadOptions>(),
       })});
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &empty_output_layouts_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("Output 0 out of range")));

  // Null output layout entry -> InternalError instead of falling back to
  // BuildDefaultLayout.
  CompiledAtomProgramMap null_output_layout_map;
  null_output_layout_map.insert(
      {atom_name,
       std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
           /*executable=*/std::make_shared<FakeUnloadedExecutable>(
               atom_name,
               /*parameter_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{
                   param_layout_},
               /*output_layouts=*/
               std::vector<std::shared_ptr<const xla::PjRtLayout>>{nullptr},
               /*hlo_module=*/nullptr),
           /*load_options=*/std::make_unique<LoadOptions>(),
       })});
  EXPECT_THAT(
      PopulateIfrtIrProgramLayouts(
          state.ifrt_ir_program->mlir_module, &client_, empty_bound_executables,
          &null_output_layout_map, absl::MakeSpan(state.in_specs),
          absl::MakeSpan(state.out_specs), state.device_list),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("Output 0 layout")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
