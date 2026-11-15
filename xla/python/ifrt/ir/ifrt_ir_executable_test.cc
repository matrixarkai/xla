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
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_layout.h"
#include "xla/python/ifrt/array.h"
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
#include "xla/python/ifrt/ir/compilation/compiled_ifrt_ir_program.h"
#include "xla/python/ifrt/ir/compilation/compiler.h"
#include "xla/python/ifrt/ir/ifrt_dialect.h"
#include "xla/python/ifrt/ir/ifrt_ir_loaded_executable.h"
#include "xla/python/ifrt/ir/ifrt_ir_program.h"
#include "xla/python/ifrt/ir/ifrt_ir_test_base.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/mock.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/topology.h"
#include "xla/python/ifrt/user_context.h"
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

// Minimal concrete `Topology` for `IfrtIrProgramCompiler::Compile` unit tests.
class TestTopology final : public RTTIExtends<TestTopology, Topology> {
 public:
  absl::string_view platform_name() const override { return "cpu"; }
  absl::string_view platform_version() const override { return "test"; }
  PjRtPlatformId platform_id() const override { return 0; }
  const std::shared_ptr<const xla::PjRtTopologyDescription>& description()
      const override {
    return description_;
  }
  std::vector<std::unique_ptr<const PjRtDeviceDescription>> DeviceDescriptions()
      const override {
    return {};
  }
  absl::StatusOr<xla::Layout> GetDefaultLayout(
      PrimitiveType element_type,
      absl::Span<const int64_t> dims) const override {
    return xla::LayoutUtil::MakeDescendingLayout(dims.size());
  }
  absl::StatusOr<uint64_t> Fingerprint() const override { return 0; }
  const AttributeMap& Attributes() const override { return attributes_; }

  static char ID;  // NOLINT

 private:
  std::shared_ptr<const xla::PjRtTopologyDescription> description_;
  AttributeMap attributes_ = AttributeMap(AttributeMap::Map());
};

char TestTopology::ID = 0;

// Minimal unloaded `Executable` representing a compiled XLA atom program.
class TestUnloadedAtomExecutable final
    : public RTTIExtends<TestUnloadedAtomExecutable, Executable> {
 public:
  TestUnloadedAtomExecutable(std::string name,
                             std::shared_ptr<const xla::PjRtLayout> layout,
                             std::shared_ptr<xla::HloModule> hlo_module)
      : name_(std::move(name)),
        layout_(std::move(layout)),
        hlo_module_(std::move(hlo_module)) {}

  absl::string_view name() const override { return name_; }
  absl::StatusOr<std::optional<std::string>> Fingerprint() const override {
    return std::optional<std::string>("test_fp");
  }
  absl::StatusOr<std::string> Serialize() const override {
    return "serialized_unloaded_atom";
  }
  int num_devices() const override { return 2; }
  int64_t SizeOfGeneratedCodeInBytes() const override { return 64; }
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
    if (layout_ == nullptr) {
      return absl::UnimplementedError("Layouts not supported");
    }
    return std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout_};
  }
  absl::StatusOr<std::vector<std::shared_ptr<const xla::PjRtLayout>>>
  GetOutputLayouts() const override {
    if (layout_ == nullptr) {
      return absl::UnimplementedError("Layouts not supported");
    }
    return std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout_};
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
    map.insert({"flops", AttributeMap::FloatValue(128.0f)});
    return AttributeMap(std::move(map));
  }

  static char ID;  // NOLINT

 private:
  std::string name_;
  std::shared_ptr<const xla::PjRtLayout> layout_;
  std::shared_ptr<xla::HloModule> hlo_module_;
};

char TestUnloadedAtomExecutable::ID = 0;

// Counters tracking compile-only vs. compile-and-load `AtomProgramCompiler`
// invocations.
struct AtomCompilerCounters {
  std::atomic<int> compile_and_load_xla_calls{0};
  std::atomic<int> compile_xla_calls{0};
  std::atomic<int> compile_and_load_mpmd_reshard_calls{0};
  std::atomic<int> compile_mpmd_reshard_calls{0};
};

// Recording `AtomProgramCompiler` that tracks compile-only vs. compile-and-load
// invocations.
class RecordingAtomProgramCompiler final : public AtomProgramCompiler {
 public:
  RecordingAtomProgramCompiler(DeviceListRef devices,
                               std::shared_ptr<const xla::PjRtLayout> layout,
                               AtomCompilerCounters* absl_nonnull counters)
      : devices_(std::move(devices)),
        layout_(std::move(layout)),
        counters_(counters) {}

  tsl::Future<LoadedExecutableRef> CompileAndLoadXla(
      std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks)
      override {
    counters_->compile_and_load_xla_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
    return absl::UnimplementedError("Unexpected CompileAndLoadXla call");
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
    ExecutableRef unloaded_executable =
        std::make_shared<TestUnloadedAtomExecutable>(atom_name, layout_,
                                                     std::move(hlo_module));
    auto load_options = std::make_unique<const LoadOptions>(
        devices_, /*outputs_bundle_slice_sizes=*/std::nullopt);
    return std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
        /*executable=*/std::move(unloaded_executable),
        /*load_options=*/std::move(load_options),
    });
  }

  tsl::Future<LoadedExecutableRef> CompileAndLoadMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) override {
    counters_->compile_and_load_mpmd_reshard_calls.fetch_add(
        1, std::memory_order_relaxed);
    return absl::UnimplementedError(
        "Unexpected CompileAndLoadMpmdReshard call");
  }

  tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileMpmdReshard(std::vector<DType> dtypes, std::vector<Shape> shapes,
                     std::vector<IfrtArrayType> in_array_types,
                     std::vector<IfrtArrayType> out_array_types) override {
    counters_->compile_mpmd_reshard_calls.fetch_add(1,
                                                    std::memory_order_relaxed);
    ExecutableRef unloaded_reshard =
        std::make_shared<TestUnloadedAtomExecutable>(
            "mpmd_reshard", /*layout=*/nullptr, /*hlo_module=*/nullptr);
    auto load_options = std::make_unique<const LoadOptions>(
        devices_, /*outputs_bundle_slice_sizes=*/std::nullopt);
    return std::make_shared<const CompiledAtomProgram>(CompiledAtomProgram{
        /*executable=*/std::move(unloaded_reshard),
        /*load_options=*/std::move(load_options),
    });
  }

 private:
  DeviceListRef devices_;
  std::shared_ptr<const xla::PjRtLayout> layout_;
  AtomCompilerCounters* absl_nonnull counters_;
};

class IfrtIrExecutableTest : public test_util::IfrtIrTestBase {
 protected:
  void SetUp() override {
    layout_ = std::make_shared<xla::PjRtLayout>(
        xla::LayoutUtil::MakeDescendingLayout(2));

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
        .WillByDefault(Return(layout_));
    ON_CALL(client_, DeleteValues(testing::_))
        .WillByDefault(Return(tsl::Future<>(absl::OkStatus())));
  }

  // Creates a mock loaded atom executable with serialization and layout stubs.
  std::shared_ptr<NiceMock<MockLoadedExecutable>> MakeLoadedAtomExecutable(
      absl::string_view module_name) {
    auto executable = std::make_shared<NiceMock<MockLoadedExecutable>>();
    auto hlo_module = std::make_shared<xla::HloModule>(std::string(module_name),
                                                       xla::HloModuleConfig());
    xla::OpSharding tile_sharding =
        xla::HloSharding::IotaTile({2, 1}).ToProto();
    ON_CALL(*executable, client()).WillByDefault(Return(&client_));
    ON_CALL(*executable, devices())
        .WillByDefault(Return(std::optional<DeviceListRef>(device_list_)));
    ON_CALL(*executable, Serialize())
        .WillByDefault(Return(absl::StrCat("serialized_", module_name)));
    ON_CALL(*executable, GetParameterShardings())
        .WillByDefault(Return(
            std::optional<std::vector<xla::OpSharding>>{{tile_sharding}}));
    ON_CALL(*executable, GetOutputShardings())
        .WillByDefault(Return(
            std::optional<std::vector<xla::OpSharding>>{{tile_sharding}}));
    ON_CALL(*executable, GetParameterLayouts())
        .WillByDefault(Return(
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout_}));
    ON_CALL(*executable, GetOutputLayouts())
        .WillByDefault(Return(
            std::vector<std::shared_ptr<const xla::PjRtLayout>>{layout_}));
    ON_CALL(*executable, GetHloModules())
        .WillByDefault(
            Return(std::vector<std::shared_ptr<xla::HloModule>>{hlo_module}));
    return executable;
  }

  AttributeMap client_attributes_ = AttributeMap(AttributeMap::Map());
  NiceMock<MockClient> client_;
  NiceMock<MockDevice> device0_;
  NiceMock<MockDevice> device1_;
  std::vector<Device*> all_devices_;
  DeviceListRef device_list_;
  std::shared_ptr<const xla::PjRtLayout> layout_;
};

TEST_F(IfrtIrExecutableTest,
       CompilesWithoutLoadingAndSupportsConcurrentMultiLoadSerialization) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @compile_only_module {
  func.func @main(%arg0: !array {ifrt.donated}) -> (!array)
      attributes {ifrt.function} {
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
  AtomCompilerCounters counters;
  IfrtIrProgramCompiler compiler(
      &client_,
      [&](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<RecordingAtomProgramCompiler>(
            device_list_, layout_, &counters);
      });

  std::shared_ptr<NiceMock<MockLoadedExecutable>> bound_mock =
      MakeLoadedAtomExecutable("bound_op");
  LoadedExecutableRef bound_executable = bound_mock;
  absl::flat_hash_map<std::string, LoadedExecutableRef> bindings;
  bindings.insert({"bound_op", bound_executable});
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)}, std::move(bindings));

  TestTopology topology;
  ASSERT_OK_AND_ASSIGN(ExecutableRef executable,
                       compiler
                           .Compile(std::make_unique<IfrtIRProgram>(*module),
                                    topology, std::move(compile_options))
                           .Await());
  EXPECT_EQ(counters.compile_xla_calls.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(counters.compile_and_load_xla_calls.load(std::memory_order_relaxed),
            0);
  EXPECT_EQ(counters.compile_and_load_mpmd_reshard_calls.load(
                std::memory_order_relaxed),
            0);
  EXPECT_EQ(counters.compile_mpmd_reshard_calls.load(std::memory_order_relaxed),
            0);

  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_EQ(ifrt_ir_executable->name(), "compile_only_module");
  EXPECT_EQ(ifrt_ir_executable->num_devices(), 2);
  EXPECT_EQ(ifrt_ir_executable->SizeOfGeneratedCodeInBytes(), -1);
  ASSERT_OK_AND_ASSIGN(std::optional<std::string> fingerprint,
                       ifrt_ir_executable->Fingerprint());
  EXPECT_EQ(fingerprint, std::nullopt);
  EXPECT_THAT(ifrt_ir_executable->Serialize(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ifrt_ir_executable->GetCompiledMemoryStats(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ifrt_ir_executable->GetCostAnalysis(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ifrt_ir_executable->donatable_input_indices(), ElementsAre(0));
  EXPECT_THAT(ifrt_ir_executable->bound_atom_program_executables(), SizeIs(1));
  ASSERT_THAT(ifrt_ir_executable->compiled_atom_programs(), SizeIs(1));
  ASSERT_TRUE(ifrt_ir_executable->GetParameterShardings().has_value());
  EXPECT_THAT(*ifrt_ir_executable->GetParameterShardings(), SizeIs(1));
  ASSERT_TRUE(ifrt_ir_executable->GetOutputShardings().has_value());
  EXPECT_THAT(*ifrt_ir_executable->GetOutputShardings(), SizeIs(1));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> param_layouts,
      ifrt_ir_executable->GetParameterLayouts());
  EXPECT_THAT(param_layouts, ElementsAre(layout_));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<const xla::PjRtLayout>> output_layouts,
      ifrt_ir_executable->GetOutputLayouts());
  EXPECT_THAT(output_layouts, ElementsAre(layout_));
  ASSERT_OK_AND_ASSIGN(
      std::vector<std::vector<absl::string_view>> output_memory_kinds,
      ifrt_ir_executable->GetOutputMemoryKinds());
  ASSERT_THAT(output_memory_kinds, SizeIs(1));
  EXPECT_THAT(output_memory_kinds.front(), ElementsAre("device"));
  ASSERT_OK_AND_ASSIGN(std::vector<std::shared_ptr<xla::HloModule>> hlo_modules,
                       ifrt_ir_executable->GetHloModules());
  EXPECT_THAT(hlo_modules, SizeIs(2));

  const std::string compiled_atom_name =
      ifrt_ir_executable->compiled_atom_programs().begin()->first;
  std::shared_ptr<NiceMock<MockLoadedExecutable>> loaded_add_one_1 =
      MakeLoadedAtomExecutable("loaded_add_one_1");
  std::shared_ptr<NiceMock<MockLoadedExecutable>> loaded_add_one_2 =
      MakeLoadedAtomExecutable("loaded_add_one_2");
  AtomExecutableMap load1_atoms;
  load1_atoms.insert({compiled_atom_name, loaded_add_one_1});
  AtomExecutableMap load2_atoms;
  load2_atoms.insert({compiled_atom_name, loaded_add_one_2});

  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<CompiledIfrtIrProgram> compiled_program_1,
      ifrt_ir_executable->CreateCompiledIfrtIrProgram(&client_,
                                                      std::move(load1_atoms)));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<CompiledIfrtIrProgram> compiled_program_2,
      ifrt_ir_executable->CreateCompiledIfrtIrProgram(&client_,
                                                      std::move(load2_atoms)));

  // Release `IfrtIrExecutable` before using or destroying the loaded programs
  // to verify each `CompiledIfrtIrProgram` owns an independent `MLIRContext`.
  executable.reset();

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_executable_1,
      IfrtIrLoadedExecutable::Create(&client_, compiled_program_1));
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutableRef loaded_executable_2,
      IfrtIrLoadedExecutable::Create(&client_, compiled_program_2));

  absl::StatusOr<std::string> serialized_1;
  absl::StatusOr<std::string> serialized_2;
  {
    // `loaded_executable_1`, `loaded_executable_2`, `serialized_1`, and
    // `serialized_2` outlive `thread_1` and `thread_2` because `~Thread()`
    // joins on scope exit.
    std::unique_ptr<tsl::Thread> thread_1 =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "serialize_1",
            [&]() { serialized_1 = loaded_executable_1->Serialize(); }));
    std::unique_ptr<tsl::Thread> thread_2 =
        absl::WrapUnique(tsl::Env::Default()->StartThread(
            tsl::ThreadOptions(), "serialize_2",
            [&]() { serialized_2 = loaded_executable_2->Serialize(); }));
  }
  ASSERT_OK(serialized_1);
  ASSERT_OK(serialized_2);
  EXPECT_FALSE(serialized_1->empty());
  EXPECT_FALSE(serialized_2->empty());

  ArrayRef input_1 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef bound_out_1 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef final_out_1 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef input_2 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef bound_out_2 = tsl::MakeRef<NiceMock<MockArray>>();
  ArrayRef final_out_2 = tsl::MakeRef<NiceMock<MockArray>>();

  EXPECT_CALL(*bound_mock,
              Execute(ElementsAre(input_1), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{bound_out_1},
      }));
  EXPECT_CALL(*loaded_add_one_1,
              Execute(ElementsAre(bound_out_1), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{final_out_1},
      }));
  EXPECT_CALL(*bound_mock,
              Execute(ElementsAre(input_2), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{bound_out_2},
      }));
  EXPECT_CALL(*loaded_add_one_2,
              Execute(ElementsAre(bound_out_2), testing::_, testing::_))
      .WillOnce(Return(LoadedExecutable::ExecuteResult{
          /*status=*/tsl::Future<>(absl::OkStatus()),
          /*outputs=*/{final_out_2},
      }));

  LoadedExecutable::ExecuteOptions execute_options;
  execute_options.fill_status = true;
  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult exec_result_1,
      loaded_executable_1->Execute(absl::MakeSpan(&input_1, 1), execute_options,
                                   /*devices=*/std::nullopt));
  ASSERT_OK(exec_result_1.status.Await());
  EXPECT_THAT(exec_result_1.outputs, ElementsAre(final_out_1));

  ASSERT_OK_AND_ASSIGN(
      LoadedExecutable::ExecuteResult exec_result_2,
      loaded_executable_2->Execute(absl::MakeSpan(&input_2, 1), execute_options,
                                   /*devices=*/std::nullopt));
  ASSERT_OK(exec_result_2.status.Await());
  EXPECT_THAT(exec_result_2.outputs, ElementsAre(final_out_2));

  // Clearing `loaded_exec_binding` on load 1 must not mutate load 2's bindings.
  compiled_program_1->compile_options->loaded_exec_binding.clear();
  EXPECT_THAT(compiled_program_2->compile_options->loaded_exec_binding,
              SizeIs(1));
}

TEST_F(IfrtIrExecutableTest,
       CompilesMpmdReshardWithoutLoadingAndSupportsDotGraphDump) {
  constexpr absl::string_view kSource = R"mlir(
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                      [0]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
module @mpmd_reshard_module {
  func.func @main(%arg0: !array0) -> (!array1) attributes {ifrt.function} {
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

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)});
  const std::string dot_dump_dir =
      absl::StrCat(testing::TempDir(), "/ifrt_ir_dot_dump");
  compile_options->dot_graph_dump_to = dot_dump_dir;

  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(std::make_unique<IfrtIRProgram>(*module),
                               std::move(compile_options), &client_,
                               std::make_shared<RecordingAtomProgramCompiler>(
                                   device_list_, layout_, &counters))
          .Await());
  EXPECT_EQ(counters.compile_xla_calls.load(std::memory_order_relaxed), 1);
  EXPECT_EQ(counters.compile_and_load_xla_calls.load(std::memory_order_relaxed),
            0);
  EXPECT_EQ(counters.compile_mpmd_reshard_calls.load(std::memory_order_relaxed),
            1);
  EXPECT_EQ(counters.compile_and_load_mpmd_reshard_calls.load(
                std::memory_order_relaxed),
            0);
  std::vector<std::string> dot_files;
  ASSERT_OK(tsl::Env::Default()->GetMatchingPaths(
      absl::StrCat(dot_dump_dir, "/ifrt_mpmd_reshard_module_*.dot"),
      &dot_files));
  EXPECT_THAT(dot_files, SizeIs(1));

  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_TRUE(ifrt_ir_executable->bound_atom_program_executables().empty());
  ASSERT_THAT(ifrt_ir_executable->compiled_atom_programs(), SizeIs(2));
  EXPECT_THAT(ifrt_ir_executable->GetParameterLayouts(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ifrt_ir_executable->GetHloModules(),
              StatusIs(absl::StatusCode::kUnimplemented));

  AtomExecutableMap partial_loaded_atoms;
  partial_loaded_atoms.insert(
      {ifrt_ir_executable->compiled_atom_programs().begin()->first,
       MakeLoadedAtomExecutable("partial_atom")});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &client_, partial_loaded_atoms),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 loaded atom executables")));

  AtomExecutableMap loaded_atoms;
  for (const auto& [name, unused_compiled_atom] :
       tsl::KeySortedRange(ifrt_ir_executable->compiled_atom_programs())) {
    loaded_atoms.insert({name, MakeLoadedAtomExecutable(name)});
  }
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<CompiledIfrtIrProgram> compiled_program,
                       ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                           &client_, std::move(loaded_atoms)));
  EXPECT_THAT(*compiled_program->atom_program_executables, SizeIs(2));
}

TEST_F(IfrtIrExecutableTest, RejectsInvalidCreateCompiledIfrtIrProgramInputs) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @validation_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_,
          std::make_shared<RecordingAtomProgramCompiler>(device_list_, layout_,
                                                         &counters))
          .Await());
  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  const std::string compiled_atom_name =
      ifrt_ir_executable->compiled_atom_programs().begin()->first;

  AtomExecutableMap wrong_name_atoms;
  wrong_name_atoms.insert(
      {"unexpected_atom", MakeLoadedAtomExecutable("unexpected")});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &client_, std::move(wrong_name_atoms)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected atom executable")));

  AtomExecutableMap null_atoms;
  null_atoms.insert({compiled_atom_name, nullptr});
  EXPECT_THAT(ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                  &client_, std::move(null_atoms)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("must not be null")));
}

TEST_F(IfrtIrExecutableTest,
       OwnsLoweredModuleAfterCallerDestroysInputModulePostCompile) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @ephemeral_caller_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  AtomCompilerCounters counters;
  ExecutableRef executable;
  {
    ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                         LoadFromSource(kSource));
    ASSERT_OK_AND_ASSIGN(
        executable, IfrtIrExecutable::Create(
                        std::make_unique<IfrtIRProgram>(*module),
                        std::make_unique<IfrtIRCompileOptions>(
                            std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                        &client_,
                        std::make_shared<RecordingAtomProgramCompiler>(
                            device_list_, layout_, &counters))
                        .Await());
  }

  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_TRUE(ifrt_ir_executable->program()->OwnsMlirContext());
  const std::string compiled_atom_name =
      ifrt_ir_executable->compiled_atom_programs().begin()->first;
  AtomExecutableMap loaded_atoms;
  loaded_atoms.insert(
      {compiled_atom_name, MakeLoadedAtomExecutable("loaded_add_one")});

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<CompiledIfrtIrProgram> compiled_program,
                       ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                           &client_, std::move(loaded_atoms)));
  EXPECT_NE(compiled_program, nullptr);
}

TEST_F(IfrtIrExecutableTest,
       SupportsPreBoundMpmdReshardExecutableWithoutSymbolCollision) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @bound_mpmd_reshard_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @bound_reshard(%arg0)
        {ifrt.module_type = "mpmd_reshard"} : (!array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @bound_reshard on devices [0,1] : (!array) -> !array
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  absl::flat_hash_map<std::string, LoadedExecutableRef> bindings;
  bindings.insert({"bound_reshard", MakeLoadedAtomExecutable("bound_reshard")});
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)}, std::move(bindings));

  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(std::make_unique<IfrtIRProgram>(*module),
                               std::move(compile_options), &client_,
                               std::make_shared<RecordingAtomProgramCompiler>(
                                   device_list_, layout_, &counters))
          .Await());
  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_TRUE(ifrt_ir_executable->compiled_atom_programs().empty());
  EXPECT_THAT(ifrt_ir_executable->bound_atom_program_executables(), SizeIs(1));
  EXPECT_EQ(counters.compile_mpmd_reshard_calls.load(std::memory_order_relaxed),
            0);
  EXPECT_EQ(counters.compile_and_load_mpmd_reshard_calls.load(
                std::memory_order_relaxed),
            0);

  ASSERT_OK_AND_ASSIGN(std::shared_ptr<CompiledIfrtIrProgram> compiled_program,
                       ifrt_ir_executable->CreateCompiledIfrtIrProgram(
                           &client_, /*newly_loaded_atom_executables=*/{}));
  EXPECT_THAT(*compiled_program->atom_program_executables, SizeIs(1));
}

TEST_F(IfrtIrExecutableTest, RejectsNullLoadedExecBinding) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @null_binding_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  IfrtIrProgramCompiler compiler(
      &client_,
      [&](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<RecordingAtomProgramCompiler>(
            device_list_, layout_, &counters);
      });
  auto compile_options = std::make_unique<IfrtIRCompileOptions>(
      std::vector<DeviceId>{DeviceId(10), DeviceId(11)},
      AtomExecutableMap{{"unused_bound", nullptr}});
  TestTopology topology;
  EXPECT_THAT(compiler
                  .Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                           std::move(compile_options))
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Bound executable `unused_bound` must not be "
                                 "null")));
}

TEST_F(IfrtIrExecutableTest, RejectsUnboundCallLoadedExecutable) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @unbound_call_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.CallLoadedExecutable @missing_bound(%arg0)
        : (!array) -> !array
    return %0 : !array
  }

  ifrt.LoadedExecutable @missing_bound on devices [0,1] : (!array) -> !array
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  EXPECT_THAT(IfrtIrExecutable::Create(
                  std::make_unique<IfrtIRProgram>(*module),
                  std::make_unique<IfrtIRCompileOptions>(
                      std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
                  &client_,
                  std::make_shared<RecordingAtomProgramCompiler>(
                      device_list_, layout_, &counters))
                  .Await(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Could not find bound or compiled executable "
                                 "for `missing_bound`")));
}

TEST_F(IfrtIrExecutableTest,
       ResetsPartialParameterLayoutsWhenOutputLayoutResolutionFails) {
  constexpr absl::string_view kSource = R"mlir(
!array0 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                      [0,1]>
!array1 = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<1x1 to [0] on 1>,
                      [0]>
module @partial_layout_reset_module {
  func.func @main(%arg0: !array0) -> (!array1) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array0) -> !array0
    %1, %ctrl_1 = ifrt.Reshard(%0) after %ctrl_0 : (!array0) -> !array1
    return %1 : !array1
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  AtomCompilerCounters counters;
  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_,
          std::make_shared<RecordingAtomProgramCompiler>(device_list_, layout_,
                                                         &counters))
          .Await());
  auto* ifrt_ir_executable =
      xla::ifrt::cast<IfrtIrExecutable>(executable.get());
  EXPECT_THAT(ifrt_ir_executable->layout_status(),
              StatusIs(absl::StatusCode::kUnimplemented));
  ASSERT_THAT(ifrt_ir_executable->in_specs(), SizeIs(1));
  EXPECT_EQ(ifrt_ir_executable->in_specs()[0].layout, nullptr);
  ASSERT_THAT(ifrt_ir_executable->out_specs(), SizeIs(1));
  EXPECT_EQ(ifrt_ir_executable->out_specs()[0].layout, nullptr);
  EXPECT_THAT(ifrt_ir_executable->GetParameterLayouts(),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(ifrt_ir_executable->GetOutputLayouts(),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(IfrtIrExecutableTest,
       PreservesUserContextDuringAsyncExecutableCreation) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @user_context_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    return %arg0 : !array
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  UserContextRef expected_context = BasicUserContext::Create("compile_ctx");
  UserContextRef observed_context;
  EXPECT_CALL(client_, GetDefaultPjRtLayout(testing::_, testing::_, testing::_,
                                            testing::_))
      .WillOnce([&](DType, absl::Span<const int64_t>, Device*, MemoryKind) {
        observed_context = UserContextScope::current();
        return layout_;
      });

  AtomCompilerCounters counters;
  IfrtIrProgramCompiler compiler(
      &client_,
      [&](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<RecordingAtomProgramCompiler>(
            device_list_, layout_, &counters);
      });
  TestTopology topology;
  UserContextScope user_context_scope(expected_context);
  ASSERT_OK_AND_ASSIGN(
      ExecutableRef executable,
      compiler
          .Compile(std::make_unique<IfrtIRProgram>(*module), topology,
                   std::make_unique<IfrtIRCompileOptions>(
                       std::vector<DeviceId>{DeviceId(10), DeviceId(11)}))
          .Await());
  EXPECT_NE(executable, nullptr);
  EXPECT_EQ(observed_context, expected_context);
}

TEST_F(IfrtIrExecutableTest, CompileRejectsNonIfrtIrProgram) {
  AtomCompilerCounters counters;
  IfrtIrProgramCompiler compiler(
      &client_,
      [&](const IfrtIRCompileOptions& unused_opts)
          -> absl::StatusOr<std::unique_ptr<AtomProgramCompiler>> {
        return std::make_unique<RecordingAtomProgramCompiler>(
            device_list_, layout_, &counters);
      });
  TestTopology topology;
  EXPECT_THAT(
      compiler
          .Compile(/*program=*/nullptr, topology,
                   std::make_unique<IfrtIRCompileOptions>(
                       std::vector<DeviceId>{DeviceId(10), DeviceId(11)}))
          .Await(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("requires an IFRT IR program")));
}

// `AtomProgramCompiler` returning a null `CompiledAtomProgram` for negative
// testing.
class NullReturningAtomProgramCompiler final : public AtomProgramCompiler {
 public:
  tsl::Future<LoadedExecutableRef> CompileAndLoadXla(
      std::unique_ptr<HloProgram> computation, xla::CompileOptions options,
      std::vector<tsl::RCReference<LoadedHostCallback>> loaded_host_callbacks)
      override {
    return absl::UnimplementedError("Unexpected CompileAndLoadXla");
  }

  tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileXla(std::unique_ptr<HloProgram> computation,
             xla::CompileOptions options,
             std::vector<tsl::RCReference<LoadedHostCallback>>
                 loaded_host_callbacks) override {
    return std::shared_ptr<const CompiledAtomProgram>(nullptr);
  }

  tsl::Future<LoadedExecutableRef> CompileAndLoadMpmdReshard(
      std::vector<DType> dtypes, std::vector<Shape> shapes,
      std::vector<IfrtArrayType> in_array_types,
      std::vector<IfrtArrayType> out_array_types) override {
    return absl::UnimplementedError("Unexpected CompileAndLoadMpmdReshard");
  }

  tsl::Future<absl_nonnull std::shared_ptr<const CompiledAtomProgram>>
  CompileMpmdReshard(std::vector<DType> dtypes, std::vector<Shape> shapes,
                     std::vector<IfrtArrayType> in_array_types,
                     std::vector<IfrtArrayType> out_array_types) override {
    return std::shared_ptr<const CompiledAtomProgram>(nullptr);
  }
};

TEST_F(IfrtIrExecutableTest,
       CreateRejectsNullCompiledAtomProgramFromAtomCompiler) {
  constexpr absl::string_view kSource = R"mlir(
!array = !ifrt.array<tensor<2x2xi32>, #ifrt.sharding_param<2x1 to [0] on 2>,
                     [0,1]>
module @null_compiled_atom_module {
  func.func @main(%arg0: !array) -> (!array) attributes {ifrt.function} {
    %0, %ctrl_0 = ifrt.Call @add_one(%arg0) on devices [0,1]
        : (!array) -> !array
    return %0 : !array
  }

  func.func private @add_one(%arg0: tensor<2x2xi32>) -> tensor<2x2xi32> {
    return %arg0 : tensor<2x2xi32>
  }
}
  )mlir";

  ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                       LoadFromSource(kSource));
  EXPECT_THAT(
      IfrtIrExecutable::Create(
          std::make_unique<IfrtIRProgram>(*module),
          std::make_unique<IfrtIRCompileOptions>(
              std::vector<DeviceId>{DeviceId(10), DeviceId(11)}),
          &client_, std::make_shared<NullReturningAtomProgramCompiler>())
          .Await(),
      StatusIs(absl::StatusCode::kInternal,
               HasSubstr("returned null compiled atom program")));
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
