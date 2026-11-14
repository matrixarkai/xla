/* Copyright 2025 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/serdes.h"
#include "xla/python/ifrt/serdes.pb.h"
#include "xla/python/ifrt/serdes_test_util.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"
#include "xla/python/pjrt_ifrt/xla_sharding_spec.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ShardingSpecSerDesTestParam = std::tuple<SerDesVersion, int>;

class ShardingSpecSerDesTest
    : public testing::TestWithParam<ShardingSpecSerDesTestParam> {
 public:
  SerDesVersion version() const { return std::get<0>(GetParam()); }
  int num_shards() const { return std::get<1>(GetParam()); }
};

TEST_P(ShardingSpecSerDesTest, HloShardingSpecRoundTrip) {
  ASSERT_EQ(num_shards() % 2, 0);
  auto xla_hlo_sharding =
      xla::HloSharding::Tile(xla::TileAssignment({2, num_shards() / 2}));
  auto sharding_spec = HloShardingSpec::Create(num_shards(), xla_hlo_sharding);

  ASSERT_OK_AND_ASSIGN(
      auto serialized,
      Serialize(*sharding_spec, std::make_unique<SerializeOptions>(version())));

  ASSERT_OK_AND_ASSIGN(auto deserialized, Deserialize<ShardingSpec>(
                                              serialized, /*options=*/nullptr));

  const auto* deserialized_spec = dyn_cast<HloShardingSpec>(deserialized.get());
  ASSERT_NE(deserialized_spec, nullptr);
  EXPECT_EQ(deserialized_spec->num_shards(), num_shards());
  EXPECT_EQ(deserialized_spec->xla_hlo_sharding(), xla_hlo_sharding);
}

TEST_P(ShardingSpecSerDesTest, RejectMalformedDeserializationPayloads) {
  auto wrap_proto = [](const HloShardingSpecProto& proto) {
    Serialized outer;
    outer.set_type_name("xla::ifrt::HloShardingSpec");
    outer.set_data(proto.SerializeAsString());
    return outer;
  };

  // Negative num_shards.
  {
    HloShardingSpecProto proto;
    proto.set_version_number(0);
    proto.set_num_shards(-3);
    proto.mutable_xla_op_sharding()->set_type(xla::OpSharding::REPLICATED);
    EXPECT_THAT(
        Deserialize<ShardingSpec>(wrap_proto(proto), /*options=*/nullptr),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("non-negative")));
  }

  // Duplicate tile_assignment_devices.
  {
    HloShardingSpecProto proto;
    proto.set_version_number(0);
    proto.set_num_shards(4);
    xla::OpSharding* op = proto.mutable_xla_op_sharding();
    op->set_type(xla::OpSharding::OTHER);
    op->add_tile_assignment_dimensions(4);
    op->add_tile_assignment_devices(0);
    op->add_tile_assignment_devices(0);
    op->add_tile_assignment_devices(2);
    op->add_tile_assignment_devices(3);
    EXPECT_THAT(
        Deserialize<ShardingSpec>(wrap_proto(proto), /*options=*/nullptr),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("not unique")));
  }

  // Mismatched num_shards vs num_devices.
  {
    HloShardingSpecProto proto;
    proto.set_version_number(0);
    proto.set_num_shards(2);
    *proto.mutable_xla_op_sharding() =
        xla::HloSharding::IotaTile({4}).ToProto();
    EXPECT_THAT(
        Deserialize<ShardingSpec>(wrap_proto(proto), /*options=*/nullptr),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("does not match")));
  }

  // Tuple OpSharding.
  {
    HloShardingSpecProto proto;
    proto.set_version_number(0);
    proto.set_num_shards(2);
    *proto.mutable_xla_op_sharding() =
        xla::HloSharding::FlatTuple(
            {xla::HloSharding::Replicate(), xla::HloSharding::Manual()})
            .ToProto();
    EXPECT_THAT(
        Deserialize<ShardingSpec>(wrap_proto(proto), /*options=*/nullptr),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("Tuple HloSharding")));
  }
}

INSTANTIATE_TEST_SUITE_P(
    SerDesVersion_NumShards, ShardingSpecSerDesTest,
    testing::Combine(testing::ValuesIn(test_util::AllSupportedSerDesVersions()),
                     testing::Values(2, 4)),
    [](const testing::TestParamInfo<ShardingSpecSerDesTestParam>& info) {
      return absl::StrCat("version_",
                          std::get<0>(info.param).version_number().value(),
                          "_num_shards_", std::get<1>(info.param));
    });

}  // namespace
}  // namespace ifrt
}  // namespace xla
