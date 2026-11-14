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

#include "xla/python/pjrt_ifrt/xla_sharding_spec.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/hash/hash_testing.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/python/ifrt/basic_device_list.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/device_test_util.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/tsl/platform/errors.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {
namespace {

using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::SizeIs;

class HloShardingSpecTest : public testing::Test {
 public:
  HloShardingSpecTest()
      : fixture_(test_util::DeviceTestParam{/*num_devices=*/6,
                                            /*num_addressable_devices=*/4}) {}

  DeviceListRef GetDevices(absl::Span<const int> device_indices) {
    return fixture_.GetDevices(device_indices);
  }

 private:
  test_util::DeviceTestFixture fixture_;
};

TEST_F(HloShardingSpecTest, CreateWithNumShardsMismatch) {
  auto xla_hlo_sharding = xla::HloSharding::IotaTile({2, 3});
  EXPECT_DEATH(
      HloShardingSpec::Create(/*num_shards=*/5, xla_hlo_sharding),
      "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not match");
}

TEST_F(HloShardingSpecTest, IsFullyReplicated) {
  int num_shards = 6;
  {
    // Fully replicated HloSharding is fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::Replicate();
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Single-tile HloSharding is fully replicated.
    int num_shards = 1;
    auto xla_hlo_sharding = xla::HloSharding::IotaTile({1, 1});
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Multi-tile HloSharding with last_dim_replicate where all replices are on
    // the last tile dimension is fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::PartialTile(
        xla::TileAssignment(xla::IotaTileAssignment::Create({1, 6})));
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Multi-tile HloSharding with last_dim_replicate where not all replices are
    // on the last tile dimension is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::PartialTile(
        xla::TileAssignment(xla::IotaTileAssignment::Create({2, 3})));
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Multi-tile HloSharding with no last_dim_replicate is not fully
    // replicated.
    auto xla_hlo_sharding = xla::HloSharding::IotaTile({1, 6});
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Maximal HloSharding with a single device is fully replicated.
    int num_shards = 1;
    auto xla_hlo_sharding = xla::HloSharding::SingleDevice(/*device_id=*/0);
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_TRUE(spec->IsFullyReplicated());
  }
  {
    // Maximal HloSharding with more than one device is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::SingleDevice(/*device_id=*/0);
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Manual HloSharding is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::Manual();
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
  {
    // Unknown HloSharding is not fully replicated.
    auto xla_hlo_sharding = xla::HloSharding::Unknown();
    std::shared_ptr<const HloShardingSpec> spec =
        HloShardingSpec::Create(num_shards, xla_hlo_sharding);
    EXPECT_FALSE(spec->IsFullyReplicated());
  }
}

TEST_F(HloShardingSpecTest, GetShardShape) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::IotaTile({2, 3});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);
  EXPECT_THAT(spec->GetShardShape(Shape({6, 6})),
              absl_testing::IsOkAndHolds(Shape({3, 2})));
  EXPECT_THAT(spec->GetShardShape(Shape({6, 6, 6})),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Numbers of dimensions don't match. From "
                            "Shape [6,6,6] vs from HloSharding "
                            "{devices=[2,3]<=[6]}")));
}

TEST_F(HloShardingSpecTest, HasSamePartitioning) {
  int num_shards0 = 6;
  auto xla_hlo_sharding0 = xla::HloSharding::IotaTile({2, 3});
  std::shared_ptr<const HloShardingSpec> spec0 =
      HloShardingSpec::Create(num_shards0, xla_hlo_sharding0);

  EXPECT_TRUE(spec0->HasSamePartitioning(*spec0));
  {
    // Different number of shards.
    int num_shards1 = 3;
    auto xla_hlo_sharding1 = xla::HloSharding::IotaTile({3, 1});
    std::shared_ptr<const HloShardingSpec> spec1 =
        HloShardingSpec::Create(num_shards1, xla_hlo_sharding1);
    EXPECT_FALSE(spec0->HasSamePartitioning(*spec1));
  }
  // Different HloSharding.
  {
    auto xla_hlo_sharding1 = xla::HloSharding::IotaTile({3, 2});
    std::shared_ptr<const HloShardingSpec> spec1 =
        HloShardingSpec::Create(num_shards0, xla_hlo_sharding1);
    EXPECT_FALSE(spec0->HasSamePartitioning(*spec1));
  }

  // Replicated sharding with different numbers of devices.
  {
    int num_shards1 = 3;
    std::shared_ptr<const HloShardingSpec> spec0 =
        HloShardingSpec::Create(num_shards0, xla::HloSharding::Replicate());
    std::shared_ptr<const HloShardingSpec> spec1 =
        HloShardingSpec::Create(num_shards1, xla::HloSharding::Replicate());
    EXPECT_FALSE(spec0->HasSamePartitioning(*spec1));
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithReplication) {
  int num_shards = 6;
  // Fully replicated.
  auto xla_hlo_sharding = xla::HloSharding::Replicate();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(
      index_domains,
      ElementsAre(IndexDomain(shape), IndexDomain(shape), IndexDomain(shape),
                  IndexDomain(shape), IndexDomain(shape), IndexDomain(shape)));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, UniqueIndexDomainsWithReplication) {
  int num_shards = 6;
  // Fully replicated.
  auto xla_hlo_sharding = xla::HloSharding::Replicate();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(spec->UniqueIndexDomains(shape),
              absl_testing::IsOkAndHolds(ElementsAre(FieldsAre(
                  IndexDomain(shape), ElementsAre(0, 1, 2, 3, 4, 5)))));
  EXPECT_THAT(spec->ShardToUniqueIndexDomainIndex(),
              absl_testing::IsOkAndHolds(ElementsAre(0, 0, 0, 0, 0, 0)));
}

TEST_F(HloShardingSpecTest, DisassembleWithReplication) {
  int num_shards = 6;
  // Fully replicated.
  auto xla_hlo_sharding = xla::HloSharding::Replicate();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({10, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({12, 20});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({2, 20})),
                          IndexDomain(Index({2, 0}), Shape({2, 20})),
                          IndexDomain(Index({4, 0}), Shape({2, 20})),
                          IndexDomain(Index({6, 0}), Shape({2, 20})),
                          IndexDomain(Index({8, 0}), Shape({2, 20})),
                          IndexDomain(Index({10, 0}), Shape({2, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, UniqueIndexDomainsWithTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({12, 20});
  EXPECT_THAT(
      spec->UniqueIndexDomains(shape),
      absl_testing::IsOkAndHolds(ElementsAre(
          FieldsAre(IndexDomain(Index({0, 0}), Shape({2, 20})), ElementsAre(0)),
          FieldsAre(IndexDomain(Index({2, 0}), Shape({2, 20})), ElementsAre(1)),
          FieldsAre(IndexDomain(Index({4, 0}), Shape({2, 20})), ElementsAre(2)),
          FieldsAre(IndexDomain(Index({6, 0}), Shape({2, 20})), ElementsAre(3)),
          FieldsAre(IndexDomain(Index({8, 0}), Shape({2, 20})), ElementsAre(4)),
          FieldsAre(IndexDomain(Index({10, 0}), Shape({2, 20})),
                    ElementsAre(5)))));
  EXPECT_THAT(spec->ShardToUniqueIndexDomainIndex(),
              absl_testing::IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5)));
}

TEST_F(HloShardingSpecTest, DisassembleWithTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({12, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({2, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithUnevenTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({11, 20});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({2, 20})),
                          IndexDomain(Index({2, 0}), Shape({2, 20})),
                          IndexDomain(Index({4, 0}), Shape({2, 20})),
                          IndexDomain(Index({6, 0}), Shape({2, 20})),
                          IndexDomain(Index({8, 0}), Shape({2, 20})),
                          IndexDomain(Index({10, 0}), Shape({1, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, UniqueIndexDomainsWithUnevenTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({11, 20});
  EXPECT_THAT(
      spec->UniqueIndexDomains(shape),
      absl_testing::IsOkAndHolds(ElementsAre(
          FieldsAre(IndexDomain(Index({0, 0}), Shape({2, 20})), ElementsAre(0)),
          FieldsAre(IndexDomain(Index({2, 0}), Shape({2, 20})), ElementsAre(1)),
          FieldsAre(IndexDomain(Index({4, 0}), Shape({2, 20})), ElementsAre(2)),
          FieldsAre(IndexDomain(Index({6, 0}), Shape({2, 20})), ElementsAre(3)),
          FieldsAre(IndexDomain(Index({8, 0}), Shape({2, 20})), ElementsAre(4)),
          FieldsAre(IndexDomain(Index({10, 0}), Shape({1, 20})),
                    ElementsAre(5)))));
  EXPECT_THAT(spec->ShardToUniqueIndexDomainIndex(),
              absl_testing::IsOkAndHolds(ElementsAre(0, 1, 2, 3, 4, 5)));
}

TEST_F(HloShardingSpecTest, DisassembleWithUnevenTile) {
  int num_shards = 6;
  // 6-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({6, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({11, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    if (i < 5) {
      EXPECT_EQ(shape, Shape({2, 20}));
    } else {
      EXPECT_EQ(shape, Shape({1, 20}));
    }
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithPartialTile) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding =
      xla::HloSharding::PartialTile(xla::TileAssignment({2, 1, 3}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, UniqueIndexDomainsWithPartialTile) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding =
      xla::HloSharding::PartialTile(xla::TileAssignment({2, 1, 3}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(spec->UniqueIndexDomains(shape),
              absl_testing::IsOkAndHolds(ElementsAre(
                  FieldsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                            ElementsAre(0, 1, 2)),
                  FieldsAre(IndexDomain(Index({5, 0}), Shape({5, 20})),
                            ElementsAre(3, 4, 5)))));
  EXPECT_THAT(spec->ShardToUniqueIndexDomainIndex(),
              absl_testing::IsOkAndHolds(ElementsAre(0, 0, 0, 1, 1, 1)));
}

TEST_F(HloShardingSpecTest, UniqueIndexDomainsManualFails) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Manual();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(
      spec->UniqueIndexDomains(shape),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("Manual sharding does not support UniqueIndexDomains")));
  EXPECT_THAT(
      spec->ShardToUniqueIndexDomainIndex(),
      absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                             HasSubstr("Manual sharding does not support "
                                       "ShardToUniqueIndexDomainIndex")));
}

TEST_F(HloShardingSpecTest, DisassembleWithPartialTile) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding =
      xla::HloSharding::PartialTile(xla::TileAssignment({2, 1, 3}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithSubgroupReplicated) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::REPLICATED});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithSubgroupReplicated) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // replicated by 3 times.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::REPLICATED});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithSubgroupMaximalSlowPath) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // maximal-replicated by 3 times, device#0 in each replication is maximal.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::MAXIMAL});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(index_domains,
              ElementsAre(IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({0, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20})),
                          IndexDomain(Index({5, 0}), Shape({5, 20}))));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, DisassembleWithSubgroupMaximalSlowPath) {
  int num_shards = 6;
  // 2-way sharded along axis 0, 1-way sharded along axis 1, each shard
  // maximal-replicated by 3 times, device#0 in each replication is maximal.
  auto xla_hlo_sharding = xla::HloSharding::Subgroup(
      xla::TileAssignment({2, 1, 3}), {xla::OpSharding::MAXIMAL});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({5, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, IndexDomainsWithTileTranspose) {
  int num_shards = 4;
  auto xla_hlo_sharding =
      xla::HloSharding::IotaTile(/*tile_assignment_dims=*/{2, 2},
                                 /*reshape_dims=*/{2, 2},
                                 /*transpose_perm=*/{1, 0});
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);
  Shape shape({4, 4});
  ASSERT_OK_AND_ASSIGN(auto index_domains, spec->IndexDomains(shape));
  EXPECT_THAT(
      index_domains,
      ElementsAreArray(TEST_HloShardingSpecIndexDomainsSlowPath(*spec, shape)));
}

TEST_F(HloShardingSpecTest, IndexDomainsWithUnreduced) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Unreduced();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(
      spec->IndexDomains(shape).status(),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("Unreduced sharding does not support IndexDomains")));
}

TEST_F(HloShardingSpecTest, IndexDomainsWithManual) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Manual();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  EXPECT_THAT(spec->IndexDomains(shape).status(),
              absl_testing::StatusIs(
                  tsl::error::INVALID_ARGUMENT,
                  HasSubstr("Manual sharding does not support IndexDomains")));
}

TEST_F(HloShardingSpecTest, DisassembleWithManual) {
  int num_shards = 6;
  auto xla_hlo_sharding = xla::HloSharding::Manual();
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10, 20});
  ASSERT_OK_AND_ASSIGN(auto disassembled, spec->Disassemble(shape));
  ASSERT_THAT(disassembled, SizeIs(6));
  for (int i = 0; i < 6; ++i) {
    const auto& [shape, sharding] = disassembled[i];
    EXPECT_EQ(shape, Shape({10, 20}));
    EXPECT_EQ(*sharding, *SingleDeviceShardingSpec::Create());
  }
}

TEST_F(HloShardingSpecTest, DisassembleFailsWithMismatchingShapeDimsSize) {
  int num_shards = 2;
  // 2-way sharded along axis 0, 1-way sharded along axis 1.
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({2, 1}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  Shape shape({10});
  EXPECT_THAT(
      spec->Disassemble(shape),
      absl_testing::StatusIs(
          tsl::error::INVALID_ARGUMENT,
          HasSubstr("shape must have 2 dimensions, but has 1 dimensions")));
}

TEST_F(HloShardingSpecTest, DisassembleFailsWithDynamicShape) {
  int num_shards = 2;
  auto xla_hlo_sharding =
      xla::HloSharding::Tile(xla::TileAssignment(absl::Span<const int64_t>{2}));
  std::shared_ptr<const HloShardingSpec> spec =
      HloShardingSpec::Create(num_shards, xla_hlo_sharding);

  ASSERT_OK_AND_ASSIGN(
      DynamicShape dynamic_shape,
      DynamicShape::Create(Shape({10}), BoundedDynamicShapeTag({true})));
  EXPECT_THAT(
      spec->Disassemble(dynamic_shape),
      absl_testing::StatusIs(tsl::error::INVALID_ARGUMENT,
                             HasSubstr("can only disassemble static shape")));
}

TEST_F(HloShardingSpecTest, Hash) {
  EXPECT_TRUE(absl::VerifyTypeImplementsAbslHashCorrectly({
      HloShardingSpec::Create(6, xla::HloSharding::Replicate()),
      HloShardingSpec::Create(1, xla::HloSharding::Replicate()),
      HloShardingSpec::Create(6,
                              xla::HloSharding::SingleDevice(/*device_id=*/0)),
      HloShardingSpec::Create(
          6, xla::HloSharding::PartialTile(
                 xla::TileAssignment(xla::IotaTileAssignment::Create({2, 3})))),
  }));
}

TEST_F(HloShardingSpecTest, ToSharding) {
  auto xla_hlo_sharding = xla::HloSharding::Tile(xla::TileAssignment({2, 1}));
  ShardingSpecRef spec = HloShardingSpec::Create(2, xla_hlo_sharding);

  ASSERT_OK_AND_ASSIGN(ShardingRef sharding,
                       spec->ToSharding(GetDevices({0, 1}), MemoryKind()));
  EXPECT_EQ(sharding->sharding_spec(), spec);

  EXPECT_THAT(spec->ToSharding(GetDevices({0}), MemoryKind()),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-shared_ptr instance fallback.
  std::unique_ptr<HloShardingSpec> non_shared_spec =
      HloShardingSpec::Create(2, xla_hlo_sharding);
  ASSERT_OK_AND_ASSIGN(
      ShardingRef non_shared_sharding,
      non_shared_spec->ToSharding(GetDevices({0, 1}), MemoryKind()));
  EXPECT_EQ(*non_shared_sharding->sharding_spec(), *non_shared_spec);

  // Empty device list is rejected even when num_shards is 0.
  ShardingSpecRef zero_shard_spec =
      HloShardingSpec::Create(0, xla::HloSharding::Replicate());
  EXPECT_THAT(
      zero_shard_spec->ToSharding(BasicDeviceList::Create({}), MemoryKind()),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             HasSubstr("at least one device")));
}

TEST_F(HloShardingSpecTest, CreateRejectsNegativeShardsAndTuple) {
  EXPECT_DEATH(HloShardingSpec::Create(-1, xla::HloSharding::Replicate()),
               "must be non-negative");
  xla::HloSharding tuple_sharding =
      xla::HloSharding::FlatTuple({xla::HloSharding::Replicate()});
  EXPECT_DEATH(HloShardingSpec::Create(1, tuple_sharding),
               "Tuple HloSharding is not supported");
  EXPECT_DEATH(HloShardingSpec::Create(4, xla::HloSharding::IotaTile({-2, -2})),
               "does not match");
  EXPECT_DEATH(HloShardingSpec::Create(
                   4, xla::HloSharding::IotaTile({4611686018427387905LL, 4})),
               "does not match");
}

TEST_F(HloShardingSpecTest, RejectsMalformedTileAssignmentsAndRanks) {
  // Duplicate device IDs in HloSharding constructed in C++.
  {
    ShardingSpecRef spec = HloShardingSpec::Create(
        4, xla::HloSharding::Tile(xla::Array<int64_t>({4}, {0, 0, 2, 3})));
    EXPECT_THAT(spec->IndexDomains(Shape({5})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Duplicate device id")));
    EXPECT_THAT(spec->UniqueIndexDomains(Shape({5})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Duplicate device id")));
    EXPECT_THAT(spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Duplicate device id")));
  }

  // Out of range and negative device IDs in ShardToUniqueIndexDomainIndex.
  {
    ShardingSpecRef spec = HloShardingSpec::Create(
        2, xla::HloSharding::Tile(xla::Array<int64_t>({2}, {100000, -5})));
    EXPECT_THAT(spec->IndexDomains(Shape({5})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Out of range device id")));
    EXPECT_THAT(spec->UniqueIndexDomains(Shape({5})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Out of range device id")));
    EXPECT_THAT(spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Out of range device id")));
  }

  // Zero or negative tile dimensions and negative Shape dimensions return
  // InvalidArgumentError without division by zero.
  {
    ShardingSpecRef zero_tile_spec = HloShardingSpec::Create(
        0, xla::HloSharding::Tile(xla::Array<int64_t>({0})));
    EXPECT_THAT(zero_tile_spec->GetShardShape(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("must be positive")));
    EXPECT_THAT(zero_tile_spec->Disassemble(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("must be positive")));
    EXPECT_THAT(zero_tile_spec->IndexDomains(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("positive")));
    EXPECT_THAT(zero_tile_spec->UniqueIndexDomains(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("positive")));
    EXPECT_THAT(zero_tile_spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("positive")));

    ShardingSpecRef valid_spec =
        HloShardingSpec::Create(4, xla::HloSharding::IotaTile({2, 2}));
    EXPECT_THAT(valid_spec->GetShardShape(Shape({-4, 8})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("must be nonnegative")));
    EXPECT_THAT(valid_spec->IndexDomains(Shape({-4, 8})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("must be nonnegative")));

    // Malformed IotaTile constructed directly in C++ (negative reshape_dims,
    // duplicate or out of range transpose_perm, and overflowing dimensions)
    // returns InvalidArgumentError instead of corrupting memory.
    ShardingSpecRef neg_reshape_spec = HloShardingSpec::Create(
        4, xla::HloSharding::IotaTile({4}, {-2, -2}, {1, 0}));
    EXPECT_THAT(neg_reshape_spec->IndexDomains(Shape({9})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("reshape_dims")));
    EXPECT_THAT(neg_reshape_spec->UniqueIndexDomains(Shape({9})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("reshape_dims")));
    EXPECT_THAT(neg_reshape_spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("reshape_dims")));

    ShardingSpecRef dup_perm_spec = HloShardingSpec::Create(
        16, xla::HloSharding::IotaTile({16}, {4, 2, 2}, {0, 2, 2}));
    EXPECT_THAT(dup_perm_spec->IndexDomains(Shape({17})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("transpose_perm")));
    EXPECT_THAT(dup_perm_spec->UniqueIndexDomains(Shape({17})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("transpose_perm")));
    EXPECT_THAT(dup_perm_spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("transpose_perm")));

    ShardingSpecRef oob_perm_spec = HloShardingSpec::Create(
        4, xla::HloSharding::IotaTile({4}, {2, 2}, {99, 0}));
    EXPECT_THAT(oob_perm_spec->IndexDomains(Shape({5})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("transpose_perm")));
  }

  // Rank mismatch with nonreplicated subgroup returns InvalidArgumentError
  // instead of crashing inside TileOffsetForDevice.
  {
    ShardingSpecRef subgroup_spec = HloShardingSpec::Create(
        6, xla::HloSharding::Subgroup(xla::TileAssignment({2, 1, 3}),
                                      {xla::OpSharding::MAXIMAL}));
    EXPECT_THAT(
        subgroup_spec->IndexDomains(Shape({10})),
        absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                               HasSubstr("shape must have 2 dimensions")));
    // Valid zero element Shape with dimension > INT64_MAX / 4 succeeds without
    // crashing MakeShapeWithDescendingLayout or overflowing int64_t.
    EXPECT_OK(subgroup_spec->IndexDomains(Shape({3000000000000000001LL, 0})));
  }

  // Large dimension near INT64_MAX clamps cleanly in UniqueIndexDomains and
  // IndexDomains without signed overflow.
  {
    constexpr int64_t kMaxDim = 9223372036854775807LL;
    ShardingSpecRef large_spec =
        HloShardingSpec::Create(3, xla::HloSharding::IotaTile({3}));
    ASSERT_OK_AND_ASSIGN(std::vector<IndexDomain> domains,
                         large_spec->IndexDomains(Shape({kMaxDim})));
    ASSERT_EQ(domains.size(), 3);
    EXPECT_EQ(domains[2].origin(), Index({6148914691236517206LL}));
    EXPECT_EQ(domains[2].shape(), Shape({3074457345618258601LL}));
    ASSERT_OK_AND_ASSIGN(auto unique_domains,
                         large_spec->UniqueIndexDomains(Shape({kMaxDim})));
    ASSERT_EQ(unique_domains.size(), 3);
    EXPECT_EQ(unique_domains[2].index_domain, domains[2]);
  }

  // Unknown sharding returns InvalidArgumentError across all query methods.
  {
    ShardingSpecRef unknown_spec =
        HloShardingSpec::Create(2, xla::HloSharding::Unknown());
    EXPECT_THAT(unknown_spec->IndexDomains(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Unknown sharding")));
    EXPECT_THAT(unknown_spec->Disassemble(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Unknown sharding")));
    EXPECT_THAT(unknown_spec->UniqueIndexDomains(Shape({4})),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Unknown sharding")));
    EXPECT_THAT(unknown_spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                       HasSubstr("Unknown sharding")));
  }

  // Partially replicated V3 NamedSharding leaf works across IndexDomains,
  // Disassemble, UniqueIndexDomains, and ShardToUniqueIndexDomainIndex.
  {
    xla::HloSharding v3_partial = xla::HloSharding::ToV3Sharding(
        xla::HloSharding::PartialTile(xla::TileAssignment({2, 3})));
    ShardingSpecRef v3_spec = HloShardingSpec::Create(6, v3_partial);
    EXPECT_THAT(v3_spec->IndexDomains(Shape({5})),
                absl_testing::IsOkAndHolds(
                    ElementsAre(IndexDomain(Index({0}), Shape({3})),
                                IndexDomain(Index({0}), Shape({3})),
                                IndexDomain(Index({0}), Shape({3})),
                                IndexDomain(Index({3}), Shape({2})),
                                IndexDomain(Index({3}), Shape({2})),
                                IndexDomain(Index({3}), Shape({2})))));
    EXPECT_THAT(v3_spec->UniqueIndexDomains(Shape({5})),
                absl_testing::IsOkAndHolds(
                    ElementsAre(FieldsAre(IndexDomain(Index({0}), Shape({3})),
                                          ElementsAre(0, 1, 2)),
                                FieldsAre(IndexDomain(Index({3}), Shape({2})),
                                          ElementsAre(3, 4, 5)))));
    EXPECT_THAT(v3_spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::IsOkAndHolds(ElementsAre(0, 0, 0, 1, 1, 1)));
  }

  // V3 NamedSharding sharded only along a size 1 axis collapses to Replicate in
  // V3ToV2Sharding and still works across all query methods.
  {
    xla::NamedSharding size1_named(
        xla::Mesh({1, 2}, {"a", "b"}),
        {xla::NamedSharding::DimensionSharding({xla::AxisRef(0)},
                                               /*is_closed=*/true)});
    ShardingSpecRef v3_size1_spec =
        HloShardingSpec::Create(2, xla::HloSharding(size1_named));
    EXPECT_THAT(v3_size1_spec->IndexDomains(Shape({5})),
                absl_testing::IsOkAndHolds(ElementsAre(
                    IndexDomain(Shape({5})), IndexDomain(Shape({5})))));
    EXPECT_THAT(v3_size1_spec->UniqueIndexDomains(Shape({5})),
                absl_testing::IsOkAndHolds(ElementsAre(
                    FieldsAre(IndexDomain(Shape({5})), ElementsAre(0, 1)))));
    EXPECT_THAT(v3_size1_spec->ShardToUniqueIndexDomainIndex(),
                absl_testing::IsOkAndHolds(ElementsAre(0, 0)));
  }
}

}  // namespace
}  // namespace ifrt
}  // namespace xla
