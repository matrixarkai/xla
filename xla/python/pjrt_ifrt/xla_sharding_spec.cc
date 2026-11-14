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

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/overflow_util.h"
#include "xla/python/ifrt/device_list.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.h"
#include "xla/python/ifrt/memory.h"
#include "xla/python/ifrt/rtti.h"
#include "xla/python/ifrt/shape.h"
#include "xla/python/ifrt/sharding_spec.h"
#include "xla/python/pjrt_ifrt/xla_sharding.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace ifrt {

char XlaCompatibleShardingSpec::ID = 0;  // NOLINT
char HloShardingSpec::ID = 0;            // NOLINT

namespace {

absl::Status ValidateShape(const Shape& shape) {
  for (int64_t dim : shape.dims()) {
    if (dim < 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Shape dimensions must be nonnegative: shape=%v", shape));
    }
  }
  return absl::OkStatus();
}

absl::Status ValidatePositiveShardingDims(
    const xla::HloSharding& hlo_sharding) {
  for (int64_t dim : hlo_sharding.dimensions()) {
    if (dim <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "HloSharding tile dimensions must be positive, got %d; sharding=%s",
          dim, hlo_sharding.ToString()));
    }
  }
  return absl::OkStatus();
}

int64_t ClampedMultiply(int64_t a, int64_t b, int64_t limit) {
  auto [product, overflow] = xla::OverflowSafeMultiply(a, b);
  return overflow ? limit : std::min(product, limit);
}

IndexDomain ComputeTileIndexDomainForV2Device(
    const xla::HloSharding& v2_sharding, const Shape& shape, int device_idx) {
  if (v2_sharding.IsReplicatedOrSingleDevice()) {
    return IndexDomain(shape);
  }
  const absl::Span<const int64_t> shape_dims = shape.dims();
  const absl::Span<const int64_t> sharding_dims = v2_sharding.dimensions();
  const std::vector<int64_t> tile_index =
      v2_sharding.TileIndexForDevice(device_idx);
  Index::Elements origin(shape_dims.size());
  Shape::Dimensions shard_shape(shape_dims.size());
  for (size_t i = 0; i < shape_dims.size(); ++i) {
    const int64_t tile_dim = xla::CeilOfRatio(shape_dims[i], sharding_dims[i]);
    const int64_t offset =
        ClampedMultiply(tile_index[i], tile_dim, shape_dims[i]);
    const int64_t limit =
        ClampedMultiply(tile_index[i] + 1, tile_dim, shape_dims[i]);
    origin[i] = offset;
    shard_shape[i] = limit - offset;
  }
  return IndexDomain(Index(std::move(origin)), Shape(std::move(shard_shape)));
}

// Generates IndexDomains for an HloShardingSpec without constructing xla::Shape
// (which rejects valid zero element or large dimension ifrt::Shapes).
std::vector<IndexDomain> IndexDomainsSlowPath(
    const xla::HloSharding& hlo_sharding, int num_shards, const Shape& shape) {
  if (num_shards <= 0) {
    return {};
  }
  if (num_shards > 8) {
    LOG_FIRST_N(WARNING, 1) << "Taking a slow path for "
                               "HloShardingSpec::IndexDomains(). This will not "
                               "scale for a large number of devices.";
  }

  std::optional<xla::HloSharding> converted_v2;
  const xla::HloSharding* effective_sharding = &hlo_sharding;
  if (hlo_sharding.UseNamedShardingLeaf()) {
    converted_v2 = xla::HloSharding::V3ToV2Sharding(hlo_sharding);
    effective_sharding = &*converted_v2;
  }

  std::vector<IndexDomain> result;
  if (effective_sharding->IsReplicatedOrSingleDevice()) {
    result.resize(num_shards, IndexDomain(shape));
    return result;
  }
  if (num_shards == 1) {
    const absl::Span<const int64_t> shape_dims = shape.dims();
    const absl::Span<const int64_t> sharding_dims =
        effective_sharding->dimensions();
    Shape::Dimensions shard_shape(shape_dims.size());
    for (size_t i = 0; i < shape_dims.size(); ++i) {
      shard_shape[i] = xla::CeilOfRatio(shape_dims[i], sharding_dims[i]);
    }
    result.push_back(IndexDomain(Shape(std::move(shard_shape))));
    return result;
  }

  result.reserve(num_shards);
  for (int device_idx = 0; device_idx < num_shards; ++device_idx) {
    result.push_back(ComputeTileIndexDomainForV2Device(*effective_sharding,
                                                       shape, device_idx));
  }
  return result;
}

}  // namespace

std::unique_ptr<HloShardingSpec> HloShardingSpec::Create(
    int num_shards, xla::HloSharding xla_hlo_sharding) {
  CHECK_GE(num_shards, 0) << "`num_shards` must be non-negative: "
                          << num_shards;
  CHECK(!xla_hlo_sharding.IsTuple())
      << "Tuple HloSharding is not supported: " << xla_hlo_sharding.ToString();
  if (!xla_hlo_sharding.IsReplicated() && !xla_hlo_sharding.IsUnreduced() &&
      xla_hlo_sharding.IsTiled()) {
    CHECK_EQ(num_shards, xla_hlo_sharding.num_devices())
        << "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not "
           "match: "
        << num_shards << " vs. " << xla_hlo_sharding.num_devices()
        << "; sharding=" << xla_hlo_sharding.ToString();
  }
  return std::unique_ptr<HloShardingSpec>(
      new HloShardingSpec(num_shards, std::move(xla_hlo_sharding)));
}

HloShardingSpec::HloShardingSpec(int num_shards,
                                 xla::HloSharding xla_hlo_sharding)
    : RTTIExtends<HloShardingSpec, XlaCompatibleShardingSpec>(
          num_shards, /*is_fully_replicated=*/false),
      xla_hlo_sharding_(std::move(xla_hlo_sharding)) {
  is_fully_replicated_ =
      xla_hlo_sharding_.IsReplicated() ||
      ((xla_hlo_sharding_.IsTiled() || xla_hlo_sharding_.IsSingleDevice()) &&
       num_shards_ == 1);
}

HloShardingSpec::HloShardingSpec(const HloShardingSpec& other)
    : RTTIExtends<HloShardingSpec, XlaCompatibleShardingSpec>(other),
      xla_hlo_sharding_(other.xla_hlo_sharding_),
      hash_(other.hash_.load(std::memory_order_relaxed)) {}

absl::StatusOr<ShardingRef> HloShardingSpec::ToSharding(
    DeviceListRef devices, MemoryKind memory_kind) const {
  if (devices == nullptr || devices->devices().empty()) {
    return absl::InvalidArgumentError(
        "HloSharding requires at least one device");
  }
  if (devices->size() != num_shards()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloShardingSpec requires %d devices, but received %d devices; "
        "sharding=%s",
        num_shards(), devices->size(), xla_hlo_sharding_.ToString()));
  }
  std::shared_ptr<const HloShardingSpec> spec =
      std::static_pointer_cast<const HloShardingSpec>(weak_from_this().lock());
  if (spec == nullptr) {
    spec = HloShardingSpec::Create(num_shards(), xla_hlo_sharding());
  }
  return std::unique_ptr<HloSharding>(
      new HloSharding(std::move(devices), memory_kind, std::move(spec)));
}

absl::StatusOr<Shape> HloShardingSpec::GetShardShape(const Shape& shape) const {
  ABSL_RETURN_IF_ERROR(ValidateShape(shape));
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice() ||
      xla_hlo_sharding_.IsManual() || xla_hlo_sharding_.IsUnreduced() ||
      xla_hlo_sharding_.IsUnknown()) {
    return shape;
  }
  if (shape.dims().size() != xla_hlo_sharding_.TiledDataRank()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Numbers of dimensions don't match. From Shape %v vs from "
        "HloSharding %s",
        shape, xla_hlo_sharding_.ToString()));
  }
  ABSL_RETURN_IF_ERROR(ValidatePositiveShardingDims(xla_hlo_sharding_));
  const absl::Span<const int64_t> sharding_dims =
      xla_hlo_sharding_.dimensions();
  Shape::Dimensions tile_shape;
  tile_shape.reserve(shape.dims().size());
  for (int64_t i = 0; i < shape.dims().size(); ++i) {
    tile_shape.push_back(xla::CeilOfRatio(shape.dims()[i], sharding_dims[i]));
  }
  return Shape(std::move(tile_shape));
}

bool HloShardingSpec::HasSamePartitioning(const ShardingSpec& other) const {
  if (this == &other) {
    return true;
  }
  if (num_shards() != other.num_shards()) {
    return false;
  }
  const auto* other_hlo_sharding_spec = dyn_cast<HloShardingSpec>(&other);
  if (!other_hlo_sharding_spec) {
    return false;
  }
  return xla_hlo_sharding_ == other_hlo_sharding_spec->xla_hlo_sharding_;
}

absl::StatusOr<std::vector<std::pair<Shape, ShardingSpecRef>>>
HloShardingSpec::Disassemble(const Shape& shape) const {
  ABSL_RETURN_IF_ERROR(ValidateShape(shape));
  bool is_even_sharding = false;
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice() ||
      xla_hlo_sharding_.IsUnreduced()) {
    is_even_sharding = true;
  } else if (xla_hlo_sharding_.IsTiled()) {
    const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
    if (shape.dims().size() != tiled_data_rank) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "shape must have %d dimensions, but has %d dimensions: "
          "shape=%v, sharding=%s",
          tiled_data_rank, shape.dims().size(), shape,
          xla_hlo_sharding_.ToString()));
    }
    ABSL_RETURN_IF_ERROR(ValidatePositiveShardingDims(xla_hlo_sharding_));

    is_even_sharding = true;
    for (int i = 0; i < tiled_data_rank; ++i) {
      if (shape.dims()[i] % xla_hlo_sharding_.dimension(i) != 0) {
        is_even_sharding = false;
        break;
      }
    }
  } else if (xla_hlo_sharding_.IsManual()) {
    // By convention, MANUAL sharding has the same global/shard shapes.
    is_even_sharding = true;
  }

  if (is_even_sharding) {
    ABSL_ASSIGN_OR_RETURN(Shape shard_shape, GetShardShape(shape));
    std::vector<std::pair<Shape, ShardingSpecRef>> result;
    result.reserve(num_shards_);
    for (int i = 0; i < num_shards_; ++i) {
      result.push_back({
          shard_shape,
          SingleDeviceShardingSpec::Create(),
      });
    }
    return result;
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<IndexDomain> index_domains,
                        IndexDomains(shape));
  CHECK_EQ(index_domains.size(), num_shards_);
  std::vector<std::pair<Shape, ShardingSpecRef>> result;
  result.reserve(num_shards_);
  for (int i = 0; i < index_domains.size(); ++i) {
    result.push_back({
        index_domains[i].shape(),
        SingleDeviceShardingSpec::Create(),
    });
  }
  return result;
}

absl::StatusOr<std::vector<std::pair<DynamicShape, ShardingSpecRef>>>
HloShardingSpec::Disassemble(const DynamicShape& dynamic_shape) const {
  return absl::InvalidArgumentError(absl::StrFormat(
      "HloShardingSpec can only disassemble static shape, but was asked "
      "to disassemble dynamic shape %v",
      dynamic_shape));
}

absl::Status HloShardingSpec::ValidateTileAssignment() const {
  if (num_shards_ <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Tiled HloShardingSpec must have positive `num_shards`, got %d; "
        "sharding=%s",
        num_shards_, xla_hlo_sharding_.ToString()));
  }
  int64_t sharding_elements = 1;
  for (int64_t dim : xla_hlo_sharding_.dimensions()) {
    if (dim <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "HloSharding tile dimensions must be positive, got %d; sharding=%s",
          dim, xla_hlo_sharding_.ToString()));
    }
    auto [next, overflow] = xla::OverflowSafeMultiply(sharding_elements, dim);
    if (overflow) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Product of HloSharding dimensions overflows int64_t; sharding=%s",
          xla_hlo_sharding_.ToString()));
    }
    sharding_elements = next;
  }
  const xla::TileAssignment& raw_assignment =
      xla_hlo_sharding_.UseNamedShardingLeaf()
          ? xla_hlo_sharding_.named_sharding().mesh().device_assignment()
          : xla_hlo_sharding_.tile_assignment();
  int64_t assignment_elements = 1;
  for (int64_t dim : raw_assignment.dimensions()) {
    if (dim <= 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "HloSharding device_assignment dimensions must be positive, got %d; "
          "sharding=%s",
          dim, xla_hlo_sharding_.ToString()));
    }
    auto [next, overflow] = xla::OverflowSafeMultiply(assignment_elements, dim);
    if (overflow) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Product of HloSharding device_assignment dimensions overflows "
          "int64_t; sharding=%s",
          xla_hlo_sharding_.ToString()));
    }
    assignment_elements = next;
  }
  if (assignment_elements != num_shards_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not match: "
        "%d vs. %d; sharding=%s",
        num_shards_, assignment_elements, xla_hlo_sharding_.ToString()));
  }
  if (!xla_hlo_sharding_.UseNamedShardingLeaf() &&
      sharding_elements != num_shards_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not match: "
        "%d vs. %d; sharding=%s",
        num_shards_, sharding_elements, xla_hlo_sharding_.ToString()));
  }
  if (raw_assignment.iota().has_value()) {
    const xla::IotaTileAssignment& iota = *raw_assignment.iota();
    const absl::Span<const int64_t> reshape_dims = iota.reshape_dims();
    const absl::Span<const int> transpose_perm = iota.transpose_perm();
    if (reshape_dims.size() != transpose_perm.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "HloSharding iota reshape_dims size (%d) must match transpose_perm "
          "size (%d); sharding=%s",
          reshape_dims.size(), transpose_perm.size(),
          xla_hlo_sharding_.ToString()));
    }
    int64_t reshape_elements = 1;
    for (int64_t dim : reshape_dims) {
      if (dim <= 0) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "HloSharding iota reshape_dims must be positive, got %d; "
            "sharding=%s",
            dim, xla_hlo_sharding_.ToString()));
      }
      auto [next, overflow] = xla::OverflowSafeMultiply(reshape_elements, dim);
      if (overflow) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Product of HloSharding iota reshape_dims overflows int64_t; "
            "sharding=%s",
            xla_hlo_sharding_.ToString()));
      }
      reshape_elements = next;
    }
    if (reshape_elements != num_shards_) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Product of HloSharding iota reshape_dims (%d) must match "
          "`num_shards` (%d); sharding=%s",
          reshape_elements, num_shards_, xla_hlo_sharding_.ToString()));
    }
    std::vector<bool> seen_perm(transpose_perm.size(), false);
    for (int p : transpose_perm) {
      if (p < 0 || static_cast<size_t>(p) >= transpose_perm.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "HloSharding iota transpose_perm entry %d out of range [0, %d); "
            "sharding=%s",
            p, transpose_perm.size(), xla_hlo_sharding_.ToString()));
      }
      if (seen_perm[p]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "HloSharding iota transpose_perm entry %d is not unique; "
            "sharding=%s",
            p, xla_hlo_sharding_.ToString()));
      }
      seen_perm[p] = true;
    }
    return absl::OkStatus();
  }
  const xla::Array<int64_t>& array = raw_assignment.array();
  if (array.num_elements() != num_shards_) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "`num_shards` and `xla_hlo_sharding`'s `num_devices` does not match: "
        "%d vs. %d; sharding=%s",
        num_shards_, array.num_elements(), xla_hlo_sharding_.ToString()));
  }
  std::vector<bool> seen(num_shards_, false);
  for (int64_t device_id : array) {
    if (device_id < 0 || device_id >= num_shards_) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Out of range device id in device_assignment: %d; valid range: "
          "[0, %d)",
          device_id, num_shards_));
    }
    if (seen[device_id]) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Duplicate device id in device_assignment: %d", device_id));
    }
    seen[device_id] = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<IndexDomain>> HloShardingSpec::IndexDomains(
    const Shape& shape) const {
  std::vector<IndexDomain> result;
  ABSL_RETURN_IF_ERROR(ValidateShape(shape));

  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Manual sharding does not support IndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unreduced sharding does not support IndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnknown()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unknown sharding does not support IndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice()) {
    // Fast path for a fully replicated or maximal sharding.
    IndexDomain element(shape);
    result.resize(/*count=*/num_shards_, /*value=*/element);
    return result;
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%v, sharding=%s",
                        tiled_data_rank, shape.dims().size(), shape,
                        xla_hlo_sharding_.ToString()));
  }
  ABSL_RETURN_IF_ERROR(ValidateTileAssignment());

  if (!xla_hlo_sharding_.IsTiled() ||
      xla_hlo_sharding_.HasNonReplicatedSubgroup()) {
    return IndexDomainsSlowPath(xla_hlo_sharding_, num_shards_, shape);
  }

  ABSL_ASSIGN_OR_RETURN(Shape tile_shape, GetShardShape(shape));

  const absl::Span<const int64_t> shape_dims = shape.dims();
  std::vector<std::optional<IndexDomain>> all(num_shards_);
  ABSL_RETURN_IF_ERROR(xla_hlo_sharding_.EachTile(
      shape_dims, [shape_dims, &all](int device_index,
                                     absl::Span<const int64_t> tile_offset,
                                     absl::Span<const int64_t> tile_limit) {
        Shape::Dimensions tile_shape;
        tile_shape.reserve(shape_dims.size());
        for (int i = 0; i < shape_dims.size(); ++i) {
          tile_shape.push_back(tile_limit[i] - tile_offset[i]);
        }
        all[device_index] =
            IndexDomain(Index(tile_offset), Shape(std::move(tile_shape)));
      }));

  result.reserve(num_shards_);
  for (int device_idx = 0; device_idx < num_shards_; ++device_idx) {
    if (!all[device_idx].has_value()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "HloSharding tile assignment does not cover shard %d: sharding=%s",
          device_idx, xla_hlo_sharding_.ToString()));
    }
    result.push_back(*std::move(all[device_idx]));
  }

  return result;
}

absl::StatusOr<absl::InlinedVector<ShardingSpec::IndexDomainAndShardIndices, 1>>
HloShardingSpec::UniqueIndexDomains(const Shape& shape) const {
  ABSL_RETURN_IF_ERROR(ValidateShape(shape));
  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Manual sharding does not support UniqueIndexDomains: "
                        "sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unreduced sharding does not support UniqueIndexDomains: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnknown()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unknown sharding does not support UniqueIndexDomains: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.HasNonReplicatedSubgroup()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Non-replicated subgroup (e.g., manual or unreduced subgroup) sharding "
        "does not support UniqueIndexDomains: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice()) {
    if (num_shards_ == 0) {
      return absl::InlinedVector<IndexDomainAndShardIndices, 1>();
    }
    absl::call_once(unique_shard_indices_once_, [this] {
      cached_shard_indices_.reserve(num_shards_);
      for (int i = 0; i < num_shards_; ++i) {
        cached_shard_indices_.push_back(i);
      }
    });
    return absl::InlinedVector<IndexDomainAndShardIndices, 1>{
        IndexDomainAndShardIndices{
            /*index_domain=*/IndexDomain(shape),
            /*shard_indices=*/absl::MakeConstSpan(cached_shard_indices_),
        },
    };
  }

  const int64_t tiled_data_rank = xla_hlo_sharding_.TiledDataRank();
  if (shape.dims().size() != tiled_data_rank) {
    return absl::InvalidArgumentError(
        absl::StrFormat("shape must have %d dimensions, but has %d dimensions: "
                        "shape=%v, sharding=%s",
                        tiled_data_rank, shape.dims().size(), shape,
                        xla_hlo_sharding_.ToString()));
  }
  ABSL_RETURN_IF_ERROR(ValidateTileAssignment());

  std::optional<xla::HloSharding> converted_v2;
  const xla::HloSharding* effective_sharding = &xla_hlo_sharding_;
  if (xla_hlo_sharding_.UseNamedShardingLeaf()) {
    converted_v2 = xla::HloSharding::V3ToV2Sharding(xla_hlo_sharding_);
    effective_sharding = &*converted_v2;
  }

  const int64_t num_unique_tiles = effective_sharding->NumTiles();
  if (num_unique_tiles <= 0 || num_shards_ % num_unique_tiles != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloShardingSpec has %d shards, but HloSharding has %d unique tiles, "
        "which is not a divisor of the number of shards; sharding=%s",
        num_shards_, num_unique_tiles, xla_hlo_sharding_.ToString()));
  }
  const int64_t num_replicas = num_shards_ / num_unique_tiles;

  absl::call_once(unique_shard_indices_once_, [&, this] {
    const int64_t* flat_tile_assignment =
        effective_sharding->tile_assignment().array().data();
    cached_shard_indices_.reserve(num_shards_);
    for (int64_t i = 0; i < num_shards_; ++i) {
      cached_shard_indices_.push_back(
          static_cast<int>(flat_tile_assignment[i]));
    }
  });

  absl::InlinedVector<IndexDomainAndShardIndices, 1> unique_domains;
  unique_domains.reserve(num_unique_tiles);
  for (int64_t tile_idx = 0; tile_idx < num_unique_tiles; ++tile_idx) {
    const int first_shard = cached_shard_indices_[tile_idx * num_replicas];
    unique_domains.push_back(IndexDomainAndShardIndices{
        /*index_domain=*/
        ComputeTileIndexDomainForV2Device(*effective_sharding, shape,
                                          first_shard),
        /*shard_indices=*/
        absl::MakeConstSpan(cached_shard_indices_)
            .subspan(tile_idx * num_replicas, num_replicas),
    });
  }

  return unique_domains;
}

absl::StatusOr<absl::Span<const int>>
HloShardingSpec::ShardToUniqueIndexDomainIndex() const {
  if (xla_hlo_sharding_.IsManual()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Manual sharding does not support "
                        "ShardToUniqueIndexDomainIndex: sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnreduced()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unreduced sharding does not support "
                        "ShardToUniqueIndexDomainIndex: sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsUnknown()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unknown sharding does not support "
                        "ShardToUniqueIndexDomainIndex: sharding=%s",
                        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.HasNonReplicatedSubgroup()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Non-replicated subgroup (e.g., manual or unreduced subgroup) sharding "
        "does not support ShardToUniqueIndexDomainIndex: sharding=%s",
        xla_hlo_sharding_.ToString()));
  }
  if (xla_hlo_sharding_.IsReplicatedOrSingleDevice()) {
    absl::call_once(shard_to_unique_index_domain_index_once_, [this] {
      cached_shard_to_unique_index_domain_index_.assign(num_shards_, 0);
    });
    return absl::MakeConstSpan(cached_shard_to_unique_index_domain_index_);
  }
  ABSL_RETURN_IF_ERROR(ValidateTileAssignment());

  std::optional<xla::HloSharding> converted_v2;
  const xla::HloSharding* effective_sharding = &xla_hlo_sharding_;
  if (xla_hlo_sharding_.UseNamedShardingLeaf()) {
    converted_v2 = xla::HloSharding::V3ToV2Sharding(xla_hlo_sharding_);
    effective_sharding = &*converted_v2;
  }

  const int64_t num_unique_tiles = effective_sharding->NumTiles();
  if (num_unique_tiles <= 0 || num_shards_ % num_unique_tiles != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "HloShardingSpec has %d shards, but HloSharding has %d unique tiles, "
        "which is not a divisor of the number of shards; sharding=%s",
        num_shards_, num_unique_tiles, xla_hlo_sharding_.ToString()));
  }
  const int64_t num_replicas = num_shards_ / num_unique_tiles;

  absl::call_once(shard_to_unique_index_domain_index_once_, [&, this] {
    cached_shard_to_unique_index_domain_index_.resize(num_shards_);
    const int64_t* flat_tile_assignment =
        effective_sharding->tile_assignment().array().data();
    for (int64_t tile_idx = 0; tile_idx < num_unique_tiles; ++tile_idx) {
      const int64_t offset = tile_idx * num_replicas;
      for (int64_t i = 0; i < num_replicas; ++i) {
        const int device_idx =
            static_cast<int>(flat_tile_assignment[offset + i]);
        cached_shard_to_unique_index_domain_index_[device_idx] = tile_idx;
      }
    }
  });
  return absl::MakeConstSpan(cached_shard_to_unique_index_domain_index_);
}

std::string HloShardingSpec::DebugString() const {
  return absl::StrFormat("HloShardingSpec(num_shards: %d, hlo_sharding: %s)",
                         num_shards_, xla_hlo_sharding_.ToString());
}

void HloShardingSpec::Hash(absl::HashState state) const {
  uint64_t hash = hash_.load(std::memory_order_relaxed);
  if (hash == kUnsetHash) {
    hash = absl::HashOf(num_shards_, xla_hlo_sharding_);
    if (ABSL_PREDICT_FALSE(hash == kUnsetHash)) {
      ++hash;
    }
    hash_.store(hash, std::memory_order_relaxed);
  }
  absl::HashState::combine(std::move(state), hash);
}

std::vector<IndexDomain> TEST_HloShardingSpecIndexDomainsSlowPath(
    const HloShardingSpec& sharding_spec, const Shape& shape) {
  CHECK_OK(ValidateShape(shape));
  const xla::HloSharding& hlo_sharding = sharding_spec.xla_hlo_sharding();
  CHECK(!hlo_sharding.IsManual());
  CHECK(!hlo_sharding.IsUnreduced());
  CHECK(!hlo_sharding.IsUnknown());
  if (!hlo_sharding.IsReplicatedOrSingleDevice()) {
    CHECK_EQ(shape.dims().size(), hlo_sharding.TiledDataRank());
    CHECK_OK(sharding_spec.ValidateTileAssignment());
  }
  return IndexDomainsSlowPath(hlo_sharding, sharding_spec.num_shards(), shape);
}

}  // namespace ifrt
}  // namespace xla
