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

#include "xla/hlo/ir/mesh_and_axis.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/array.h"
#include "xla/hlo/ir/tile_assignment.h"
#include "xla/overflow_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

absl::Status Mesh::Validate() {
  if (device_assignment_.num_dimensions() != axes_names_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of axes names must match number of dimensions in the device "
        "assignment. Number of axes names: ",
        axes_names_.size(),
        ", Number of dimensions: ", device_assignment_.dimensions().size()));
  }

  if (device_assignment_.num_dimensions() == 0) {
    // Empty mesh or single-device mesh.
    if (device_assignment_.num_elements() >= 0 &&
        device_assignment_.num_elements() <= 1) {
      return absl::OkStatus();
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Single-device mesh must have at most 1 device id. Number of "
        "device ids: ",
        device_assignment_.num_elements()));
  }

  int64_t num_elements = 1;
  for (int64_t dim : device_assignment_.dimensions()) {
    if (dim <= 0) {
      return absl::InvalidArgumentError("Mesh axis size must be positive.");
    }
    auto [next, overflow] = OverflowSafeMultiply(num_elements, dim);
    if (overflow) {
      return absl::InvalidArgumentError(
          "Product of mesh axis sizes overflows int64_t.");
    }
    num_elements = next;
  }

  absl::flat_hash_set<std::string> seen_axis_names;
  for (const std::string& axis_name : axes_names_) {
    if (!seen_axis_names.insert(axis_name).second) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mesh has duplicate axis names. Duplicate axis name: ", axis_name));
    }
  }

  // Validate iota parameters or explicit device id permutation.
  if (device_assignment_.iota().has_value()) {
    const IotaTileAssignment& iota = *device_assignment_.iota();
    const absl::Span<const int64_t> reshape_dims = iota.reshape_dims();
    const absl::Span<const int> transpose_perm = iota.transpose_perm();
    if (reshape_dims.empty()) {
      return absl::InvalidArgumentError(
          "Mesh iota_transform reshape_dims must not be empty.");
    }
    if (reshape_dims.size() != transpose_perm.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Mesh iota_transform reshape_dims size (%d) must match "
          "transpose_perm size (%d).",
          reshape_dims.size(), transpose_perm.size()));
    }
    int64_t reshape_elements = 1;
    for (int64_t dim : reshape_dims) {
      if (dim <= 0) {
        return absl::InvalidArgumentError(
            "Mesh iota_transform reshape_dims must be positive.");
      }
      auto [next, overflow] = OverflowSafeMultiply(reshape_elements, dim);
      if (overflow) {
        return absl::InvalidArgumentError(
            "Product of mesh iota_transform reshape_dims overflows int64_t.");
      }
      reshape_elements = next;
    }
    if (reshape_elements != num_elements) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Product of mesh iota_transform reshape_dims (%d) must match "
          "product of mesh axis sizes (%d).",
          reshape_elements, num_elements));
    }
    std::vector<bool> seen_perm(transpose_perm.size(), false);
    for (int p : transpose_perm) {
      if (p < 0 || static_cast<size_t>(p) >= transpose_perm.size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Mesh iota_transform transpose_perm entry %d out of range "
            "[0, %d).",
            p, transpose_perm.size()));
      }
      if (seen_perm[p]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Mesh iota_transform transpose_perm entry %d is not unique.", p));
      }
      seen_perm[p] = true;
    }
    return absl::OkStatus();
  }
  std::vector<int64_t> device_ids(device_assignment_.array().begin(),
                                  device_assignment_.array().end());
  for (int64_t device_id : device_ids) {
    if (device_id < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Mesh device ids must be non-negative. Device id: ", device_id));
    }
  }
  std::vector<int64_t> iota(device_ids.size());
  absl::c_iota(iota, 0);

  // For non-iota cases the device ids should be a non-identity permutation
  // of iota.
  if (device_ids == iota) {
    return absl::InvalidArgumentError(
        "Non-iota device assignment has iota device id list [0,1,2,3...].");
  }
  absl::c_sort(device_ids);
  if (device_ids != iota) {
    return absl::InvalidArgumentError(
        "Device ids must be a permutation of [0,1,2,3...].");
  }
  return absl::OkStatus();
}

Mesh::Mesh(TileAssignment device_assignment,
           absl::Span<const absl::string_view> axes_names)
    : device_assignment_(std::move(device_assignment)),
      axes_names_(axes_names.begin(), axes_names.end()) {
  CHECK_OK(Validate());
}

std::string Mesh::ToString() const {
  if (IsSingleDevice()) {
    return absl::StrCat(
        "single_device[device_id=", device_assignment_.array()(0), "]");
  }

  std::string mesh_str = "mesh";
  // Add the mesh axes names and sizes.
  std::vector<std::string> formatted_axes_names;
  formatted_axes_names.reserve(axes_names_.size());
  for (int64_t i = 0; i < axes_names_.size(); ++i) {
    formatted_axes_names.push_back(
        absl::StrCat("'", axes_names_[i], "'", "=", device_assignment_.dim(i)));
  }

  // Add the device assignment if it is not an iota case.
  std::optional<IotaTileAssignment> iota = device_assignment_.iota();
  std::string device_assignment_str = "";
  bool simple_iota = iota.has_value() && iota->reshape_dims().size() == 1;
  if (!simple_iota && device_assignment_.num_elements() != 0) {
    device_assignment_str =
        absl::StrCat(", device_ids=(", device_assignment_.ArrayToString(), ")");
  }
  absl::StrAppend(&mesh_str, "[", absl::StrJoin(formatted_axes_names, ","), "]",
                  device_assignment_str);
  return mesh_str;
}

MeshProto Mesh::ToProto() const {
  MeshProto proto;

  if (num_axes() == 0) {
    if (device_assignment_.num_elements() == 0) {
      return MeshProto();
    }
    // Single-device mesh
    // TODO(b/454008727): Validate device_ids_size is 1.
    proto.add_device_ids(*device_assignment_.array().begin());
    return proto;
  }

  std::vector<MeshProto::MeshAxis> axes;
  axes.reserve(num_axes());

  for (int64_t i = 0; i < axes_names_.size(); ++i) {
    MeshProto::MeshAxis axis;
    axis.set_name(axes_names_[i]);
    axis.set_size(device_assignment_.dim(i));
    axes.push_back(std::move(axis));
  }
  proto.mutable_axes()->Assign(axes.begin(), axes.end());

  // Serialize device IDs.
  std::optional<IotaTileAssignment> iota = device_assignment_.iota();
  if (iota.has_value() && iota->reshape_dims().size() != 1) {
    proto.mutable_iota_transform()->mutable_reshape_dims()->Assign(
        iota->reshape_dims().begin(), iota->reshape_dims().end());
    for (int elem : iota.value().transpose_perm()) {
      proto.mutable_iota_transform()->add_transpose_perm(
          static_cast<int64_t>(elem));
    }
  } else if (!iota.has_value()) {
    proto.mutable_device_ids()->Assign(device_assignment_.array().begin(),
                                       device_assignment_.array().end());
  }
  // The proto must not have defined both a device IDs list and have an
  // iota transform, since the iota transform makes no sense in that case.
  CHECK(!(proto.has_iota_transform() && proto.device_ids_size() != 0))
      << "Mesh must not have an iota transform and a device ID list";
  return proto;
}

Mesh Mesh::FromProto(const MeshProto& proto) {
  absl::StatusOr<Mesh> mesh = TryFromProto(proto);
  CHECK_OK(mesh);
  return *std::move(mesh);
}

absl::StatusOr<Mesh> Mesh::TryFromProto(const MeshProto& proto) {
  if (proto.axes_size() == 0) {
    if (proto.has_iota_transform()) {
      return absl::InvalidArgumentError(
          "Mesh without axes must not have an iota transform.");
    }
    if (proto.device_ids_size() == 0) {
      return Mesh();
    }
    if (proto.device_ids_size() != 1) {
      return absl::InvalidArgumentError(
          "Maximal mesh must have exactly 1 device id.");
    }
    if (proto.device_ids(0) < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Mesh device ids must be non-negative. Device id: ",
                       proto.device_ids(0)));
    }
    return Mesh(proto.device_ids(0));
  }

  std::vector<int64_t> mesh_axis_sizes;
  std::vector<std::string> mesh_axis_names;
  mesh_axis_sizes.reserve(proto.axes_size());
  mesh_axis_names.reserve(proto.axes_size());
  int64_t num_elements = 1;
  for (const MeshProto::MeshAxis& axis : proto.axes()) {
    if (axis.size() <= 0) {
      return absl::InvalidArgumentError("Mesh axis size must be positive.");
    }
    auto [next_product, overflow] =
        OverflowSafeMultiply(num_elements, axis.size());
    if (overflow) {
      return absl::InvalidArgumentError(
          "Product of mesh axis sizes overflows int64_t.");
    }
    num_elements = next_product;
    mesh_axis_sizes.push_back(axis.size());
    mesh_axis_names.push_back(axis.name());
  }

  Mesh mesh;
  mesh.axes_names_ = std::move(mesh_axis_names);

  // If device ids are not specified, create a mesh with iota tiling.
  if (proto.device_ids_size() == 0) {
    if (proto.has_iota_transform()) {
      const MeshProto::IotaTransform& transform = proto.iota_transform();
      if (transform.reshape_dims().empty()) {
        return absl::InvalidArgumentError(
            "Mesh iota_transform reshape_dims must not be empty.");
      }
      if (transform.reshape_dims_size() != transform.transpose_perm_size()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Mesh iota_transform reshape_dims size (%d) must match "
            "transpose_perm size (%d).",
            transform.reshape_dims_size(), transform.transpose_perm_size()));
      }
      int64_t reshape_elements = 1;
      for (int64_t dim : transform.reshape_dims()) {
        if (dim <= 0) {
          return absl::InvalidArgumentError(
              "Mesh iota_transform reshape_dims must be positive.");
        }
        auto [next_product, overflow] =
            OverflowSafeMultiply(reshape_elements, dim);
        if (overflow) {
          return absl::InvalidArgumentError(
              "Product of mesh iota_transform reshape_dims overflows int64_t.");
        }
        reshape_elements = next_product;
      }
      if (reshape_elements != num_elements) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Product of mesh iota_transform reshape_dims (%d) must match "
            "product of mesh axis sizes (%d).",
            reshape_elements, num_elements));
      }
      const int perm_size = transform.transpose_perm_size();
      std::vector<bool> seen_perm(perm_size, false);
      for (int64_t perm_dim : transform.transpose_perm()) {
        if (perm_dim < 0 || perm_dim >= perm_size) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Mesh iota_transform transpose_perm entry %d out of range "
              "[0, %d).",
              perm_dim, perm_size));
        }
        if (seen_perm[perm_dim]) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Mesh iota_transform transpose_perm entry %d is not unique.",
              perm_dim));
        }
        seen_perm[perm_dim] = true;
      }
      mesh.device_assignment_ = TileAssignment(
          IotaTileAssignment::Create(mesh_axis_sizes, transform));
    } else {
      mesh.device_assignment_ =
          TileAssignment(IotaTileAssignment::Create(mesh_axis_sizes));
    }
    ABSL_RETURN_IF_ERROR(mesh.Validate());
    return mesh;
  }

  // The proto must not have defined both a device IDs list and have an
  // iota transform, since the iota transform makes no sense in this case.
  if (proto.has_iota_transform()) {
    return absl::InvalidArgumentError(
        "Mesh must not have an iota transform and a device ID list");
  }

  if (proto.device_ids_size() != num_elements) {
    return absl::InvalidArgumentError(
        "Number of device ids must match the product of mesh axis sizes.");
  }
  Array<int64_t> device_ids_array(mesh_axis_sizes);
  absl::c_copy(proto.device_ids(), device_ids_array.begin());
  mesh.device_assignment_ = TileAssignment(
      std::make_shared<Array<int64_t>>(std::move(device_ids_array)));
  ABSL_RETURN_IF_ERROR(mesh.Validate());
  return mesh;
}

bool Mesh::ContainsAllMeshAxesInOrder(absl::Span<const AxisRef> axes) const {
  int64_t axes_idx = 0;
  for (int64_t i = 0; i < num_axes(); ++i) {
    if (axes_idx < axes.size() && axes[axes_idx].mesh_axis_index() == i) {
      if (axes[axes_idx].sub_axis_info().has_value()) {
        return false;
      }
      axes_idx++;
    } else if (axis_size(i) != 1) {
      return false;
    }
  }
  return axes_idx == axes.size();
}

std::string AxisRef::ToString(const Mesh* mesh) const {
  // TODO(b/474013054): Remove these checks if they have significant overhead.
  CHECK_GE(mesh_axis_index_, 0);
  if (mesh) {
    CHECK_LT(mesh_axis_index_, mesh->num_axes());
  }
  std::string axis_str =
      mesh ? absl::StrCat("'", mesh->axis_names()[mesh_axis_index_], "'")
           : std::to_string(mesh_axis_index_);
  if (sub_axis_info_.has_value()) {
    absl::StrAppend(&axis_str, ":(", sub_axis_info_->pre_size, ")",
                    sub_axis_info_->size);
  }
  return axis_str;
}

AxisRefProto AxisRef::ToProto() const {
  AxisRefProto proto;
  proto.set_mesh_axis_index(mesh_axis_index_);
  if (sub_axis_info_.has_value()) {
    proto.mutable_sub_axis_info()->set_pre_size(sub_axis_info_->pre_size);
    proto.mutable_sub_axis_info()->set_size(sub_axis_info_->size);
  }
  return proto;
}

AxisRef AxisRef::FromProto(const AxisRefProto& proto) {
  absl::StatusOr<AxisRef> axis_ref = TryFromProto(proto);
  CHECK_OK(axis_ref);
  return *std::move(axis_ref);
}

absl::StatusOr<AxisRef> AxisRef::TryFromProto(const AxisRefProto& proto) {
  if (proto.mesh_axis_index() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Axis index must be non-negative. Axis index: ",
                     proto.mesh_axis_index()));
  }
  if (proto.has_sub_axis_info()) {
    const int64_t pre_size = proto.sub_axis_info().pre_size();
    const int64_t size = proto.sub_axis_info().size();
    if (pre_size < 1) {
      return absl::InvalidArgumentError("sub-axis pre-size must be >= 1");
    }
    if (size <= 1) {
      return absl::InvalidArgumentError("sub-axis size must be > 1");
    }
    if (OverflowSafeMultiply(pre_size, size).second) {
      return absl::InvalidArgumentError(
          "sub-axis next_pre_size overflows int64_t");
    }
    return AxisRef(proto.mesh_axis_index(), SubAxis{pre_size, size});
  }
  return AxisRef(proto.mesh_axis_index());
}

AxisRef::AxisRef(int64_t mesh_axis_index) : mesh_axis_index_(mesh_axis_index) {}

AxisRef::AxisRef(int64_t mesh_axis_index, SubAxis sub_axis_info)
    : mesh_axis_index_(mesh_axis_index), sub_axis_info_(sub_axis_info) {
  CHECK_GT(sub_axis_info_->pre_size, 0) << "sub-axis pre-size must be >= 1";
  CHECK_GT(sub_axis_info_->size, 1) << "sub-axis size must be > 1";
}

namespace {

bool CanSubAxesCoexist(int64_t min_pre_size, int64_t max_pre_size,
                       int64_t min_next_pre_size, int64_t max_next_pre_size) {
  if (min_next_pre_size > max_pre_size) {
    // Sub-axes overlap, check if overlapping and non-overlapping parts are
    // valid.
    return min_next_pre_size % max_pre_size == 0 &&
           max_pre_size % min_pre_size == 0 &&
           max_next_pre_size % min_next_pre_size == 0;
  }
  // Sub-axes don't overlap, check if the gap is valid.
  return max_pre_size % min_next_pre_size == 0;
}

}  // namespace

bool AxisRef::CanCoexist(const AxisRef& other) const {
  if (mesh_axis_index() != other.mesh_axis_index()) {
    return true;
  }
  if (!sub_axis_info_.has_value() || !other.sub_axis_info_.has_value()) {
    // One of the axes is full
    return true;
  }
  const SubAxis& this_sub = *sub_axis_info_;
  const SubAxis& other_sub = *other.sub_axis_info_;

  auto [min_pre_size, max_pre_size] =
      std::minmax(this_sub.pre_size, other_sub.pre_size);
  auto [min_next_pre_size, max_next_pre_size] =
      std::minmax({this_sub.next_pre_size(), other_sub.next_pre_size()});

  return CanSubAxesCoexist(min_pre_size, max_pre_size, min_next_pre_size,
                           max_next_pre_size);
}

bool AxisRef::Overlaps(const AxisRef& other) const {
  if (mesh_axis_index() != other.mesh_axis_index()) {
    return false;
  }
  if (!sub_axis_info_.has_value() || !other.sub_axis_info_.has_value()) {
    // One of the axes is full
    return true;
  }
  const SubAxis& this_sub = *sub_axis_info_;
  const SubAxis& other_sub = *other.sub_axis_info_;

  return this_sub.pre_size < other_sub.next_pre_size() &&
         other_sub.pre_size < this_sub.next_pre_size();
}

std::optional<AxisRef> AxisRef::GetPrefixWithoutOverlap(
    const AxisRef& other) const {
  if (!CanCoexist(other)) {
    return std::nullopt;
  }
  if (!Overlaps(other)) {
    return *this;
  }

  int64_t this_pre_size =
      sub_axis_info_.has_value() ? sub_axis_info_->pre_size : 1;
  int64_t other_pre_size =
      other.sub_axis_info_.has_value() ? other.sub_axis_info_->pre_size : 1;

  if (this_pre_size >= other_pre_size) {
    return std::nullopt;
  }
  return AxisRef(mesh_axis_index_,
                 SubAxis{this_pre_size, other_pre_size / this_pre_size});
}

bool AxisRef::CanCoexistWithoutOverlap(const AxisRef& other) const {
  // Check if the axes are on different mesh dimensions. If so, they can always
  // coexist and never overlap.
  if (mesh_axis_index() != other.mesh_axis_index()) {
    return true;
  }

  // If one AxisRef is a full axis it will always overlap the other axis on the
  // same dimension.
  if (!sub_axis_info_.has_value() || !other.sub_axis_info_.has_value()) {
    return false;
  }

  const SubAxis& this_sub_axis = sub_axis_info_.value();
  const SubAxis& other_sub_axis = other.sub_axis_info_.value();

  int64_t this_pre_size = this_sub_axis.pre_size;
  int64_t other_pre_size = other_sub_axis.pre_size;
  int64_t this_next_pre_size = this_sub_axis.next_pre_size();
  int64_t other_next_pre_size = other_sub_axis.next_pre_size();

  // Check for overlapping sub-axes
  bool overlaps = (this_next_pre_size > other_pre_size) &&
                  (other_next_pre_size > this_pre_size);
  if (overlaps) {
    return false;
  }
  // Assert that sub-axes can coexist.
  auto [min_pre_size, max_pre_size] =
      std::minmax(this_pre_size, other_pre_size);
  auto [min_next_pre_size, max_next_pre_size] =
      std::minmax(this_next_pre_size, other_next_pre_size);

  // Sub-axes don't overlap, check if the gap is valid.
  return max_pre_size % min_next_pre_size == 0;
}

bool AxisRef::CanMerge(const AxisRef& other) const {
  if (mesh_axis_index_ != other.mesh_axis_index()) {
    return false;
  }
  if (!sub_axis_info_.has_value() || !other.sub_axis_info_.has_value()) {
    return false;
  }
  return sub_axis_info_->next_pre_size() == other.sub_axis_info_->pre_size;
}

bool AxisRef::Merge(const AxisRef& other, const Mesh& mesh) {
  if (!CanMerge(other)) {
    return false;
  }

  sub_axis_info_->size *= other.sub_axis_info_->size;
  if (sub_axis_info_->size == mesh.axis_size(mesh_axis_index_)) {
    assert(sub_axis_info_->pre_size == 1);
    sub_axis_info_ = std::nullopt;
  }
  return true;
}

absl::Status AxisRef::Validate(const Mesh& mesh) const {
  if (mesh_axis_index_ < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Axis index must be non-negative. Axis index: ", mesh_axis_index_));
  }
  if (mesh_axis_index_ >= mesh.num_axes()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Axis index must be less than number of axes. Axis index: ",
        mesh_axis_index_, ", Number of axes: ", mesh.axis_names().size()));
  }
  if (!sub_axis_info_.has_value()) {
    return absl::OkStatus();
  }

  int64_t axis_size = mesh.axis_size(mesh_axis_index_);
  if (axis_size % sub_axis_info_->next_pre_size() != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Sub-axis next_pre_size must divide the full axis size. Next "
        "pre-size: ",
        sub_axis_info_->next_pre_size(), ", Axis size: ", axis_size));
  }
  if (sub_axis_info_->size >= axis_size) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Sub-axis size must be strictly less than the full axis size. Sub-axis "
        "size: ",
        sub_axis_info_->size, ", Axis size: ", axis_size));
  }
  return absl::OkStatus();
}

int64_t AxisRef::size(const Mesh& mesh) const {
  if (sub_axis_info_.has_value()) {
    return sub_axis_info_->size;
  }

  return mesh.axis_size(mesh_axis_index_);
}

std::ostream& operator<<(std::ostream& out, const Mesh& mesh) {
  return out << mesh.ToString();
}

std::ostream& operator<<(std::ostream& out, const AxisRef& axis) {
  return out << axis.ToString();
}

bool AxesCanCoexistWithoutOverlap(absl::Span<const AxisRef> axes) {
  if (axes.size() < 2) {
    return true;
  }
  for (auto it1 = axes.begin(); it1 != std::prev(axes.end()); ++it1) {
    for (auto it2 = std::next(it1); it2 != axes.end(); ++it2) {
      if (!it1->CanCoexistWithoutOverlap(*it2)) {
        return false;
      }
    }
  }
  return true;
}

absl::Status ValidateSpanOfAxes(absl::Span<const AxisRef> axes,
                                const Mesh& mesh,
                                bool allow_mergeable_neighbors) {
  if (axes.empty()) {
    return absl::OkStatus();
  }
  for (const AxisRef& axis : axes) {
    ABSL_RETURN_IF_ERROR(axis.Validate(mesh));
  }
  if (!AxesCanCoexistWithoutOverlap(axes)) {
    return absl::InvalidArgumentError("Axes cannot coexist or axes overlap.");
  }
  if (allow_mergeable_neighbors) {
    return absl::OkStatus();
  }
  for (auto it = axes.begin(); it != std::prev(axes.end()); ++it) {
    if (it->CanMerge(*std::next(it))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Adjacent axes in dimension sharding can be merged: ",
          it->ToString(&mesh), ", ", std::next(it)->ToString(&mesh)));
    }
  }
  return absl::OkStatus();
}

void MergeAxes(std::vector<AxisRef>& axes, const Mesh& mesh) {
  if (axes.empty()) {
    return;
  }

  auto current = axes.begin();
  for (auto next = current + 1; next != axes.end(); ++next) {
    if (current->Overlaps(*next)) {
      LOG(FATAL) << "Axes should not overlap: " << current->ToString(&mesh)
                 << " and " << next->ToString(&mesh);
    }
    if (current->CanMerge(*next)) {
      CHECK(current->Merge(*next, mesh));
    } else {
      current++;
      *current = *next;
    }
  }
  axes.erase(current + 1, axes.end());
}

void SortAndMergeAxes(std::vector<AxisRef>& axes, const Mesh& mesh) {
  if (axes.empty()) {
    return;
  }

  absl::c_sort(axes);
  MergeAxes(axes, mesh);
}

bool TruncateAxesByRemovingOverlaps(std::vector<AxisRef>& axes,
                                    absl::Span<const AxisRef> other_axis_refs) {
  for (size_t i = 0; i < axes.size(); ++i) {
    std::optional<AxisRef> prefix = axes[i];
    for (const AxisRef& other : other_axis_refs) {
      prefix = prefix->GetPrefixWithoutOverlap(other);
      if (!prefix) {
        break;
      }
    }

    if (!prefix) {
      axes.erase(axes.begin() + i, axes.end());
      return true;
    }
    if (axes[i] != *prefix) {
      axes[i] = *prefix;
      axes.erase(axes.begin() + i + 1, axes.end());
      return true;
    }
  }
  return false;
}

}  // namespace xla
