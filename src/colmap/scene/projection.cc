// Copyright (c) 2023, ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include "colmap/scene/projection.h"

#include "colmap/geometry/pose.h"
#include "colmap/math/matrix.h"

namespace colmap {

double CalculateSquaredReprojectionError(const Eigen::Vector2d& point2D,
                                         const Eigen::Vector3d& point3D,
                                         const Rigid3d& cam_from_world,
                                         const Camera& camera) {
  const Eigen::Vector3d point3D_in_cam = cam_from_world * point3D;

  const double c1 = camera.PrincipalPointX();
  const double c2 = camera.PrincipalPointY();

  
  const double pi = M_PI;
  const double two = 2;


  Eigen::Vector3d coords_3D;
  // Compute spherical coordinates for predicted point
  const double theta = (point2D[0] - c1) * pi / c1;
  const double phi = (point2D[1] - c2) * pi / (two * c2);

  coords_3D[0] = cos(phi) * sin(theta);
  coords_3D[1] = sin(phi);
  coords_3D[2] = cos(phi) * cos(theta);
  double M_dot_m = coords_3D.dot(point3D_in_cam.normalized());
  // LOG(INFO) << "[DEBUG] Reprojection " << 10*(coords_3D - point3D_in_cam.normalized()).squaredNorm() ;
  return 4* (1-M_dot_m)/(1+M_dot_m);
  // return (camera.ImgFromCam(point3D_in_cam.hnormalized()) - point2D)
  //     .squaredNorm();
}

double CalculateSquaredReprojectionError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world,
    const Camera& camera) {
  const double proj_z = cam_from_world.row(2).dot(point3D.homogeneous());
  const double proj_x = cam_from_world.row(0).dot(point3D.homogeneous());
  const double proj_y = cam_from_world.row(1).dot(point3D.homogeneous());
  Eigen::Vector3d coords_3D;
  coords_3D[0] = proj_x;
  coords_3D[1] = proj_y;
  coords_3D[2] = proj_z;
  coords_3D = coords_3D.normalized();
  // const double inv_proj_z = 1.0 / proj_z;

  const double c1 = camera.PrincipalPointX();
  const double c2 = camera.PrincipalPointY();

  
  const double pi = M_PI;
  const double two = 2;


  Eigen::Vector3d coords_3D_1;
  // Compute spherical coordinates for predicted point
  const double theta = (point2D[0] - c1) * pi / c1;
  const double phi = (point2D[1] - c2) * pi / (two * c2);

  coords_3D_1[0] = cos(phi) * sin(theta);
  coords_3D_1[1] = sin(phi);
  coords_3D_1[2] = cos(phi) * cos(theta);
  double M_dot_m = coords_3D.dot(coords_3D_1);

  return 4* (1-M_dot_m)/(1+M_dot_m);;
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Rigid3d& cam_from_world,
                             const Camera& camera) {
  return CalculateNormalizedAngularError(
      camera.CamFromImg(point2D), point3D, cam_from_world);
}

double CalculateAngularError(const Eigen::Vector2d& point2D,
                             const Eigen::Vector3d& point3D,
                             const Eigen::Matrix3x4d& cam_from_world,
                             const Camera& camera) {
  return CalculateNormalizedAngularError(
      camera.CamFromImg(point2D), point3D, cam_from_world);
}

double CalculateNormalizedAngularError(const Eigen::Vector2d& point2D,
                                       const Eigen::Vector3d& point3D,
                                       const Rigid3d& cam_from_world) {
  const Eigen::Vector3d ray1 = point2D.homogeneous();
  const Eigen::Vector3d ray2 = cam_from_world * point3D;
  return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

double CalculateNormalizedAngularError(
    const Eigen::Vector2d& point2D,
    const Eigen::Vector3d& point3D,
    const Eigen::Matrix3x4d& cam_from_world) {
  const Eigen::Vector3d ray1 = point2D.homogeneous();
  const Eigen::Vector3d ray2 = cam_from_world * point3D.homogeneous();
  return std::acos(ray1.normalized().transpose() * ray2.normalized());
}

bool HasPointPositiveDepth(const Eigen::Matrix3x4d& cam_from_world,
                           const Eigen::Vector3d& point3D) {
  return cam_from_world.row(2).dot(point3D.homogeneous()) >=
         std::numeric_limits<double>::epsilon();
}

}  // namespace colmap
