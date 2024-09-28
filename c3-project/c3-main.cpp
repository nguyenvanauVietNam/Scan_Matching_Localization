#include <carla/client/Client.h>
#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Map.h>
#include <carla/geom/Location.h>
#include <carla/geom/Transform.h>
#include <carla/client/Sensor.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <thread>

#include <carla/client/Vehicle.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h> // TicToc

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

struct ControlState {
    float t;
    float s;
    float b;

    ControlState(float throttle, float steer, float brake) : t(throttle), s(steer), b(brake) {}
};

std::chrono::time_point<std::chrono::system_clock> currentTime;
std::vector<ControlState> cs;

bool refresh_view = false;
PointCloudT pclCloud;

void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void *viewer) {
    if (event.getKeySym() == "Right" && event.keyDown()) {
        cs.push_back(ControlState(0, -0.02, 0));
    } else if (event.getKeySym() == "Left" && event.keyDown()) {
        cs.push_back(ControlState(0, 0.02, 0));
    }
    if (event.getKeySym() == "Up" && event.keyDown()) {
        cs.push_back(ControlState(0.1, 0, 0));
    } else if (event.getKeySym() == "Down" && event.keyDown()) {
        cs.push_back(ControlState(-0.1, 0, 0));
    }
    if (event.getKeySym() == "a" && event.keyDown()) {
        refresh_view = true;
    }
}

void Accuate(ControlState response, cc::Vehicle::Control &state) {
    if (response.t > 0) {
        if (!state.reverse) {
            state.throttle = std::min(state.throttle + response.t, 1.0f);
        } else {
            state.reverse = false;
            state.throttle = std::min(response.t, 1.0f);
        }
    } else if (response.t < 0) {
        response.t = -response.t;
        if (state.reverse) {
            state.throttle = std::min(state.throttle + response.t, 1.0f);
        } else {
            state.reverse = true;
            state.throttle = std::min(response.t, 1.0f);
        }
    }
    state.steer = std::min(std::max(state.steer + response.s, -1.0f), 1.0f);
    state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr &viewer) {
    BoxQ box;
    box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
    renderBox(viewer, box, num, color, alpha);
}

Eigen::Matrix4d ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose) {
    Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
    Eigen::Matrix4d initTransform = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch,
                                                 startingPose.rotation.roll, startingPose.position.x,
                                                 startingPose.position.y, startingPose.position.z);
    PointCloudT::Ptr transformSource(new PointCloudT);
    pcl::transformPointCloud(*source, *transformSource, initTransform);

    pcl::console::TicToc time;
    time.tic();
    pcl::IterativeClosestPoint<PointT, PointT> icp;
    icp.setTransformationEpsilon(1e-8);
    int iterations = 60;
    icp.setMaximumIterations(iterations);
    icp.setInputSource(transformSource);
    icp.setInputTarget(target);

    PointCloudT::Ptr cloud_icp(new PointCloudT);  // ICP output point cloud
    icp.align(*cloud_icp);
    cout << "ICP has converged: " << icp.hasConverged() << " score: " << icp.getFitnessScore() << " time: " << time.toc() << " ms" << endl;

    if (icp.hasConverged()) {
        transformation_matrix = icp.getFinalTransformation().cast<double>();
        transformation_matrix = transformation_matrix * initTransform;
        return transformation_matrix;
    } else {
        cout << "WARNING: ICP did not converge" << endl;
    }
    return transformation_matrix;
}

Eigen::Matrix4d NDT(PointCloudT::Ptr mapCloud, PointCloudT::Ptr source, Pose startingPose) {
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(1e-8);
    ndt.setResolution(1);
    ndt.setInputTarget(mapCloud);

    pcl::console::TicToc time;
    time.tic();
    Eigen::Matrix4f init_guess = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch,
                                              startingPose.rotation.roll, startingPose.position.x,
                                              startingPose.position.y, startingPose.position.z).cast<float>();

    int iterations = 60;
    ndt.setMaximumIterations(iterations);
    ndt.setInputSource(source);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_ndt(new pcl::PointCloud<pcl::PointXYZ>);
    ndt.align(*cloud_ndt, init_guess);
    cout << "NDT has converged: " << ndt.hasConverged() << " score: " << ndt.getFitnessScore() << " time: " << time.toc() << " ms" << endl;
    Eigen::Matrix4d transformation_matrix = ndt.getFinalTransformation().cast<double>();
    return transformation_matrix;
}

void print_transform(Eigen::Matrix4d transform) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%f\t", transform(i, j));
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int USE_NDT = 1;
    if (argc == 2 && strcmp(argv[1], "2") == 0) USE_NDT = 0;
    printf("\n%s\n\n", USE_NDT ? "Using Algorithm 1: Normal Distributions Transform (NDT)" : "Using Algorithm 2: Iterative Closest Point (ICP)");

    auto client = cc::Client("localhost", 2000);
    client.SetTimeout(2s);
    auto world = client.GetWorld();
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicles = blueprint_library->Filter("vehicle");
    auto map = world.GetMap();
    auto transform = map->GetRecommendedSpawnPoints()[1];
    auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

    auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
    lidar_bp.SetAttribute("upper_fov", "15");
    lidar_bp.SetAttribute("lower_fov", "-25");
    lidar_bp.SetAttribute("channels", "32");
    lidar_bp.SetAttribute("range", "30");
    lidar_bp.SetAttribute("rotation_frequency", "60");
    lidar_bp.SetAttribute("points_per_second", "500000");

    auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8));
    auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
    auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
    bool new_scan = true;
    std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    viewer->registerKeyboardCallback(keyboardEventOccurred, (void *)&viewer);

    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
    Pose pose(Point(0, 0, 0), Rotate(0, 0, 0));

    PointCloudT::Ptr mapCloud(new PointCloudT);
    pcl::io::loadPCDFile("map.pcd", *mapCloud);
    cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;

    while (!viewer->wasStopped()) {
        if (new_scan) {
            currentTime = std::chrono::system_clock::now();
            lastScanTime = currentTime;

            auto lidar_data = lidar->Listen([&](const csd::LidarMeasurement &measurement) {
                auto points = measurement.GetPointCloud();
                pclCloud.clear();
                for (const auto &point : points) {
                    pclCloud.push_back(PointT(point.x, point.y, point.z));
                }
                pclCloud.width = pclCloud.size();
                pclCloud.height = 1;

                if (refresh_view) {
                    viewer->removeAllPointClouds();
                    viewer->addPointCloud<PointT>(mapCloud, "mapCloud");
                    viewer->addPointCloud<PointT>(&pclCloud, "lidarCloud");
                    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lidarCloud");
                    viewer->spinOnce();
                    refresh_view = false;
                }
                new_scan = false;
            });

            while (!new_scan) {
                auto current = std::chrono::system_clock::now();
                if (std::chrono::duration_cast<std::chrono::milliseconds>(current - lastScanTime).count() > 1000) {
                    cout << "No new scan data received for 1 second." << endl;
                    break;
                }
            }

            // Add control state management here
            ControlState response(0, 0, 0);
            if (!cs.empty()) {
                response = cs.front();
                cs.erase(cs.begin());
            }
            cc::Vehicle::Control control;
            Accuate(response, control);
            vehicle->ApplyControl(control);
            pose = getPose(vehicle->GetTransform());

            if (USE_NDT) {
                Eigen::Matrix4d transform = NDT(mapCloud, pclCloud.makeShared(), pose);
                print_transform(transform);
            } else {
                Eigen::Matrix4d transform = ICP(mapCloud, pclCloud.makeShared(), pose);
                print_transform(transform);
            }
        }
    }
    return 0;
}
