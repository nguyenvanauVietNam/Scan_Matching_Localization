
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

//pcl code
//#include "render/render.h"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

using namespace std;

#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>
#include "helper.h"
#include <sstream>
#include <chrono> 
#include <ctime> 
#include <pcl/registration/icp.h>
#include <pcl/registration/ndt.h>
#include <pcl/console/time.h>   // TicToc

PointCloudT pclCloud;
cc::Vehicle::Control control;
std::chrono::time_point<std::chrono::system_clock> currentTime;
vector<ControlState> cs;

bool refresh_view = false;
void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer)
{

  	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = *static_cast<boost::shared_ptr<pcl::visualization::PCLVisualizer> *>(viewer_void);
	if (event.getKeySym() == "Right" && event.keyDown()){
		cs.push_back(ControlState(0, -0.02, 0));
  	}
	else if (event.getKeySym() == "Left" && event.keyDown()){
		cs.push_back(ControlState(0, 0.02, 0)); 
  	}
  	if (event.getKeySym() == "Up" && event.keyDown()){
		cs.push_back(ControlState(0.1, 0, 0));
  	}
	else if (event.getKeySym() == "Down" && event.keyDown()){
		cs.push_back(ControlState(-0.1, 0, 0)); 
  	}
	if(event.getKeySym() == "a" && event.keyDown()){
		refresh_view = true;
	}
}

void Accuate(ControlState response, cc::Vehicle::Control& state){

	if(response.t > 0){
		if(!state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = false;
			state.throttle = min(response.t, 1.0f);
		}
	}
	else if(response.t < 0){
		response.t = -response.t;
		if(state.reverse){
			state.throttle = min(state.throttle+response.t, 1.0f);
		}
		else{
			state.reverse = true;
			state.throttle = min(response.t, 1.0f);

		}
	}
	state.steer = min( max(state.steer+response.s, -1.0f), 1.0f);
	state.brake = response.b;
}

void drawCar(Pose pose, int num, Color color, double alpha, pcl::visualization::PCLVisualizer::Ptr& viewer){

	BoxQ box;
	box.bboxTransform = Eigen::Vector3f(pose.position.x, pose.position.y, 0);
    box.bboxQuaternion = getQuaternion(pose.rotation.yaw);
    box.cube_length = 4;
    box.cube_width = 2;
    box.cube_height = 2;
	renderBox(viewer, box, num, color, alpha);
}

Eigen::Matrix4d ICP(PointCloudT::Ptr target, PointCloudT::Ptr source, Pose startingPose)
{
    // Initialize transformation matrix to identity
    Eigen::Matrix4d transf_matrix = Eigen::Matrix4d::Identity();
	// Setup and configure the Generalized Iterative Closest Point (GICP) algorithm
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icpDetection;
	// Set the maximum transformation epsilon
	icpDetection.setTransformationEpsilon(1e-8); 

	

    // Calculate initial transformation based on the starting pose
    Eigen::Matrix4d initTransf = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, 
                                            startingPose.position.x, startingPose.position.y, startingPose.position.z).cast<float>();

    // Apply the initial transformation to the source point cloud
    // PointCloudT::Ptr transformSource(new PointCloudT);
    // pcl::transformPointCloud(*source, *transformSource, initTransf);

	// // Start the timer
	// pcl::console::TicToc time;
	// time.tic();
    // Setup and configure the Iterative Closest Point (ICP) algorithm
    pcl::console::TicToc time;
    time.tic();

	// Set the maximum number of iterations
	int iterations = 100;
	
	//using GICP
	// Set the transformation epsilon
	icpDetection.setMaximumIterations(iterations);
	// Set the input point clouds
	icpDetection.setInputSource(source);
	// Set the target point clouds
	icpDetection.setInputTarget(target);

	// Output point cloud after ICP alignment using ICP  base
    // pcl::IterativeClosestPoint<PointT, PointT> icp;
    // icp.setMaximumIterations(iterations); // Set max number of iterations
    // icp.setInputSource(transformSource);  // Set transformed source cloud
    // icp.setInputTarget(target);           // Set target cloud

	
    // Output point cloud after ICP alignment
    PointCloudT::Ptr cloud_icp(new PointCloudT); 
    //icp.align(*cloud_icp, initTransf); // Align source to target
	icpDetection.align(*cloud_icp, initTransf); // Align source to target

    // Check if ICP has successfully converged
    if (icpDetection.hasConverged())
    {
        // Get the final transformation matrix
		std::cout << "\nICP took " << time.toc() << " ms" << std::endl;
		std::cout << "\nICP score : " << icp.getFitnessScore() << std::endl;
        // Get the final transformation matrix and combine it with the initial transform
        transf_matrix = icpDetection.getFinalTransformation().cast<double>();
        //transf_matrix = transf_matrix * initTransf;
		// Return the final transformation matrix
        return transf_matrix;
    }

	// If ICP has not converged, display an error message
	std::cerr << "\nICP did not converge" << std::endl;
	// Return the initial transformation matrix
    return transf_matrix;
}

Eigen::Matrix4d NDT(PointCloudT::Ptr mapCloud, PointCloudT::Ptr source, Pose startingPose)
{
   	// Setup and configure the Normal Distributions Transform (NDT) algorithm
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndtDetection;
	// Initialize transformation matrix to identity
    Eigen::Matrix4d transf_matrix = Eigen::Matrix4d::Identity();
	// Calculate initial transformation based on the starting pose
    Eigen::Matrix4d initTransf = transform3D(startingPose.rotation.yaw, startingPose.rotation.pitch, startingPose.rotation.roll, 
                                            startingPose.position.x, startingPose.position.y, startingPose.position.z).cast<float>();

	// Apply the initial transformation to the source point cloud
	PointCloudT::Ptr transformSource(new PointCloudT);
	pcl::transformPointCloud(*source, *transformSource, initTransf);


    // Start the timer
    pcl::console::TicToc time;
    time.tic();

	// Setup and configure the Normal Distributions Transform (NDT) algorithm
    // Set the maximum transformation epsilon
    ndtDetection.setTransformationEpsilon(1e-8);
	// Set the maximum number of iterations
	int iterations = 60;

    // Set the resolution of the NDT grid (VoxelGridCovariance)

    ndtDetection.setResolution(1.0); 
    ndtDetection.setInputTarget(mapCloud); // Set the target map point cloud
	ndtDetection.setMaximumIterations(iterations);
	ndtDetection.setStepSize(0.1);
	ndtDetection.setTransformationEpsilon(0.01);
	ndtDetection.setRANSACIterations(0);
	ndtDetection.setInputSource(transformSource); // Set transformed source cloud

	// Output point cloud after NDT alignment
	PointCloudT::Ptr cloud_ndt(new PointCloudT);
	ndtDetection.align(*cloud_ndt); // Align source to target

	// Check if NDT has successfully converged
	if (ndtDetection.hasConverged())
	{
		// Print the convergence result and score
		std::cout << "\nNDT took " << time.toc() << " ms" << std::endl;
		std:cout<<"\nNDT score : " << ndt.getFitnessScore() << std::endl;

		// Get the final transformation matrix
		Eigen::Matrix4d final_transf = ndt.getFinalTransformation().cast<double>();
		// Return the final transformation matrix
		return final_transf;
	}

	// If NDT has not converged, display an error message
	std::cerr << "\nNDT did not converge" << std::endl;
	// Return the final transformation matrix
    return final_transf;
}
//using int argc, char *argv[] in main

// int main(){
int main(int argc, char *argv[])
{
	//Declare the variable using ICP algorithm and NDT algorithm
	int choose_ICP = 0; // 0 for ICP, 1 for NDT. Default is ICP
	if(argc == 0 && strcmp(argv[1], "2") == 0)
	{
		choose_ICP = 1;//using NDT algorithm
	}
	//print the usage algorithm in console
	if(choose_ICP ==0)
	{
		std::cout<<"Using ICP algorithm"<<std::endl;
	}else if (choose_ICP ==1)
	{
		std::cout<<"Using NDT algorithm"<<std::endl;
	}
	else
	{
		std::cout<<"Wrong input, using ICP algorithm"<<std::endl;
		choose_ICP = 0;
	}

	auto client = cc::Client("localhost", 2000);
	client.SetTimeout(2s);
	auto world = client.GetWorld();

	auto blueprint_library = world.GetBlueprintLibrary();
	auto vehicles = blueprint_library->Filter("vehicle");

	auto map = world.GetMap();
	auto transform = map->GetRecommendedSpawnPoints()[1];
	auto ego_actor = world.SpawnActor((*vehicles)[12], transform);

	//Create lidar
	auto lidar_bp = *(blueprint_library->Find("sensor.lidar.ray_cast"));
	// CANDO: Can modify lidar values to get different scan resolutions
	lidar_bp.SetAttribute("upper_fov", "15");
    lidar_bp.SetAttribute("lower_fov", "-25");
    lidar_bp.SetAttribute("channels", "32");
    lidar_bp.SetAttribute("range", "30");
	lidar_bp.SetAttribute("rotation_frequency", "60");
	lidar_bp.SetAttribute("points_per_second", "500000");

	auto user_offset = cg::Location(0, 0, 0);
	auto lidar_transform = cg::Transform(cg::Location(-0.5, 0, 1.8) + user_offset);
	auto lidar_actor = world.SpawnActor(lidar_bp, lidar_transform, ego_actor.get());
	auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);
	bool new_scan = true;
	std::chrono::time_point<std::chrono::system_clock> lastScanTime, startTime;

	pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  	viewer->setBackgroundColor (0, 0, 0);
	viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);

	auto vehicle = boost::static_pointer_cast<cc::Vehicle>(ego_actor);
	Pose pose(Point(0,0,0), Rotate(0,0,0));

	// Load map
	PointCloudT::Ptr mapCloud(new PointCloudT);
  	pcl::io::loadPCDFile("map.pcd", *mapCloud);
  	cout << "Loaded " << mapCloud->points.size() << " data points from map.pcd" << endl;
	renderPointCloud(viewer, mapCloud, "map", Color(0,0,1)); 

	typename pcl::PointCloud<PointT>::Ptr cloudFiltered (new pcl::PointCloud<PointT>);
	typename pcl::PointCloud<PointT>::Ptr scanCloud (new pcl::PointCloud<PointT>);

	lidar->Listen([&new_scan, &lastScanTime, &scanCloud](auto data){

		if(new_scan){
			auto scan = boost::static_pointer_cast<csd::LidarMeasurement>(data);
			for (auto detection : *scan){
				// if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
				// 	pclCloud.points.push_back(PointT(detection.x, detection.y, detection.z));
				// }
				//Fix by https://knowledge.udacity.com/questions/997414
				// if((detection.point.x*detection.point.x + detection.point.y*detection.point.y + detection.point.z*detection.point.z) > 8.0){ // Don't include points touching ego
				// 	pclCloud.points.push_back(PointT(detection.point.x, detection.point.y, detection.point.z));
				// }
				//QnA https://knowledge.udacity.com/questions/1052767

				// Ignore points that are too close to the ego vehicle
				// by checking if the distance between the detection and
				// the origin (0,0,0) is greater than a certain value (8.0 in this case)
				// If the distance is greater than 8.0, add the point to the pclCloud
				if((detection.x*detection.x + detection.y*detection.y + detection.z*detection.z) > 8.0){
					pclCloud.points.push_back(PointT(detection.x, detection.y, detection.z));
				}
			}
			if(pclCloud.points.size() > 5000){ // CANDO: Can modify this value to get different scan resolutions
				lastScanTime = std::chrono::system_clock::now();
				*scanCloud = pclCloud;
				new_scan = false;
			}
		}
	});
	
	Pose poseRef(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180));
	double maxError = 0;

	while (!viewer->wasStopped())
  	{
		while(new_scan){
			std::this_thread::sleep_for(0.1s);
			world.Tick(1s);
		}
		if(refresh_view){
			viewer->setCameraPosition(pose.position.x, pose.position.y, 60, pose.position.x+1, pose.position.y+1, 0, 0, 0, 1);
			refresh_view = false;
		}
		
		viewer->removeShape("box0");
		viewer->removeShape("boxFill0");
		Pose truePose = Pose(Point(vehicle->GetTransform().location.x, vehicle->GetTransform().location.y, vehicle->GetTransform().location.z), Rotate(vehicle->GetTransform().rotation.yaw * pi/180, vehicle->GetTransform().rotation.pitch * pi/180, vehicle->GetTransform().rotation.roll * pi/180)) - poseRef;
		drawCar(truePose, 0,  Color(1,0,0), 0.7, viewer);
		double theta = truePose.rotation.yaw;
		double stheta = control.steer * pi/4 + theta;
		viewer->removeShape("steer");
		renderRay(viewer, Point(truePose.position.x+2*cos(theta), truePose.position.y+2*sin(theta),truePose.position.z),  Point(truePose.position.x+4*cos(stheta), truePose.position.y+4*sin(stheta),truePose.position.z), "steer", Color(0,1,0));


		ControlState accuate(0, 0, 1);
		if(cs.size() > 0){
			accuate = cs.back();
			cs.clear();

			Accuate(accuate, control);
			vehicle->ApplyControl(control);
		}

  		viewer->spinOnce ();
		
		if(!new_scan){
			
			new_scan = true;
			std::cout << "New scan received" << std::endl;

			// TODO: (Filter scan using voxel filter)
			float resolution = 0.1;
			pcl::VoxelGrid<PointT> voxel_filter;
			voxel_filter.setInputCloud(scanCloud);//scanCloud);
			// voxel_filter.setLeafSize(0.1f, 0.1f, 0.1f); 
			voxel_filter.setLeafSize(resolution, resolution, resolution); // 0.1m x 0.1m x 0.1m


			// FilterCloud is a pointer to a new PointCloud object that is going to hold the filtered points
			// voxel_filter.filter(*FilterCloud) is going to filter the points in scanCloud and store the result in FilterCloud
			// The filtered points are going to be those that are not too close together, i.e. those that are at least 0.1m away from each other
			typename pcl::PointCloud<PointT>::Ptr FilterCloud(new pcl::PointCloud<PointT>);
			voxel_filter.filter(*FilterCloud);

			// TODO: Find pose transform by using ICP or NDT matching
			//pose = ....
			Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
			// check usage of ICP or NDT algorithm
			if(choose_ICP){
				//using ICP algorithm
				transformation = ICP(mapCloud, FilterCloud, pose);
			}
			else
			{
				//Using NDT algorithm
				transformation = NDT(mapCloud, FilterCloud, pose);
			}
			pose = getPose(transformation);

			// TODO: Transform scan so it aligns with ego's actual pose and render that scan
			// pcl::transformPointCloud(*alignedCloud, *scanCloud, transformation);
			// corrected is a pointer to a new PointCloud object that is going to hold the filtered points
			PointCloudT::Ptr corrected(new PointCloudT); 
			// scanCloud is a pointer to a new PointCloud object that is going to hold the filtered points
			pcl::transformPointCloud(*FilterCloud, *corrected, transformation);
			viewer->removePointCloud("scan");

			pclCloud = *corrected;
			viewer->addPointCloud<pcl::PointXYZ>(corrected, "scan");

			// TODO: Change `scanCloud` below to your transformed scan
			renderPointCloud(viewer, scanCloud, "scan", Color(1, 0, 0));

			viewer->removeAllShapes();
			drawCar(pose, 1,  Color(0,1,0), 0.35, viewer);
          
          	double poseError = sqrt( (truePose.position.x - pose.position.x) * (truePose.position.x - pose.position.x) + (truePose.position.y - pose.position.y) * (truePose.position.y - pose.position.y) );
			if(poseError > maxError)
				maxError = poseError;
			double distDriven = sqrt( (truePose.position.x) * (truePose.position.x) + (truePose.position.y) * (truePose.position.y) );
			viewer->removeShape("maxE");
			viewer->addText("Max Error: "+to_string(maxError)+" m", 200, 100, 32, 1.0, 1.0, 1.0, "maxE",0);
			viewer->removeShape("derror");
			viewer->addText("Pose error: "+to_string(poseError)+" m", 200, 150, 32, 1.0, 1.0, 1.0, "derror",0);
			viewer->removeShape("dist");
			viewer->addText("Distance: "+to_string(distDriven)+" m", 200, 200, 32, 1.0, 1.0, 1.0, "dist",0);

			if(maxError > 1.2 || distDriven >= 170.0 ){
				viewer->removeShape("eval");
			if(maxError > 1.2){
				viewer->addText("Try Again", 200, 50, 32, 1.0, 0.0, 0.0, "eval",0);
			}
			else{
				viewer->addText("Passed!", 200, 50, 32, 0.0, 1.0, 0.0, "eval",0);
			}
		}

			pclCloud.points.clear();
		}
  	}
	return 0;
}