This directory contains code for a command line tool that uniformly samples a point cloud on a mesh. It is a modified version of `pcl_mesh_sampling`. To use it:
1. Install [CMake](https://cmake.org/download/), [PCL](http://pointclouds.org/downloads/) and [VTK](https://vtk.org/download/).
2. Make a build directory: `makedir build & cd build`.
3. Build the code by running `cmake ..` and then `make`.
4. Run `./mesh_sampling` to see the command line usage.