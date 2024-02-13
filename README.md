# OxVoxNNS - Oxidised Voxelised Nearest Neighbour Search
A performant (for large numbers of points) nearest neighbour search implemented in rust

## TODO
### PyPI
- Investigate building release binaries and pushing with pipx (manual)
- Investigate automated with cibuildwheel

### Performance testing
- Create a function to generate test data, with the following parameters:
  - Number of search points
  - Range of point coordinate values (probably as a scalar, e.g. 15 => XYZ values in range [0, 15])
  - (Harder, do later) Distribution of points, i.e. some way of making the points less evenly spread (this can severely impact the performance of KDTrees)

Useful function: `np.random.random((num_points, 3))`

### Plotting
- Compare KDTree implementations - run the same nearest neighbour search using different libraries and plot results (x-axis: num points, y-axis: processing time)
  - scipy.spatial.KDTree (native python)
  - sklearn.neighbours.KDTree (C++)
  - open3d.core.nns.NearestNeighbourSearch (multithreaded C++)
  - OxVoxNNS (rust)
