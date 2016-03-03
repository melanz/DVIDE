#--------------------------------------------------------------
# Always use full RPATH (differentiating between the build and install trees)

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")


#--------------------------------------------------------------
# Add executables

cuda_add_executable(FlexibleNet main.cu)
target_link_libraries(FlexibleNet ImplicitBeamsGPU)
add_dependencies(FlexibleNet ImplicitBeamsGPU)

cuda_add_executable(test_edge test_edge.cu)
target_link_libraries(test_edge ImplicitBeamsGPU)
add_dependencies(test_edge ImplicitBeamsGPU)

cuda_add_executable(test_draft test_draft.cu)
target_link_libraries(test_draft ImplicitBeamsGPU)
add_dependencies(test_draft ImplicitBeamsGPU)

cuda_add_executable(test_contact test_contact.cu)
target_link_libraries(test_contact ImplicitBeamsGPU)
add_dependencies(test_contact ImplicitBeamsGPU)

cuda_add_executable(test_shear test_shear.cu)
target_link_libraries(test_shear ImplicitBeamsGPU)
add_dependencies(test_shear ImplicitBeamsGPU)

cuda_add_executable(test_beam test_beam.cu)
target_link_libraries(test_beam ImplicitBeamsGPU)
add_dependencies(test_beam ImplicitBeamsGPU)

cuda_add_executable(test_plate test_plate.cu)
target_link_libraries(test_plate ImplicitBeamsGPU)
add_dependencies(test_plate ImplicitBeamsGPU)

cuda_add_executable(test_hub test_hub.cu)
target_link_libraries(test_hub ImplicitBeamsGPU)
add_dependencies(test_hub ImplicitBeamsGPU)

cuda_add_executable(test_oneBall test_oneBall.cu)
target_link_libraries(test_oneBall ImplicitBeamsGPU)
add_dependencies(test_oneBall ImplicitBeamsGPU)

cuda_add_executable(test_bikeWheel test_bikeWheel.cu)
target_link_libraries(test_bikeWheel ImplicitBeamsGPU)
add_dependencies(test_bikeWheel ImplicitBeamsGPU)

cuda_add_executable(test_compaction test_compaction.cu)
target_link_libraries(test_compaction ImplicitBeamsGPU)
add_dependencies(test_compaction ImplicitBeamsGPU)
