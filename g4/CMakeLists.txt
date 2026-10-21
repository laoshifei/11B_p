# Setup the project
cmake_minimum_required(VERSION 3.16...3.21)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

project(HB)

#----------------------------------------------------------------------------
# Find Geant4 package, activating all available UI and Vis drivers by default.
# Set WITH_GEANT4_UIVIS=OFF to build a batch-mode-only executable.
option(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
if(WITH_GEANT4_UIVIS)
  find_package(Geant4 REQUIRED ui_all vis_all)
else()
  find_package(Geant4 REQUIRED)
endif()

# Setup Geant4 include directories and compile definitions.
include(${Geant4_USE_FILE})

# Find ROOT.
find_package(ROOT QUIET)
if(ROOT_FOUND)
  message(STATUS "ROOT found.")
  add_definitions(-DG4ANALYSIS_USE_ROOT)
else()
  message(FATAL_ERROR "ROOT not found.")
endif()

# Project include directories.
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${Geant4_INCLUDE_DIR}
  ${ROOT_INCLUDE_DIRS}
)

# Source files.
file(GLOB sources CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB headers CONFIGURE_DEPENDS ${PROJECT_SOURCE_DIR}/include/*.hh)

# Executable.
add_executable(HB HB.cc ${sources} ${headers})
target_link_libraries(HB ${Geant4_LIBRARIES} ${ROOT_LIBRARIES})

# Explicitly requested validation target; uses the production 162 amplitude.
add_executable(validate_162_bose EXCLUDE_FROM_ALL
  validation/validate_162_bose_driver.cc src/H11B162BoseModel.cc
  src/CoulombPenetrability.cc src/CoulombPenetrabilityData.cc)
