# Require CMake 3.15+ (matching scikit-build-core) Use new versions of all
# policies up to CMake 4.0
cmake_minimum_required(VERSION 3.15...4.0)

set(CMAKE_C_COMPILER /usr/bin/gcc-14)
set(CMAKE_CXX_COMPILER /usr/bin/g++-14)
set(CMAKE_CUDA_HOST_COMPILER /usr/bin/gcc-14)

project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
	LANGUAGES CXX CUDA)

find_package(pybind11 CONFIG REQUIRED)

# pysword_mod : function for setting up pybind11 mdouels easily given our format
#---------------------------------------------------------------------------
function(pysword_mod module_name top_level_dir)
	# C
	file(GLOB MOD_C_SOURCES "${top_level_dir}/src/*.cpp")
	file(GLOB MOD_CUDA_SOURCES "${top_level_dir}/src/*.cu")
	python_add_library(${module_name} MODULE csrc/${module_name}/src/main.cpp 
											WITH_SOABI)
	set_property(TARGET ${module_name} PROPERTY CUDA_ARCHITECTURES 80-real)
	set_target_properties(${module_name}
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
	target_sources(${module_name} PRIVATE ${MOD_C_SOURCES})
	target_sources(${module_name} PRIVATE ${MOD_CUDA_SOURCES})
	target_link_libraries(${module_name} PRIVATE pybind11::headers)


	target_include_directories(${module_name}
		PRIVATE
			${top_level_dir}/lib
	)
	target_compile_definitions(${module_name} PRIVATE VERSION_INFO=${PROJECT_VERSION})
	install(TARGETS ${module_name} DESTINATION .)
endfunction()



#---------------------------------------------------------------------------

pysword_mod(gputensor csrc/gputensor)
#pysword_mod(basicmath csrc/basicmath)
