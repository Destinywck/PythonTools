dg-master

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

# 指定cmake版本
cmake_minimum_required(VERSION 2.8)

# 指定项目名称 
project(dg)

# 在当前文件中包含另一个CMake脚本文件的内容,用来载入CMakeLists.txt文件，也用于载入预定义的cmake模块.
include(CTest)

# 头文件目录
include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/src )

# 源文件目录
# AUX_SOURCE_DIRECTORY(src DIR_SRCS)

# 添加要编译的可执行文件
# ADD_EXECUTABLE(${PROJECT_NAME} ${TEST_MATH})

# set()
# 设置环境变量，编译用到的源文件全部都要放到这里，否则编译能够通过，但是执行的时候会出现各种问题，比如"symbol lookup error xxxxx , undefined symbol".
# we need C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 
OPTION(LLVM_DG "Support for LLVM Dependency graph" ON)
OPTION(ENABLE_CFG "Add support for CFG edges to the graph" ON)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if (LLVM_DG)
	# for llvm dg we need cfg and postdom edges
	if (NOT ENABLE_CFG)
		message(STATUS "Enabling CFG edges due to llvm dg")
	endif()

	set(ENABLE_CFG ON)

	find_package(LLVM REQUIRED CONFIG)

	message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
	message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
	# 可能有问题
	message(STATUS "LLVM binaries: ${LLVM_TOOLS_BINARY_DIR}")

	# 可能有问题
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_DIR}")
	set(LLVMConfig "LLVMConfig" "${CMAKE_MODULE_PATH}\\LLVMConfig.cmake")
	set(AddLLVM "AddLLVM" "${CMAKE_MODULE_PATH}\\AddLLVM.cmake")
	message(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")
	message(STATUS "LLVMConfig: ${LLVMConfig}")
	message(STATUS "AddLLVM: ${AddLLVM}")
	include(LLVMConfig)
	include(AddLLVM)

	message(STATUS "LLVM include dir: ${LLVM_INCLUDE_DIRS}")
	# 可能有问题
	message(STATUS "LLVM libraries dir: ${LLVM_LIBRARY_DIRS}")
	message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")

	include_directories(${LLVM_INCLUDE_DIRS})
	link_directories(${LLVM_LIBRARY_DIRS})

	# if we were provided a path to custom sources
	# use that path
	if (LLVM_SRC_PATH)
	  include_directories(${LLVM_SRC_PATH}\\include)
	  message(STATUS "Looking for headers in given: ${LLVM_SRC_PATH}\\include")
	else()
	  include_directories(${LLVM_INCLUDE_DIRS})
	  message(STATUS "Looking for headers in: ${LLVM_INCLUDE_DIRS}")
	endif()

	# if we were provided a path to custom build directory
	# use that
	if (LLVM_BUILD_PATH)
	  link_directories(${LLVM_BUILD_PATH}\\lib)

	  # llvm-config.h
	  include_directories("${LLVM_BUILD_PATH}/include")
	  message(STATUS "Looking for libraries in given: ${LLVM_BUILD_PATH}\\lib")
	else()
	  link_directories(${LLVM_LIBRARY_DIRS})
	  message(STATUS "Looking for libraries in: ${LLVM_LIBRARY_DIRS}")
	endif(LLVM_BUILD_PATH)

	add_definitions(${LLVM_DEFINITIONS})
	add_definitions(-DHAVE_LLVM)

	# 可能有问题
	if (${LLVM_PACKAGE_VERSION} VERSION_GREATER "3.4")
		llvm_map_components_to_libnames(llvm_core core)
		llvm_map_components_to_libnames(llvm_irreader irreader)
		llvm_map_components_to_libnames(llvm_bitwriter bitwriter)
		llvm_map_components_to_libnames(llvm_analysis analysis)
		llvm_map_components_to_libnames(llvm_support support)
	else()
		llvm_map_components_to_libraries(llvm_core core)
		llvm_map_components_to_libraries(llvm_irreader irreader)
		llvm_map_components_to_libraries(llvm_bitwriter bitwriter)
		llvm_map_components_to_libraries(llvm_analysis analysis)
		llvm_map_components_to_libraries(llvm_support support)
	endif()
endif(LLVM_DG)

if (ENABLE_CFG)
	add_definitions(-DENABLE_CFG)
endif()

message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")

# explicitly add -std=c++11 and -fno-rtti
# we have CMAKE_CXX_STANDARD, but for some reason it does not
# put the -std=c++11 or -std=gnu++11 to the flags
# on some systems. For the -fno-rtti: LLVM still got problems
# with turning RTTI off...
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -std=c++11 -Wall -Wextra")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -std=c++11")

if (USE_CLANG_SANITIZERS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined -fsanitize=address")
endif()

# Debug Release RelWithDebInfo MinSizeRel.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_definitions(-DDEBUG_ENABLED)
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	message(STATUS
		"Using compilation flags: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
endif ()

message(STATUS "Additional compilation flags: ${CMAKE_CXX_FLAGS}")

include(GNUInstallDirs)
message(STATUS "GNUInstallDirs: ${GNUInstallDirs}")
message(STATUS "CMAKE_INSTALL_LIBDIR: \"${CMAKE_INSTALL_LIBDIR}\"")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: \"${CMAKE_INSTALL_INCLUDEDIR}\"")

add_subdirectory(src)
add_subdirectory(tests EXCLUDE_FROM_ALL)
add_subdirectory(tools)

src

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

if (LLVM_DG)

add_library(PTA SHARED
	analysis/SubgraphNode.h
	analysis/Offset.h
	analysis/Offset.cpp
	analysis/PointsTo/Pointer.h
	analysis/PointsTo/Pointer.cpp
	analysis/PointsTo/PointerSubgraph.h
	analysis/PointsTo/PointerAnalysis.h
	analysis/PointsTo/PointerAnalysis.cpp
	analysis/PointsTo/PointsToFlowInsensitive.h
	analysis/PointsTo/PointsToFlowSensitive.h
	analysis/PointsTo/PointerSubgraphValidator.h
	analysis/PointsTo/PointerSubgraphValidator.cpp
)

add_library(RD SHARED
	analysis/SubgraphNode.h
	analysis/Offset.h
	analysis/Offset.cpp
	analysis/ReachingDefinitions/ReachingDefinitions.h
	analysis/ReachingDefinitions/ReachingDefinitions.cpp
	analysis/ReachingDefinitions/RDMap.h
	analysis/ReachingDefinitions/RDMap.cpp
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFI.h
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFI.cpp
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFS.h
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFS.cpp
)

add_library(LLVMpta SHARED
	llvm/MemAllocationFuncs.cpp
	llvm/analysis/PointsTo/PointsTo.h
	llvm/analysis/PointsTo/PointerSubgraph.h
	llvm/analysis/PointsTo/PointerSubgraph.cpp
	llvm/analysis/PointsTo/PointerSubgraphValidator.h
	llvm/analysis/PointsTo/PointerSubgraphValidator.cpp
	llvm/analysis/PointsTo/Structure.cpp
	llvm/analysis/PointsTo/Globals.cpp
)

target_link_libraries(LLVMpta PUBLIC PTA)

add_library(LLVMrd SHARED
	llvm/analysis/ReachingDefinitions/ReachingDefinitions.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilder.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderSemisparse.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderSemisparse.cpp
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderDense.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderDense.cpp
)
target_link_libraries(LLVMrd
			PUBLIC LLVMpta
			PUBLIC RD)

add_library(LLVMdg SHARED
	BBlock.h
	Node.h
	DependenceGraph.h
	ADT/DGContainer.h
	# -- LLVM
	llvm/LLVMNode.h
	llvm/LLVMNode.cpp
	llvm/LLVMDependenceGraph.h
	llvm/LLVMDependenceGraph.cpp
	llvm/LLVMDGVerifier.h
	llvm/LLVMDGVerifier.cpp
	llvm/Slicer.h
	llvm/llvm-utils.h
	# -- LLVM analysis
	llvm/analysis/PostDominators.cpp
	llvm/analysis/DefUse.h
	llvm/analysis/DefUse.cpp
)

target_link_libraries(LLVMdg
			PUBLIC LLVMpta
			PUBLIC LLVMrd)
			
message(STATUS "DESTINATION: ${CMAKE_INSTALL_LIBDIR}")

if(WIN32)
	install(TARGETS LLVMdg LLVMpta LLVMrd PTA RD
		RUNTIME DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
	install(TARGETS LLVMdg LLVMpta LLVMrd PTA RD
		LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()

message(STATUS "DESTINATION: ${CMAKE_INSTALL_INCLUDEDIR}")

install(FILES
	ADT/Queue.h
        ADT/DGContainer.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/ADT/)
install(FILES
	analysis/Offset.h
	analysis/SCC.h
	analysis/SubgraphNode.h
        analysis/Analysis.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/analysis/)
install(FILES
        analysis/ControlExpression/ControlExpression.h
        analysis/ControlExpression/CENode.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/analysis/ControlExpression/)
install(FILES
	analysis/PointsTo/PointerAnalysis.h
	analysis/PointsTo/Pointer.h
	analysis/PointsTo/PointerSubgraph.h
	analysis/PointsTo/PointerSubgraphValidator.h
	analysis/PointsTo/PointsToFlowInsensitive.h
	analysis/PointsTo/PointsToFlowSensitive.h
	analysis/PointsTo/PointsToWithInvalidate.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/analysis/PointsTo/)
install(FILES
	llvm/llvm-utils.h
	llvm/MemAllocationFuncs.h
	llvm/LLVMNode.h
	llvm/LLVMDependenceGraph.h
	llvm/LLVMDGAssemblyAnnotationWriter.h
	llvm/LLVMDGVerifier.h
	llvm/Slicer.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/llvm/)
install(FILES
	llvm/analysis/PointsTo/PointerSubgraph.h
	llvm/analysis/PointsTo/PointerSubgraphValidator.h
	llvm/analysis/PointsTo/PointsTo.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/llvm/analysis/PointsTo/)
install(FILES
        Node.h
        DGParameters.h
        BBlock.h
        DependenceGraph.h
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/llvm-dg/)

endif(LLVM_DG)


tools

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")

message(STATUS "LLVM INFO: ")
message(STATUS "--" "llvm_analysis: ${llvm_analysis}")
message(STATUS "--" "llvm_irreader: ${llvm_irreader}")
message(STATUS "--" "llvm_bitwriter: ${llvm_bitwriter}")
message(STATUS "--" "llvm_core: ${llvm_core}")
message(STATUS "--" "llvm_support: ${llvm_support}")

if (LLVM_DG)
	# generate a git-version.h with a HEAD commit hash tag
	# (if it changed)
	message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
	add_custom_target(
		gitversion ALL sh ${CMAKE_CURRENT_SOURCE_DIR}/git-version.sh
	)
	message(STATUS "CMAKE_CURRENT_BINARY_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
	include_directories(${CMAKE_CURRENT_BINARY_DIR})

	add_executable(llvm-dg-dump llvm-dg-dump.cpp)
	target_link_libraries(llvm-dg-dump
				PRIVATE LLVMdg
				PRIVATE ${llvm_support}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_bitwriter}
				PRIVATE ${llvm_core})

	add_executable(llvm-slicer llvm-slicer.cpp)
	target_link_libraries(llvm-slicer PRIVATE LLVMdg)
	target_link_libraries(llvm-slicer
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_bitwriter}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_support}
				PRIVATE ${llvm_core})
	add_dependencies(llvm-slicer gitversion)

	add_executable(llvm-ps-dump llvm-ps-dump.cpp)
	target_link_libraries(llvm-ps-dump PRIVATE LLVMpta)
	target_link_libraries(llvm-ps-dump
				PRIVATE ${llvm_core}
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_support})

	add_executable(llvm-pta-compare llvm-pta-compare.cpp)
	target_link_libraries(llvm-pta-compare PRIVATE LLVMpta)
	target_link_libraries(llvm-pta-compare
				PRIVATE ${llvm_core}
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_support})

	add_executable(llvm-rd-dump llvm-rd-dump.cpp)
	target_link_libraries(llvm-rd-dump PRIVATE LLVMrd)
	target_link_libraries(llvm-rd-dump
				PRIVATE ${llvm_core}
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_support})

	add_executable(llvm-to-source llvm-to-source.cpp)
	target_link_libraries(llvm-to-source
				PRIVATE ${llvm_core}
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_support})

	install(TARGETS llvm-dg-dump llvm-slicer
		RUNTIME DESTINATION bin)
endif (LLVM_DG)

include_directories(../src)


