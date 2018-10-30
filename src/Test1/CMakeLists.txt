set_property(GLOBAL PROPERTY USE_FOLDERS On)
cmake_minimum_required(VERSION 2.8)
project(dg)

include(CTest)

# we need C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED on)

# for llvm dg we need cfg and postdom edges
OPTION(LLVM_DG "Support for LLVM Dependency graph" ON)
OPTION(ENABLE_CFG "Add support for CFG edges to the graph" ON)

if (LLVM_DG)

	find_package(LLVM REQUIRED CONFIG)

	message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
	message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
	message(STATUS "LLVM binaries: ${LLVM_TOOLS_BINARY_DIR}")

	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${LLVM_DIR}")
	include(LLVMConfig)
	include(AddLLVM)

	message(STATUS "LLVM include dir: ${LLVM_INCLUDE_DIRS}")
	message(STATUS "LLVM libraries dir: ${LLVM_LIBRARY_DIRS}")
	message(STATUS "LLVM definitions: ${LLVM_DEFINITIONS}")
	
	link_directories(${LLVM_LIBRARY_DIRS})
	message(STATUS "Looking for libraries in: ${LLVM_LIBRARY_DIRS}")

	add_definitions(${LLVM_DEFINITIONS})
	add_definitions(-DHAVE_LLVM)

	if (${LLVM_PACKAGE_VERSION} VERSION_GREATER "3.4")
		llvm_map_components_to_libnames(llvm_core core)
		llvm_map_components_to_libnames(llvm_passes passes)
		llvm_map_components_to_libnames(llvm_object object)
		llvm_map_components_to_libnames(llvm_asmparser asmparser)
		llvm_map_components_to_libnames(llvm_irreader irreader)
		llvm_map_components_to_libnames(llvm_bitwriter bitwriter)
		llvm_map_components_to_libnames(llvm_bitreader bitreader)
		llvm_map_components_to_libnames(llvm_analysis analysis)
		llvm_map_components_to_libnames(llvm_support support)
	else()
		llvm_map_components_to_libraries(llvm_core core)
		llvm_map_components_to_libraries(llvm_passes passes)
		llvm_map_components_to_libraries(llvm_object object)
		llvm_map_components_to_libraries(llvm_asmparser asmparser)
		llvm_map_components_to_libraries(llvm_irreader irreader)
		llvm_map_components_to_libraries(llvm_bitwriter bitwriter)
		llvm_map_components_to_libraries(llvm_bitreader bitreader)
		llvm_map_components_to_libraries(llvm_analysis analysis)
		llvm_map_components_to_libraries(llvm_support support)
	endif()
	
	message(STATUS 
	"LLVM Libraries: \n"
	"--   ${llvm_core}\n"
	"--   ${llvm_irreader}\n"
	"--   ${llvm_bitwriter}\n"
	"--   ${llvm_analysis}\n"
	"--   ${llvm_support}\n"
	"--   ${llvm_passes}\n"
	"--   ${llvm_object}\n"
	"--   ${llvm_asmparser}\n"
	"--   ${llvm_bitreader}")
endif(LLVM_DG)

if (ENABLE_CFG)
	add_definitions(-DENABLE_CFG)
endif()

# path for boost
if (NOT BOOST_INCLUDEDIR)
	message(FATAL_ERROR "BOOST_INCLUDEDIR not specified!")
endif(NOT BOOST_INCLUDEDIR)

if (BOOST_LIBRARYDIR)
	# 设置boost使用静态链接
	set(Boost_USE_STATIC_LIBS ON)
	# 设置需要的boost 组件
	set(BOOST_COMPONENTS regex graph)
	# 使用cmake find_package 查找boost库位置
	find_package(Boost REQUIRED COMPONENTS ${BOOST_COMPONENTS})
	
	if(NOT Boost_FOUND)
		message(FATAL_ERROR "Not found Boost")
	endif()
else()
	message(FATAL_ERROR "BOOST_LIBRARYDIR not specified!")
endif(BOOST_LIBRARYDIR)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")	
message(STATUS "Using compiler: ${CMAKE_CXX_COMPILER}")

# explicitly add -std=c++11 and -fno-rtti
# we have CMAKE_CXX_STANDARD, but for some reason it does not
# put the -std=c++11 or -std=gnu++11 to the flags
# on some systems. For the -fno-rtti: LLVM still got problems
# with turning RTTI off...
if (WIN32)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
else()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti -std=c++11 -Wall -Wextra")
endif(WIN32)

if (USE_CLANG_SANITIZERS)
	message(WARNING "USE_CLANG_SANITIZERS has been renamed to USE_SANITIZERS")
	set(USE_SANITIZERS on)
endif()

if (USE_SANITIZERS)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined,address")
	add_definitions(-DUSING_SANITIZERS)
endif()

# Debug Release RelWithDebInfo MinSizeRel.
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
	add_definitions(-DDEBUG_ENABLED)
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_DEBUG}")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_RELEASE}")
elseif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
elseif (CMAKE_BUILD_TYPE STREQUAL "MinSizeRel")
	message(STATUS "Using compilation flags: ${CMAKE_CXX_FLAGS_MINSIZEREL}")
endif ()

message(STATUS "Additional compilation flags: ${CMAKE_CXX_FLAGS}")

# 设置可执行文件输出目录
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
# 设置库文件输出目录
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# 设置Debug版本的Release版本可执行文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/Debug/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/Release/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PROJECT_BINARY_DIR}/RelWithDebInfo/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${PROJECT_BINARY_DIR}/MinSizeRel/bin)
# 设置Debug版本的Release版本库文件的输出目录
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/Debug/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/Release/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${PROJECT_BINARY_DIR}/RelWithDebInfo/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_MINSIZEREL ${PROJECT_BINARY_DIR}/MinSizeRel/lib)

include(GNUInstallDirs)
message(STATUS "CMAKE_INSTALL_LIBDIR: \"${CMAKE_INSTALL_LIBDIR}\"")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: \"${CMAKE_INSTALL_INCLUDEDIR}\"")

add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(tests)

install(DIRECTORY include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})




tools:

include_directories(${CMAKE_SOURCE_DIR}/include)
# these Tools can access the private headers
include_directories(${CMAKE_SOURCE_DIR}/lib)

include_directories(${LLVM_INCLUDE_DIRS})
message(STATUS "Looking for headers in: ${LLVM_INCLUDE_DIRS}")

include_directories(${BOOST_INCLUDEDIR})
message(STATUS "Looking for headers in: ${BOOST_INCLUDEDIR}")

if (LLVM_DG)
		
	add_executable(llvm-dg-dump llvm-dg-dump.cpp)
	set_target_properties(llvm-dg-dump PROPERTIES FOLDER "Tools")
	target_link_libraries(llvm-dg-dump PRIVATE XMLParser)
	target_link_libraries(llvm-dg-dump PRIVATE Utils)
	target_link_libraries(llvm-dg-dump PRIVATE LLVMdg)
	target_link_libraries(llvm-dg-dump
				PRIVATE ${llvm_support}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_bitwriter}
				PRIVATE ${llvm_core}
				PRIVATE ${llvm_passes}
				PRIVATE ${llvm_object}
				PRIVATE ${llvm_asmparser}
				PRIVATE ${llvm_bitreader})

	add_executable(llvm-slicer llvm-slicer.cpp)
	set_target_properties(llvm-slicer PROPERTIES FOLDER "Tools")
	target_link_libraries(llvm-slicer PRIVATE XMLParser)
	target_link_libraries(llvm-slicer PRIVATE Utils)
	target_link_libraries(llvm-slicer PRIVATE LLVMdg)
	target_link_libraries(llvm-slicer
				PRIVATE ${llvm_irreader}
				PRIVATE ${llvm_bitwriter}
				PRIVATE ${llvm_analysis}
				PRIVATE ${llvm_support}
				PRIVATE ${llvm_core}
				PRIVATE ${llvm_passes}
				PRIVATE ${llvm_object}
				PRIVATE ${llvm_asmparser}
				PRIVATE ${llvm_bitreader})

	add_executable(iso-predictor iso-predictor.cpp)
	set_target_properties(iso-predictor PROPERTIES FOLDER "Tools")
	target_link_libraries(iso-predictor
				PRIVATE BSubgraphIso
				PRIVATE Utils
				PRIVATE XMLParser)
	
	install(TARGETS llvm-dg-dump llvm-slicer iso-predictor
		RUNTIME DESTINATION bin)
endif (LLVM_DG)


lib:
include_directories(${CMAKE_SOURCE_DIR}/lib)
include_directories(${CMAKE_SOURCE_DIR}/include)

include_directories(${LLVM_INCLUDE_DIRS})
message(STATUS "Looking for headers in: ${LLVM_INCLUDE_DIRS}")

include_directories(${BOOST_INCLUDEDIR})
message(STATUS "Looking for headers in: ${BOOST_INCLUDEDIR}")
	
add_library(DGAnalysis STATIC
	${CMAKE_SOURCE_DIR}/include/analysis/Offset.h
	${CMAKE_SOURCE_DIR}/include/ADT/DGContainer.h
	${CMAKE_SOURCE_DIR}/include/ADT/Bitvector.h
	${CMAKE_SOURCE_DIR}/include/ADT/Bits.h
	${CMAKE_SOURCE_DIR}/include/ADT/NumberSet.h

	analysis/Offset.cpp
)
set_target_properties(DGAnalysis PROPERTIES FOLDER "Librarys")

add_library(PTA STATIC
	${CMAKE_SOURCE_DIR}/include/analysis/SubgraphNode.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/Pointer.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/PointsToSet.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/MemoryObject.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/PointerSubgraph.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/PointerAnalysis.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/PointerAnalysisFI.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/PointerAnalysisFS.h
	${CMAKE_SOURCE_DIR}/include/analysis/PointsTo/PointerSubgraphValidator.h

	analysis/PointsTo/Pointer.cpp
	analysis/PointsTo/PointerAnalysis.cpp
	analysis/PointsTo/PointerSubgraphValidator.cpp
)
set_target_properties(PTA PROPERTIES FOLDER "Librarys")
target_link_libraries(PTA PUBLIC DGAnalysis)

add_library(RD STATIC
	${CMAKE_SOURCE_DIR}/include/analysis/ReachingDefinitions/ReachingDefinitions.h
	${CMAKE_SOURCE_DIR}/include/analysis/ReachingDefinitions/RDMap.h

	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFI.h
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFS.h

	analysis/ReachingDefinitions/RDMap.cpp
	analysis/ReachingDefinitions/ReachingDefinitions.cpp
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFI.cpp
	analysis/ReachingDefinitions/Srg/MarkerSRGBuilderFS.cpp
)
set_target_properties(RD PROPERTIES FOLDER "Librarys")
target_link_libraries(RD PUBLIC DGAnalysis)


if (LLVM_DG)

add_library(LLVMpta STATIC
	${CMAKE_SOURCE_DIR}/include/llvm/analysis/PointsTo/LLVMPointerSubgraph.h

	llvm/MemAllocationFuncs.h
	llvm/MemAllocationFuncs.cpp
	llvm/analysis/PointsTo/LLVMPointerSubgraphValidator.h
	llvm/analysis/PointsTo/LLVMPointerSubgraph.cpp
	llvm/analysis/PointsTo/LLVMPointerSubgraphValidator.cpp
	llvm/analysis/PointsTo/Structure.cpp
	llvm/analysis/PointsTo/Globals.cpp
	llvm/analysis/PointsTo/Constants.cpp
	llvm/analysis/PointsTo/Instructions.cpp
	llvm/analysis/PointsTo/Calls.cpp
)
set_target_properties(LLVMpta PROPERTIES FOLDER "Librarys")
target_link_libraries(LLVMpta PUBLIC PTA)

add_library(LLVMrd STATIC
	llvm/MemAllocationFuncs.h
	llvm/MemAllocationFuncs.cpp
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderSemisparse.cpp
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderDense.cpp
	
	${CMAKE_SOURCE_DIR}/include/llvm/analysis/ReachingDefinitions/LLVMReachingDefinitions.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilder.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderSemisparse.h
	llvm/analysis/ReachingDefinitions/LLVMRDBuilderDense.h
)
set_target_properties(LLVMrd PROPERTIES FOLDER "Librarys")
target_link_libraries(LLVMrd
			PUBLIC LLVMpta
			PUBLIC RD)

add_library(LLVMdg STATIC
	${CMAKE_SOURCE_DIR}/include/BBlock.h
	${CMAKE_SOURCE_DIR}/include/Node.h
	${CMAKE_SOURCE_DIR}/include/DependenceGraph.h
	${CMAKE_SOURCE_DIR}/include/llvm/LLVMNode.h
	${CMAKE_SOURCE_DIR}/include/llvm/LLVMDependenceGraph.h
	${CMAKE_SOURCE_DIR}/include/llvm/slicer.h
	${CMAKE_SOURCE_DIR}/include/llvm/analysis/DefUse/DefUse.h

	llvm/LLVMDGVerifier.h
	llvm/llvm-utils.h

	llvm/LLVMNode.cpp
	llvm/LLVMDependenceGraph.cpp
	llvm/LLVMDGVerifier.cpp
	llvm/analysis/Dominators/PostDominators.cpp
	llvm/analysis/DefUse/DefUse.cpp
)
set_target_properties(LLVMdg PROPERTIES FOLDER "Librarys")
target_link_libraries(LLVMdg
			PUBLIC LLVMpta
			PUBLIC LLVMrd)
			
add_library(Utils STATIC
	${CMAKE_SOURCE_DIR}/include/Utils/Logger.h
	${CMAKE_SOURCE_DIR}/include/Utils/Strings.h
	${CMAKE_SOURCE_DIR}/include/Utils/Systems.h
	${CMAKE_SOURCE_DIR}/include/Utils/Timer.h
	
	Utils/Logger.cpp
	Utils/Strings.cpp
	Utils/Systems.cpp
	Utils/Timer.cpp
)
set_target_properties(Utils PROPERTIES FOLDER "Librarys")

add_library(LLParser STATIC
	${CMAKE_SOURCE_DIR}/include/LLParser/asmParser/instFlags.h
	${CMAKE_SOURCE_DIR}/include/LLParser/asmParser/instParser.h
	${CMAKE_SOURCE_DIR}/include/LLParser/asmParser/irParser.h
	${CMAKE_SOURCE_DIR}/include/LLParser/asmParser/stringParser.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/allocaInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/bitcastInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/branchInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/callInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/callInstFamily.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/catchInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/divideInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/fcmpInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/icmpInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/getelementptrInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/invokeInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/loadInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/phiInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/inst/storeInst.h
	${CMAKE_SOURCE_DIR}/include/LLParser/ir/LabelValue.h
	${CMAKE_SOURCE_DIR}/include/LLParser/ir/shadow.h
	
	LLParser/asmParser/instFlags.cpp
	LLParser/asmParser/instParser.cpp
	LLParser/asmParser/irParser.cpp
	LLParser/asmParser/stringParser.cpp
	LLParser/inst/allocaInst.cpp
	LLParser/inst/bitcastInst.cpp
	LLParser/inst/branchInst.cpp
	LLParser/inst/callInst.cpp
	LLParser/inst/callInstFamily.cpp
	LLParser/inst/catchInst.cpp
	LLParser/inst/divideInst.cpp
	LLParser/inst/fcmpInst.cpp
	LLParser/inst/icmpInst.cpp
	LLParser/inst/getelementptrInst.cpp
	LLParser/inst/invokeInst.cpp
	LLParser/inst/loadInst.cpp
	LLParser/inst/phiInst.cpp
	LLParser/inst/storeInst.cpp
	LLParser/ir/LabelValue.cpp
	LLParser/ir/shadow.cpp
)
set_target_properties(LLParser PROPERTIES FOLDER "Librarys")
target_link_libraries(LLParser PUBLIC Utils)

add_library(SubgraphIso STATIC
	${CMAKE_SOURCE_DIR}/include/GraphIso/graph/Container.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/graph/DotGraphReader.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/graph/Edge.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/graph/Graph.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/graph/Vertex.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/vf2/State.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/vf2/VF2.h

	GraphIso/graph/DotGraphReader.cpp
	GraphIso/graph/Edge.cpp
	GraphIso/graph/Graph.cpp
	GraphIso/graph/Vertex.cpp
	GraphIso/vf2/State.cpp
	GraphIso/vf2/VF2.cpp
)
set_target_properties(SubgraphIso PROPERTIES FOLDER "Librarys")
target_link_libraries(SubgraphIso PUBLIC Utils)

add_library(XMLParser STATIC
	${CMAKE_SOURCE_DIR}/include/xml/slicer/XmlSlicer.h
	${CMAKE_SOURCE_DIR}/include/xml/pugixml.hpp
	
	${CMAKE_SOURCE_DIR}/include/xml/pugixml.cpp
	xml/slicer/XmlSlicer.cpp
)
set_target_properties(XMLParser PROPERTIES FOLDER "Librarys")
target_link_libraries(XMLParser PUBLIC Utils)
	
add_library(BSubgraphIso STATIC
	${BOOST_INCLUDEDIR}/boost/graph/vf2_sub_graph_iso.hpp
	${BOOST_INCLUDEDIR}/boost/graph/graphviz.hpp
	${BOOST_INCLUDEDIR}/boost/graph/graph_traits.hpp
	${BOOST_INCLUDEDIR}/boost/graph/adjacency_list.hpp
	${BOOST_INCLUDEDIR}/boost/pending/property.hpp
	${CMAKE_SOURCE_DIR}/include/GraphIso/boost/GraphReader.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/boost/PropertyMap.h
	${CMAKE_SOURCE_DIR}/include/GraphIso/boost/SubgraphIso.h
	
	GraphIso/boost/GraphReader.cpp
	GraphIso/boost/PropertyMap.cpp
	GraphIso/boost/SubgraphIso.cpp
)
set_target_properties(BSubgraphIso PROPERTIES FOLDER "Librarys")
target_link_libraries(BSubgraphIso 
	PUBLIC LLParser
	PUBLIC Utils
	PUBLIC ${Boost_LIBRARIES})
	
if(WIN32)
  install(TARGETS Utils LLVMdg LLVMpta LLVMrd PTA RD DGAnalysis 
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
else()
  install(TARGETS Utils LLVMdg LLVMpta LLVMrd PTA RD DGAnalysis 
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()  

endif(LLVM_DG)


