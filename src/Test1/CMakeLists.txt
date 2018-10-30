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


