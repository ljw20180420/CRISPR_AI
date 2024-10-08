# Install script for directory: /home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin" TYPE EXECUTABLE FILES "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelmap")
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmap")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin" TYPE EXECUTABLE FILES "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelgen")
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin" TYPE EXECUTABLE FILES "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelgentarget")
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgentarget")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin" TYPE EXECUTABLE FILES "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelgen_i1")
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelgen_i1")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin" TYPE EXECUTABLE FILES "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/indelmh")
  if(EXISTS "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/bin/indelmh")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/ljw/wuqiang/sx/SelfTarget/indel_analysis/indelmap/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
