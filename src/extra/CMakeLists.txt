add_executable(test_kpm test_kpm.cpp)
target_link_libraries(test_kpm ${LIBS})

add_executable(test_BdG test_BdG.cpp)
target_link_libraries(test_BdG ${LIBS})

add_executable(conductivity conductivity.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries (conductivity ${LIBS})

add_executable(test_conductivity test_conductivity.cpp $<TARGET_OBJECTS:kondo_base>)
target_link_libraries (test_conductivity ${LIBS})
